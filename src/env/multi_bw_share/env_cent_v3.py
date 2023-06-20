# add queuing delay into halo
import numpy as np

from util.constants import DEFAULT_QUALITY, REBUF_PENALTY, SMOOTH_PENALTY, VIDEO_BIT_RATE, BUFFER_NORM_FACTOR, \
    BITRATE_WEIGHT, CHUNK_TIL_VIDEO_END_CAP, M_IN_K, PAST_LEN, A_DIM, PAST_LEN, BITRATE_REWARD, PAST_SAT_LOG_LEN, \
    MAX_SAT
from util.encode import encode_other_sat_info
from . import core_cent_time as abrenv
from . import load_trace as load_trace

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6

NUM_AGENTS = None
REWARD_FUNC = None


class ABREnv():
    def __init__(self, random_seed=RANDOM_SEED, num_agents=NUM_AGENTS, reward_func=REWARD_FUNC, train_traces=None):
        self.num_users = num_agents
        global S_INFO
        S_INFO = 9 + 8 * (self.num_users - 1) + (self.num_users - 1) * PAST_SAT_LOG_LEN + MAX_SAT - 2
        # SAT_DIM = num_agents
        # A_SAT = num_agents
        # SAT_DIM = num_agents + 1

        self.is_handover = False

        np.random.seed(random_seed)
        if train_traces:
            all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(train_traces)
        else:
            all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          random_seed=random_seed,
                                          num_agents=self.num_users)

        self.last_bit_rate = [DEFAULT_QUALITY for _ in range(self.num_users)]
        self.buffer_size = [0 for _ in range(self.num_users)]
        self.rebuf = [0 for _ in range(self.num_users)]
        self.video_chunk_size = [0 for _ in range(self.num_users)]
        self.delay = [0 for _ in range(self.num_users)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_users)]
        self.video_chunk_remain = [0 for _ in range(self.num_users)]
        self.next_sat_bw_logs = [[] for _ in range(self.num_users)]
        self.cur_sat_bw_logs = [[] for _ in range(self.num_users)]
        self.connected_time = [{} for _ in range(self.num_users)]
        self.cur_sat_id = [0 for _ in range(self.num_users)]
        self.next_sat_id = [0 for _ in range(self.num_users)]
        self.other_ids = [[] for _ in range(self.num_users)]

        self.state = [np.zeros((S_INFO, PAST_LEN)) for _ in range(self.num_users)]
        self.reward_func = reward_func

    def seed(self, num):
        np.random.seed(num)

    def reset_agent(self, agent):
        bit_rate = [DEFAULT_QUALITY] * self.num_users

        delay, sleep_time, self.buffer_size[agent], rebuf, video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, is_handover, num_of_user_sat, next_sat_bandwidth, next_sat_bw_logs, \
            cur_sat_user_num, next_sat_user_num, cur_sat_bw_logs, connected_time, cur_sat_id, next_sat_id, _, _, _, _, \
        other_sat_users, other_sat_bw_logs, other_buffer_sizes = \
            self.net_env.get_video_chunk(bit_rate[agent], agent, None)

        state = np.roll(self.state[agent], -1, axis=1)

        self.video_chunk_size[agent] = video_chunk_size
        self.delay[agent] = delay
        self.next_video_chunk_sizes[agent] = next_video_chunk_sizes
        self.video_chunk_remain[agent] = video_chunk_remain
        self.next_sat_bw_logs[agent] = next_sat_bw_logs
        self.cur_sat_bw_logs[agent] = cur_sat_bw_logs
        self.connected_time[agent] = connected_time
        self.cur_sat_id[agent] = cur_sat_id
        self.next_sat_id[agent] = next_sat_id

        other_user_sat_decisions, other_sat_num_users, other_sat_bws, cur_user_sat_decisions, other_ids \
            = encode_other_sat_info(self.net_env.sat_decision_log, self.num_users, self.cur_sat_id[agent], self.next_sat_id[agent],
                                    agent, other_sat_users, other_sat_bw_logs, PAST_SAT_LOG_LEN)
        self.other_ids[agent] = other_ids

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[self.last_bit_rate[agent]] / \
                               float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size[agent] / BUFFER_NORM_FACTOR  # 10 sec
        if self.delay[agent] != 0:
            state[2, -1] = float(self.video_chunk_size[agent]) / \
                                   float(self.delay[agent]) / M_IN_K  # kilo byte / ms
        else:
            state[2, -1] = 0
        state[3, -1] = float(self.delay[agent]) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        # state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        if self.next_video_chunk_sizes[agent]:
            state[4, :A_DIM] = np.array(
                [self.next_video_chunk_sizes[agent][index] for index in [0, 2, 4]]) / M_IN_K / M_IN_K  # mega byte
        else:
            state[4, :A_DIM] = [0, 0, 0]
        state[5, -1] = np.minimum(self.video_chunk_remain[agent],
                                          CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        if len(self.next_sat_bw_logs[agent]) < PAST_LEN:
            self.next_sat_bw_logs[agent] = [0] * (PAST_LEN - len(self.next_sat_bw_logs[agent])) + self.next_sat_bw_logs[agent]

        state[6, :PAST_LEN] = np.array(self.next_sat_bw_logs[agent][:PAST_LEN]) / 10

        if len(self.cur_sat_bw_logs[agent]) < PAST_LEN:
            self.cur_sat_bw_logs[agent] = [0] * (PAST_LEN - len(self.cur_sat_bw_logs[agent])) + self.cur_sat_bw_logs[agent]

        state[7, :PAST_LEN] = np.array(self.cur_sat_bw_logs[agent][:PAST_LEN]) / 10
        # if self.is_handover:
        #     state[8, 0:PAST_LEN] = np.zeros((1, PAST_LEN))
        #     state[9, 0:PAST_LEN] = np.zeros((1, PAST_LEN))

        # state[8, -1] = np.array(cur_sat_user_num) / 10
        # state[9, -1] = np.array(next_sat_user_num) / 10
        if self.connected_time[agent] and self.other_ids[agent]:
            state[8, :MAX_SAT] = [float(self.connected_time[agent][self.cur_sat_id[agent]]) / BUFFER_NORM_FACTOR / 10,
                                     float(self.connected_time[agent][self.next_sat_id[agent]]) / BUFFER_NORM_FACTOR / 10,
                                  float(self.connected_time[agent][self.other_ids[agent][0]]) / BUFFER_NORM_FACTOR / 10,
                                  float(self.connected_time[agent][self.other_ids[agent][1]]) / BUFFER_NORM_FACTOR / 10]
        else:
            state[8, :MAX_SAT] = [0] * MAX_SAT

        i = 0
        for u_id in range(self.num_users-1):
            if u_id == agent:
                continue
            state[9 + 8*i, -1] = VIDEO_BIT_RATE[self.last_bit_rate[u_id]] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            state[10 + 8*i, -1] = self.buffer_size[u_id] / BUFFER_NORM_FACTOR  # 10 sec
            if self.delay[u_id] != 0:
                state[11 + 8*i, -1] = float(self.video_chunk_size[u_id]) / \
                    float(self.delay[u_id]) / M_IN_K  # kilo byte / ms
            else:
                state[11 + 8*i, -1] = 0
            state[12 + 8*i, -1] = float(self.delay[u_id]) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            # state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[13 + 8*i, -1] = np.minimum(self.video_chunk_remain[u_id],
                                    CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            if len(self.next_sat_bw_logs[u_id]) < PAST_LEN:
                self.next_sat_bw_logs[u_id] = [0] * (PAST_LEN - len(self.next_sat_bw_logs[u_id])) + self.next_sat_bw_logs[u_id]

            state[14 + 8*i, :PAST_LEN] = np.array(self.next_sat_bw_logs[u_id][:PAST_LEN]) / 10

            if len(self.cur_sat_bw_logs[u_id]) < PAST_LEN:
                self.cur_sat_bw_logs[u_id] = [0] * (PAST_LEN - len(self.cur_sat_bw_logs[u_id])) + self.cur_sat_bw_logs[u_id]

            state[15 + 8*i, :PAST_LEN] = np.array(self.cur_sat_bw_logs[u_id][:PAST_LEN]) / 10
            if self.connected_time[u_id]:
                state[16 + 8 * i, -1] = float(self.connected_time[u_id][self.cur_sat_id[u_id]]) / BUFFER_NORM_FACTOR / 10
            else:
                state[16 + 8 * i, -1] = 0
            i += 1

        state[9 + 8 * (self.num_users - 1):(9 + 8 * (self.num_users - 1) + (self.num_users - 1) * PAST_SAT_LOG_LEN),
        0:2] = np.reshape(other_user_sat_decisions, (-1, 2))
        for i, sat_bw in enumerate(other_sat_bws):
            state[(9 + 8 * (self.num_users - 1) + (self.num_users - 1) * PAST_SAT_LOG_LEN) + i, :PAST_LEN] = np.array(sat_bw) / 10

        # if len(next_sat_user_nums) < PAST_LEN:
        #     next_sat_user_nums = [0] * (PAST_LEN - len(next_sat_user_nums)) + next_sat_user_nums

        # state[agent][8, :PAST_LEN] = next_sat_user_nums[:5]

        self.state[agent] = state

        return self.state[agent]

    def reset(self):
        # self.net_env.reset_ptr()
        self.net_env.reset()
        self.time_stamp = 0
        self.last_bit_rate = [DEFAULT_QUALITY for _ in range(self.num_users)]
        self.state = [np.zeros((S_INFO, PAST_LEN)) for _ in range(self.num_users)]
        self.buffer_size = [0 for _ in range(self.num_users)]
        self.rebuf = [0 for _ in range(self.num_users)]
        self.video_chunk_size = [0 for _ in range(self.num_users)]
        self.delay = [0 for _ in range(self.num_users)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_users)]
        self.video_chunk_remain = [0 for _ in range(self.num_users)]
        self.next_sat_bw_logs = [[] for _ in range(self.num_users)]
        self.cur_sat_bw_logs = [[] for _ in range(self.num_users)]
        self.connected_time = [{} for _ in range(self.num_users)]
        self.other_ids = [[] for _ in range(self.num_users)]

        return self.state

    def get_first_agent(self):
        return self.net_env.get_first_agent()

    def check_end(self):
        return self.net_env.check_end()

    def render(self):
        return

    def set_sat(self, agent, sat):
        sat_id = None

        if sat == 0:
            self.is_handover = False
        elif sat == 1:
            self.is_handover = True
        else:
            assert sat >= 2
            sat_id = self.other_ids[agent][sat-2]
        self.net_env.set_satellite(agent, sat, sat_id=sat_id)

    def step(self, action, agent):
        bit_rate = int(action) % A_DIM
        # sat = int(action) // A_DIM

        # For testing with mpc
        # bit_rate /= BITRATE_WEIGHT
        # bit_rate = int(bit_rate)
        bit_rate *= BITRATE_WEIGHT

        # 0 -> select current satellite // 1 -> select another satellite
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size[agent], rebuf, video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, is_handover, num_of_user_sat, next_sat_bandwidth, next_sat_bw_logs, \
        cur_sat_user_num, next_sat_user_num, cur_sat_bw_logs, connected_time, cur_sat_id, next_sat_id, _, _, _, _, \
        other_sat_users, other_sat_bw_logs, other_buffer_sizes = \
            self.net_env.get_video_chunk(bit_rate, agent, None)
        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        if self.reward_func == "LIN":
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[self.last_bit_rate[agent]]) / M_IN_K
        elif self.reward_func == "HD":
            reward = BITRATE_REWARD[bit_rate] \
                     - 8 * rebuf - np.abs(BITRATE_REWARD[bit_rate] - BITRATE_REWARD[self.last_bit_rate[agent]])
        else:
            raise Exception

        self.last_bit_rate[agent] = bit_rate
        state = np.roll(self.state[agent], -1, axis=1)

        self.video_chunk_size[agent] = video_chunk_size
        self.delay[agent] = delay
        self.next_video_chunk_sizes[agent] = next_video_chunk_sizes
        self.video_chunk_remain[agent] = video_chunk_remain
        self.next_sat_bw_logs[agent] = next_sat_bw_logs
        self.cur_sat_bw_logs[agent] = cur_sat_bw_logs
        self.connected_time[agent] = connected_time
        self.cur_sat_id[agent] = cur_sat_id
        self.next_sat_id[agent] = next_sat_id

        other_user_sat_decisions, other_sat_num_users, other_sat_bws, cur_user_sat_decisions, other_ids \
            = encode_other_sat_info(self.net_env.sat_decision_log, self.num_users, self.cur_sat_id[agent], self.next_sat_id[agent] ,
                                    agent, other_sat_users, other_sat_bw_logs, PAST_SAT_LOG_LEN)

        self.other_ids[agent] = other_ids
        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[self.last_bit_rate[agent]] / \
                       float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size[agent] / BUFFER_NORM_FACTOR  # 10 sec

        if self.delay[agent] != 0:
            state[2, -1] = float(self.video_chunk_size[agent]) / \
                           float(self.delay[agent]) / M_IN_K  # kilo byte / ms
        else:
            state[2, -1] = 0
        state[3, -1] = float(self.delay[agent]) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        # state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        if self.next_video_chunk_sizes[agent]:
            state[4, :A_DIM] = np.array(
                [self.next_video_chunk_sizes[agent][index] for index in [0, 2, 4]]) / M_IN_K / M_IN_K  # mega byte
        else:
            state[4, :A_DIM] = [0, 0, 0]
        state[5, -1] = np.minimum(self.video_chunk_remain[agent],
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        if len(self.next_sat_bw_logs[agent]) < PAST_LEN:
            self.next_sat_bw_logs[agent] = [0] * (PAST_LEN - len(self.next_sat_bw_logs[agent])) + self.next_sat_bw_logs[
                agent]

        state[6, :PAST_LEN] = np.array(self.next_sat_bw_logs[agent][:PAST_LEN]) / 10

        if len(self.cur_sat_bw_logs[agent]) < PAST_LEN:
            self.cur_sat_bw_logs[agent] = [0] * (PAST_LEN - len(self.cur_sat_bw_logs[agent])) + self.cur_sat_bw_logs[
                agent]

        state[7, :PAST_LEN] = np.array(self.cur_sat_bw_logs[agent][:PAST_LEN]) / 10

        if self.connected_time[agent] and self.other_ids[agent]:
            state[8, :MAX_SAT] = [float(self.connected_time[agent][self.cur_sat_id[agent]]) / BUFFER_NORM_FACTOR / 10,
                                  float(self.connected_time[agent][self.next_sat_id[agent]]) / BUFFER_NORM_FACTOR / 10,
                                  float(self.connected_time[agent][self.other_ids[agent][0]]) / BUFFER_NORM_FACTOR / 10,
                                  float(self.connected_time[agent][self.other_ids[agent][1]]) / BUFFER_NORM_FACTOR / 10]
        else:
            state[8, :MAX_SAT] = [0] * MAX_SAT

        i = 0
        for u_id in range(self.num_users - 1):
            if u_id == agent:
                continue
            state[9 + 8 * i, -1] = VIDEO_BIT_RATE[self.last_bit_rate[u_id]] / \
                                    float(np.max(VIDEO_BIT_RATE))  # last quality
            state[10 + 8 * i, -1] = self.buffer_size[u_id] / BUFFER_NORM_FACTOR  # 10 sec
            if self.delay[u_id] != 0:
                state[11 + 8 * i, -1] = float(self.video_chunk_size[u_id]) / \
                                        float(self.delay[u_id]) / M_IN_K  # kilo byte / ms
            else:
                state[11 + 8 * i, -1] = 0
            state[12 + 8 * i, -1] = float(self.delay[u_id]) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            # state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[13 + 8 * i, -1] = np.minimum(self.video_chunk_remain[u_id],
                                               CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            if len(self.next_sat_bw_logs[u_id]) < PAST_LEN:
                self.next_sat_bw_logs[u_id] = [0] * (PAST_LEN - len(self.next_sat_bw_logs[u_id])) + self.next_sat_bw_logs[u_id]

            state[14 + 8 * i, :PAST_LEN] = np.array(self.next_sat_bw_logs[u_id][:PAST_LEN]) / 10

            if len(self.cur_sat_bw_logs[u_id]) < PAST_LEN:
                self.cur_sat_bw_logs[u_id] = [0] * (PAST_LEN - len(self.cur_sat_bw_logs[u_id])) + self.cur_sat_bw_logs[u_id]

            state[15 + 8 * i, :PAST_LEN] = np.array(self.cur_sat_bw_logs[u_id][:PAST_LEN]) / 10
            if self.connected_time[u_id]:
                state[16 + 8 * i, -1] = float(self.connected_time[u_id][self.cur_sat_id[u_id]]) / BUFFER_NORM_FACTOR / 10
            else:
                state[16 + 8 * i, -1] = 0
            i += 1

        state[
        9 + 8 * (self.num_users - 1):(9 + 8 * (self.num_users - 1) + (self.num_users - 1) * PAST_SAT_LOG_LEN),
        0:2] = np.reshape(other_user_sat_decisions, (-1, 2))
        # if len(next_sat_user_nums) < PAST_LEN:
        #     next_sat_user_nums = [0] * (PAST_LEN - len(next_sat_user_nums)) + next_sat_user_nums
        for i, sat_bw in enumerate(other_sat_bws):
            state[(9 + 8 * (self.num_users - 1) + (self.num_users - 1) * PAST_SAT_LOG_LEN) + i, :PAST_LEN] = np.array(sat_bw) / 10

        # state[agent][8, :PAST_LEN] = next_sat_user_nums[:5]
        self.state[agent] = state

        # observation, reward, done, info = ppo_spec.step(action)
        return state, reward, end_of_video, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf}
