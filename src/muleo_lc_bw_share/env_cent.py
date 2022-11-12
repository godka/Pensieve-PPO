# add queuing delay into halo
import os
import numpy as np
from . import core_cent as abrenv
from . import load_trace

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
#  = 10 + 1 + 3 + 6 * 5 # Original + nums of sat + bw of sats + decisions of users
S_LEN = 8  # take how many frames in the past
A_DIM = 6
PAST_LEN = 5
A_SAT = 2
MAX_SAT = 5
TRAIN_SEQ_LEN = 1000  # take as a train batch
MODEL_SAVE_INTERVAL = 1000
VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps
# VIDEO_BIT_RATE = [10000, 20000, 30000, 60000, 90000, 140000]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6

NUM_AGENTS = None
SAT_DIM = A_SAT


class ABREnv():

    def __init__(self, random_seed=RANDOM_SEED, num_agents=NUM_AGENTS):
        self.num_agents = num_agents
        S_INFO = 10 + 1 + 3 + self.num_agents * PAST_LEN
        # SAT_DIM = num_agents
        # A_SAT = num_agents
        # SAT_DIM = num_agents + 1

        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          random_seed=random_seed,
                                          num_agents=self.num_agents)

        self.last_bit_rate = DEFAULT_QUALITY
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.state = [np.zeros((S_INFO, S_LEN))for _ in range(self.num_agents)]
        self.sat_decision_log = [[-1,-1,-1,-1,-1] for _ in range(self.num_agents)]
        
    def seed(self, num):
        np.random.seed(num)

    def reset_agent(self, agent):
        bit_rate = DEFAULT_QUALITY
        delay, sleep_time, self.buffer_size[agent], rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, \
            next_sat_bw, next_sat_bw_logs, cur_sat_user_num, next_sat_user_nums, cur_sat_bw_logs, connected_time, \
            cur_sat_id, next_sat_id, other_sat_users, other_sat_bw_logs = self.net_env.get_video_chunk(bit_rate, agent)
        state = np.roll(self.state[agent], -1, axis=1)

        # this should be S_INFO number of terms`
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size[agent] / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        if len(next_sat_bw_logs) < PAST_LEN:
            next_sat_bw_logs = [0] * (PAST_LEN - len(next_sat_bw_logs)) + next_sat_bw_logs

        state[6, :PAST_LEN] = np.array(next_sat_bw_logs[:PAST_LEN]) / 10

        if len(cur_sat_bw_logs) < PAST_LEN:
            cur_sat_bw_logs = [0] * (PAST_LEN - len(cur_sat_bw_logs)) + cur_sat_bw_logs

        state[7, :PAST_LEN] = np.array(cur_sat_bw_logs[:PAST_LEN]) / 10

        other_user_sat_decisions, other_sat_num_users, other_sat_bws \
            = self.encode_other_sat_info(cur_sat_id, next_sat_id, agent, other_sat_users, other_sat_bw_logs)

        state[8, :A_SAT] = np.array([cur_sat_user_num, next_sat_user_nums]) / 10
        state[9, :A_SAT] = [float(connected_time[0]) / BUFFER_NORM_FACTOR,
                                   float(connected_time[1]) / BUFFER_NORM_FACTOR]

        state[10, :MAX_SAT - A_SAT] = np.array(other_sat_num_users) / 10

        state[11:(11 + MAX_SAT - A_SAT), :PAST_LEN] = np.array(other_sat_bws) / 10

        state[(11 + MAX_SAT - A_SAT):(11 + MAX_SAT - A_SAT + (self.num_agents) * PAST_LEN),
        0:PAST_LEN] = np.reshape(other_user_sat_decisions, (-1, 5))

        self.state[agent] = state
        
        return self.state[agent]

    def reset(self):
        # self.net_env.reset_ptr()
        self.net_env.reset()
        self.time_stamp = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = [np.zeros((S_INFO, S_LEN)) for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]

        # for agent in range(self.num_agents):
        #     delay, sleep_time, self.buffer_size[agent], rebuf, \
        #         video_chunk_size, next_video_chunk_sizes, \
        #         end_of_video, video_chunk_remain, \
        #         next_sat_bw = \
        #         self.net_env.get_video_chunk(bit_rate, agent, sat)
        #     state = np.roll(self.state[agent], -1, axis=1)

        #     # this should be S_INFO number of terms
        #     state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
        #         float(np.max(VIDEO_BIT_RATE))  # last quality
        #     state[1, -1] = self.buffer_size[agent] / BUFFER_NORM_FACTOR  # 10 sec
        #     state[2, -1] = float(video_chunk_size) / \
        #         float(delay) / M_IN_K  # kilo byte / ms
        #     state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        #     state[4, :A_DIM] = np.array(
        #         next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        #     state[5, -1] = np.minimum(video_chunk_remain,
        #                             CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        #     state[6, :SAT_DIM] = np.array(
        #         next_sat_bw) * B_IN_MB / BITS_IN_BYTE  # mega byte

        #     self.state[agent] = state
        return self.state

    def get_first_agent(self):
        return self.net_env.get_first_agent()
    
    def check_end(self):
        return self.net_env.check_end()

    def render(self):
        return

    def set_sat(self, agent, sat):
        changed_sat_id = self.net_env.set_satellite(agent, sat)
        self.sat_decision_log[agent].append(changed_sat_id)

    def step(self, action, agent):
        bit_rate = int(action) % A_DIM
        sat = int(action) // A_DIM
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size[agent], rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, \
            next_sat_bw, next_sat_bw_logs, cur_sat_user_num, next_sat_user_nums, cur_sat_bw_logs, connected_time, \
            cur_sat_id, next_sat_id, other_sat_users, other_sat_bw_logs = self.net_env.get_video_chunk(bit_rate, agent)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K

        self.last_bit_rate = bit_rate
        state = np.roll(self.state[agent], -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size[agent] / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        if len(next_sat_bw_logs) < PAST_LEN:
            next_sat_bw_logs = [0] * (PAST_LEN - len(next_sat_bw_logs)) + next_sat_bw_logs

        state[6, :PAST_LEN] = np.array(next_sat_bw_logs[:PAST_LEN]) / 10

        if len(cur_sat_bw_logs) < PAST_LEN:
            cur_sat_bw_logs = [0] * (PAST_LEN - len(cur_sat_bw_logs)) + cur_sat_bw_logs

        state[7, :PAST_LEN] = np.array(cur_sat_bw_logs[:PAST_LEN]) / 10

        other_user_sat_decisions, other_sat_num_users, other_sat_bws \
            = self.encode_other_sat_info(cur_sat_id, next_sat_id, agent, other_sat_users, other_sat_bw_logs)

        state[8, :A_SAT] = np.array([cur_sat_user_num, next_sat_user_nums]) / 10
        state[9, :A_SAT] = [float(connected_time[0]) / BUFFER_NORM_FACTOR,
                            float(connected_time[1]) / BUFFER_NORM_FACTOR]

        state[10, :MAX_SAT - A_SAT] = np.array(other_sat_num_users) / 10

        state[11:(11 + MAX_SAT - A_SAT), :PAST_LEN] = np.array(other_sat_bws) / 10
        state[(11 + MAX_SAT - A_SAT):(11 + MAX_SAT - A_SAT + (self.num_agents) * PAST_LEN),
        0:PAST_LEN] = np.reshape(other_user_sat_decisions, (-1, 5))

        # if len(next_sat_user_nums) < PAST_LEN:
        #     next_sat_user_nums = [0] * (PAST_LEN - len(next_sat_user_nums)) + next_sat_user_nums

        # state[agent][8, :PAST_LEN] = next_sat_user_nums[:5]

        self.state[agent] = state
        #observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf}

    def encode_other_sat_info(self, cur_sat_id, next_sat_id, agent, other_sat_users, other_sat_bw_logs):
        # self.sat_decision_log
        # one hot encoding by bw strength
        # MAX_SAT
        assert len(other_sat_users.keys()) == len(other_sat_bw_logs.keys())
        other_user_sat_decisions = []
        other_sat_num_users = []
        other_sat_bws = []
        other_sat_id_bw = {}
        other_index_ids = {}
        """
        for sat_id in other_sat_bw_logs.keys():
            avg_bw = sum(other_sat_bw_logs[sat_id]) / len(other_sat_bw_logs[sat_id])
            other_sat_id_bw[sat_id] = avg_bw
        """
        other_ids = sorted(other_sat_users, reverse=True)
        other_ids = other_ids[:MAX_SAT-2]
        for i in range(MAX_SAT-2):
            if len(other_ids) <= i:
                break
            other_index_ids[other_ids[i]] = i

        for i in range(MAX_SAT-2):
            if len(other_sat_users.keys()) <= i:
                other_sat_num_users.append(0)
                other_sat_bws.append([0, 0, 0, 0, 0])
                continue
            other_sat_num_users.append(other_sat_users[other_ids[i]])
            if len(other_sat_bw_logs[other_ids[i]]) < PAST_LEN:
                tmp_len = len(other_sat_bw_logs[other_ids[i]])
                other_sat_bws.append([0] * (PAST_LEN - tmp_len) + other_sat_bw_logs[other_ids[i]])
            else:
                other_sat_bws.append(other_sat_bw_logs[other_ids[i]])

        for index, i_agent in enumerate(range(self.num_agents)):
            # Exclude the current user's deicision
            # if i_agent == agent:
            #     continue
            sat_logs = self.sat_decision_log[i_agent][-PAST_LEN:]
            tmp_logs = []
            for log_data in sat_logs:
                if log_data == cur_sat_id:
                    encoded_logs = [1, 0, 0, 0, 0]
                elif log_data == next_sat_id:
                    encoded_logs = [0, 1, 0, 0, 0]
                elif log_data in other_index_ids.keys():
                    tmp_array = [0, 0, 0, 0, 0]
                    tmp_array[other_index_ids[log_data] + 2] = 1
                    encoded_logs = tmp_array
                else:
                    # print("Cannot happen")
                    encoded_logs = [0, 0, 0, 0, 0]
                # encoded_logs = encoded_logs + [0] * 3
                tmp_logs.append(encoded_logs)
            other_user_sat_decisions.append(tmp_logs)

        return other_user_sat_decisions, other_sat_num_users, other_sat_bws

