# add queuing delay into halo
import os
import numpy as np
from . import core as abrenv
from . import load_trace

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6 + 1
S_LEN = 8  # take how many frames in the past
A_DIM = 6
A_SAT = 2
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
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

NUM_AGENTS = 2
SAT_DIM = A_SAT

class ABREnv():

    def __init__(self, random_seed=RANDOM_SEED, num_agents=NUM_AGENTS):
        self.num_agents = num_agents
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
        
    def seed(self, num):
        np.random.seed(num)

    def reset_agent(self, agent):
        bit_rate = DEFAULT_QUALITY
        delay, sleep_time, self.buffer_size[agent], rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, \
            next_sat_bw = \
            self.net_env.get_video_chunk(bit_rate, agent)
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
        state[6, :SAT_DIM] = np.array(next_sat_bw)

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
        self.net_env.set_satellite(agent, sat)

    def step(self, action, agent):
        bit_rate = int(action) % A_DIM
        sat = int(action) // A_DIM
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size[agent], rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, \
            next_sat_bw = \
            self.net_env.get_video_chunk(bit_rate, agent)

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
        state[6, :SAT_DIM] = np.array(next_sat_bw)

        self.state[agent] = state

        #observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf}
