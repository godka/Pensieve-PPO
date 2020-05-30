# add queuing delay into halo
import os
import numpy as np
import env as abrenv

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 7
S_LEN = 10  # take how many frames in the past
A_DIM = 10

TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = np.array([200., 300, 450, 750, 1200,
                  1850, 2850, 4300, 6000, 8000])  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
M_IN_B = 1000000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6
VIDEO_FOLDER = './videos/'
COOKED_TRACE_FOLDER = './cooked_traces/'


class ABREnv():

    def __init__(self, random_seed=RANDOM_SEED, fixed_env=False,
                 trace_folder=COOKED_TRACE_FOLDER, video_folder=VIDEO_FOLDER):
        np.random.seed(random_seed)
        # all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        self.net_env = abrenv.Environment(
            video_folder=video_folder,
            trace_folder=trace_folder,
            fixed_env=fixed_env,
            random_seed=random_seed)

        self.last_action = None
        self.mask = None
        self.last_bit_rate = DEFAULT_QUALITY

        self.buffer_size = 0.
        self.state = np.zeros((S_INFO, S_LEN))
        self.reset()

    def seed(self, num):
        np.random.seed(num)

    def bitrate_to_action(self, bitrate, mask, a_dim=A_DIM):
        assert len(mask) == a_dim
        assert bitrate >= 0
        assert bitrate < np.sum(mask)
        cumsum_mask = np.cumsum(mask) - 1
        action = np.where(cumsum_mask == bitrate)[0][0]
        return action
        
    def reset(self):
        # self.net_env.reset_ptr()
        self.time_stamp = 0

        action = self.bitrate_to_action(
            DEFAULT_QUALITY, self.net_env.video_masks[self.net_env.video_idx])
        self.last_action = action

        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.

        delay, sleep_time, self.buffer_size, \
                rebuf, video_chunk_size, end_of_video, \
                video_chunk_remain, video_num_chunks, \
                next_video_chunk_size, self.mask = \
                self.net_env.get_video_chunk(self.last_action)

        state = np.roll(self.state, -1, axis=1)
        
        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[action] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K
        state[4, -1] = video_chunk_remain / float(video_num_chunks)
        state[5, :] = -1
        nxt_chnk_cnt = 0
        for i in range(A_DIM):
            if self.mask[i] == 1:
                state[5, i] = next_video_chunk_size[nxt_chnk_cnt] / M_IN_B
                nxt_chnk_cnt += 1
        assert(nxt_chnk_cnt) == np.sum(self.mask)
        state[6, -A_DIM:] = self.mask


        self.state = state
        return state
        # return state.reshape((1, S_INFO*S_LEN))

    def render(self):
        return

    def step(self, action):

        delay, sleep_time, self.buffer_size, \
                rebuf, video_chunk_size, end_of_video, \
                video_chunk_remain, video_num_chunks, \
                next_video_chunk_size, self.mask = \
                self.net_env.get_video_chunk(action)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        
        reward = VIDEO_BIT_RATE[action] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[action] -
                                        VIDEO_BIT_RATE[self.last_action]) / M_IN_K

        self.last_action = action
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[action] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K
        state[4, -1] = video_chunk_remain / float(video_num_chunks)
        state[5, :] = -1
        nxt_chnk_cnt = 0
        for i in range(A_DIM):
            if self.mask[i] == 1:
                state[5, i] = next_video_chunk_size[nxt_chnk_cnt] / M_IN_B
                nxt_chnk_cnt += 1
        assert(nxt_chnk_cnt) == np.sum(self.mask)
        state[6, -A_DIM:] = self.mask

        self.state = state
        #observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, {'bitrate': VIDEO_BIT_RATE[action], \
            'rebuffer': rebuf}
