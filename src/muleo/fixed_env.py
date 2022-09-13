import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
VIDEO_SIZE_FILE = './envivio/video_size_'

# LEO SETTINGS
HANDOVER_DELAY = 0.2  # sec
HANDOVER_WEIGHT = 1
SCALE_VIDEO_SIZE_FOR_TEST = 20
SCALE_VIDEO_LEN_FOR_TEST = 2

# Multi-user setting
NUM_AGENTS = 2

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED, num_agents=NUM_AGENTS):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)
        self.num_agents = num_agents

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.connection = {}
        for sat_id, sat_bw in self.cooked_bw.items():
            self.connection[sat_id] = -1

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1
        
        # multiuser setting
        self.cur_sat_id = []
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent)
            self.connection[cur_sat_id] = agent
            self.cur_sat_id.append(cur_sat_id)

        self.available_sat_list = self.get_available_sats_id()
        self.delay = [0 for _ in range(self.num_agents)]
        self.rebuf = [0 for _ in range(self.num_agents)]
        self.state = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter = [0 for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.sleep_time = [0 for _ in range(self.num_agents)]
        self.video_chunk_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter_sent = [0 for _ in range(self.num_agents)]
        self.last_mahimahi_time = [self.cooked_time[self.mahimahi_ptr - 1] for _ in range(self.num_agents)]
        self.return_buffer_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_remain = [0 for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        self.take_action = [False for _ in range(self.num_agents)]
        
        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def chunk_end(self, agent):
        self.return_buffer_size[agent] = self.buffer_size[agent]

        self.video_chunk_counter[agent] += 1
        self.video_chunk_remain[agent] = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter[agent]

        if self.video_chunk_counter[agent] >= TOTAL_VIDEO_CHUNCK:
            self.end_of_video[agent] = True
            self.buffer_size[agent] = 0
            self.video_chunk_counter[agent] = 0
            
        self.next_video_chunk_sizes[agent] = []
        for i in range(BITRATE_LEVELS):
            self.next_video_chunk_sizes[agent].append(self.video_size[i][self.video_chunk_counter[agent]])

        # Mark the end of chunk
        self.take_action[agent] = True
        self.state[agent] = 2


    def rebuffing(self, agent):
        duration = self.cooked_time[self.mahimahi_ptr] \
                    - self.last_mahimahi_time[agent]
        if duration > self.sleep_time[agent] / MILLISECONDS_IN_SECOND:
            self.last_mahimahi_time[agent] += self.sleep_time[agent] / MILLISECONDS_IN_SECOND
            self.chunk_end(agent)
         
        self.sleep_time[agent] -= duration * MILLISECONDS_IN_SECOND
        self.last_mahimahi_time[agent] = self.cooked_time[self.mahimahi_ptr]

    def buffering(self, agent):
        
        self.delay[agent] *= MILLISECONDS_IN_SECOND
        self.delay[agent] += LINK_RTT

        # rebuffer time
        self.rebuf[agent] = np.maximum(self.delay[agent] - self.buffer_size[agent], 0.0)

        # update the buffer
        self.buffer_size[agent] = np.maximum(self.buffer_size[agent] - self.delay[agent], 0.0)

        # add in the new chunk
        self.buffer_size[agent] += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        self.sleep_time[agent] = 0
        if self.buffer_size[agent] > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size[agent] - BUFFER_THRESH
            self.sleep_time[agent] = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size[agent] -= self.sleep_time[agent]
        else:
            self.chunk_end(agent)

    def downloading(self, agent):
        cur_sat_id = self.cur_sat_id[agent]
        throughput = self.cooked_bw[cur_sat_id][self.mahimahi_ptr] \
                        * B_IN_MB / BITS_IN_BYTE
                        
        if throughput == 0.0:
            cur_sat_id = self.get_best_sat_id(agent)
            pre_sat_id = self.cur_sat_id[agent]
            self.connection[pre_sat_id] = -1
            self.connection[cur_sat_id] = agent
            self.cur_sat_id[agent] = cur_sat_id
            self.delay[agent] += HANDOVER_DELAY

        duration = self.cooked_time[self.mahimahi_ptr] \
                    - self.last_mahimahi_time[agent]

        packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION
        
        if self.video_chunk_counter_sent[agent] + packet_payload > self.video_chunk_size[agent]:
            fractional_time = (self.video_chunk_size[agent] - self.video_chunk_counter_sent[agent]) / \
                                throughput / PACKET_PAYLOAD_PORTION
                                
            self.delay[agent] += fractional_time
            self.last_mahimahi_time[agent] += fractional_time
            self.state[agent] = 1
            self.buffering(agent)
            
        self.video_chunk_counter_sent[agent] += packet_payload
        self.delay[agent] += duration
        self.last_mahimahi_time[agent] = self.cooked_time[self.mahimahi_ptr]

    def step_agent(self, agent):
        if self.end_of_video[agent] == False:
            if self.state[agent] == 0:
                self.downloading(agent)
            elif self.state[agent] == 2:
                self.delay[agent] = 0
                self.video_chunk_counter_sent[agent] = 0
                self.state[agent] = 0
                self.downloading(agent)
            else:
                self.rebuffing(agent)
    
    def step(self):
        for i in range(self.num_agents):
            self.step_agent(i)
        
        # print('--------------------------', self.mahimahi_ptr)
        # print([self.cooked_bw[id][self.mahimahi_ptr] for id in self.cur_sat_id])
        self.mahimahi_ptr += 1
        if self.mahimahi_ptr >= len(self.cooked_bw[self.cur_sat_id[0]]):
            # print('--------------------------')
            # print(self.mahimahi_ptr, len(self.cooked_bw[self.cur_sat_id[0]]))
            # loop back in the beginning
            # note: trace file starts with time 0
            self.mahimahi_ptr = 1
            self.last_mahimahi_time = [0 for _ in range(self.num_agents)]
            
        for agent in range(self.num_agents):
            if self.end_of_video[agent] == False:
                return False
        
        self.trace_idx += 1
        if self.trace_idx >= len(self.all_cooked_time):
            self.trace_idx = 0            

        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the video
        # note: trace file starts with time 0
        self.mahimahi_ptr = self.mahimahi_start_ptr
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        return True

    def reset(self):
        # Refresh satellite info
        self.connection = {}
        for sat_id, sat_bw in self.cooked_bw.items():
            self.connection[sat_id] = -1

        # multiuser setting
        
        self.cur_sat_id = []
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent)
            self.connection[cur_sat_id] = agent
            self.cur_sat_id.append(cur_sat_id)

        self.available_sat_list = self.get_available_sats_id()
        self.delay = [0 for _ in range(self.num_agents)]
        self.rebuf = [0 for _ in range(self.num_agents)]
        self.state = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter = [0 for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.sleep_time = [0 for _ in range(self.num_agents)]
        self.video_chunk_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter_sent = [0 for _ in range(self.num_agents)]
        self.last_mahimahi_time = [self.cooked_time[self.mahimahi_ptr - 1] for _ in range(self.num_agents)]
        self.return_buffer_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_remain = [0 for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        self.take_action = [False for _ in range(self.num_agents)]

    def get_result(self, agent):
        return self.delay[agent], \
            self.sleep_time[agent], \
            self.return_buffer_size[agent] / MILLISECONDS_IN_SECOND, \
            self.rebuf[agent] / MILLISECONDS_IN_SECOND, \
            self.video_chunk_size[agent], \
            self.next_video_chunk_sizes[agent], \
            self.end_of_video[agent], \
            self.video_chunk_remain[agent]

    def get_action(self):
        return self.take_action
            
    def set_video_chunk(self, quality, agent):
        assert quality >= 0
        assert quality < BITRATE_LEVELS

        self.video_chunk_size[agent] = self.video_size[quality][self.video_chunk_counter[agent]]

        # use the delivery opportunity in mahimahi
        self.delay[agent] = 0.0  # in ms
        self.video_chunk_counter_sent[agent] = 0  # in bytes
        self.take_action[agent] = False
    
    def get_best_sat_id(self, agent, mahimahi_ptr=None):
        best_sat_id = None
        best_sat_bw = 0

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr

        for sat_id, sat_bw in self.cooked_bw.items():
            if self.connection[sat_id] == -1:
                if best_sat_bw < sat_bw[mahimahi_ptr]:
                    best_sat_id = sat_id
                    best_sat_bw = sat_bw[mahimahi_ptr]
        
        return best_sat_id

    def get_available_sats_id(self, mahimahi_ptr=None):
        sats_id = []
        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr
        for sat_id, sat_bw in self.cooked_bw.items():
            if sat_bw[mahimahi_ptr] > 0:
                sats_id.append(sat_id)
        return sats_id