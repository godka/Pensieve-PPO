import itertools

import structlog
from scipy.optimize import minimize, LinearConstraint
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd
import numpy as np
import copy

from env.object.satellite import Satellite
from env.object.user import User
from util.constants import EPSILON, MPC_FUTURE_CHUNK_COUNT, QUALITY_FACTOR, REBUF_PENALTY, SMOOTH_PENALTY, \
    MPC_PAST_CHUNK_COUNT, HO_NUM, TOTAL_VIDEO_CHUNKS, CHUNK_TIL_VIDEO_END_CAP, DEFAULT_QUALITY, SNR_MIN, BUF_RATIO, \
    VIDEO_CHUNCK_LEN, BITRATE_LEVELS, B_IN_MB, BITS_IN_BYTE, M_IN_K, MILLISECONDS_IN_SECOND

RANDOM_SEED = 42
PAST_LEN = 8
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = 'data/video_data/envivio/video_size_'

# LEO SETTINGS
HANDOVER_DELAY = 0.2  # sec
HANDOVER_WEIGHT = 1

# Multi-user setting
NUM_AGENTS = None

SAT_STRATEGY = "resource-fair"
# SAT_STRATEGY = "ratio-based"


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED, num_agents=NUM_AGENTS):
        assert len(all_cooked_time) == len(all_cooked_bw)
        self.log = structlog.get_logger()

        np.random.seed(random_seed)
        self.num_agents = num_agents

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # self.last_quality = DEFAULT_QUALITY
        self.last_quality = [DEFAULT_QUALITY for _ in range(self.num_agents)]

        # randomize the start point of the trace
        # note: trace file starts with time 0

        self.mahimahi_ptr = [np.random.randint(1, len(self.cooked_time) - TOTAL_VIDEO_CHUNKS)] * self.num_agents

        self.last_mahimahi_time = [self.mahimahi_ptr[i] - 1 for i in range(self.num_agents)]

        # Centralization
        self.user_qoe_log = [{} for _ in range(self.num_agents)]
        self.num_of_user_sat = {}
        self.num_sat_info = {}
        self.cur_satellite = {}

        for sat_id, sat_bw in self.cooked_bw.items():
            self.num_sat_info[sat_id] = [0 for _ in range(len(sat_bw))]
            self.num_of_user_sat[sat_id] = 0
            self.cur_satellite[sat_id] = Satellite(sat_id, sat_bw, SAT_STRATEGY)

        self.cur_user = [User(0, SNR_MIN) for _ in range(self.num_agents)]

        # print(self.num_sat_info)
        self.prev_best_combos = [[DEFAULT_QUALITY] * MPC_FUTURE_CHUNK_COUNT] * self.num_agents

        # multiuser setting
        self.cur_sat_id = []
        self.prev_sat_id = [None for _ in range(self.num_agents)]
        cur_sat_id = self.get_best_sat_id(0)
        self.cur_sat_id.append(cur_sat_id)
        self.update_sat_info(cur_sat_id, self.last_mahimahi_time[0], 0, 1)

        self.video_chunk_counter = [0 for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter_sent = [0 for _ in range(self.num_agents)]
        self.video_chunk_remain = [TOTAL_VIDEO_CHUNKS for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        # self.next_sat_bandwidth = [[] for _ in range(self.num_agents)]
        self.next_sat_id = [[] for _ in range(self.num_agents)]
        self.delay = [0 for _ in range(self.num_agents)]

        self.bit_rate = None
        self.download_bw = [[] for _ in range(self.num_agents)]
        self.past_download_ests = [[] for _ in range(self.num_agents)]
        self.past_download_bw_errors = [[] for _ in range(self.num_agents)]
        self.past_bw_ests = [{} for _ in range(self.num_agents)]
        self.past_bw_errors = [{} for _ in range(self.num_agents)]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        self.last_delay = [MPC_PAST_CHUNK_COUNT for _ in range(self.num_agents)]
        self.unexpected_change = True

    def get_video_chunk(self, quality, agent, model_type, runner_up_sat_id=None, ho_stamp=None, do_mpc=False):

        assert quality >= 0
        assert quality < BITRATE_LEVELS
        assert quality in [0, 2, 4]

        runner_up_sat_ids, ho_stamps, best_combos, best_user_info, final_rate = None, None, None, None, None

        # update noise of agent SNR
        self.cur_user[agent].update_snr_noise()

        video_chunk_size = self.video_size[quality][self.video_chunk_counter[agent]]

        # use the delivery opportunity in mahimahi
        delay = self.delay[agent]  # in ms
        self.delay[agent] = 0
        video_chunk_counter_sent = 0  # in bytes
        end_of_network = False
        is_handover = False

        self.last_quality[agent] = quality

        while True:  # download video chunk over mahimahi
            throughput = self.cur_satellite[self.cur_sat_id[0]].data_rate(self.cur_user[agent],
                                                                              self.mahimahi_ptr[
                                                                                  agent]) * B_IN_MB / BITS_IN_BYTE
            if throughput == 0.0:
                if ho_stamp and ho_stamp == "MVT":
                    sat_id = self.get_mvt_sat_id(agent, self.mahimahi_ptr[agent])
                else:
                    sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent])
                self.log.debug("Forced Handover1", cur_sat_id=self.cur_sat_id[0], next_sat_id=sat_id,
                              mahimahi_ptr=self.mahimahi_ptr[agent], agent=agent,
                              cur_bw=self.cooked_bw[self.cur_sat_id[0]][self.mahimahi_ptr[agent]-3:self.mahimahi_ptr[agent]+3],
                              next_bw=self.cooked_bw[
                                  sat_id][self.mahimahi_ptr[agent] - 3:self.mahimahi_ptr[agent] + 3]
                              )
                assert self.cur_sat_id[0] != sat_id
                # self.update_sat_info(self.cur_sat_id[0], self.last_mahimahi_time[agent], agent, -1)
                # self.update_sat_info(sat_id, self.last_mahimahi_time[agent], agent, 1)

                # bself.switch_sat(agent, sat_id)
                delay += HANDOVER_DELAY
                is_handover = True
                self.download_bw[agent] = []
                self.unexpected_change = True
                throughput = self.cur_satellite[sat_id].data_rate(self.cur_user[agent],
                                                                                  self.mahimahi_ptr[
                                                                                      agent]) * B_IN_MB / BITS_IN_BYTE
                assert throughput != 0
            duration = self.cooked_time[self.mahimahi_ptr[agent]] \
                       - self.last_mahimahi_time[agent]

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time[agent] += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time[agent] = self.cooked_time[self.mahimahi_ptr[agent]]

            self.mahimahi_ptr[agent] += 1

            if self.mahimahi_ptr[agent] >= len(self.cooked_bw[self.cur_sat_id[0]]):
                # loop back in the beginning
                # note: trace file starts with time 0
                # self.mahimahi_ptr[agent] = 1
                # self.last_mahimahi_time[agent] = 0
                # self.end_of_video[agent] = True
                end_of_network = True
                break
        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size[agent], 0.0)

        # update the buffer
        self.buffer_size[agent] = np.maximum(self.buffer_size[agent] - delay, 0.0)

        # add in the new chunk
        self.buffer_size[agent] += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size[agent] > BUFFER_THRESH:
            self.log.info("Buffer exceed!", buffer_size=self.buffer_size[agent], mahimahi_ptr=self.mahimahi_ptr[agent],
                          agent=agent)
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size[agent] - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size[agent] -= sleep_time

            while True:
                if self.mahimahi_ptr[agent] >= len(self.cooked_bw[self.cur_sat_id[0]]):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    # self.mahimahi_ptr[agent] = 1
                    # self.last_mahimahi_time[agent] = 0
                    # self.end_of_video[agent] = True
                    end_of_network = True
                    break

                duration = self.cooked_time[self.mahimahi_ptr[agent]] \
                           - self.last_mahimahi_time[agent]
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time[agent] += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time[agent] = self.cooked_time[self.mahimahi_ptr[agent]]
                self.mahimahi_ptr[agent] += 1

                throughput = self.cur_satellite[self.cur_sat_id[0]].data_rate(self.cur_user[agent],
                                                                                  self.mahimahi_ptr[agent])* B_IN_MB / BITS_IN_BYTE
                if throughput == 0.0:
                    # Connect the satellite that has the best serving time
                    sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent])
                    assert sat_id != self.cur_sat_id[0]
                    self.log.debug("Forced Handover2", cur_sat_id=self.cur_sat_id[0], sat_id=sat_id,
                                  mahimahi_ptr=self.mahimahi_ptr[agent], agent=agent)
                    assert sat_id != self.cur_sat_id[0]
                    self.update_sat_info(sat_id, self.last_mahimahi_time[agent], agent, 1)
                    self.update_sat_info(self.cur_sat_id[0], self.last_mahimahi_time[agent], agent, -1)
                    self.switch_sat(agent, sat_id)
                    is_handover = True
                    delay += HANDOVER_DELAY
                    throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent],
                                                                                      self.mahimahi_ptr[agent])* B_IN_MB / BITS_IN_BYTE

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size[agent]

        self.video_chunk_counter[agent] += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.video_chunk_counter[agent]

        if self.video_chunk_counter[agent] >= TOTAL_VIDEO_CHUNKS or end_of_network:
            self.log.debug("End downloading", end_of_network=end_of_network, counter=self.video_chunk_counter[agent],
                          mahimahi_ptr=self.mahimahi_ptr[agent], agent=agent)
            self.end_of_video[agent] = True
            self.buffer_size[agent] = 0
            self.video_chunk_counter[agent] = 0
            # self.update_sat_info(self.cur_sat_id[0], self.last_mahimahi_time[agent], agent, -1)

            # Refresh satellite info
            # self.connection[self.cur_sat_id[0]] = -1
            # self.cur_sat_id[0] = None

            # wait for overall clean
            cur_sat_bw_logs, next_sat_bandwidth, next_sat_id, next_sat_bw_logs, connected_time = [], [], None, [], [0, 0]
        else:
            cur_sat_bw_logs, next_sat_bandwidth, next_sat_id, next_sat_bw_logs, connected_time = self.get_next_sat_info(
                agent, self.mahimahi_ptr[agent])

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter[agent]])

        self.video_chunk_remain[agent] = video_chunk_remain
        self.download_bw[agent].append(float(video_chunk_size) / delay / M_IN_K * BITS_IN_BYTE)

        # num of users
        cur_sat_user_num = len(self.cur_satellite[self.cur_sat_id[0]].get_ue_list(self.mahimahi_ptr[agent]))
        self.next_sat_id[agent] = next_sat_id
        if next_sat_id:
            next_sat_user_num = len(self.cur_satellite[next_sat_id].get_ue_list(self.mahimahi_ptr[agent]))
        else:
            next_sat_user_num = 0
        MPC_PAST_CHUNK_COUNT = round(delay / M_IN_K)

        return delay, \
               sleep_time, \
               return_buffer_size / MILLISECONDS_IN_SECOND, \
               rebuf / MILLISECONDS_IN_SECOND, \
               video_chunk_size, \
               next_video_chunk_sizes, \
               self.end_of_video[agent], \
               video_chunk_remain, \
               is_handover, self.get_num_of_user_sat(self.mahimahi_ptr[agent], sat_id="all"), \
               next_sat_bandwidth, next_sat_bw_logs, cur_sat_user_num, next_sat_user_num, cur_sat_bw_logs, connected_time, \
               self.cur_sat_id[0], runner_up_sat_ids, ho_stamps, best_combos, final_rate

    def reset(self):
        self.video_chunk_counter = [0 for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter_sent = [0 for _ in range(self.num_agents)]
        self.video_chunk_remain = [TOTAL_VIDEO_CHUNKS for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        # self.next_sat_bandwidth = [[] for _ in range(self.num_agents)]
        self.next_sat_id = [[] for _ in range(self.num_agents)]
        self.delay = [0 for _ in range(self.num_agents)]
        self.num_of_user_sat = {}
        self.download_bw = [[] for _ in range(self.num_agents)]
        self.cur_satellite = {}

        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        for sat_id, sat_bw in self.cooked_bw.items():
            self.num_sat_info[sat_id] = [0 for _ in range(len(sat_bw))]
            self.num_of_user_sat[sat_id] = 0
            self.cur_satellite[sat_id] = Satellite(sat_id, sat_bw, SAT_STRATEGY)

        self.cur_user = [User(0, SNR_MIN) for _ in range(self.num_agents)]

        self.mahimahi_ptr = [np.random.randint(1, len(self.cooked_time) - TOTAL_VIDEO_CHUNKS)] * self.num_agents
        self.last_mahimahi_time = [self.mahimahi_ptr[i] - 1 for i in range(self.num_agents)]

        self.cur_sat_id = []
        cur_sat_id = self.get_best_sat_id(0)
        # self.connection[cur_sat_id] = agent
        self.cur_sat_id.append(cur_sat_id)
        self.update_sat_info(cur_sat_id, self.last_mahimahi_time[0], 0, 1)

        self.last_delay = [MPC_PAST_CHUNK_COUNT for _ in range(self.num_agents)]

    def check_end(self):
        # End if all users finish
        for agent in range(self.num_agents):
            if not self.end_of_video[agent]:
                return False

        return True

    def get_first_agent(self):
        user = -1
        
        for agent in range(self.num_agents):
            if not self.end_of_video[agent]:
                if user == -1:
                    user = agent
                else:
                    if self.last_mahimahi_time[agent] < self.last_mahimahi_time[user]:
                        user = agent
                        
        return user

    def get_next_sat_info(self, agent, mahimahi_ptr=None):
        best_sat_id = None
        best_sat_bw = 0
        best_bw_list = []
        up_time_list = []
        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        list1, next_sat_id, next_sat_bws = [], [], []
        bw_list = []
        sat_bw = self.cooked_bw[self.cur_sat_id[0]]
        for i in range(PAST_LEN, 1, -1):
            if mahimahi_ptr - i >= 0:
                if len(self.cur_satellite[self.cur_sat_id[0]].get_ue_list(mahimahi_ptr)) == 0:
                    bw_list.append(sat_bw[mahimahi_ptr - i])
                else:
                    bw_list.append(
                        sat_bw[mahimahi_ptr - i] / len(self.cur_satellite[self.cur_sat_id[0]].get_ue_list(mahimahi_ptr)))

        up_time = 0
        tmp_index = mahimahi_ptr - 1
        tmp_sat_bw = sat_bw[tmp_index]
        while tmp_sat_bw != 0 and tmp_index >= 0:
            up_time += 1
            tmp_index -= 1
            tmp_sat_bw = sat_bw[tmp_index]
        up_time_list.append(up_time)
        # list1.append(bw)
        cur_sat_bws = bw_list

        runner_up_sat_id = self.get_runner_up_sat_id(agent, method="harmonic-mean", mahimahi_ptr=mahimahi_ptr)[0]
        if runner_up_sat_id:
            bw_list = []
            for i in range(PAST_LEN, 1, -1):
                if mahimahi_ptr - i >= 0 and sat_bw[mahimahi_ptr - i] != 0:
                    if len(self.cur_satellite[runner_up_sat_id].get_ue_list(mahimahi_ptr)) == 0:
                        bw_list.append(sat_bw[mahimahi_ptr - i])
                    else:
                        bw_list.append(sat_bw[mahimahi_ptr - i] / (len(self.cur_satellite[runner_up_sat_id].get_ue_list(mahimahi_ptr)) + 1))
            next_sat_bws = bw_list
            up_time = 0
            tmp_index = mahimahi_ptr - 1
            sat_bw = self.cooked_bw[runner_up_sat_id]
            tmp_sat_bw = sat_bw[tmp_index]
            while tmp_sat_bw != 0 and tmp_index >= 0:
                up_time += 1
                tmp_index -= 1
                tmp_sat_bw = sat_bw[tmp_index]
            up_time_list.append(up_time)

            next_sat_id = runner_up_sat_id
        else:
            up_time_list.append(0)
            next_sat_id = None
        # zipped_lists = zip(list1, list2)
        # sorted_pairs = sorted(zipped_lists)

        # tuples = zip(*sorted_pairs)
        # list1, list2 = [ list(tuple) for tuple in  tuples]
        # list1 = [ list1[i] for i in range(1)]
        # list2 = [ list2[i] for i in range(1)]

        return cur_sat_bws, None, next_sat_id, next_sat_bws, up_time_list

    def get_best_sat_id(self, agent, mahimahi_ptr=None):
        best_sat_id = None
        best_sat_bw = 0

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        for sat_id, sat_bw in self.cooked_bw.items():
            real_sat_bw = self.cur_satellite[sat_id].data_rate(self.cur_user[agent], mahimahi_ptr)

            if best_sat_bw < real_sat_bw:
                best_sat_id = sat_id
                best_sat_bw = real_sat_bw

        return best_sat_id

    def switch_sat(self, agent, cur_sat_id):
        pre_sat_id = self.cur_sat_id[0]
        self.prev_sat_id[0] = pre_sat_id

        # self.connection[pre_sat_id] = -1
        # self.connection[cur_sat_id] = agent

        self.cur_sat_id[0] = cur_sat_id

    def update_sat_info(self, sat_id, mahimahi_ptr, agent, variation):
        # update sat info
        self.log.debug("update_sat_info", agent=agent, sat_id=sat_id, mahimahi_ptr=mahimahi_ptr, variation=variation)
        total_num = 0
        for i in range(self.num_agents):
            if variation == 1:
                self.cur_user[i].update_sat_log(sat_id, mahimahi_ptr)
                self.cur_satellite[sat_id].add_ue(i, mahimahi_ptr)

            elif variation == -1:
                self.cur_satellite[sat_id].remove_ue(i, mahimahi_ptr)

            if sat_id in self.num_of_user_sat.keys():
                self.num_of_user_sat[sat_id] += variation
            else:
                self.num_of_user_sat[sat_id] = variation
            total_num += self.num_of_user_sat[sat_id]
        assert self.num_of_user_sat[sat_id] >= 0

    def get_num_of_user_sat(self, mahimahi_ptr, sat_id):
        # update sat info
        if sat_id == "all":
            filtered_num_of_user_sat = {}
            for tmp_sat_id in self.cur_satellite.keys():
                if len(self.cur_satellite[tmp_sat_id].get_ue_list(mahimahi_ptr)) != 0:
                    filtered_num_of_user_sat[tmp_sat_id] = len(self.cur_satellite[tmp_sat_id].get_ue_list(mahimahi_ptr))
            return filtered_num_of_user_sat
        if sat_id in self.cur_satellite.keys():
            return len(self.cur_satellite[sat_id].get_ue_list(mahimahi_ptr))

    def get_runner_up_sat_id(self, agent, method="holt-winter", mahimahi_ptr=None, cur_sat_id=None, plus=False):
        best_sat_id = None
        best_sat_bw = 0
        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]
        if cur_sat_id is None:
            cur_sat_id = self.cur_sat_id[0]

        for sat_id, sat_bw in self.cooked_bw.items():
            real_sat_bw = None

            # Pass the previously connected satellite
            # if sat_id == cur_sat_id or sat_id == self.prev_sat_id[0]:
            if sat_id == cur_sat_id or sat_id == self.prev_sat_id[0]:
                continue

            if method == "harmonic-mean":
                target_sat_bw = self.predict_bw_num(sat_id, agent, mahimahi_ptr=mahimahi_ptr, plus=True)
                real_sat_bw = target_sat_bw  # / (self.get_num_of_user_sat(sat_id) + 1)
            else:
                print("Cannot happen")
                raise Exception

            assert (real_sat_bw is not None)
            if best_sat_bw < real_sat_bw:
                best_sat_id = sat_id
                best_sat_bw = real_sat_bw

        return best_sat_id, best_sat_bw

    def predict_bw_num(self, sat_id, agent, robustness=True, plus=False, mahimahi_ptr=None):
        curr_error = 0
        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        if plus:
            num_of_user_sat = len(self.cur_satellite[sat_id].get_ue_list(mahimahi_ptr)) + 1
        else:
            num_of_user_sat = len(self.cur_satellite[sat_id].get_ue_list(mahimahi_ptr))

        # past_bw = self.cooked_bw[self.cur_sat_id][self.mahimahi_ptr - 1]
        if num_of_user_sat == 0:
            past_bw = self.cooked_bw[sat_id][mahimahi_ptr]
        else:
            past_bw = self.cooked_bw[sat_id][mahimahi_ptr] / num_of_user_sat
        if past_bw == 0:
            return 0

        if sat_id in self.past_bw_ests[agent].keys() and len(self.past_bw_ests[agent][sat_id]) > 0:
            curr_error = abs(self.past_bw_ests[agent][sat_id][-1] - past_bw) / float(past_bw)
        if sat_id not in self.past_bw_errors[agent].keys():
            self.past_bw_errors[agent][sat_id] = []
        self.past_bw_errors[agent][sat_id].append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        start_index = mahimahi_ptr - MPC_PAST_CHUNK_COUNT
        if start_index < 0:
            start_index = 0

        past_bws = []
        for index in range(start_index, mahimahi_ptr):
            if num_of_user_sat == 0:
                past_bws.append(self.cooked_bw[sat_id][index])
            else:
                past_bws.append(self.cooked_bw[sat_id][index] / num_of_user_sat)

        # Newly possible satellite case
        if all(v == 0.0 for v in past_bws):
            if num_of_user_sat == 0:
                return self.cooked_bw[sat_id][mahimahi_ptr]
            else:
                return self.cooked_bw[sat_id][mahimahi_ptr] / num_of_user_sat

        while past_bws[0] == 0.0:
            past_bws = past_bws[1:]

        bandwidth_sum = 0
        for past_val in past_bws:
            bandwidth_sum += (1 / float(past_val))

        harmonic_bw = 1.0 / (bandwidth_sum / len(past_bws))

        if sat_id not in self.past_bw_ests[agent].keys():
            self.past_bw_ests[agent][sat_id] = []
        self.past_bw_ests[agent][sat_id].append(harmonic_bw)

        if robustness:
            # future bandwidth prediction
            # divide by 1 + max of last 5 (or up to 5) errors
            error_pos = -MPC_PAST_CHUNK_COUNT
            if sat_id in self.past_bw_errors[agent].keys() and len(
                    self.past_bw_errors[agent][sat_id]) < MPC_PAST_CHUNK_COUNT:
                error_pos = -len(self.past_bw_errors[agent][sat_id])
            max_error = float(max(self.past_bw_errors[agent][sat_id][error_pos:]))
            harmonic_bw = harmonic_bw / (1 + max_error)  # robustMPC here

        return harmonic_bw

    def set_satellite(self, agent, sat=0, id_list=None):
        if id_list is None:
            sat_id = self.next_sat_id[agent]

        if sat == 1 and agent == 0:
            if sat_id == self.cur_sat_id[0] or sat_id is None:
                # print("Can't do handover. Only one visible satellite")
                return
            self.log.debug("set_satellite", cur_sat_id=self.cur_sat_id[0], next_sat_id=sat_id,
                          mahimahi_ptr=self.mahimahi_ptr[agent], agent=agent)

            self.update_sat_info(sat_id, self.last_mahimahi_time[agent], agent, 1)
            self.update_sat_info(self.cur_sat_id[0], self.last_mahimahi_time[agent], agent, -1)

            self.prev_sat_id[0] = self.cur_sat_id[0]
            self.cur_sat_id[0] = sat_id
            self.download_bw[agent] = []
            for i in range(self.num_agents):
                self.delay[i] = HANDOVER_DELAY
            return sat_id
