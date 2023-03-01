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
    VIDEO_CHUNCK_LEN, BITRATE_LEVELS, B_IN_MB, BITS_IN_BYTE, M_IN_K, MILLISECONDS_IN_SECOND, PAST_LEN, VIDEO_SIZE_FILE

RANDOM_SEED = 42
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1

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
        self.sat_decision_log = [[-1, -1, -1, -1, -1] for _ in range(self.num_agents)]

        for sat_id, sat_bw in self.cooked_bw.items():
            self.num_sat_info[sat_id] = [0 for _ in range(len(sat_bw))]
            self.num_of_user_sat[sat_id] = 0
            self.cur_satellite[sat_id] = Satellite(sat_id, sat_bw, SAT_STRATEGY)

        self.cur_user = [User(i, SNR_MIN) for i in range(self.num_agents)]

        # print(self.num_sat_info)
        self.prev_best_combos = [[DEFAULT_QUALITY] * MPC_FUTURE_CHUNK_COUNT] * self.num_agents

        # multiuser setting
        self.cur_sat_id = []
        self.prev_sat_id = [None for _ in range(self.num_agents)]
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent)
            self.cur_sat_id.append(cur_sat_id)
            self.update_sat_info(cur_sat_id, self.last_mahimahi_time[agent], agent, 1)

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

        # Do All users' handover

        self.last_quality[agent] = quality

        if ho_stamp and ho_stamp == "MRSS":
            tmp_best_id = self.get_max_sat_id(agent)
            if tmp_best_id != self.cur_sat_id[agent]:
                is_handover = True
                delay += HANDOVER_DELAY
                self.update_sat_info(self.cur_sat_id[agent], self.last_mahimahi_time[agent], agent, -1)
                self.update_sat_info(tmp_best_id, self.last_mahimahi_time[agent], agent, 1)
                self.prev_sat_id[agent] = self.cur_sat_id[agent]
                self.cur_sat_id[agent] = tmp_best_id
                self.download_bw[agent] = []

            throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent], self.mahimahi_ptr[
                agent]) * B_IN_MB / BITS_IN_BYTE
            # assert throughput != 0
        elif ho_stamp and ho_stamp == "MRSS-Smart":
            tmp_best_id = self.get_max_sat_id(agent, past_len=PAST_LEN)
            if tmp_best_id != self.cur_sat_id[agent]:
                is_handover = True
                delay += HANDOVER_DELAY
                self.update_sat_info(self.cur_sat_id[agent], self.last_mahimahi_time[agent], agent, -1)
                self.update_sat_info(tmp_best_id, self.last_mahimahi_time[agent], agent, 1)
                self.prev_sat_id[agent] = self.cur_sat_id[agent]
                self.cur_sat_id[agent] = tmp_best_id
                self.download_bw[agent] = []

            throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent], self.mahimahi_ptr[
                agent]) * B_IN_MB / BITS_IN_BYTE
            # assert throughput != 0

        while True:  # download video chunk over mahimahi
            throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent],
                                                                              self.mahimahi_ptr[
                                                                                  agent]) * B_IN_MB / BITS_IN_BYTE
            if throughput == 0.0:
                if ho_stamp and ho_stamp == "MVT":
                    sat_id = self.get_mvt_sat_id(agent, self.mahimahi_ptr[agent])
                else:
                    sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent])
                self.log.debug("Forced Handover1", cur_sat_id=self.cur_sat_id[agent], next_sat_id=sat_id,
                               mahimahi_ptr=self.mahimahi_ptr[agent], agent=agent,
                               cur_bw=self.cooked_bw[self.cur_sat_id[agent]][
                                      self.mahimahi_ptr[agent] - 3:self.mahimahi_ptr[agent] + 3],
                               next_bw=self.cooked_bw[
                                           sat_id][self.mahimahi_ptr[agent] - 3:self.mahimahi_ptr[agent] + 3]
                               )
                assert self.cur_sat_id[agent] != sat_id
                self.update_sat_info(self.cur_sat_id[agent], self.last_mahimahi_time[agent], agent, -1)
                self.update_sat_info(sat_id, self.last_mahimahi_time[agent], agent, 1)

                self.switch_sat(agent, sat_id)
                delay += HANDOVER_DELAY
                is_handover = True
                self.download_bw[agent] = []
                self.unexpected_change = True
                throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent],
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

            if self.mahimahi_ptr[agent] >= len(self.cooked_bw[self.cur_sat_id[agent]]):
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
                if self.mahimahi_ptr[agent] >= len(self.cooked_bw[self.cur_sat_id[agent]]):
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

                throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent],
                                                                                  self.mahimahi_ptr[
                                                                                      agent]) * B_IN_MB / BITS_IN_BYTE
                if throughput == 0.0:
                    # Do the forced handover
                    # Connect the satellite that has the best serving time
                    if ho_stamp and ho_stamp == "MVT":
                        sat_id = self.get_mvt_sat_id(agent, self.mahimahi_ptr[agent])
                    else:
                        sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent])

                    assert sat_id != self.cur_sat_id[agent]
                    self.log.debug("Forced Handover2", cur_sat_id=self.cur_sat_id[agent], sat_id=sat_id,
                                   mahimahi_ptr=self.mahimahi_ptr[agent], agent=agent)
                    assert sat_id != self.cur_sat_id[agent]
                    self.update_sat_info(sat_id, self.last_mahimahi_time[agent], agent, 1)
                    self.update_sat_info(self.cur_sat_id[agent], self.last_mahimahi_time[agent], agent, -1)
                    self.switch_sat(agent, sat_id)
                    is_handover = True
                    delay += HANDOVER_DELAY * MILLISECONDS_IN_SECOND
                    throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent],
                                                                                      self.mahimahi_ptr[
                                                                                          agent]) * B_IN_MB / BITS_IN_BYTE

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
            self.update_sat_info(self.cur_sat_id[agent], self.last_mahimahi_time[agent], agent, -1)

            # Refresh satellite info
            # self.connection[self.cur_sat_id[agent]] = -1
            # self.cur_sat_id[agent] = None

            # wait for overall clean
            cur_sat_bw_logs, next_sat_bandwidth, next_sat_id, next_sat_bw_logs, connected_time, other_sat_users, other_sat_bw_logs = [], [], None, [], [0, 0], {}, {}
        else:
            cur_sat_bw_logs, next_sat_bandwidth, next_sat_id, next_sat_bw_logs, connected_time, other_sat_users, other_sat_bw_logs = self.get_next_sat_info(
                agent, self.mahimahi_ptr[agent])
        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter[agent]])

        self.video_chunk_remain[agent] = video_chunk_remain
        self.download_bw[agent].append(float(video_chunk_size) / delay / M_IN_K * BITS_IN_BYTE)

        # num of users
        cur_sat_user_num = len(self.cur_satellite[self.cur_sat_id[agent]].get_ue_list(self.mahimahi_ptr[agent]))
        self.next_sat_id[agent] = next_sat_id
        if next_sat_id:
            next_sat_user_num = len(self.cur_satellite[next_sat_id].get_ue_list(self.mahimahi_ptr[agent]))
        else:
            next_sat_user_num = 0

        self.sat_decision_log[agent].append(self.cur_sat_id[agent])

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
               self.cur_sat_id[
                   agent], runner_up_sat_ids, ho_stamps, best_combos, final_rate, quality, other_sat_users, other_sat_bw_logs, \
               np.delete(self.buffer_size, agent)

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
        self.sat_decision_log = [[-1, -1, -1, -1, -1] for _ in range(self.num_agents)]

        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        for sat_id, sat_bw in self.cooked_bw.items():
            self.num_sat_info[sat_id] = [0 for _ in range(len(sat_bw))]
            self.num_of_user_sat[sat_id] = 0
            self.cur_satellite[sat_id] = Satellite(sat_id, sat_bw, SAT_STRATEGY)

        self.cur_user = []
        for agent_id in range(self.num_agents):
            self.cur_user.append(User(agent_id, SNR_MIN))

        self.mahimahi_ptr = [np.random.randint(1, len(self.cooked_time) - TOTAL_VIDEO_CHUNKS)] * self.num_agents
        self.last_mahimahi_time = [self.mahimahi_ptr[i] - 1 for i in range(self.num_agents)]

        self.cur_sat_id = []
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent)
            # self.connection[cur_sat_id] = agent
            self.cur_sat_id.append(cur_sat_id)
            self.update_sat_info(cur_sat_id, self.last_mahimahi_time[agent], agent, 1)

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
        other_sat_users = {}
        other_sat_bw_logs = {}

        for sat_id, sat_bw in self.cooked_bw.items():
            bw_list = []
            if sat_id == self.cur_sat_id[agent]:
                continue
            for i in range(5, 0, -1):
                if mahimahi_ptr - i >= 0 and sat_bw[mahimahi_ptr - i] != 0:
                    if self.get_num_of_user_sat(self.mahimahi_ptr[agent], sat_id) == 0:
                        bw_list.append(sat_bw[mahimahi_ptr - i])
                    else:
                        bw_list.append(
                            sat_bw[mahimahi_ptr - i] / (self.get_num_of_user_sat(self.mahimahi_ptr[agent], sat_id)))
            if len(bw_list) == 0:
                continue
            bw = sum(bw_list) / len(bw_list)
            other_sat_users[sat_id] = self.get_num_of_user_sat(self.mahimahi_ptr[agent], sat_id)

            other_sat_bw_logs[sat_id] = bw_list

            if best_sat_bw < bw:
                best_sat_id = sat_id
                best_sat_bw = bw
                best_bw_list = bw_list

        if best_sat_id is None:
            best_sat_id = self.cur_sat_id[agent]

        if best_sat_id in other_sat_users:
            del other_sat_users[best_sat_id]
        if best_sat_id in other_sat_bw_logs:
            del other_sat_bw_logs[best_sat_id]

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        list1, next_sat_id, next_sat_bws = [], [], []
        bw_list = []
        sat_bw = self.cooked_bw[self.cur_sat_id[agent]]
        for i in range(PAST_LEN, 1, -1):
            if mahimahi_ptr - i >= 0:
                if len(self.cur_satellite[self.cur_sat_id[agent]].get_ue_list(mahimahi_ptr)) == 0:
                    bw_list.append(sat_bw[mahimahi_ptr - i])
                else:
                    bw_list.append(
                        sat_bw[mahimahi_ptr - i] / len(
                            self.cur_satellite[self.cur_sat_id[agent]].get_ue_list(mahimahi_ptr)))

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
                        bw_list.append(sat_bw[mahimahi_ptr - i] / (
                                    len(self.cur_satellite[runner_up_sat_id].get_ue_list(mahimahi_ptr)) + 1))
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

        return cur_sat_bws, None, next_sat_id, next_sat_bws, up_time_list, other_sat_users, other_sat_bw_logs

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

    def get_mvt_sat_id(self, agent, mahimahi_ptr=None):
        best_sat_id = None
        best_sat_time = 0

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        for sat_id, sat_bw in self.cooked_bw.items():
            tmp_time = 0
            while True:
                tmp_mahimahi_ptr = mahimahi_ptr
                real_sat_bw = self.cur_satellite[sat_id].data_rate_unshared(self.cur_user[agent], tmp_mahimahi_ptr)
                if real_sat_bw == 0 or tmp_mahimahi_ptr <= 0:
                    break
                tmp_mahimahi_ptr -= 1
                tmp_time += 1
            if best_sat_time < tmp_time:
                best_sat_id = sat_id
                best_sat_time = tmp_time

        return best_sat_id

    def get_max_sat_id(self, agent, mahimahi_ptr=None, past_len=None):
        best_sat_id = None
        best_sat_bw = 0

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        for sat_id, sat_bw in self.cooked_bw.items():
            if past_len:
                real_sat_bw = self.predict_bw(sat_id, agent, robustness=True, mahimahi_ptr=mahimahi_ptr,
                                              past_len=PAST_LEN)
            else:
                real_sat_bw = self.cur_satellite[sat_id].data_rate_unshared(mahimahi_ptr, self.cur_user[agent])

            if best_sat_bw < real_sat_bw:
                best_sat_id = sat_id
                best_sat_bw = real_sat_bw

        return best_sat_id

    def switch_sat(self, agent, cur_sat_id):
        pre_sat_id = self.cur_sat_id[agent]
        self.prev_sat_id[agent] = pre_sat_id

        # self.connection[pre_sat_id] = -1
        # self.connection[cur_sat_id] = agent

        self.cur_sat_id[agent] = cur_sat_id

    def update_sat_info(self, sat_id, mahimahi_ptr, agent, variation):
        # update sat info
        if variation == 1:
            self.cur_satellite[sat_id].add_ue(agent, mahimahi_ptr)
        elif variation == -1:
            self.cur_satellite[sat_id].remove_ue(agent, mahimahi_ptr)

        if sat_id in self.num_of_user_sat.keys():
            self.num_of_user_sat[sat_id] += variation
        else:
            self.num_of_user_sat[sat_id] = variation
        # print(self.num_of_user_sat)
        assert self.num_of_user_sat[sat_id] >= 0

    def get_num_of_user_sat(self, mahimahi_ptr, sat_id):
        # update sat info
        assert sat_id is not None
        if sat_id == "all":
            filtered_num_of_user_sat = {}
            for tmp_sat_id in self.cur_satellite.keys():
                if len(self.cur_satellite[tmp_sat_id].get_ue_list(mahimahi_ptr)) != 0:
                    filtered_num_of_user_sat[tmp_sat_id] = len(self.cur_satellite[tmp_sat_id].get_ue_list(mahimahi_ptr))
            return filtered_num_of_user_sat
        if sat_id in self.cur_satellite.keys():
            return len(self.cur_satellite[sat_id].get_ue_list(mahimahi_ptr))
        self.log.info("Error", sat_id=sat_id, cur_sat_ids=self.cur_satellite.keys(), mahimahi_ptr=mahimahi_ptr)
        raise Exception

    def get_runner_up_sat_id(self, agent, method="holt-winter", mahimahi_ptr=None, cur_sat_id=None, plus=False):
        best_sat_id = None
        best_sat_bw = 0
        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]
        if cur_sat_id is None:
            cur_sat_id = self.cur_sat_id[agent]

        for sat_id, sat_bw in self.cooked_bw.items():
            real_sat_bw = None

            # Pass the previously connected satellite
            # if sat_id == cur_sat_id or sat_id == self.prev_sat_id[agent]:
            if sat_id == cur_sat_id or sat_id == self.prev_sat_id[agent]:
                continue

            if method == "harmonic-mean":
                target_sat_bw = self.predict_bw_num(sat_id, agent, mahimahi_ptr=mahimahi_ptr, past_len=PAST_LEN)
                real_sat_bw = target_sat_bw  # / (self.get_num_of_user_sat(sat_id) + 1)
            else:
                print("Cannot happen")
                exit(1)

            assert (real_sat_bw is not None)
            if best_sat_bw < real_sat_bw:
                best_sat_id = sat_id
                best_sat_bw = real_sat_bw

        return best_sat_id, best_sat_bw

    def predict_bw(self, sat_id, agent, robustness=True, mahimahi_ptr=None, past_len=None):
        curr_error = 0
        if sat_id is None:
            return 0
        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        if mahimahi_ptr <= 0:
            return self.cur_satellite[sat_id].data_rate_unshared(0, self.cur_user[agent])

        if past_len:
            for i in range(past_len, 1, -1):
                if mahimahi_ptr - i > 0:
                    self.predict_bw(sat_id, agent, robustness, mahimahi_ptr=mahimahi_ptr - i)

        # num_of_user_sat = len(self.cur_satellite[sat_id].get_ue_list()) + 1

        # past_bw = self.cooked_bw[self.cur_sat_id][self.mahimahi_ptr - 1]

        past_bw = self.cur_satellite[sat_id].data_rate_unshared(mahimahi_ptr - 1, self.cur_user[agent])

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
            past_bws.append(self.cur_satellite[sat_id].data_rate_unshared(index, self.cur_user[agent]))
        # Newly possible satellite case
        if all(v == 0.0 for v in past_bws):
            return self.cur_satellite[sat_id].data_rate_unshared(mahimahi_ptr, self.cur_user[agent])

        while past_bws[0] == 0.0:
            past_bws = past_bws[1:]

        bandwidth_sum = 0
        bandwidth_index = 0
        for past_val in past_bws:
            if past_val != 0:
                bandwidth_sum += (1 / float(past_val))
                bandwidth_index += 1

        harmonic_bw = 1.0 / (bandwidth_sum / bandwidth_index)

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

    def predict_bw_num(self, sat_id, agent, robustness=True, mahimahi_ptr=None, past_len=None):
        curr_error = 0
        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        if mahimahi_ptr <= 0:
            return self.cur_satellite[sat_id].data_rate_unshared(0, self.cur_user[agent])

        if past_len:
            for i in range(past_len, 1, -1):
                if mahimahi_ptr - i > 0:
                    self.predict_bw_num(sat_id, agent, robustness, mahimahi_ptr=mahimahi_ptr - i)

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
        bandwidth_index = 0
        for past_val in past_bws:
            if past_val != 0:
                bandwidth_sum += (1 / float(past_val))
                bandwidth_index += 1

        harmonic_bw = 1.0 / (bandwidth_sum / bandwidth_index)

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

        if sat == 1:
            if sat_id == self.cur_sat_id[agent] or sat_id is None:
                # print("Can't do handover. Only one visible satellite")
                return
            self.log.debug("set_satellite", cur_sat_id=self.cur_sat_id[agent], next_sat_id=sat_id,
                           mahimahi_ptr=self.last_mahimahi_time[agent], agent=agent)

            self.update_sat_info(sat_id, self.last_mahimahi_time[agent], agent, 1)
            self.update_sat_info(self.cur_sat_id[agent], self.last_mahimahi_time[agent], agent, -1)
            self.prev_sat_id[agent] = self.cur_sat_id[agent]
            self.cur_sat_id[agent] = sat_id
            self.download_bw[agent] = []
            self.delay[agent] = HANDOVER_DELAY
            return sat_id

    def get_others_reward(self, agent, last_bit_rate, prev_sat_id, cur_sat_id):
        reward = 0
        prev_sat_id = self.prev_sat_id[agent]
        cur_sat_id = self.cur_sat_id[agent]
        for i in range(self.num_agents):
            if i == agent:
                continue
            agent_cur_sat_id = self.cur_sat_id[i]
            if agent_cur_sat_id in [cur_sat_id, prev_sat_id]:
                reward += self.get_simulated_penalty(i, last_bit_rate[i], prev_sat_id, cur_sat_id) / self.num_agents / 10

        return reward

    def get_simulated_penalty(self, agent, quality, prev_sat_id, next_sat_id):
        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter[agent]]
        last_mahimahi_time = self.last_mahimahi_time[agent]
        mahimahi_ptr = self.mahimahi_ptr[agent]
        cur_sat_id = self.cur_sat_id[agent]
        delay = self.delay[agent]
        buffer_size = self.buffer_size[agent]
        assert cur_sat_id in [prev_sat_id, next_sat_id]

        if cur_sat_id == prev_sat_id:
            reward1 = self.calculate_reward(agent, cur_sat_id, video_chunk_size, last_mahimahi_time, mahimahi_ptr,
                                            delay, buffer_size, self.get_num_of_user_sat(mahimahi_ptr, cur_sat_id) + 1)
            reward2 = self.calculate_reward(agent, cur_sat_id, video_chunk_size, last_mahimahi_time, mahimahi_ptr,
                                            delay, buffer_size, self.get_num_of_user_sat(mahimahi_ptr, cur_sat_id))
        elif cur_sat_id == next_sat_id:
            reward1 = self.calculate_reward(agent, cur_sat_id, video_chunk_size, last_mahimahi_time, mahimahi_ptr,
                                            delay, buffer_size, self.get_num_of_user_sat(mahimahi_ptr, cur_sat_id) - 1)
            reward2 = self.calculate_reward(agent, cur_sat_id, video_chunk_size, last_mahimahi_time, mahimahi_ptr,
                                            delay, buffer_size, self.get_num_of_user_sat(mahimahi_ptr, cur_sat_id))
        else:
            print("Cannot Happen")
        return reward1 - reward2

    def calculate_reward(self, agent, cur_sat_id, video_chunk_size, last_mahimahi_time, mahimahi_ptr, delay, buffer_size, num_of_user):
        video_chunk_counter_sent = 0  # in bytes
        while True:  # download video chunk over mahimahi
            if len(self.cooked_time) <= mahimahi_ptr:
                return 0
            throughput = self.cur_satellite[cur_sat_id].data_rate(self.cur_user[agent], mahimahi_ptr) * B_IN_MB / BITS_IN_BYTE

            if throughput == 0.0:
                # Do the forced handover
                # Connect the satellite that has the best serving time
                cur_sat_id = self.get_best_sat_id(agent, mahimahi_ptr)
                # self.update_sat_info(cur_sat_id, self.mahimahi_ptr[agent], 1)
                # self.update_sat_info(self.cur_sat_id[agent], self.mahimahi_ptr[agent], -1)

                # self.cur_sat_id[agent] = cur_sat_id
                delay += HANDOVER_DELAY

            duration = self.cooked_time[mahimahi_ptr] \
                       - last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration

            last_mahimahi_time = self.cooked_time[mahimahi_ptr]
            mahimahi_ptr += 1
            # self.step_ahead(agent)
            if mahimahi_ptr >= len(self.cooked_bw[cur_sat_id]):
                # loop back in the beginning
                # note: trace file starts with time 0
                mahimahi_ptr = 1
                last_mahimahi_time = 0
                end_of_video = True
                break

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # rebuffer time
        rebuf = np.maximum(delay - buffer_size, 0.0)

        M_IN_K = 1000.0
        REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
        SMOOTH_PENALTY = 1
        DEFAULT_QUALITY = 1  # default video quality without agent

        reward = REBUF_PENALTY * rebuf / MILLISECONDS_IN_SECOND

        return reward

