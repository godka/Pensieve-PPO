import itertools
from typing import List, Any, Dict
import numpy as np
import pandas as pd

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
DEFAULT_QUALITY = 1
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
DEFAULT_NUMBER_OF_USERS = 30

# MPC
MPC_FUTURE_CHUNK_COUNT = 5
QUALITY_FACTOR = 1
REBUF_PENALTY = 4.3  # pensieve: 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED,
                 video_chunk_len=None, video_bit_rate=None):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        self.last_quality = DEFAULT_QUALITY
        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        
        # connect the satellite that has the best performance at first
        self.prev_sat_id = None
        self.cur_sat_id = self.get_best_sat_id()
        self.available_sat_list = self.get_available_sats_id()
        

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        self.video_bit_rate = video_bit_rate

    def get_video_chunk(self, quality, sat):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        if sat == 1:
            self.cur_sat_id = self.get_best_sat_id()
            self.available_sat_list = self.get_available_sats_id()
            delay += HANDOVER_DELAY

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.cur_sat_id][self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            if throughput == 0.0:
                # Do the forced handover
                # Connect the satellite that has the best performance at first
                # print("Forced Handover")
                # print(self.cooked_bw[self.cur_sat_id][self.mahimahi_ptr-3:self.mahimahi_ptr+3])
                # print(self.mahimahi_ptr)
                self.cur_sat_id = self.get_best_sat_id()
                self.available_sat_list = self.get_available_sats_id()

                delay += HANDOVER_DELAY
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw[self.cur_sat_id]):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw[self.cur_sat_id]):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            
            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0            

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

            # Refresh satellite info
            self.cur_sat_id = self.get_best_sat_id()
            self.available_sat_list = self.get_available_sats_id()

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
        self.last_quality = quality

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain

    def get_best_sat_id(self, mahimahi_ptr=None):
        best_sat_id = None
        best_sat_bw = 0

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr

        for sat_id, sat_bw in self.cooked_bw.items():
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
    

    def get_video_chunk_handover(self, quality, handover_type="naive"):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        end_of_network = False

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.cur_sat_id][self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            if throughput == 0.0:
                # Do the forced handover
                # Connect the satellite that has the best performance at first
                # print("Forced Handover")
                # print(self.cooked_bw[self.cur_sat_id][self.mahimahi_ptr-3:self.mahimahi_ptr+3])
                # print(self.mahimahi_ptr)
                self.prev_sat_id = self.cur_sat_id
                self.cur_sat_id = self.get_best_sat_id()
                self.available_sat_list = self.get_available_sats_id()

                delay += HANDOVER_DELAY
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw[self.cur_sat_id]):
                end_of_network = True
                break

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                if self.mahimahi_ptr >= len(self.cooked_bw[self.cur_sat_id]):
                    end_of_network = True
                    break
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw[self.cur_sat_id]):
                    end_of_network = True
                    break

        # Check Handover
        if not end_of_network:
            handover_result, new_sat_id = self.check_handover(handover_type)
            if handover_result:
                # print("handover")
                delay += HANDOVER_DELAY * MILLISECONDS_IN_SECOND
                self.prev_sat_id = self.cur_sat_id
                self.cur_sat_id = new_sat_id

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            
            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0            

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

            # Refresh satellite info
            self.prev_sat_id = self.cur_sat_id
            self.cur_sat_id = self.get_best_sat_id()
            self.available_sat_list = self.get_available_sats_id()

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain

    def handover_greedy(self, cur_sat_id, mahimahi_ptr):
        harmonic_bw: dict[int:float] = {}
        is_handover = False

        # default assumes that this is the first request so error is 0 since we have never predicted bandwidth
        # past_download = float(video_chunk_size) / float(delay) / M_IN_K
        # Past BW
        # harmonic_bw[cur_sat_id] = self.predict_bw(cur_sat_id, mahimahi_ptr, robustness=True)
        harmonic_bw[cur_sat_id] = self.predict_bw_holt_winter(cur_sat_id, mahimahi_ptr)
        # pred_download_bw = self.predict_download_bw(robustness=True)
        # past_bw = self.cooked_bw[cur_sat_id][mahimahi_ptr - 1]

        best_bw = harmonic_bw[cur_sat_id]
        best_sat_id = cur_sat_id

        # Past Download
        # Calculate the harmonic bw for all satellites
        # Find best satellite & bandwidth
        for sat_id, sat_bw in self.cooked_bw.items():
            if sat_id == cur_sat_id:
                continue
            else:
                # Check if it is visible now
                if self.cooked_bw[sat_id][mahimahi_ptr-1] != 0.0 and self.cooked_bw[sat_id][mahimahi_ptr] != 0.0:
                    # Pass the previously connected satellite
                    if sat_id == self.prev_sat_id:
                        continue
                    """"
                    # Predict the past bw for error estimates of MPC
                    for i in reversed(range(1, MPC_FUTURE_CHUNK_COUNT+1)):
                        self.predict_bw(sat_id, mahimahi_ptr - i)
                    """
                    # harmonic_bw[sat_id] = self.predict_bw(sat_id, mahimahi_ptr, robustness=True)
                    harmonic_bw[sat_id] = self.predict_bw_holt_winter(sat_id, mahimahi_ptr)
                    harmonic_bw[sat_id] -= harmonic_bw[cur_sat_id] * HANDOVER_DELAY * HANDOVER_WEIGHT

                    if best_bw < harmonic_bw[sat_id]:
                        best_bw = harmonic_bw[sat_id]
                        best_sat_id = sat_id
                        is_handover = True

        if is_handover:
            self.prev_sat_id = best_sat_id

        return is_handover, best_sat_id
    

    def predict_bw_holt_winter(self, cur_sat_id, mahimahi_ptr, m=172):
        start_index = mahimahi_ptr - MPC_FUTURE_CHUNK_COUNT
        if start_index < 0:
            start_index = 0
        cur_sat_past_list = [item for item
                             in self.cooked_bw[cur_sat_id][start_index:start_index+MPC_FUTURE_CHUNK_COUNT]]
        while len(cur_sat_past_list) != 0 and cur_sat_past_list[0] == 0.0:
            cur_sat_past_list = cur_sat_past_list[1:]

        if len(cur_sat_past_list) <= 1 or any(v == 0 for v in cur_sat_past_list):
            # Just past bw
            return self.cooked_bw[cur_sat_id][mahimahi_ptr-1]

        cur_sat_past_bws = pd.Series(cur_sat_past_list)
        cur_sat_past_bws.index.freq = 's'

        # alpha = 1 / (2 * m)
        fitted_model = ExponentialSmoothing(cur_sat_past_bws, trend='add').fit()
        # fitted_model = ExponentialSmoothing(cur_sat_past_bws, trend='mul').fit()

        # fitted_model = ExponentialSmoothing(cur_sat_past_bws
        # test_predictions = fitted_model.forecast(5)
        test_predictions = fitted_model.forecast(1)

        pred_bw = sum(test_predictions) / len(test_predictions)
        return pred_bw
    
    def handover_qoe_v2(self, cur_sat_id, mahimahi_ptr, only_runner_up=True):
        is_handover = False
        best_sat_id = cur_sat_id

        ho_sat_id, ho_stamp, best_combo, max_reward = self.calculate_mpc_with_handover(cur_sat_id, mahimahi_ptr,
                                                                                       only_runner_up=only_runner_up)
        # print(cur_sat_id, ho_sat_id, ho_stamp, best_combo, max_reward)
        if ho_stamp == 0:
            is_handover = True
            best_sat_id = ho_sat_id

        return is_handover, best_sat_id

    def get_runner_up_sat_id(self, mahimahi_ptr=None, method="holt-winter"):
        best_sat_id = None
        best_sat_bw = 0

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr

        for sat_id, sat_bw in self.cooked_bw.items():
            target_sat_bw = None
            if sat_id == self.cur_sat_id:
                continue

            if method == "harmonic-mean":
                target_sat_bw = self.predict_bw(sat_id, mahimahi_ptr)
            elif method == "holt-winter":
                target_sat_bw = self.predict_bw_holt_winter(sat_id, mahimahi_ptr)
            else:
                print("Cannot happen")
                exit(1)

            assert(target_sat_bw is not None)
            # Pass the previously connected satellite
            if sat_id == self.prev_sat_id:
                continue
            if best_sat_bw < target_sat_bw:
                best_sat_id = sat_id
                best_sat_bw = target_sat_bw

        return best_sat_id, best_sat_bw

    def calculate_mpc_with_handover(self, cur_sat_id, mahimahi_ptr, robustness=True, only_runner_up=True,
                                    method="holt-winter"):
        # future chunks length (try 4 if that many remaining)
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter
        last_index = TOTAL_VIDEO_CHUNCK - video_chunk_remain

        chunk_combo_option = []
        # make chunk combination options
        for combo in itertools.product(list(range(BITRATE_LEVELS)), repeat=MPC_FUTURE_CHUNK_COUNT):
            chunk_combo_option.append(combo)

        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if video_chunk_remain < MPC_FUTURE_CHUNK_COUNT:
            future_chunk_length = video_chunk_remain

        max_reward = -10000000
        best_combo = ()
        ho_sat_id = cur_sat_id
        ho_stamp = -1
        cur_harmonic_bw, runner_up_sat_id = None, None
        if method == "harmonic-mean":
            cur_harmonic_bw = self.predict_bw(cur_sat_id, mahimahi_ptr, robustness)
            runner_up_sat_id, _ = self.get_runner_up_sat_id(mahimahi_ptr, method="harmonic-mean")
        elif method == "holt-winter":
            cur_harmonic_bw = self.predict_bw_holt_winter(cur_sat_id, mahimahi_ptr)
            runner_up_sat_id, _ = self.get_runner_up_sat_id(mahimahi_ptr, method="holt-winter")
        else:
            print("Cannot happen")
            exit(1)
        start_buffer = self.buffer_size / MILLISECONDS_IN_SECOND

        if future_chunk_length == 0:
            return ho_sat_id, ho_stamp, best_combo, max_reward

        for next_sat_id, next_sat_bw in self.cooked_bw.items():
            if next_sat_id == cur_sat_id:
                continue
            else:
                # Check if it is visible now
                if self.cooked_bw[next_sat_id][mahimahi_ptr - 1] != 0.0 and self.cooked_bw[next_sat_id][mahimahi_ptr] != 0.0:
                    # Pass the previously connected satellite
                    if next_sat_id == self.prev_sat_id:
                        continue

                    if only_runner_up and runner_up_sat_id != next_sat_id:
                        # Only consider the next-best satellite
                        continue

                    # Based on the bw, not download bw
                    next_harmonic_bw = None
                    if method == "harmonic-mean":
                        next_harmonic_bw = self.predict_bw(next_sat_id, mahimahi_ptr, robustness)
                    elif method == "holt-winter":
                        next_harmonic_bw = self.predict_bw_holt_winter(next_sat_id, mahimahi_ptr)
                    else:
                        print("Cannot happen")
                        exit(1)

                    for ho_index in range(MPC_FUTURE_CHUNK_COUNT+1):
                        # all possible combinations of 5 chunk bitrates for 6 bitrate options (6^5 options)
                        # iterate over list and for each, compute reward and store max reward combination
                        # ho_index: 0-4 -> Do handover, 5 -> Do not handover
                        for full_combo in chunk_combo_option:
                            # Break at the end of the chunk
                            combo = full_combo[0: future_chunk_length]
                            # calculate total rebuffer time for this combination (start with start_buffer and subtract
                            # each download time and add 2 seconds in that order)
                            curr_rebuffer_time = 0
                            curr_buffer = start_buffer
                            bitrate_sum = 0
                            smoothness_diffs = 0
                            last_quality = self.last_quality
                            is_impossible = False

                            for position in range(0, len(combo)):
                                chunk_quality = combo[position]
                                index = last_index + position  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                                download_time = 0
                                if ho_index > position:
                                    harmonic_bw = cur_harmonic_bw

                                elif ho_index == position:
                                    harmonic_bw = next_harmonic_bw
                                    # Give them a penalty
                                    download_time += HANDOVER_DELAY
                                else:
                                    harmonic_bw = next_harmonic_bw

                                download_time += (self.video_size[chunk_quality][index] / B_IN_MB) \
                                                / harmonic_bw * BITS_IN_BYTE  # this is MB/MB/s --> seconds

                                if curr_buffer < download_time:
                                    curr_rebuffer_time += (download_time - curr_buffer)
                                    curr_buffer = 0.0
                                else:
                                    curr_buffer -= download_time
                                curr_buffer += TOTAL_VIDEO_CHUNCK / MILLISECONDS_IN_SECOND

                                # bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                                # smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                                bitrate_sum += self.video_bit_rate[chunk_quality]
                                smoothness_diffs += abs(
                                    self.video_bit_rate[chunk_quality] - self.video_bit_rate[last_quality])
                                last_quality = chunk_quality
                            # compute reward for this combination (one reward per 5-chunk combo)

                            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

                            # 10~140 - 0~100 - 0~130
                            reward = bitrate_sum * QUALITY_FACTOR - (REBUF_PENALTY * curr_rebuffer_time) \
                                     - SMOOTH_PENALTY * smoothness_diffs

                            if reward > max_reward:
                                best_combo = combo
                                max_reward = reward
                                ho_sat_id = next_sat_id
                                ho_stamp = ho_index
                            elif reward == max_reward and (sum(combo) > sum(best_combo) or ho_index == MPC_FUTURE_CHUNK_COUNT):
                                best_combo = combo
                                max_reward = reward
                                ho_sat_id = next_sat_id
                                ho_stamp = ho_index

        return ho_sat_id, ho_stamp, best_combo, max_reward


    def check_handover(self, handover_type, cur_sat_id=None, mahimahi_ptr=None):
        # Check Handover
        is_handover = False

        if cur_sat_id is None:
            cur_sat_id = self.cur_sat_id

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr

        new_sat_id = cur_sat_id
        if handover_type == "naive":
            is_handover, new_sat_id = self.handover_naive(cur_sat_id, mahimahi_ptr)
        elif handover_type == "truth-mpc":
            is_handover, new_sat_id = self.handover_mpc_truth(cur_sat_id, mahimahi_ptr)
        elif handover_type == "greedy":
            is_handover, new_sat_id = self.handover_greedy(cur_sat_id, mahimahi_ptr)
        elif handover_type == "mpc-greedy":
            is_handover, new_sat_id = self.handover_mpc_greedy(cur_sat_id, mahimahi_ptr)
        elif handover_type == "QoE-all":
            is_handover, new_sat_id = self.handover_qoe_v2(cur_sat_id, mahimahi_ptr, only_runner_up=False)
        elif handover_type == "QoE-pruned":
            is_handover, new_sat_id = self.handover_qoe_v2(cur_sat_id, mahimahi_ptr, only_runner_up=True)
        else:
            print("Cannot happen!")
            exit(-1)
        return is_handover, new_sat_id


