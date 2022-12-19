import numpy as np
import structlog

from muleo_lc_bw_share import load_trace
from muleo_lc_bw_share import fixed_env_exhaustive as env
import matplotlib.pyplot as plt
import itertools
import os
import logging

from util.constants import VIDEO_BIT_RATE, BUFFER_NORM_FACTOR, CHUNK_TIL_VIDEO_END_CAP, M_IN_K, REBUF_PENALTY, \
    SMOOTH_PENALTY, DEFAULT_QUALITY, MPC_FUTURE_CHUNK_COUNT

VIDEO_CHOICES = 6

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
TOTAL_VIDEO_CHUNKS = 48
RANDOM_SEED = 42
RAND_RANGE = 1000000
SUMMARY_DIR = 'test_results_reduced_resource_n4/'
LOG_FILE = SUMMARY_DIR + 'log_sim_cent'
TEST_TRACES = './test_tight/'
SUMMARY_PATH = SUMMARY_DIR + 'summary'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

CHUNK_COMBO_OPTIONS = []

import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--user', type=int, default=4)
args = parser.parse_args()
NUM_AGENTS = args.user

# past errors in bandwidth

past_errors = [[] for _ in range(NUM_AGENTS)]
past_bandwidth_ests = [[] for _ in range(NUM_AGENTS)]

size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522,
               2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469,
               2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074,
               2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102,
               2124950, 2246506, 1961140, 2155012, 1433658]
size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548,
               1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126,
               1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081,
               1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250,
               1464921, 1483527, 1231456, 1364537, 889412]
size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851,
               1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935,
               1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587,
               908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282,
               687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335,
               696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884,
               587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351,
               434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700,
               425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327,
               390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746,
               179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938,
               181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254,
               149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

MPC_TYPE = "DualMPC"
MPC_TYPE = "DualMPC-Centralization-Exhaustive"
MPC_TYPE = "DualMPC-Centralization-Reduced"

# DualMPC-Centralization

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
)


def get_chunk_size(quality, index):
    if index < 0 or index > 48:
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index],
             1: size_video5[index], 0: size_video6[index]}
    return sizes[quality]


def main():
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES, NUM_AGENTS)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              num_agents=NUM_AGENTS)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]

    os.system('rm -r ' + SUMMARY_DIR)
    # os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    log_file = open(log_path, 'w')

    time_stamp = [0 for _ in range(NUM_AGENTS)]

    last_bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
    bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]

    action_vec = [np.zeros(A_DIM) for _ in range(NUM_AGENTS)]
    for i in range(NUM_AGENTS):
        action_vec[i][DEFAULT_QUALITY] = 1

    s_batch = [[np.zeros((S_INFO, S_LEN))] for _ in range(NUM_AGENTS)]
    a_batch = [[action_vec] for _ in range(NUM_AGENTS)]
    r_batch = [[] for _ in range(NUM_AGENTS)]
    entropy_record = [[] for _ in range(NUM_AGENTS)]

    video_count = 0

    results = []
    tmp_results = []
    best_user_infos = []
    do_mpc = False
    end_of_video = False

    # make chunk combination options
    for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
        CHUNK_COMBO_OPTIONS.append(combo)
    ho_stamps_log = [MPC_FUTURE_CHUNK_COUNT for _ in range(NUM_AGENTS)]
    combo_log = [[DEFAULT_QUALITY] for _ in range(NUM_AGENTS)]
    next_sat_log = [None for _ in range(NUM_AGENTS)]

    while True:  # serve video forever
        agent = net_env.get_first_agent()

        if agent == -1 or end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
            bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
            net_env.reset()

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = [np.zeros(A_DIM) for _ in range(NUM_AGENTS)]
            for i in range(NUM_AGENTS):
                action_vec[i][bit_rate[agent]] = 1

            s_batch = [[np.zeros((S_INFO, S_LEN))] for _ in range(NUM_AGENTS)]
            a_batch = [[action_vec] for _ in range(NUM_AGENTS)]
            r_batch = [[] for _ in range(NUM_AGENTS)]
            entropy_record = [[] for _ in range(NUM_AGENTS)]

            print("network count", video_count)
            print(sum(tmp_results) / len(tmp_results))
            summary_file = open(SUMMARY_PATH, 'a')
            summary_file.write(str(best_user_infos))
            summary_file.write('\n')
            summary_file.write(str(sum(tmp_results) / len(tmp_results)))
            summary_file.close()

            results += tmp_results
            tmp_results = []
            best_user_infos = []
            video_count += 1
            # break

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

            ho_stamps_log = [MPC_FUTURE_CHUNK_COUNT for _ in range(NUM_AGENTS)]
            ho_point = MPC_FUTURE_CHUNK_COUNT
            combo_log = [[DEFAULT_QUALITY] for _ in range(NUM_AGENTS)]
            next_sat_log = [None for _ in range(NUM_AGENTS)]
            end_of_video = False
            continue
        else:
            bit_rate[agent] = combo_log[agent].pop(0)
            ho_point = ho_stamps_log[agent]
            ho_stamps_log[agent] -= 1
            if np.isnan(bit_rate[agent]):
                bit_rate[agent] = DEFAULT_QUALITY

            # np.delete(combo_log[agent], 0)
            # if len(combo_log[agent]) == 1 and agent == net_env.get_first_agent():
            if not combo_log[agent]:
                do_mpc = True
            # do_mpc = True

        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, is_handover, sat_status, _, _, _, _, _, _, cur_sat_id, \
        runner_up_sat_ids, ho_stamps, best_combos, best_user_info \
            = net_env.get_video_chunk(bit_rate[agent], agent, MPC_TYPE, next_sat_log[agent], ho_point, do_mpc)

        is_handover = True if ho_point == 0 else False

        if agent == 0 or do_mpc is True:
            do_mpc = False

            ho_stamps_log = ho_stamps
            combo_log = best_combos
            next_sat_log = runner_up_sat_ids

        time_stamp[agent] += delay  # in ms
        time_stamp[agent] += sleep_time  # in ms

        if best_user_info:
            best_user_info["time"] = time_stamp[agent] / M_IN_K
            best_user_infos.append(best_user_info)

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate[agent]] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate[agent]] -
                                           VIDEO_BIT_RATE[last_bit_rate[agent]]) / M_IN_K

        r_batch[agent].append(reward)
        tmp_results.append(reward)

        # print(net_env.video_chunk_counter)
        # print(len(net_env.cooked_bw[1161]))
        # if agent == 0:
        #     print(reward, bit_rate[agent], delay, sleep_time, buffer_size, rebuf, \
        #         video_chunk_size, next_video_chunk_sizes, \
        #         end_of_video, video_chunk_remain)

        last_bit_rate[agent] = bit_rate[agent]

        if agent is not None:
            # log time_stamp, bit_rate, buffer_size, reward
            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write("{: <15} {: <10} {: <10} {: <15} {: <15} {: <15}"
                           " {: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15}\n"
                           .format(str(round(time_stamp[agent] / M_IN_K, 3)), str(agent),
                                   str(VIDEO_BIT_RATE[bit_rate[agent]]), str(round(buffer_size, 3)),
                                   str(round(rebuf, 3)),
                                   str(round(video_chunk_size, 3)), str(round(delay, 3)), str(round(reward, 3)),
                                   str(cur_sat_id), str(is_handover), str(sat_status), str(ho_stamps), str(best_user_info)))
            log_file.flush()

        # retrieve previous state
        if len(s_batch[agent]) == 0:
            state = [[np.zeros((S_INFO, S_LEN))] for _ in range(NUM_AGENTS)]
        else:
            state = [np.array(s_batch[agent][-1], copy=True) for agent in range(NUM_AGENTS)]

        # dequeue history record
        state[agent] = np.roll(state[agent], -1, axis=1)

        # this should be S_INFO number of terms
        state[agent][0, -1] = VIDEO_BIT_RATE[bit_rate[agent]] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[agent][1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[agent][2, -1] = rebuf
        state[agent][3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[agent][4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch[agent].append(state[agent])

    # print(results, sum(results))
    print(sum(results) / len(results))


if __name__ == '__main__':
    main()
