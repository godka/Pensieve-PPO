import numpy as np
import structlog
import sys
import pathlib
import os

from env.multi_bw_share_multi_session import fixed_env_time as env, load_trace as load_trace
import itertools
import logging

from util.constants import VIDEO_BIT_RATE, BUFFER_NORM_FACTOR, CHUNK_TIL_VIDEO_END_CAP, M_IN_K, REBUF_PENALTY, \
    SMOOTH_PENALTY, DEFAULT_QUALITY, MPC_FUTURE_CHUNK_COUNT, size_video1, size_video2, size_video3, size_video4, \
    size_video5, size_video6, NUM_AGENTS

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
RANDOM_SEED = 42
RAND_RANGE = 1000000
SUMMARY_DIR = 'test_results_cent_res/'
LOG_FILE = SUMMARY_DIR + 'log_sim_cent'
TEST_TRACES = 'data/sat_data/test_tight/'
SUMMARY_PATH = SUMMARY_DIR + 'summary'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

CHUNK_COMBO_OPTIONS = []

import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--user', type=int, default=3)
args = parser.parse_args()

# MPC_TYPE = "DualMPC-v1"
# MPC_TYPE = "DualMPC-Centralization-Exhaustive"
MPC_TYPE = "DualMPC-Centralization-Reduced-v2"

# DualMPC-Centralization

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
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

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

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

        if agent == -1:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
            bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
            net_env.reset()

            print("network count", video_count)
            print(sum(tmp_results) / len(tmp_results))
            summary_file = open(SUMMARY_PATH, 'a')
            summary_file.write('\n')
            summary_file.write(str(best_user_infos))
            summary_file.write('\n')
            summary_file.write(str(sum(tmp_results) / len(tmp_results)))
            summary_file.close()

            results += tmp_results
            tmp_results = []
            best_user_infos = []
            video_count += 1
            time_stamp = [0 for _ in range(NUM_AGENTS)]

            # break

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

            ho_stamps_log = [MPC_FUTURE_CHUNK_COUNT for _ in range(NUM_AGENTS)]
            combo_log = [[DEFAULT_QUALITY] for _ in range(NUM_AGENTS)]
            next_sat_log = [None for _ in range(NUM_AGENTS)]
            end_of_video = False
            continue
        else:
            if combo_log[agent]:
                bit_rate[agent] = combo_log[agent].pop(0)
            else:
                do_mpc = True

            ho_point = ho_stamps_log

            do_mpc = True
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, is_handover, sat_status, _, _, _, _, _, _, cur_sat_id, \
        runner_up_sat_ids, ho_stamps, best_combos, best_user_info, quality \
            = net_env.get_video_chunk(bit_rate[agent], agent, MPC_TYPE, next_sat_log[0], ho_point, do_mpc)

        if best_combos:
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
        reward = VIDEO_BIT_RATE[quality] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[quality] -
                                           VIDEO_BIT_RATE[last_bit_rate[agent]]) / M_IN_K

        tmp_results.append(reward)

        # print(net_env.video_chunk_counter)
        # print(len(net_env.cooked_bw[1161]))
        # if agent == 0:
        #     print(reward, bit_rate[agent], delay, sleep_time, buffer_size, rebuf, \
        #         video_chunk_size, next_video_chunk_sizes, \
        #         end_of_video, video_chunk_remain)

        last_bit_rate[agent] = quality

        if agent is not None:
            # log time_stamp, bit_rate, buffer_size, reward
            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write("{: <15} {: <10} {: <10} {: <15} {: <15} {: <15}"
                           " {: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15}\n"
                           .format(str(round(time_stamp[agent] / M_IN_K, 3)), str(agent),
                                   str(VIDEO_BIT_RATE[quality]), str(round(buffer_size, 3)),
                                   str(round(rebuf, 3)),
                                   str(round(video_chunk_size, 3)), str(round(delay, 3)), str(round(reward, 3)),
                                   str(cur_sat_id), str(is_handover), str(sat_status), str(ho_stamps), str(best_user_info)))
            log_file.flush()

    # print(results, sum(results))
    print(sum(results) / len(results))

    summary_file = open(SUMMARY_PATH, 'a')
    summary_file.write('\n')
    summary_file.write(str(sum(results) / len(results)))
    summary_file.close()


if __name__ == '__main__':
    main()
