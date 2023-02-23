import os
import sys

from util.constants import CHUNK_TIL_VIDEO_END_CAP, BUFFER_NORM_FACTOR, VIDEO_BIT_RATE, REBUF_PENALTY, SMOOTH_PENALTY, \
    DEFAULT_QUALITY, BITRATE_WEIGHT, M_IN_K, A_DIM, S_LEN, PAST_LEN, BITRATE_REWARD

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow.compat.v1 as tf
from env.multi_bw_share import fixed_env_time as env
from env.multi_bw_share import load_trace_tight as load_trace
from models.rl_multi_bw_share.ppo_spec import pensieve as network
import structlog
import logging

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
# A_SAT = 2
ACTOR_LR_RATE = 1e-4
# CRITIC_LR_RATE = 0.001
RANDOM_SEED = 42
TEST_TRACES = 'data/sat_data/test_tight/'
NN_MODEL = sys.argv[1]
USERS = int(sys.argv[2])
HO_TYPE = str(sys.argv[3])
SUMMARY_DIR = './test_results_pensieve_tight' + str(USERS)

LOG_FILE = SUMMARY_DIR + '/log_sim_pensieve'
SUMMARY_PATH = SUMMARY_DIR + '/summary'

# A_SAT = NUM_AGENTS
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

log = structlog.get_logger()
log.debug('Test init')

REWARD_FUNC = "LIN"


def main():
    np.random.seed(RANDOM_SEED)

    # assert len(VIDEO_BIT_RATE) == A_DIM

    is_handover = False

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              all_cooked_name=all_file_names,
                              num_agents=USERS)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    time_stamp = [0 for _ in range(USERS)]

    results = []
    tmp_results = []

    reward_1 = []
    reward_2 = []
    reward_3 = []

    with tf.Session() as sess:

        actor = network.Network(sess,
                                state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp = [0 for _ in range(USERS)]

        last_bit_rate = [DEFAULT_QUALITY for _ in range(USERS)]
        bit_rate = [DEFAULT_QUALITY for _ in range(USERS)]
        sat = [0 for _ in range(USERS)]

        action_vec = [np.zeros(A_DIM) for _ in range(USERS)]
        for i in range(USERS):
            action_vec[i][bit_rate] = 1

        s_batch = [[np.zeros((S_INFO, S_LEN))] for _ in range(USERS)]
        a_batch = [[action_vec] for _ in range(USERS)]
        r_batch = [[] for _ in range(USERS)]
        state = [[np.zeros((S_INFO, S_LEN))] for _ in range(USERS)]
        entropy_record = [[] for _ in range(USERS)]
        entropy_ = 0.5
        video_count = 0
        tmp_reward_1 = []
        tmp_reward_2 = []
        tmp_reward_3 = []
        while True:  # serve video forever

            agent = net_env.get_first_agent()

            if agent == -1:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = [DEFAULT_QUALITY for _ in range(USERS)]
                bit_rate = [DEFAULT_QUALITY for _ in range(USERS)]
                net_env.reset()

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = [np.zeros(A_DIM) for _ in range(USERS)]
                for i in range(USERS):
                    action_vec[i][bit_rate[agent]] = 1

                s_batch = [[np.zeros((S_INFO, S_LEN))] for _ in range(USERS)]
                a_batch = [[action_vec] for _ in range(USERS)]
                r_batch = [[] for _ in range(USERS)]
                entropy_record = [[] for _ in range(USERS)]

                state = [[np.zeros((S_INFO, S_LEN))] for _ in range(USERS)]

                print("network count", video_count)
                print(sum(tmp_results[1:]) / len(tmp_results[1:]))
                summary_file = open(SUMMARY_PATH, 'a')
                summary_file.write(net_env.get_file_name())
                summary_file.write('\n')
                summary_file.write(str(sum(tmp_results[1:]) / len(tmp_results[1:])))
                summary_file.write('\n')
                summary_file.close()
                results += tmp_results[1:]
                tmp_results = []
                time_stamp = [0 for _ in range(USERS)]
                reward_1.append(np.mean(tmp_reward_1[1:]))
                reward_2.append(np.mean(tmp_reward_2[1:]))
                reward_3.append(np.mean(tmp_reward_3[1:]))

                tmp_reward_1 = []
                tmp_reward_2 = []
                tmp_reward_3 = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'w')
                continue

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, is_handover, _, _, next_sat_bw_logs, \
            cur_sat_user_num, next_sat_user_num, cur_sat_bw_logs, connected_time, cur_sat_id, _, _, _, _, _ = \
                net_env.get_video_chunk(bit_rate[agent], agent, None, ho_stamp=HO_TYPE)

            time_stamp[agent] += delay  # in ms
            time_stamp[agent] += sleep_time  # in ms

            # reward is video quality - rebuffer penalty
            if REWARD_FUNC == "LIN":
                reward = VIDEO_BIT_RATE[bit_rate[agent]] / M_IN_K \
                         - REBUF_PENALTY * rebuf \
                         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate[agent]] -
                                                   VIDEO_BIT_RATE[last_bit_rate[agent]]) / M_IN_K
                tmp_reward_1.append(VIDEO_BIT_RATE[bit_rate[agent]] / M_IN_K)
                tmp_reward_2.append(-REBUF_PENALTY * rebuf)
                tmp_reward_3.append(- SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate[agent]] -
                                                              VIDEO_BIT_RATE[last_bit_rate[agent]]) / M_IN_K)
            elif REWARD_FUNC == "HD":
                reward = BITRATE_REWARD[bit_rate[agent]] \
                         - 8 * rebuf - np.abs(BITRATE_REWARD[bit_rate[agent]] - BITRATE_REWARD[last_bit_rate[agent]])

                tmp_reward_1.append(BITRATE_REWARD[bit_rate[agent]])
                tmp_reward_2.append(-8 * rebuf)
                tmp_reward_3.append(-np.abs(BITRATE_REWARD[bit_rate[agent]] - BITRATE_REWARD[last_bit_rate[agent]]))
            else:
                raise Exception

            r_batch[agent].append(reward)
            tmp_results.append(reward)

            last_bit_rate[agent] = bit_rate[agent]

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write("{: <15} {: <10} {: <10} {: <15} {: <15} {: <15}"
                           " {: <15} {: <15} {: <15} {: <15} {: <15} {: <15} {: <15}\n"
                           .format(str(round(time_stamp[agent] / M_IN_K, 3)),
                                   str(agent),
                                   str(VIDEO_BIT_RATE[bit_rate[agent]]),
                                   str(round(buffer_size, 3)),
                                   str(round(rebuf, 3)),
                                   str(round(video_chunk_size, 3)),
                                   str(round(delay, 3)),
                                   str(round(reward, 3)),
                                   str(cur_sat_id), str(is_handover), str(0), str(0),
                                   str(0)))
            log_file.flush()

            # retrieve previous state
            if len(s_batch[agent]) == 0:
                state[agent] = [np.zeros((S_INFO, S_LEN))]
            else:
                state[agent] = np.array(s_batch[agent][-1], copy=True)

            # dequeue history record
            state[agent] = np.roll(state[agent], -1, axis=1)

            # this should be S_INFO number of terms
            state[agent][0, -1] = VIDEO_BIT_RATE[bit_rate[agent]] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[agent][1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[agent][2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[agent][3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            # state[agent][4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[agent][4, :A_DIM] = np.array(
                [next_video_chunk_sizes[index] for index in [0, 2, 4]]) / M_IN_K / M_IN_K  # mega byte
            state[agent][5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
                
            action_prob = actor.predict(np.reshape(state[agent], (1, S_INFO, S_LEN)))
            noise = np.random.gumbel(size=len(action_prob))
            action = np.argmax(np.log(action_prob) + noise)

            # sat[agent] = action // A_DIM
            bit_rate[agent] = action % A_DIM

            # Testing for mpc
            # bit_rate[agent] /= BITRATE_WEIGHT
            # bit_rate[agent] = int(bit_rate[agent])
            # bit_rate[agent] *= BITRATE_WEIGHT
            bit_rate[agent] *= BITRATE_WEIGHT

            s_batch[agent].append(state[agent])

            entropy_ = -np.dot(action_prob, np.log(action_prob))
            entropy_record.append(entropy_)

    # print(results)
    print(sum(results) / len(results))

    summary_file = open(SUMMARY_PATH, 'a')
    summary_file.write('\n')
    summary_file.write(str(sum(results) / len(results)))
    summary_file.close()

    reward_file = open(SUMMARY_PATH + '_reward_parts', 'w')
    reward_file.write(' '.join(str(elem) for elem in reward_1))
    reward_file.write('\n')
    reward_file.write(' '.join(str(elem) for elem in reward_2))
    reward_file.write('\n')
    reward_file.write(' '.join(str(elem) for elem in reward_3))
    reward_file.write('\n')


if __name__ == '__main__':
    main()
