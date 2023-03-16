import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir + '/../')
from util.constants import CHUNK_TIL_VIDEO_END_CAP, BUFFER_NORM_FACTOR, VIDEO_BIT_RATE, REBUF_PENALTY, SMOOTH_PENALTY, \
    DEFAULT_QUALITY, BITRATE_WEIGHT, M_IN_K, A_DIM, PAST_TEST_LEN, PAST_LEN, BITRATE_REWARD, MAX_SAT, PAST_SAT_LOG_LEN, \
    TEST_TRACES
from util.encode import encode_other_sat_info, one_hot_encode

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow.compat.v1 as tf
from env.multi_bw_share import fixed_env_time as env
from env.multi_bw_share import load_trace as load_trace
from models.rl_multi_bw_share.ppo_spec import ppo_cent_his as network
import structlog
import logging

A_SAT = 2
ACTOR_LR_RATE = 1e-4
# CRITIC_LR_RATE = 0.001
RANDOM_SEED = 42
NN_MODEL = sys.argv[1]
USERS = int(sys.argv[2])
SUMMARY_DIR = './test_results_imp_cent_one_reward' + str(USERS)

if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
LOG_FILE = SUMMARY_DIR + '/log_sim_ppo'
SUMMARY_PATH = SUMMARY_DIR + '/summary'

# A_SAT = NUM_AGENTS
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

log = structlog.get_logger()
log.debug('Test init')

REWARD_FUNC = "LIN"
S_INFO = 11 + USERS-1 + (USERS-1) * PAST_SAT_LOG_LEN + (USERS-1)*2


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
    prev_buffer_size = [0 for _ in range(USERS)]
    prev_rebuf = [0 for _ in range(USERS)]
    prev_video_chunk_size = [0 for _ in range(USERS)]
    prev_delay = [0 for _ in range(USERS)]
    prev_next_video_chunk_sizes = [[] for _ in range(USERS)]
    prev_video_chunk_remain = [0 for _ in range(USERS)]
    prev_next_sat_bw_logs = [[] for _ in range(USERS)]
    prev_cur_sat_bw_logs = [[] for _ in range(USERS)]
    prev_connected_time = [[] for _ in range(USERS)]

    with tf.Session() as sess:

        actor = network.Network(sess,
                                state_dim=[S_INFO, PAST_TEST_LEN], action_dim=A_DIM * A_SAT,
                                learning_rate=ACTOR_LR_RATE, num_of_users=USERS)

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

        action_vec = [np.zeros(A_DIM * A_SAT) for _ in range(USERS)]
        for i in range(USERS):
            action_vec[i][bit_rate] = 1

        s_batch = [[np.zeros((S_INFO, PAST_TEST_LEN))] for _ in range(USERS)]
        a_batch = [[action_vec] for _ in range(USERS)]
        r_batch = [[] for _ in range(USERS)]
        state = [[np.zeros((S_INFO, PAST_TEST_LEN))] for _ in range(USERS)]
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

                s_batch = [[np.zeros((S_INFO, PAST_TEST_LEN))] for _ in range(USERS)]
                a_batch = [[action_vec] for _ in range(USERS)]
                r_batch = [[] for _ in range(USERS)]
                entropy_record = [[] for _ in range(USERS)]

                state = [[np.zeros((S_INFO, PAST_TEST_LEN))] for _ in range(USERS)]

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
            cur_sat_user_num, next_sat_user_num, cur_sat_bw_logs, connected_time, cur_sat_id, next_sat_ids, _, _, _, _,\
            other_sat_users, other_sat_bw_logs, other_buffer_sizes = \
                net_env.get_video_chunk(bit_rate[agent], agent, model_type=None)

            time_stamp[agent] += delay  # in ms
            time_stamp[agent] += sleep_time  # in ms

            prev_buffer_size[agent] = buffer_size
            prev_rebuf[agent] = rebuf
            prev_video_chunk_size[agent] = video_chunk_size
            prev_delay[agent] = delay
            prev_next_video_chunk_sizes[agent] = next_video_chunk_sizes
            prev_video_chunk_remain[agent] = video_chunk_remain
            prev_next_sat_bw_logs[agent] = next_sat_bw_logs
            prev_cur_sat_bw_logs[agent] = cur_sat_bw_logs
            prev_connected_time[agent] = connected_time

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

                # reward += self.net_env.get_others_reward(agent, self.last_bit_rate)
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
                state[agent] = [np.zeros((S_INFO, PAST_TEST_LEN))]
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

            state[agent][5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
                CHUNK_TIL_VIDEO_END_CAP)
            if len(next_sat_bw_logs) < PAST_LEN:
                next_sat_bw_logs = [0] * (PAST_LEN - len(next_sat_bw_logs)) + next_sat_bw_logs

            state[agent][6, :PAST_LEN] = np.array(next_sat_bw_logs[:PAST_LEN]) / 10

            if len(cur_sat_bw_logs) < PAST_LEN:
                cur_sat_bw_logs = [0] * (PAST_LEN - len(cur_sat_bw_logs)) + cur_sat_bw_logs

            state[agent][7, :PAST_LEN] = np.array(cur_sat_bw_logs[:PAST_LEN]) / 10

            if is_handover:
                state[agent][8:9, 0:PAST_TEST_LEN] = np.zeros((1, PAST_TEST_LEN))
                state[agent][9:10, 0:PAST_TEST_LEN] = np.zeros((1, PAST_TEST_LEN))
            state[agent][8:9, -1] = np.array(cur_sat_user_num) / 10
            state[agent][9:10, -1] = np.array(next_sat_user_num) / 10
            state[agent][10, :2] = [float(connected_time[0]) / BUFFER_NORM_FACTOR / 10,
                                    float(connected_time[1]) / BUFFER_NORM_FACTOR / 10]
            next_sat_id = None
            if next_sat_ids is not None:
                next_sat_id = next_sat_ids[agent]
            other_user_sat_decisions, other_sat_num_users, other_sat_bws, cur_user_sat_decisions \
                = encode_other_sat_info(net_env.sat_decision_log, USERS, cur_sat_id, next_sat_id, agent,
                                        other_sat_users, other_sat_bw_logs, PAST_SAT_LOG_LEN)

            # state[agent][11:11+MAX_SAT - A_SAT, -1] = np.reshape(np.array(other_sat_num_users), (MAX_SAT - A_SAT, 1)) / 10
            state[agent][11:(11 + USERS-1), -1:] = np.reshape(np.array(other_buffer_sizes) / BUFFER_NORM_FACTOR, (-1, 1))
            state[agent][(11 + USERS-1):(11 + USERS-1 + (USERS-1) * PAST_SAT_LOG_LEN),
            0:2] = np.reshape(other_user_sat_decisions, (-1, 2))

            others_last_bit_rate = np.delete(np.array(last_bit_rate), agent)
            for i in others_last_bit_rate:
                state[agent][(11 + USERS-1 + (USERS-1) * PAST_SAT_LOG_LEN) + i:
                             (11 + USERS-1 + (USERS-1) * PAST_SAT_LOG_LEN + (USERS-1)) + i, -1] \
                    = VIDEO_BIT_RATE[i] / float(np.max(VIDEO_BIT_RATE))
            i = 0
            for u_id in range(USERS):
                if u_id == agent:
                    continue
                if len(prev_cur_sat_bw_logs[u_id]) < PAST_LEN:
                    prev_cur_sat_bw_logs[u_id] = [0] * (PAST_LEN - len(prev_cur_sat_bw_logs[u_id])) + \
                                                  prev_cur_sat_bw_logs[u_id]
                state[agent][(11 + USERS-1 + (USERS-1) * PAST_SAT_LOG_LEN + (USERS-1))+i, :PAST_LEN] = np.array(prev_cur_sat_bw_logs[u_id][:PAST_LEN])

                i += 1
            # if len(next_sat_user_num) < PAST_LEN:
            #     next_sat_user_num = [0] * (PAST_LEN - len(next_sat_user_num)) + next_sat_user_num

            # state[agent][8, :PAST_LEN] = next_sat_user_num[:5]

            action_prob = actor.predict(np.reshape(state[agent], (1, S_INFO, PAST_TEST_LEN)))
            noise = np.random.gumbel(size=len(action_prob))
            action = np.argmax(np.log(action_prob) + noise)

            sat[agent] = action // A_DIM
            bit_rate[agent] = action % A_DIM

            # Testing for mpc
            # bit_rate[agent] /= BITRATE_WEIGHT
            # bit_rate[agent] = int(bit_rate[agent])
            # bit_rate[agent] *= BITRATE_WEIGHT
            bit_rate[agent] *= BITRATE_WEIGHT

            if not end_of_video:
                changed_sat_id = net_env.set_satellite(agent, sat[agent])
                if sat[agent] == 1:
                    is_handover = True
                    # print("Handover!!")
                else:
                    is_handover = False
                    # print("X Handover")
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
