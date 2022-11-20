import os
import sys

from util.encode import encode_other_sat_info, one_hot_encode

os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow.compat.v1 as tf
from muleo_lc_bw_share import load_trace
from muleo_lc_bw_share import fixed_env_cent as env
import ppo_cent_his2 as network

# S_INFO = 10 + 1 + 3 + 6 * 5  # Original + nums of sat + bw of sats + decisions of users
S_LEN = 8  # take how many frames in the past
A_DIM = 6
PAST_SAT_LOG_LEN = 3
PAST_LEN = 8
A_SAT = 2
MAX_SAT = 8
ACTOR_LR_RATE = 1e-4
# CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
TEST_TRACES = './test/'
NN_MODEL = sys.argv[1]
NUM_AGENTS = int(sys.argv[2])
S_INFO = 12 + MAX_SAT - A_SAT + (NUM_AGENTS-1)

LOG_FILE = './test_results_cent_his' + str(NUM_AGENTS) + '/log_sim_ppo'


# A_SAT = NUM_AGENTS


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    is_handover = False

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              num_agents=NUM_AGENTS)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    results = []

    with tf.Session() as sess:

        actor = network.Network(sess,
                                state_dim=[S_INFO, S_LEN], action_dim=A_DIM * A_SAT,
                                learning_rate=ACTOR_LR_RATE, num_of_users=NUM_AGENTS)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp = [0 for _ in range(NUM_AGENTS)]

        last_bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
        last_penalty = [0 for _ in range(NUM_AGENTS)]
        bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
        sat = [0 for _ in range(NUM_AGENTS)]

        action_vec = [np.zeros(A_DIM * A_SAT) for _ in range(NUM_AGENTS)]
        for i in range(NUM_AGENTS):
            action_vec[i][bit_rate] = 1

        s_batch = [[np.zeros((S_INFO, S_LEN))]for _ in range(NUM_AGENTS)]
        a_batch = [[action_vec]for _ in range(NUM_AGENTS)]
        r_batch = [[]for _ in range(NUM_AGENTS)]
        state = [[np.zeros((S_INFO, S_LEN))]for _ in range(NUM_AGENTS)]
        entropy_record = [[]for _ in range(NUM_AGENTS)]
        entropy_ = 0.5
        video_count = 0
        
        while True:  # serve video forever
            
            agent = net_env.get_first_agent()
            
            if agent == -1:
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

                s_batch = [[np.zeros((S_INFO, S_LEN))]for _ in range(NUM_AGENTS)]
                a_batch = [[action_vec]for _ in range(NUM_AGENTS)]
                r_batch = [[]for _ in range(NUM_AGENTS)]
                entropy_record = [[]for _ in range(NUM_AGENTS)]
                
                state = [[np.zeros((S_INFO, S_LEN))]for _ in range(NUM_AGENTS)]

                print("network count", video_count)
                print(sum(results) / len(results))
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
            end_of_video, video_chunk_remain, _, _, _, _, \
            _, next_sat_bw_logs, cur_sat_user_num, next_sat_user_num, cur_sat_bw_logs, connected_time, \
                cur_sat_id, next_sat_id, other_sat_users, other_sat_bw_logs, other_buffer_sizes = \
                net_env.get_video_chunk(bit_rate[agent], agent, model_type=None)

            time_stamp[agent] += delay  # in ms
            time_stamp[agent] += sleep_time  # in ms
            
            # reward is video quality - rebuffer penalty
            reward = VIDEO_BIT_RATE[bit_rate[agent]] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate[agent]] -
                                            VIDEO_BIT_RATE[last_bit_rate[agent]]) / M_IN_K
            last_penalty[agent] = REBUF_PENALTY * rebuf

            results.append(reward)

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp[agent] / M_IN_K) + '\t' +
                        str(agent) + '\t' +
                        str(sat[agent]) + '\t' +
                        str(cur_sat_id) + '\t' +
                        str(VIDEO_BIT_RATE[bit_rate[agent]]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(float(video_chunk_size) / float(delay) / M_IN_K) + '\t' +
                        str(delay) + '\t' +
                        str(reward) + '\n')
            log_file.flush()

            reward += net_env.get_others_reward(agent, last_bit_rate, last_penalty)

            r_batch[agent].append(reward)
            last_bit_rate[agent] = bit_rate[agent]
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
            state[agent][4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[agent][5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            if len(next_sat_bw_logs) < PAST_LEN:
                next_sat_bw_logs = [0] * (PAST_LEN - len(next_sat_bw_logs)) + next_sat_bw_logs

            state[agent][6, :PAST_LEN] = np.array(next_sat_bw_logs[:PAST_LEN]) / 10

            if len(cur_sat_bw_logs) < PAST_LEN:
                cur_sat_bw_logs = [0] * (PAST_LEN - len(cur_sat_bw_logs)) + cur_sat_bw_logs

            state[agent][7, :PAST_LEN] = np.array(cur_sat_bw_logs[:PAST_LEN]) / 10
            if is_handover:
                state[agent][8:9, 0:S_LEN] = np.zeros((1, S_LEN))
                state[agent][9:10, 0:S_LEN] = np.zeros((1, S_LEN))
            state[agent][8:9, -1] = np.array(cur_sat_user_num) / 10
            state[agent][9:10, -1] = np.array(next_sat_user_num) / 10
            state[agent][10, :A_SAT] = [float(connected_time[0]) / BUFFER_NORM_FACTOR / 10, float(connected_time[1]) / BUFFER_NORM_FACTOR / 10]

            other_user_sat_decisions, other_sat_num_users, other_sat_bws, cur_user_sat_decisions \
                = encode_other_sat_info(net_env.sat_decision_log, NUM_AGENTS, cur_sat_id, next_sat_id, agent,
                                        other_sat_users, other_sat_bw_logs)

            # state[agent][11:11+MAX_SAT - A_SAT, -1] = np.reshape(np.array(other_sat_num_users), (MAX_SAT - A_SAT, 1)) / 10
            state[agent][11:12, 0:NUM_AGENTS - 1] = np.array(other_buffer_sizes) / BUFFER_NORM_FACTOR
            state[agent][12:(12 + MAX_SAT - A_SAT), 0:PAST_LEN] = np.array(other_sat_bws) / 10

            # state[agent][(12 + MAX_SAT - A_SAT):(12 + MAX_SAT - A_SAT + PAST_SAT_LOG_LEN),
            # 0:3] = np.reshape(cur_user_sat_decisions, (-1, 3))

            # state[agent][(12 + MAX_SAT - A_SAT + PAST_SAT_LOG_LEN):(12 + MAX_SAT - A_SAT + PAST_SAT_LOG_LEN
            #                                                         + (NUM_AGENTS-1) * PAST_SAT_LOG_LEN),
            # 0:3] = np.reshape(other_user_sat_decisions, (-1, 3))

            others_last_bit_rate = np.delete(np.array(last_bit_rate), agent)
            state[agent][(12 + MAX_SAT - A_SAT):
                         (12 + MAX_SAT - A_SAT + (NUM_AGENTS-1)),
            0:len(VIDEO_BIT_RATE)] = np.reshape(one_hot_encode(others_last_bit_rate, len(VIDEO_BIT_RATE)), (-1, len(VIDEO_BIT_RATE)))
            # if len(next_sat_user_num) < PAST_LEN:
            #     next_sat_user_num = [0] * (PAST_LEN - len(next_sat_user_num)) + next_sat_user_num

            # state[agent][8, :PAST_LEN] = next_sat_user_num[:5]

            action_prob = actor.predict(np.reshape(state[agent], (1, S_INFO, S_LEN)))
            noise = np.random.gumbel(size=len(action_prob))
            action = np.argmax(np.log(action_prob) + noise)
            
            sat[agent] = action // A_DIM
            bit_rate[agent] = action % A_DIM

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


if __name__ == '__main__':
    main()
