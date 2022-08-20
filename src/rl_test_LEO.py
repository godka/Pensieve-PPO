import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
from leo import load_trace
from leo import test_env as env
import ppo2 as network

os.environ['CUDA_VISIBLE_DEVICES'] = ''

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [10000, 20000, 30000, 60000, 90000, 140000]  # Kbps
HD_REWARD = [1, 2, 3, 6, 9, 14]
VIDEO_BIT_RATE = HD_REWARD
BUFFER_NORM_FACTOR = 1.0
# CHUNK_TIL_VIDEO_END_CAP = 151.0
MILLI_IN_SECOND = 1000.0
M_IN_K = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
LOG_FILE = 'test_results/log_sim_ppo'
VIDEO_SIZE_FILE = 'envivio/video_size_'
# Should be changed to separate file
TEST_TRACES = 'simulated_trace/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = sys.argv[1]

# BITRATE_REWARD = [1, 2, 4, 6, 9, 15]
QUALITY_FACTOR = 1.5
REBUF_PENALTY = 10  # pensieve: 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
VIDEO_CHUNK_LEN = 2

CHUNK_COMBO_OPTION = []
HANDOVER_TYPE = "QoE"
MPC_FUTURE_CHUNK_COUNT = 5
ROBUSTNESS = True


def main():
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM
    # all_cooked_bw, all_cooked_time, all_file_names = load_trace_LEO.load_trace(TEST_TRACES, split_condition="test")

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  video_size_file=VIDEO_SIZE_FILE,
                                  video_chunk_len=VIDEO_CHUNK_LEN,
                                  video_bit_rate=VIDEO_BIT_RATE)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')
    print(log_path)

    # Log formatting
    log_file.write("{: <15} {: <10} {: <10} {: <15} {: <15} {: <15} {: <15}"
                   " {: <15} {: <15} {: <15} {: <15} {: <15}\n"
                   .format("time_stamp", "is_handover", "cur_sat_id", "download_bw", "pred_download_bw",
                           "bit_rate_sel",
                           "buffer_size", "rebuf_time",
                           "chunk_size", "delay_time", "reward", "pred_bw"))

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

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        reward_log = []
        entropy_ = 0.5
        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video_or_network, video_chunk_remain, cur_sat_id, avg_throughput, is_handover = \
                net_env.get_video_chunk(bit_rate, handover_type=HANDOVER_TYPE)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            r_batch.append(reward)
            reward_log.append(reward)

            last_bit_rate = bit_rate

            if not end_of_video_or_network:
                avg_download_throughput = float(video_chunk_size) / float(delay) / M_IN_K * BITS_IN_BYTE  # Bit / SEC
                pred_bw, pred_download_bw = net_env.predict_future_bw(robustness=ROBUSTNESS)

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write("{: <15} {: <10} {: <10} {: <15} {: <15} {: <15} {: <15}"
                           " {: <15} {: <15} {: <15} {: <15} {: <15}\n"
                           .format(str(round(time_stamp, 3)), str(is_handover), str(cur_sat_id),
                                   str(round(avg_download_throughput, 3)), str(round(pred_download_bw, 3)),
                                   str(VIDEO_BIT_RATE[bit_rate]), str(round(buffer_size, 3)), str(round(rebuf, 3)),
                                   str(round(video_chunk_size, 3)), str(round(delay, 3)), str(round(reward, 3)),
                                   str(pred_bw)))
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, net_env.get_total_video_chunk()) \
                           / float(net_env.get_total_video_chunk())

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            noise = np.random.gumbel(size=len(action_prob))
            bit_rate = np.argmax(np.log(action_prob) + noise)
            
            s_batch.append(state)
            entropy_ = -np.dot(action_prob, np.log(action_prob))
            entropy_record.append(entropy_)

            if end_of_video_or_network:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'w')

                # Log formatting
                log_file.write("{: <15} {: <10} {: <10} {: <15} {: <15} {: <15} {: <15}"
                               " {: <15} {: <15} {: <15} {: <15} {: <15}\n"
                               .format("time_stamp", "is_handover", "cur_sat_id", "avg_download_bw", "avg_throughput",
                                       "bit_rate_sel",
                                       "buffer_size", "rebuf_time",
                                       "chunk_size", "delay_time", "reward", "pred_bw"))
                
                break

        print("avg QoE: ", sum(reward_log) / len(reward_log))


if __name__ == '__main__':
    main()