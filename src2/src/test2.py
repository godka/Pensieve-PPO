import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow.compat.v1 as tf
from muleo import load_trace
from muleo import fixed_env as env
import ppo2 as network
import time


S_INFO = 6 + 1  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
A_SAT = 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
TEST_TRACES =  './noaa_test/'
LOG_FILE = './test_results/log_sim_ppo'
# NN_MODEL = sys.argv[1]
# NUM_AGENTS = int(sys.argv[2])

import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--NN_MODEL', type=str, default="nn_model_ep_48300.ckpt")
parser.add_argument('--NUM_AGENTS', type=int, default=1)
parser.add_argument('--trace', type=str, default="starlink")
args = parser.parse_args()

NN_MODEL = args.NN_MODEL
NUM_AGENTS = args.NUM_AGENTS

if args.trace == "starlink":
    TEST_TRACES =  './test/'
    location = 'london'
    SCALE_FOR_TEST = 1/30
elif args.trace == "noaa":
    TEST_TRACES =  './noaa_test/'
    location = 'Real'
    SCALE_FOR_TEST = 1
elif args.trace == "london":
    TEST_TRACES =  './london/'
    location = 'london'
    SCALE_FOR_TEST = 1/30
elif args.trace == "new_york":
    TEST_TRACES =  './newyork/'
    location = 'NewYork'
    SCALE_FOR_TEST = 1/30
elif args.trace == "shanghai":
    TEST_TRACES =  './shanghai/'
    location = 'shanghai'
    SCALE_FOR_TEST = 1/30
elif args.trace == "sydney":
    TEST_TRACES =  './sydney/'
    location = 'sydney'
    SCALE_FOR_TEST = 1/30
elif args.trace == "HK":
    TEST_TRACES =  './HK/'
    location = 'Hong_Kong'
    SCALE_FOR_TEST = 1/30
elif args.trace == "sanfan":
    TEST_TRACES =  './sanfan/'
    location = 'San_Francisco'
    SCALE_FOR_TEST = 1/30
else:
    TEST_TRACES =  './test/'
    location = 'london'
    SCALE_FOR_TEST = 1/30
    

# A_SAT = NUM_AGENTS

def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES, SCALE_FOR_TEST=SCALE_FOR_TEST)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              num_agents=NUM_AGENTS)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    results = []
    result_traces = []
    result_Qua = []
    result_Rebuf = []
    result_Smooth = []
    time_list = []


    with tf.Session() as sess:

        actor = network.Network(sess,
                                state_dim=[S_INFO, S_LEN], action_dim=A_DIM * A_SAT,
                                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp = [0 for _ in range(NUM_AGENTS)]

        last_bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
        bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
        sat = [0 for _ in range(NUM_AGENTS)]

        action_vec = [np.zeros(A_DIM * A_SAT) for _ in range(NUM_AGENTS)]
        for i in range(NUM_AGENTS):
            action_vec[i][bit_rate] = 1

        s_batch = [[np.zeros((S_INFO, S_LEN))]for _ in range(NUM_AGENTS)]
        a_batch = [[action_vec]for _ in range(NUM_AGENTS)]
        r_batch = [[]for _ in range(NUM_AGENTS)]
        
        Qua_batch = [[]for _ in range(NUM_AGENTS)]
        Rebuf_batch = [[]for _ in range(NUM_AGENTS)]
        Smooth_batch = [[]for _ in range(NUM_AGENTS)]
        entropy_record = [[]for _ in range(NUM_AGENTS)]
        entropy_ = 0.5
        video_count = 0
        
        t = time.time()
        
        while True:  # serve video forever
            
            agent = net_env.get_first_agent()
            
            if agent == -1:
                
                time_list.append(time.time() - t)
                t = time.time()
                log_file.write('\n')
                log_file.close()

                last_bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
                bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
                net_env.reset()
                
                data = [sum(r_batch[agent][1:]) / len(r_batch[agent][1:]) for agent in range(NUM_AGENTS)]
                result_traces.append(sum(data)/len(data))
                
                data = [sum(Qua_batch[agent][1:]) / len(Qua_batch[agent][1:]) for agent in range(NUM_AGENTS)]
                result_Qua.append(sum(data)/len(data))
                data = [sum(Rebuf_batch[agent][1:]) / len(Rebuf_batch[agent][1:]) for agent in range(NUM_AGENTS)]
                result_Rebuf.append(sum(data)/len(data))
                data = [sum(Smooth_batch[agent][1:]) / len(Smooth_batch[agent][1:]) for agent in range(NUM_AGENTS)]
                result_Smooth.append(sum(data)/len(data))
                
                print(video_count, \
                      '{:.4f}'.format(result_traces[-1]), \
                      '{:.4f}'.format(result_Qua[-1]), \
                      '{:.4f}'.format(result_Rebuf[-1]), \
                      '{:.4f}'.format(result_Smooth[-1]), \
                      '{:.4f}'.format(time_list[-1]), \
                      location, \
                      'RL_join', \
                      'fixed-harmonic-mean', \
                      )
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                del Qua_batch[:]
                del Rebuf_batch[:]
                del Smooth_batch[:]

                action_vec = [np.zeros(A_DIM) for _ in range(NUM_AGENTS)]
                for i in range(NUM_AGENTS):
                    action_vec[i][bit_rate[agent]] = 1

                s_batch = [[np.zeros((S_INFO, S_LEN))]for _ in range(NUM_AGENTS)]
                a_batch = [[action_vec]for _ in range(NUM_AGENTS)]
                r_batch = [[]for _ in range(NUM_AGENTS)]
                entropy_record = [[]for _ in range(NUM_AGENTS)]
                Qua_batch = [[]for _ in range(NUM_AGENTS)]
                Rebuf_batch = [[]for _ in range(NUM_AGENTS)]
                Smooth_batch = [[]for _ in range(NUM_AGENTS)]

                # print("video count", video_count)
                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'w')
            
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, \
            next_sat_bw = \
                net_env.get_video_chunk(bit_rate[agent], agent)

            time_stamp[agent] += delay  # in ms
            time_stamp[agent] += sleep_time  # in ms
            
            # reward is video quality - rebuffer penalty
            reward = VIDEO_BIT_RATE[bit_rate[agent]] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate[agent]] -
                                            VIDEO_BIT_RATE[last_bit_rate[agent]]) / M_IN_K

            if r_batch[agent]:
                results.append(reward)
            r_batch[agent].append(reward)
            Qua_batch[agent].append(VIDEO_BIT_RATE[bit_rate[agent]] / M_IN_K)
            Rebuf_batch[agent].append(REBUF_PENALTY * rebuf)
            Smooth_batch[agent].append(SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate[agent]] -
                                            VIDEO_BIT_RATE[last_bit_rate[agent]]) / M_IN_K)
            
            last_bit_rate [agent]= bit_rate[agent]

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp[agent] / M_IN_K) + '\t' +
                        str(VIDEO_BIT_RATE[bit_rate[agent]]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(reward) + '\n')
            log_file.flush()

            # retrieve previous state
            if len(s_batch[agent]) == 0:
                state = [[np.zeros((S_INFO, S_LEN))]for _ in range(NUM_AGENTS)]
            else:
                state = [np.array(s_batch[agent][-1], copy=True) for agent in range(NUM_AGENTS)]

            # dequeue history record
            state[agent] = np.roll(state[agent], -1, axis=1)

            # this should be S_INFO number of terms
            state[agent][0, -1] = VIDEO_BIT_RATE[bit_rate[agent]] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[agent][1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[agent][2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[agent][3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[agent][4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[agent][5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            state[agent][6, :A_SAT] = np.array(next_sat_bw) 

                
            action_prob = actor.predict(np.reshape(state[agent], (1, S_INFO, S_LEN)))
            noise = np.random.gumbel(size=len(action_prob))
            action = np.argmax(np.log(action_prob) + noise)
            
            sat[agent] = action // A_DIM
            bit_rate[agent] = action % A_DIM
            
            net_env.set_satellite(agent, sat[agent])
            
            s_batch[agent].append(state[agent])
        
            entropy_ = -np.dot(action_prob, np.log(action_prob))
            entropy_record.append(entropy_)

    print('{:.4f}'.format(sum(result_traces) / len(result_traces)))
    print('{:.4f}'.format(np.std(result_traces)))

    # print('{:.4f}'.format(sum(result_Qua) / len(result_Qua)))
    # print('{:.4f}'.format(np.std(result_Qua)))\
    
    # print('{:.4f}'.format(sum(result_Rebuf) / len(result_Rebuf)))
    # print('{:.4f}'.format(np.std(result_Rebuf)))
    
    # print('{:.4f}'.format(sum(result_Smooth) / len(result_Smooth)))
    # print('{:.4f}'.format(np.std(result_Smooth)))
    
    # print('{:.4f}'.format(sum(time_list) / len(time_list)))
    # print('{:.4f}'.format(np.std(time_list)))
if __name__ == '__main__':
    main()
