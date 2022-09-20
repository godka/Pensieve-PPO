import numpy as np
from muleo import load_trace
import muleo.fixed_env_lc as env
import matplotlib.pyplot as plt
import itertools
import time
import argparse

VIDEO_CHOICES = 6

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 5
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
SUMMARY_DIR = './results'
LOG_FILE = './test_results/log_sim_bb'
TEST_TRACES = './test/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

CHUNK_COMBO_OPTIONS = []


parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--user', type=int, default=2)
parser.add_argument('--handover', type=str, default='MEA')
parser.add_argument('--trace', type=str, default="noaa")
args = parser.parse_args()
NUM_AGENTS = args.user

# past errors in bandwidth
past_errors = [[]for _ in range(NUM_AGENTS)]
past_bandwidth_ests = [[]for _ in range(NUM_AGENTS)]

#size_video1 = [3155849, 2641256, 2410258, 2956927, 2593984, 2387850, 2554662, 2964172, 2541127, 2553367, 2641109, 2876576, 2493400, 2872793, 2304791, 2855882, 2887892, 2474922, 2828949, 2510656, 2544304, 2640123, 2737436, 2559198, 2628069, 2626736, 2809466, 2334075, 2775360, 2910246, 2486226, 2721821, 2481034, 3049381, 2589002, 2551718, 2396078, 2869088, 2589488, 2596763, 2462482, 2755802, 2673179, 2846248, 2644274, 2760316, 2310848, 2647013, 1653424]
size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

MPC_TYPE = "DualMPC"
# DualMPC-Centralization

def get_chunk_size(quality, index):
    if ( index < 0 or index > 48 ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0:size_video6[index]}
    return sizes[quality]

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
def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES, SCALE_FOR_TEST=SCALE_FOR_TEST)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              num_agents=NUM_AGENTS)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')
    result_traces = []
    result_Qua = []
    result_Rebuf = []
    result_Smooth = []
    time_list = []

    time_stamp = [0 for _ in range(NUM_AGENTS)]

    last_bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]
    bit_rate = [DEFAULT_QUALITY for _ in range(NUM_AGENTS)]

    action_vec = [np.zeros(A_DIM) for _ in range(NUM_AGENTS)]
    for i in range(NUM_AGENTS):
        action_vec[i][DEFAULT_QUALITY] = 1

    s_batch = [[np.zeros((S_INFO, S_LEN))]for _ in range(NUM_AGENTS)]
    a_batch = [[action_vec]for _ in range(NUM_AGENTS)]
    r_batch = [[]for _ in range(NUM_AGENTS)]
    entropy_record = [[]for _ in range(NUM_AGENTS)]
    Qua_batch = [[]for _ in range(NUM_AGENTS)]
    Rebuf_batch = [[]for _ in range(NUM_AGENTS)]
    Smooth_batch = [[]for _ in range(NUM_AGENTS)]
    video_count = 0
    

    t = time.time()
    results = []
    # make chunk combination options
    for combo in itertools.product([0,1,2,3,4,5], repeat=5):
        CHUNK_COMBO_OPTIONS.append(combo)
    
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

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            
            del Qua_batch[:]
            del Rebuf_batch[:]
            del Smooth_batch[:]


            print(video_count, \
                    '{:.4f}'.format(result_traces[-1]), \
                    '{:.4f}'.format(result_Qua[-1]), \
                    '{:.4f}'.format(result_Rebuf[-1]), \
                    '{:.4f}'.format(result_Smooth[-1]), \
                    '{:.4f}'.format(time_list[-1]), \
                    'london', \
                    'greedy+' + args.handover, \
                    'fixed-harmonic-mean', \
                    )         
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

            print("network count", video_count)
            video_count += 1
            # break

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, b, is_handover, new_sat_id = \
            net_env.get_video_chunk(bit_rate[agent], agent, MPC_TYPE)

        bit_rate[agent] = b

        time_stamp[agent] += delay  # in ms
        time_stamp[agent] += sleep_time  # in ms
            
        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate[agent]] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate[agent]] -
                                        VIDEO_BIT_RATE[last_bit_rate[agent]]) / M_IN_K

        r_batch[agent].append(reward)
        results.append(reward)
        Qua_batch[agent].append(VIDEO_BIT_RATE[bit_rate[agent]] / M_IN_K)
        Rebuf_batch[agent].append(REBUF_PENALTY * rebuf)
        Smooth_batch[agent].append(SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate[agent]] -
                                                    VIDEO_BIT_RATE[last_bit_rate[agent]]) / M_IN_K)
        r_batch[agent].append(reward)
        results.append(reward)
        # print(net_env.video_chunk_counter)
            # print(len(net_env.cooked_bw[1161]))
            # if agent == 0:
            #     print(reward, bit_rate[agent], delay, sleep_time, buffer_size, rebuf, \
            #         video_chunk_size, next_video_chunk_sizes, \
            #         end_of_video, video_chunk_remain)

        last_bit_rate[agent] = bit_rate[agent]

        if agent == 0:
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

if __name__ == '__main__':
    main()
