SUPPORTED_SHARING = {'max-cap', 'resource-fair', 'ratio-based'}
CENT_MPC_MODELS = ["DualMPC-Centralization-Reduced", "DualMPC-Centralization-Exhaustive", "Oracle"]
DIST_MPC_MODELS = ["ManifoldMPC", "DualMPC", "DualMPC-Centralization"]
SEP_MPC_MODELS = ["MVT", "MRSS", "MRSS-Smart", "MB"]
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
MILLISECONDS_IN_SECOND = 1000.0

NUM_AGENTS = 1
MAX_SAT = A_SAT = 4

QUALITY_FACTOR = 1
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
# small epsilon used in denominator to avoid division by zero
EPSILON = 0.2
BIG_EPSILON = -0.001
MIN_RATIO = 0.1
MAX_RATIO = 0.9
B_IN_MB = 1000000
BITS_IN_BYTE = 8.0
BITRATE_LEVELS = 6
PAST_LEN = 8

VIDEO_CHUNCK_LEN = 2000.0  # millisec, every time add this amount to buffer

SNR_NOISE_LOW = 0.7
SNR_NOISE_HIGH = 1
SNR_NOISE_UNIT = 0.005

MPC_FUTURE_CHUNK_COUNT = 3
MPC_PAST_CHUNK_COUNT = 5

# Multi-user config in the trace files
# NUM_USERS = 10
MULTI_USER_FOLDER = 'test_multi_user/'

INNER_PROCESS_NUMS = 22
HO_NUM = 10

SNR_MIN = 70

PAST_SAT_LOG_LEN = 3
BITRATE_WEIGHT = 2
NO_EXHAUSTIVE = True
ADAPTIVE_BUF = False
TEST_TRACES = '../../data/sat_data/test/'
TRAIN_TRACES = '../../data/sat_data/train/'
TEST_REAL_TRACES = '../../data/sat_data/real_test/'
TRAIN_REAL_TRACES = '../../data/sat_data/real_train/'
TEST_TIGHT_TRACES = '../../data/sat_data/test_tight/'

TRAIN_NOAA_TRACES = '../../data/sat_data/noaa_train_trace/'
TEST_NOAA_TRACES = '../../data/sat_data/noaa_test_trace/'

VIDEO_SIZE_FILE = '../../data/video_data/envivio/video_size_'

BUF_RATIO = 0.7
BUF_RATIO_COMBO = 0.8

A_DIM = 3
PAST_LEN = 8
SAT_CANDIDATES = 5

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
