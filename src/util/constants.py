
SUPPORTED_SHARING = {'max-cap', 'resource-fair', 'ratio-based'}
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0

QUALITY_FACTOR = 1
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
# small epsilon used in denominator to avoid division by zero
EPSILON = 1e-16
MIN_RATIO = 0.1
MAX_RATIO = 0.9
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0

SNR_NOISE_LOW = 0.7
SNR_NOISE_HIGH = 1
SNR_NOISE_UNIT = 0.005


TOTAL_VIDEO_CHUNK = 20

MPC_FUTURE_CHUNK_COUNT = 3
MPC_PAST_CHUNK_COUNT = 5

# Multi-user config in the trace files
# NUM_USERS = 10
MULTI_USER_FOLDER = 'test_multi_user/'

HO_NUM = 5