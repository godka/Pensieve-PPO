
SUPPORTED_SHARING = {'max-cap', 'resource-fair', 'rate-fair', 'proportional-fair'}
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
# small epsilon used in denominator to avoid division by zero
EPSILON = 1e-16
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0

NOISE_LOW = 0.9
NOISE_HIGH = 1.1

