import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

LOG_PATH = './test_results_mpc_exhaustive/log_sim_cent_rss_Beijing_2022-9-21-00-00-00'
# LOG_PATH = 'results/log_sim_mpc_truth_naive_london'
# PLOT_SAMPLES = 300


all_time_stamp = []
all_bw = []

with open(LOG_PATH, 'r') as f:
    time_stamp = []
    is_handover = []
    avg_download = []
    pred_download = []
    bit_sel = []
    buf_size = []
    rebuf_time = []
    chunk_size = []
    delay_time = []
    qoe = []
    pred_bw = []

    first_line = True
    for line in f:
        if first_line:
            first_line = False
            continue
        parse = line.split()
        if parse:
            time_stamp.append(float(parse[0]))
            avg_download.append(float(parse[3]))
            pred_download.append(float(parse[4]))
            bit_sel.append(float(parse[5]))
            buf_size.append(float(parse[6]))
            rebuf_time.append(float(parse[7]))
            chunk_size.append(float(parse[8]))
            delay_time.append(float(parse[9]))
            qoe.append(float(parse[10]))
            if len(parse) >= 12:
                pred_bw.append(float(parse[11]))


fig, axs = plt.subplots(2, 3, figsize=(10, 5))
axs[0, 0].plot(time_stamp, avg_download, linestyle="-", marker="o", markersize=1, label="act. bw")
time_handover = list(compress(time_stamp, is_handover))
bw_handover = list(compress(avg_download, is_handover))
axs[0, 0].plot(time_handover, bw_handover, 'r*', markersize=10, label="HO point")
if pred_download:
    axs[0, 0].plot(time_stamp, pred_download, 'g--', markersize=1, label="pred. bw", alpha=0.5)
# axs[0, 0].set_ylim([50, 95])
axs[0, 0].set_title("Download BW (MBps)")

# axs[1, 0].plot(time_stamp, avg_throughput, linestyle="-", marker="o", markersize=1)
# throughput_handover = list(compress(avg_throughput, is_handover))
# axs[1, 0].plot(time_handover, throughput_handover, 'r*',markersize=10)
# axs[1, 0].set_title("Satellite BW (MB)")

axs[1, 0].plot(time_stamp, bit_sel, linestyle="-", marker="o", markersize=1)
axs[1, 0].set_title("Bitrate Selection (MB)")

axs[0, 1].plot(time_stamp, buf_size, linestyle="-", marker="o", markersize=1)
axs[0, 1].set_title("buf_size (sec)")

axs[1, 1].plot(time_stamp, rebuf_time, linestyle="-", marker="o", markersize=1)
axs[1, 1].set_title("rebuf_time (sec)")

# axs[0, 2].plot(time_stamp, avg_throughput, linestyle="-", marker="o", markersize=1)
# hroughput_handover = list(compress(avg_throughput, is_handover))
# axs[0, 2].plot(time_handover, throughput_handover, 'r*', markersize=10)
# axs[0, 2].set_title("Satellite BW (MB)")
# axs[1, 1].plot(time_stamp, chunk_size, linestyle="-", marker="o", markersize=1)
# axs[1, 1].set_title("Chunk size (MB)")

axs[1, 2].plot(time_stamp, qoe, linestyle="-", marker="o", markersize=1)
axs[1, 2].set_title("QoE")

legend = axs[0, 0].legend(loc='lower right', fontsize='x-small')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')
fig.tight_layout()

# Summry
print("Total time: ", time_stamp[-1])
print("Rebuf time: ", sum(rebuf_time))
print("Avg QoE: ", sum(qoe)/len(qoe))
print("Avg bitrate: ", sum(bit_sel)/len(bit_sel))
print("Total HO #: ", sum(is_handover))
print("Avg download bw", sum(avg_download) / len(avg_download))

plt.show()
plt.savefig('np.png')