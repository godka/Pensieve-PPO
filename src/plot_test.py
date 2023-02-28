import csv

import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

LOG_PATH = '../data/test_results_pensieve1/log_sim_pensieve_rss_Chicago_2022-9-21-10-00-00'
SAT_PATH = 'data/sat_data/test/rss_Chicago_2022-9-21-10-00-00.csv'
LOG_PATH = '../data/test_results_imp5/log_sim_ppo_rss_Chicago_2022-9-21-10-00-00'

LOG_PATH = '../data/test_results_pensieve5/log_sim_pensieve_rss_Chicago_2022-9-21-10-00-00'
# LOG_PATH = '../data/MPC_dist5/log_sim_cent_rss_Chicago_2022-9-21-10-00-00'

# LOG_PATH = 'results/log_sim_mpc_truth_naive_london'
# PLOT_SAMPLES = 300

cooked_time = []
satellite_bw = {}
with open(SAT_PATH, 'r') as f:
    csv_reader = csv.DictReader(f)
    line_count = 0

    for row in csv_reader:
        if line_count == 0:
            # Get Satellite ID
            satellite_id = list(row.keys())[2:]
            satellite_bw = {int(sat_id): [] for sat_id in satellite_id}

        line_count += 1

        for sat_id in satellite_id:
            # satellite_bw[int(sat_id)].append(float(row[sat_id]))
            satellite_bw[int(sat_id)].append(float(row[sat_id]) * 1/20)
        cooked_time.append(int(row["time"]))

all_time_stamp = []
all_bw = []
actual_bw = []
with open(LOG_PATH, 'r') as f:
    time_stamp = []
    is_handover = []
    agent_id = []
    avg_download = []
    pred_download = []
    bit_sel = []
    buf_size = []
    rebuf_time = []
    chunk_size = []
    delay_time = []
    qoe = []
    pred_bw = []
    prev_sat_id = None
    first_line = True
    for line in f:
        parse = line.split()
        if parse and float(parse[1]) == 4:
            time_stamp.append(float(parse[0]))
            agent_id.append(float(parse[1]))
            # pred_download.append(float(parse[4]))
            bit_sel.append(float(parse[2]))
            buf_size.append(float(parse[3]))
            rebuf_time.append(float(parse[4]))
            chunk_size.append(float(parse[5]))
            delay_time.append(float(parse[6]))
            avg_download.append(float(parse[5]) / float(parse[6]) / 1000)
            qoe.append(float(parse[7]))
            if prev_sat_id is None or prev_sat_id == str(parse[8]):
                is_handover.append(False)
            else:
                is_handover.append(True)
            prev_sat_id = str(parse[8])
            # is_handover.append(bool(parse[9]))
            actual_bw.append(satellite_bw[int(parse[8])][round(float(parse[0]))])


fig, axs = plt.subplots(2, 3, figsize=(10, 5))
axs[0, 0].plot(time_stamp, avg_download, linestyle="-", marker="o", markersize=1, label="act. bw")
time_handover = list(compress(time_stamp, is_handover))
bw_handover = list(compress(avg_download, is_handover))
axs[0, 0].plot(time_handover, bw_handover, 'r*', markersize=10, label="HO point")
if pred_download:
    axs[0, 0].plot(time_stamp, pred_download, 'g--', markersize=1, label="pred. bw", alpha=0.5)
axs[0, 0].set_ylim([0.0, 0.6])
axs[0, 0].set_title("Download Time")

# axs[1, 0].plot(time_stamp, avg_throughput, linestyle="-", marker="o", markersize=1)
# throughput_handover = list(compress(avg_throughput, is_handover))
# axs[1, 0].plot(time_handover, throughput_handover, 'r*',markersize=10)
# axs[1, 0].set_title("Satellite BW (MB)")

axs[1, 0].plot(time_stamp, bit_sel, linestyle="-", marker="o", markersize=1)
axs[1, 0].set_title("Bitrate Selection (MB)")

axs[0, 1].plot(time_stamp, buf_size, linestyle="-", marker="o", markersize=1)
axs[0, 1].set_title("buf_size (sec)")
axs[0, 1].set_ylim([0, 40])

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

axs[0, 2].plot(time_stamp, actual_bw, linestyle="--", marker="o", markersize=1)
# axs[0, 2].plot(time_stamp, satellite_bw[427][:len(time_stamp)], linestyle="--", marker="o", markersize=1)
# axs[0, 2].plot(time_stamp, satellite_bw[24][:len(time_stamp)], linestyle="-", marker="o", markersize=1)

axs[0, 2].set_title("Sat bw")
axs[0, 2].set_ylim([0.0, 5.0])

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
print("Avg delay_time", sum(avg_download) / len(avg_download))

plt.show()
plt.savefig('np.png')