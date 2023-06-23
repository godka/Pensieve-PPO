import csv

import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

LOG_PATH = 'real/cent_rl_cd/5/test_results_imp_agg_weight_v2_real5/log_sim_ppo_dataset_uk_11'
SAT_PATH = 'data/sat_data/real_test/dataset_uk_11.csv'
# LOG_PATH = 'real/dist_mpc/5/log_sim_cent_dataset_uk_11'

# PLOT_SAMPLES = 300

AGENT_ID = 1

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
        cooked_time.append(int(row['']))

for time_index in cooked_time:
    num_of_sats = 0
    for sat_id in satellite_bw:
        if satellite_bw[sat_id][time_index] != 0:
            num_of_sats += 1

plt.plot(cooked_time, satellite_bw[int(satellite_id[0])], linestyle="-", marker="o", markersize=3)
# plt.plot(cooked_time, satellite_bw[int(satellite_id[1])], linestyle="-", marker="o", markersize=3)
# plt.plot(cooked_time, satellite_bw[int(satellite_id[2])], linestyle="-", marker="o", markersize=3)

plt.legend()
plt.show()

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
    sat_history = [[] for _ in range(5)]
    qoe = []
    pred_bw = []
    prev_sat_id = [None for _ in range(5)]
    first_line = True
    for line in f:
        parse = line.split()
        if parse and int(float(parse[1])) == AGENT_ID:
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
            if prev_sat_id[int(float(parse[1]))] is None or prev_sat_id[int(float(parse[1]))] == str(parse[8]):
                is_handover.append(False)
            else:
                is_handover.append(True)
                sat_history[int(float(parse[1]))].append(str(parse[8]))
            if prev_sat_id[int(float(parse[1]))] is None:
                sat_history[int(float(parse[1]))].append(str(parse[8]))
            prev_sat_id[int(float(parse[1]))] = str(parse[8])
            # is_handover.append(bool(parse[9]))
            actual_bw.append(satellite_bw[int(parse[8])][round(float(parse[0]))])


fig, axs = plt.subplots(2, 3, figsize=(10, 5))
axs[0, 0].plot(time_stamp, avg_download, linestyle="-", marker="o", markersize=1, label="act. bw")
time_handover = list(compress(time_stamp, is_handover))
bw_handover = list(compress(avg_download, is_handover))
axs[0, 0].plot(time_handover, bw_handover, 'r*', markersize=10, label="HO point")

sat_log = ""
i = 0
for sat_his in sat_history[AGENT_ID]:
    sat_log += "->" + sat_his

axs[0, 0].scatter(time_stamp[0]+30, 0.05, s=15000, c=15000,
                  marker=r"$ {} $".format(sat_log), edgecolors='none')
i += 1

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
print("Avg buffer_size", sum(buf_size) / len(buf_size))

plt.show()
plt.savefig('np.png')