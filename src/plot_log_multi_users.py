import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# rss_Chicago_2022-9-21-10-00-00

# LOG_PATH = '../data/test_results_pensieve5/log_sim_pensieve_rss_Chicago_2022-9-21-10-00-00'
LOG_PATH = '../data/MPC_dist5/log_sim_cent_rss_Chicago_2022-9-21-10-00-00'
PLOT_SAMPLES = 300


time_stamp = []
bit_rates = []
buffer_occupancies = []
rebuffer_times = []
rewards = []
satlites = []
agents = []
download_bw = []


with open(LOG_PATH, 'rb') as f:
    for line in f:
        parse = line.split()
        if len(parse) > 4:
            # .format(str(round(time_stamp[agent] / M_IN_K, 3)), str(agent),
            # str(round(rebuf, 3)),
            # s str(round(reward, 3)), str(sat_status)))
            time_stamp.append(float(parse[0]))
            agents.append(int(parse[1]))
            bit_rates.append(float(parse[2]))
            buffer_occupancies.append(float(parse[3]))
            rebuffer_times.append(float(parse[4]))
            rewards.append(float(parse[7]))
            satlites.append(float(parse[8]))
            # download_bw.append(float(parse[10]))

num_of_users = max(agents) + 1

DIM_OF_PLOT = 4
f, axs = plt.subplots(num_of_users, DIM_OF_PLOT, sharex=True)

# fig, ax = plt.subplots()

# sns.histplot(rewards[-PLOT_SAMPLES:], stat="probability", ax=ax)
# ax.set_ylim(0, 0.4)
# plt.show()


for idx, ax in enumerate(axs.T.ravel()):
    if idx < num_of_users:
        if idx == 0:
            ax.set_title('Reward')
        ax.set_ylabel('User' + str(idx))

        tmp_rewards = [res for i, res in enumerate(rewards) if agents[i] == idx % num_of_users and satlites[i] != -1]
        tmp_time_stamp = [res for i, res in enumerate(time_stamp) if agents[i] == idx % num_of_users and satlites[i] != -1]
        ax.plot(tmp_time_stamp[-PLOT_SAMPLES:], tmp_rewards[-PLOT_SAMPLES:])

    elif idx < num_of_users * 2:
        if idx == num_of_users:
            ax.set_title('Rebuffering')
        tmp_rebuffer_times = [res for i, res in enumerate(rebuffer_times) if agents[i] == idx % num_of_users and satlites[i] != -1]
        tmp_time_stamp = [res for i, res in enumerate(time_stamp) if agents[i] == idx % num_of_users and satlites[i] != -1]
        ax.plot(tmp_time_stamp[-PLOT_SAMPLES:], tmp_rebuffer_times[-PLOT_SAMPLES:])

    elif idx < num_of_users * 3:
        if idx == num_of_users * 2:
            ax.set_title('Bitrate levels')
        tmp_rewards = [res for i, res in enumerate(bit_rates) if agents[i] == idx % num_of_users and satlites[i] != -1]
        tmp_time_stamp = [res for i, res in enumerate(time_stamp) if agents[i] == idx % num_of_users and satlites[i] != -1]
        ax.plot(tmp_time_stamp[-PLOT_SAMPLES:], tmp_rewards[-PLOT_SAMPLES:])

    elif idx < num_of_users * 4:
        if idx == num_of_users * 3:
            ax.set_title('Buffer status')
        tmp_rewards = [res for i, res in enumerate(buffer_occupancies) if agents[i] == idx % num_of_users and satlites[i] != -1]
        tmp_time_stamp = [res for i, res in enumerate(time_stamp) if agents[i] == idx % num_of_users and satlites[i] != -1]
        ax.plot(tmp_time_stamp[-PLOT_SAMPLES:], tmp_rewards[-PLOT_SAMPLES:])

# f.subplots_adjust(hspace=0)
f.tight_layout()
plt.show()
# plt.savefig('books_read.png')