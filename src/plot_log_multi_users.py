import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOG_PATH = './test_results/dist_wo'
PLOT_SAMPLES = 300


time_stamp = []
bit_rates = []
buffer_occupancies = []
rebuffer_times = []
rewards = []
satlites = []
agents = []

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
            buffer_occupancies.append(float(parse[2]))
            rebuffer_times.append(float(parse[3]))
            rewards.append(float(parse[-2]))
            satlites.append(float(parse[-1]))

num_of_users = max(agents)

# f, axs = plt.subplots(num_of_users, sharex=True)

fig, ax = plt.subplots()

sns.histplot(rewards[-PLOT_SAMPLES:], stat="probability", ax=ax)
ax.set_ylim(0, 0.4)
plt.show()

exit(1)

for idx, ax in enumerate(axs):
    tmp_rewards = [res for i, res in enumerate(rewards) if agents[i] == idx]
    tmp_time_stamp = [res for i, res in enumerate(time_stamp) if agents[i] == idx]

    ax.plot(tmp_time_stamp[-PLOT_SAMPLES:], tmp_rewards[-PLOT_SAMPLES:])
    ax.set_title('Average reward: ' + str(np.mean(tmp_rewards[-PLOT_SAMPLES:])))
    ax.set_ylabel('Reward')


# f.subplots_adjust(hspace=0)
f.tight_layout()
plt.show()
plt.savefig('books_read.png')