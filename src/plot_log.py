import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOG_PATH = './test_results/dist_w'
PLOT_SAMPLES = 300


time_stamp = []
bit_rates = []
buffer_occupancies = []
rebuffer_times = []
rewards = []
satlites = []

with open(LOG_PATH, 'rb') as f:
    for line in f:
        parse = line.split()
        if len(parse) > 4:
            time_stamp.append(float(parse[0]))
            bit_rates.append(float(parse[1]))
            buffer_occupancies.append(float(parse[2]))
            rebuffer_times.append(float(parse[3]))
            rewards.append(float(parse[-1]))
            satlites.append(float(parse[-3]))

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

# ax1.plot(time_stamp[-PLOT_SAMPLES:], rewards[-PLOT_SAMPLES:])
sns.displot(rewards[-PLOT_SAMPLES:], ax=ax1)
# ax1.set_title('Average reward: ' + str(np.mean(rewards[-PLOT_SAMPLES:])))
# ax1.set_ylabel('Reward')

ax2.plot(time_stamp[-PLOT_SAMPLES:], satlites[-PLOT_SAMPLES:])
ax2.set_ylabel('satlites switch')

ax3.plot(time_stamp[-PLOT_SAMPLES:], buffer_occupancies[-PLOT_SAMPLES:])
ax3.set_ylabel('buffer occupancy (sec)')

ax4.plot(time_stamp[-PLOT_SAMPLES:], bit_rates[-PLOT_SAMPLES:])
ax4.set_ylabel('bitrate selection(sec)')
ax4.set_xlabel('Time (ms)')

f.subplots_adjust(hspace=0)

plt.show()
plt.savefig('books_read.png')