import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import matplotlib
import sys
from collections import OrderedDict
import scipy.stats
plt.switch_backend('agg')

NUM_BINS = 10000
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1

# labels = SCHEMES#, 'RB']
LW = 2.
LOG = './baselines/'

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def inlist(filename, traces):
    ret = False
    for trace in traces:
        if trace in filename:
            ret = True
            break
    return ret

def bitrate_smo(outputs):
    SCHEMES = ['bb', 'rl', 'mpc', 'hyb', 'bola', 'rb', 'quetra', 'elastic', 'ppo']
    labels = ['BBA', 'Pensieve', 'RobustMPC', 'HYB', 'BOLA', 'Rate-based', 'QUETRA', 'Elastic', 'Pen-PPO']
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    lines = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                # if np.mean(arr[1:]) > -100.:
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_bit)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_smo)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            marker = markers[idx], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Bitrate Smoothness (mbps)')
    ax.set_ylabel('Video Bitrate (mbps)')
    ax.set_ylim(max_bitrate * 0.5, max_bitrate * 1.01)

    ax.grid(linestyle='--', linewidth=1.2)
    ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left')
    ax.invert_xaxis()
    # ax.invert_yaxis()

    fig.savefig(outputs + '.png')
    # fig.savefig(outputs + '.pdf')
    plt.close()

def smo_rebuf(outputs):
    # os.system('cp ./test_results/* ' + LOG)
    SCHEMES = ['bb', 'rl', 'mpc', 'hyb', 'bola', 'rb', 'quetra', 'elastic', 'ppo']
    labels = ['BBA', 'Pensieve', 'RobustMPC', 'HYB', 'BOLA', 'Rate-based', 'QUETRA', 'Elastic', 'Pen-PPO']
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    lines = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                # if np.mean(arr[1:]) > -100.:
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_smo)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_rebuf)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            marker = markers[idx], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Time Spent on Stall (%)')
    ax.set_ylabel('Bitrate Smoothness (mbps)')
    ax.set_ylim(0., max_bitrate * 1.5)

    ax.grid(linestyle='--', linewidth=1.2)
    ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left')
    ax.invert_xaxis()
    ax.invert_yaxis()

    fig.savefig(outputs + '.png')
    # fig.savefig(outputs + '.pdf')
    plt.close()

def bitrate_rebuf(outputs):
    # os.system('cp ./test_results/* ' + LOG)
    SCHEMES = ['bb', 'rl', 'mpc', 'hyb', 'bola', 'rb', 'quetra', 'elastic', 'ppo']
    labels = ['BBA', 'Pensieve', 'RobustMPC', 'HYB', 'BOLA', 'Rate-based', 'QUETRA', 'Elastic', 'Pen-PPO']
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    lines = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                # if np.mean(arr[1:]) > -100.:
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_bit)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_rebuf)
        # mean_smo_, low_smo_, high_smo_ = mean_confidence_interval(mean_smo)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            marker = markers[idx], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Time Spent on Stall (%)')
    ax.set_ylabel('Video Bitrate (mbps)')
    ax.set_ylim(max_bitrate * 0.5, max_bitrate * 1.01)

    ax.grid(linestyle='--', linewidth=1.2)
    ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left')
    ax.invert_xaxis()

    fig.savefig(outputs + '.png')
    # fig.savefig(outputs + '.pdf')
    plt.close()

if __name__ == '__main__':
    os.system('cp ./test_results/* ' + LOG)
    bitrate_rebuf('baselines-br')
    smo_rebuf('baselines-sr')
    bitrate_smo('baselines-bs')