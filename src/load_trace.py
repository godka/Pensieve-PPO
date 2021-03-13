import os
import numpy as np

COOKED_TRACE_FOLDER = './train/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names

def load_trace_virtual():
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    random_file = np.random.randint(30)
    for cooked_file in range(random_file):
        cooked_time = []
        cooked_bw = []
        max_bandwidth = np.random.uniform(0.03, 11.)
        min_bandwidth = np.random.uniform(0.01, max_bandwidth)
        for iter_ in range(300):
            cooked_time.append(float(iter_))
            cooked_bw.append(np.random.uniform(min_bandwidth, max_bandwidth))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(str(cooked_file))

    return all_cooked_time, all_cooked_bw, all_file_names
