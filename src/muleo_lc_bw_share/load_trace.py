import os
import csv

from util.constants import NUM_USERS, MULTI_USER_FOLDER

COOKED_TRACE_FOLDER = 'train/'


# LEO SETTINGS
HANDOVER_DELAY = 0.2  # sec
HANDOVER_WEIGHT = 1
SCALE_VIDEO_SIZE_FOR_TEST = 30
SCALE_VIDEO_LEN_FOR_TEST = 2
SCALE_FOR_TEST = 1 / SCALE_VIDEO_SIZE_FOR_TEST


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER, num_users=NUM_USERS, split_condition=None):
    cooked_files = os.listdir(cooked_trace_folder)
    all_satellite_bw = []
    all_cooked_time = []
    all_file_names = []

    for cooked_file in sorted(cooked_files):
        file_path = cooked_trace_folder + cooked_file
        satellite_id = []
        cooked_time = []

        with open(file_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0

            for row in csv_reader:
                """
                if line_count == 0:
                    # Get Satellite ID
                    satellite_id = list(row.keys())[2:]
                    satellite_bw = {int(sat_id): [] for sat_id in satellite_id}
                for sat_id in satellite_id:
                    # satellite_bw[int(sat_id)].append(float(row[sat_id]))
                    satellite_bw[int(sat_id)].append(float(row[sat_id]) * SCALE_FOR_TEST)
                """
                cooked_time.append(int(row["time"]))
            tmp_bw = []
            for user_id in range(num_users):
                splited_elem = cooked_file.split("_")
                user_file_name = '_'.join(splited_elem[0:2]) + "_user_" + str(user_id) + "_" + '_'.join(splited_elem[2:])
                with open(MULTI_USER_FOLDER + user_file_name, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    line_count = 0
                    satellite_bw = {}
                    for row in csv_reader:
                        if line_count == 0:
                            # Get Satellite ID
                            satellite_id = list(row.keys())[2:]
                            satellite_bw = {int(sat_id): [] for sat_id in satellite_id}
                        for sat_id in satellite_id:
                            # satellite_bw[int(sat_id)].append(float(row[sat_id]))
                            satellite_bw[int(sat_id)].append(float(row[sat_id]) * SCALE_FOR_TEST)

                        line_count += 1
                    tmp_bw.append(satellite_bw)
        all_satellite_bw.append(tmp_bw)
        all_cooked_time.append(cooked_time)
        all_file_names.append(os.path.splitext(cooked_file)[0])

    if split_condition == "train":
        for i in range(len(all_cooked_time)):
            for sat_id, sat_bw in all_satellite_bw[i].items():
                all_satellite_bw[i][sat_id] = all_satellite_bw[i][sat_id][:round(len(all_satellite_bw[i][sat_id])*0.8)]
            all_cooked_time[i] = all_cooked_time[i][:round(len(all_cooked_time[i])*0.8)]

    elif split_condition == "test":
        for i in range(len(all_cooked_time)):
            for sat_id, sat_bw in all_satellite_bw[i].items():
                all_satellite_bw[i][sat_id] = all_satellite_bw[i][sat_id][round(len(all_satellite_bw[i][sat_id])*0.8):]
            all_cooked_time[i] = all_cooked_time[i][round(len(all_cooked_time[i])*0.8):]

    return all_cooked_time, all_satellite_bw, all_file_names

