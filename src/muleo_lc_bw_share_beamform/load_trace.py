import os
import csv

COOKED_TRACE_FOLDER = 'beamformed_rss/'
COOKED_DIS_FOLDER = 'dis/'


# LEO SETTINGS
HANDOVER_DELAY = 0.2  # sec
HANDOVER_WEIGHT = 1
SCALE_VIDEO_SIZE_FOR_TEST = 20
SCALE_VIDEO_LEN_FOR_TEST = 2
SCALE_FOR_TEST = 1 / SCALE_VIDEO_SIZE_FOR_TEST


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER, split_condition=None):
    cooked_files = os.listdir(cooked_trace_folder)
    all_satellite_bw = []
    all_cooked_time = []
    all_file_names = []

    for cooked_file in sorted(cooked_files):
        file_path = cooked_trace_folder + cooked_file
        satellite_id = []
        satellite_bw = {}
        cooked_time = []

        with open(file_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    # Get Satellite ID
                    satellite_id = list(row.keys())[1:]
                    satellite_bw = {int(sat_id): [] for sat_id in satellite_id}
                for sat_id in satellite_id:
                    # satellite_bw[int(sat_id)].append(float(row[sat_id]))
                    if row[sat_id] == '':
                        raw_data = 0
                    else:
                        raw_data = float(row[sat_id])
                    satellite_bw[int(sat_id)].append(raw_data)
                cooked_time.append(int(row['']))

                line_count += 1

        # Reformat the bw data
        b_max = 100
        remove_list = []
        for sat_id in satellite_bw:
            rss_min = min(satellite_bw[sat_id])
            new_bw = []
            for bw in satellite_bw[sat_id]:
                new_bw.append(abs(b_max * bw / rss_min) * SCALE_FOR_TEST)
            satellite_bw[sat_id] = new_bw
            if all(bw == 0.0 for bw in new_bw):
                print(sat_id)
                remove_list.append(sat_id)

        for sat_id in remove_list:
            satellite_bw.pop(sat_id, None)

        all_satellite_bw.append(satellite_bw)
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
