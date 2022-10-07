
from leo import load_trace

TEST_TRACES = './test/'

SCALE_FOR_TEST = 1/30
all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES, SCALE_FOR_TEST=SCALE_FOR_TEST)

n = len(all_cooked_time)

B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0

all_cooked_remain = []
for trace_idx in range(len(all_cooked_bw)):
   all_cooked_remain.append({})
   for sat_id, sat_bw in all_cooked_bw[trace_idx].items():
         all_cooked_remain[trace_idx][sat_id] = []
         for index in range(len(sat_bw)):
            count = 0
            while index + count < len(sat_bw) and sat_bw[index] != 0:
               count += 1
            all_cooked_remain[trace_idx][sat_id].append(count)

for i in range(n):
   filename = 'lc_trace/'  + str(i) + '.log'
   cooked_time = all_cooked_time[i]
   cooked_bw = all_cooked_bw[i]
   cooked_remain = all_cooked_remain[trace_idx]
   
   length = len(cooked_time)
   
   log_file = open(filename, 'w')  
   for j in range(length):
      best_sat_bw = 0
      for sat_id, sat_bw in cooked_bw.items():
         if best_sat_bw < sat_bw[j]:
            best_sat_id = sat_id
            best_sat_bw = sat_bw[j]
      log_file.write(str(float(cooked_time[j])) + ' ' + str(float(best_sat_bw)) + '\n')
      if j > 300:
         break
