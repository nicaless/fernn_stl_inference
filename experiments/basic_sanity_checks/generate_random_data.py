import numpy as np


DATA_OUTPUT_FOLDER = 'experiments/basic_sanity_checks/'
np.random.seed(42)

# GENERATE RANDOM ROBUSTNESS VALUES
trace_length = 5
data1 = np.round(np.random.uniform(low=-5, high=5,
                                   size=(100, trace_length)), 2)
data2 = np.round(np.random.uniform(low=-5, high=5,
                                   size=(100, trace_length)), 2)
data3 = np.round(np.random.uniform(low=-5, high=5,
                                   size=(100, trace_length)), 2)

np.savetxt(DATA_OUTPUT_FOLDER + 'data1_trace.csv',
           data1, delimiter=',', fmt='%.5e')
np.savetxt(DATA_OUTPUT_FOLDER + 'data2_trace.csv',
           data2, delimiter=',', fmt='%.5e')
np.savetxt(DATA_OUTPUT_FOLDER + 'data3_trace.csv',
           data3, delimiter=',', fmt='%.5e')
