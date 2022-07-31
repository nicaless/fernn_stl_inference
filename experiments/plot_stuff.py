import json
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science', 'ieee'])

# # PLOT THE SYNTHETIC DATA
# x = np.loadtxt('basic_sanity_checks/data1_trace.csv', delimiter=',')
# y = np.loadtxt('basic_sanity_checks/data2_trace.csv', delimiter=',')
# z = np.loadtxt('basic_sanity_checks/data3_trace.csv', delimiter=',')
#
# time_range = range(0, 5)
# for i in range(10):
#     plt.plot(time_range, x[i, ])
# plt.savefig('plots/synth_data_x.png')
# plt.clf()
# for i in range(10):
#     plt.plot(time_range, y[i, ])
# plt.savefig('plots/synth_data_y.png')
# plt.clf()
# for i in range(10):
#     plt.plot(time_range, z[i, ])
# plt.savefig('plots/synth_data_z.png')
# plt.clf()

# PLOT HAPT LOSS CURVES AND DATA
x = np.loadtxt('hapt_data/good_hapt_data.csv', delimiter=' ')
y = np.loadtxt('hapt_data/bad_hapt_data.csv', delimiter=' ')
# w1 = json.load(open('hapt_data/test_1_quantize_weights.json', 'r'))
# w2_1 = json.load(open('hapt_data/test_2_quantize_weights.json', 'r'))
# w2_2 = json.load(open('hapt_data/test_2_weights.json', 'r'))

# dirname = 'hapt_data/'
# tp_files = ['test_1', 'test_2']
# titles = ['Alw (x)',
#           'ch (Alw, Ev) (x)']
# i = 0
# for fname in tp_files:
#     epoch_num = []
#     quantized_loss = []
#     unquantized_loss = []
#
#     f1 = open(dirname + fname + '_quantize.txt', 'r')
#     lines = f1.readlines()
#     get_final_weights = False
#     for line in lines:
#         if line.startswith('Epoch '):
#             epoch = int(line.split(',')[0].split('Epoch ')[1])
#             loss = float(line.split(',')[1].split('loss ')[1])
#             epoch_num.append(epoch)
#             quantized_loss.append(loss)
#
#     if fname != 'test_1':
#         f2 = open(dirname + fname + '.txt', 'r')
#         lines = f2.readlines()
#         for line in lines:
#             if line.startswith('Epoch '):
#                 loss = float(line.split(',')[1].split('loss ')[1])
#                 unquantized_loss.append(loss)
#
#     plt.plot(epoch_num, quantized_loss, label='quantized')
#
#     if fname != 'test_1':
#         plt.plot(epoch_num, unquantized_loss, label='un-quantized')
#         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w2_2['choice1']['true_weights']), fontsize=10)
#
#     plt.legend()
#     plt.title(titles[i])
#     plt.savefig('plots/loss_hapt_' + fname + '.png')
#     plt.clf()
#     i += 1

time_range = range(0, 10)
for i in range(10):
    plt.plot(time_range, x[i, ], 'g-', alpha=0.8)
for i in range(10):
    plt.plot(time_range, y[i, ], 'r-', alpha=0.3)

plt.axhline(y=-0.81, linestyle='--', label='RNN')
plt.axhline(y=-0.65, linestyle='-.', label='Enumerative')
plt.axhline(y=-0.57, xmin=0.5, xmax=1, linestyle='dotted', label='Lattice (Class)')
plt.axhline(y=0.935, xmin=0, xmax=0.5, linestyle=(0, (1, 10)), label='Lattice (Pred)')
# plt.legend(bbox_to_anchor=(0, 1), loc='lower center')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left")
# plt.legend()
plt.ylabel('std dev in x direction')
plt.xlabel('time')
plt.savefig('plots/hapt_data_qual.png')
plt.clf()


# PLOT THE CRUISE CONTROL DATA
good = np.loadtxt('cruise_control_train/Traces_normal_BIG2.csv', delimiter=',')
bad = np.loadtxt('cruise_control_train/Traces_anomaly_BIG2.csv', delimiter=',')
time_range = range(0, 101)

# w1_1 = json.load(open('cruise_control_train/test_1_weights.json', 'r'))
# w2_1 = json.load(open('cruise_control_train/test_2_quantize_weights.json', 'r'))
# w2_2 = json.load(open('cruise_control_train/test_2_weights.json', 'r'))
#
# dirname = 'cruise_control_train/'
# tp_files = ['test_1', 'test_2']
# titles = ['Alw (x)',
#           'choose (Alw, Ev) (x)']
# i = 0
# for fname in tp_files:
#     epoch_num = []
#     quantized_loss = []
#     unquantized_loss = []
#
#     f1 = open(dirname + fname + '_quantize.txt', 'r')
#     lines = f1.readlines()
#     for line in lines:
#         if line.startswith('Epoch '):
#             epoch = int(line.split(',')[0].split('Epoch ')[1])
#             loss = float(line.split(',')[1].split('loss ')[1])
#             epoch_num.append(epoch)
#             quantized_loss.append(loss)
#
#     if fname != 'test_1':
#         f2 = open(dirname + fname + '.txt', 'r')
#         lines = f2.readlines()
#         for line in lines:
#             if line.startswith('Epoch '):
#                 loss = float(line.split(',')[1].split('loss ')[1])
#                 unquantized_loss.append(loss)
#
#     plt.plot(epoch_num, quantized_loss, label='quantized')
#     if fname != 'test_1':
#         plt.plot(epoch_num, unquantized_loss, label='un-quantized')
#         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w2_2['choice1']['true_weights']), fontsize=10)
#
#     plt.legend()
#     plt.title(titles[i])
#     plt.savefig('plots/loss_cruise_control_' + fname + '.png')
#     plt.clf()
#     i += 1


for i in range(30):
    plt.plot(time_range, good[i, ], 'g-')
for i in range(30):
    plt.plot(time_range, bad[i, ], 'r-')

plt.axhline(y=39.8, linestyle='--', label='RNN')
plt.axhline(y=34.3, linestyle='-.', label='Enumerative')
plt.axhline(y=34, xmin=0.5, xmax=1, linestyle='dotted', label='Lattice (Class)')
plt.axhline(y=50, xmin=0, xmax=0.5, linestyle=(0, (1, 10)), label='Lattice (Pred)')
# plt.legend()
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left")
plt.ylabel('velocity')
plt.xlabel('time')
plt.savefig('plots/cruise_control_train_qual.png')
plt.clf()


# PLOT THE LYFT DATA
good_x = np.loadtxt('lyft_data/lyft_good_traces_x.csv', delimiter=',')
good_y = np.loadtxt('lyft_data/lyft_good_traces_y.csv', delimiter=',')
bad_x = np.loadtxt('lyft_data/bad_traces_x.csv', delimiter=',')
bad_y = np.loadtxt('lyft_data/bad_traces_y.csv', delimiter=',')

# w1_1 = json.load(open('lyft_data/test_1_quantize_weights.json', 'r'))
# w1_2 = json.load(open('lyft_data/test_1_weights.json', 'r'))
# w2_1 = json.load(open('lyft_data/test_2_quantize_weights.json', 'r'))
# w2_2 = json.load(open('lyft_data/test_2_weights.json', 'r'))
# w3_1 = json.load(open('lyft_data/test_3_quantize_weights.json', 'r'))
# w3_2 = json.load(open('lyft_data/test_3_weights.json', 'r'))
# w4_1 = json.load(open('lyft_data/test_4_quantize_weights.json', 'r'))
# w4_2 = json.load(open('lyft_data/test_4_weights.json', 'r'))

time_range = range(0, good_x.shape[1])

for i in range(20):
    plt.plot(time_range, good_x[i, ], 'g-', alpha=0.8)
for i in range(20):
    plt.plot(time_range, bad_x[i, ], 'r-', alpha=0.3)

plt.axhline(y=-10.3421, xmin=0.5, xmax=1, linestyle='dotted', label='Lattice (Class)')
plt.axhline(y=50, xmin=0, xmax=0.5, linestyle=(0, (1, 10)), label='Lattice (Pred)')
# plt.legend()
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left")
plt.ylabel('x')
plt.xlabel('time')
plt.savefig('plots/lyftx_qual.png')
plt.clf()

for i in range(20):
    plt.plot(time_range, good_y[i, ], 'g-', alpha=0.8)
for i in range(20):
    plt.plot(time_range, bad_y[i, ], 'r-', alpha=0.3)

plt.axhline(y=-1.4, linestyle='--', label='RNN')
plt.axhline(y=1.5, linestyle='-.', label='Enumerative')
# plt.legend()
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left")
plt.ylabel('y')
plt.xlabel('time')
plt.savefig('plots/lyfty_qual.png')
plt.clf()


#
# # dirname = 'lyft_data/'
# # tp_files = ['test_1', 'test_2', 'test_3', 'test_4']
# # titles = ['ch (Alw, Ev) (ax)', 'ch (Alw, Ev) (ch (x, y)',
# #           'ch (Alw, Ev) (ch(AND,OR)(x,y))',
# #           'ch (Alw, Ev) (ch(AND,OR)(ch(x, y))']
# # i = 0
# # for fname in tp_files:
# #     epoch_num = []
# #     quantized_loss = []
# #     unquantized_loss = []
# #
# #     f1 = open(dirname + fname + '_quantize.txt', 'r')
# #     lines = f1.readlines()
# #     for line in lines:
# #         if line.startswith('Epoch '):
# #             epoch = int(line.split(',')[0].split('Epoch ')[1])
# #             loss = float(line.split(',')[1].split('loss ')[1])
# #             epoch_num.append(epoch)
# #             quantized_loss.append(loss)
# #
# #     f2 = open(dirname + fname + '.txt', 'r')
# #     lines = f2.readlines()
# #     for line in lines:
# #         if line.startswith('Epoch '):
# #             loss = float(line.split(',')[1].split('loss ')[1])
# #             unquantized_loss.append(loss)
# #
# #     plt.plot(epoch_num, quantized_loss, label='quantized')
# #     plt.plot(epoch_num, unquantized_loss, label='un-quantized')
# #
# #     if fname == 'test_1':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w1_2['choice1']['true_weights']), fontsize=10)
# #     if fname == 'test_2':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w2_2['choice1']['true_weights']), fontsize=10)
# #     if fname == 'test_3':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w3_2['choice1']['true_weights']), fontsize=10)
# #     if fname == 'test_4':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w4_2['choice1']['true_weights']), fontsize=10)
# #
# #     plt.legend()
# #     plt.title(titles[i])
# #     plt.savefig('plots/loss_curve_lyft_' + fname + '.png')
# #     plt.clf()
# #     i += 1
#
#
#
# PLOT THE ECG DATA
good_x = np.loadtxt('ecg/hr_traces_len_10/1_nsr_traces_len_10.csv',
                    delimiter=',')
bad_x = np.loadtxt('ecg/hr_traces_len_10/4_afib_traces_len_10.csv',
                    delimiter=',')
good_x_diff = np.loadtxt('ecg/hr_traces_len_10_diff/1_nsr_traces_len_10.csv',
                         delimiter=',')
bad_x_diff = np.loadtxt('ecg/hr_traces_len_10_diff/4_afib_traces_len_10.csv',
                        delimiter=',')
time_range = range(0, good_x.shape[1])

# # w1_1 = json.load(open('ecg/test_1_quantize_weights.json', 'r'))
# # w1_2 = json.load(open('ecg/test_1_weights.json', 'r'))
# # w2_1 = json.load(open('ecg/test_2_quantize_weights.json', 'r'))
# # w2_2 = json.load(open('ecg/test_2_weights.json', 'r'))
# # w3_1 = json.load(open('ecg/test_3_quantize_weights.json', 'r'))
# # w3_2 = json.load(open('ecg/test_3_weights.json', 'r'))
# # w4_1 = json.load(open('ecg/test_4_quantize_weights.json', 'r'))
# # w4_2 = json.load(open('ecg/test_4_weights.json', 'r'))
# # w5_1 = json.load(open('ecg/test_5_quantize_weights.json', 'r'))
# # w5_2 = json.load(open('ecg/test_5_weights.json', 'r'))
# # w6_1 = json.load(open('ecg/test_6_quantize_weights.json', 'r'))
# # w6_2 = json.load(open('ecg/test_6_weights.json', 'r'))
# # w7_1 = json.load(open('ecg/test_7_quantize_weights.json', 'r'))
# # w7_2 = json.load(open('ecg/test_7_weights.json', 'r'))
# #
# #
# # dirname = 'ecg/'
# # tp_files = ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7']
# # titles = ['ch (Alw, Ev) (hr)', 'ch (Alw, Ev) (hr_dx)',
# #           'ch (Alw, Ev) (ch (hr, hr_dx))',
# #           'ch (Alw, Ev) (ch (hr AND hr_dx,  hr, OR hr_dx))',
# #           '(ch (Alw, Ev) (ch (hr, hr_dx))) AND (ch (Alw, Ev) (ch (hr, hr_dx)))',
# #           '(ch (Alw, Ev) (ch (hr, hr_dx))) OR (ch (Alw, Ev) (ch (hr, hr_dx)))',
# #           '(ch (Alw, Ev) (ch (hr, hr_dx))) ch(AND_OR) (ch (Alw, Ev) (ch (hr, hr_dx)))'
# #           ]
# # i = 0
# # for fname in tp_files:
# #     epoch_num = []
# #     quantized_loss = []
# #     unquantized_loss = []
# #
# #     f1 = open(dirname + fname + '_quantize.txt', 'r')
# #     lines = f1.readlines()
# #     for line in lines:
# #         if line.startswith('Epoch '):
# #             epoch = int(line.split(',')[0].split('Epoch ')[1])
# #             loss = float(line.split(',')[1].split('loss ')[1])
# #             epoch_num.append(epoch)
# #             quantized_loss.append(loss)
# #
# #     f2 = open(dirname + fname + '.txt', 'r')
# #     lines = f2.readlines()
# #     for line in lines:
# #         if line.startswith('Epoch '):
# #             loss = float(line.split(',')[1].split('loss ')[1])
# #             unquantized_loss.append(loss)
# #
# #     plt.plot(epoch_num, quantized_loss, label='quantized')
# #     plt.plot(epoch_num, unquantized_loss, label='un-quantized')
# #
# #     if fname == 'test_1':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w1_2['choice1']['true_weights']), fontsize=10)
# #     if fname == 'test_2':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w2_2['choice1']['true_weights']), fontsize=10)
# #     if fname == 'test_3':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w3_2['choice1']['true_weights']), fontsize=10)
# #     if fname == 'test_4':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w4_2['choice1']['true_weights']), fontsize=10)
# #     if fname == 'test_5':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w5_2['choice1']['true_weights']), fontsize=10)
# #     if fname == 'test_6':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w6_2['choice1']['true_weights']), fontsize=10)
# #     if fname == 'test_7':
# #         plt.text(0, -0.3, "final choice weights (unquant)\n" + str(w7_2['choice0']['true_weights']), fontsize=10)
# #
# #     plt.legend()
# #     plt.title(titles[i])
# #     plt.savefig('plots/loss_curve_ecg_' + fname + '.png')
# #     plt.clf()
# #     i += 1
#
for i in range(15):
    plt.plot(time_range, good_x[i, ], 'g-', alpha=0.8)
for i in range(15):
    plt.plot(time_range, bad_x[i, ], 'r-', alpha=0.3)

plt.axhline(y=120.2, xmin=0.5, xmax=1, linestyle='dotted', label='Lattice (Class)')
plt.axhline(y=86.7, xmin=0, xmax=0.5, linestyle=(0, (1, 10)), label='Lattice (Pred)')
# plt.legend()
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left")
plt.ylabel('Heart Rate')
plt.xlabel('time')
plt.savefig('plots/ecg_hr_qual.png')
plt.clf()

for i in range(15):
    plt.plot(time_range, good_x_diff[i, ], 'g-', alpha=0.8)
for i in range(15):
    plt.plot(time_range, bad_x_diff[i, ], 'r-', alpha=0.3)
plt.savefig('plots/ecg_data_diff.png')

plt.axhline(y=5.80, linestyle='-.', label='Enumerative')
plt.axhline(y=-3.783, linestyle='--', label='RNN')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left")
plt.ylabel('Change in Heart Rate')
plt.xlabel('time')
plt.savefig('plots/ecg_hrdiff_qual.png')
plt.clf()
