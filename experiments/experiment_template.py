import sys
sys.path.append('/home/nicaless/repos/gan_tl')

import argparse
from formulas.choose_length import choose_length
# from formulas.choose_depth import choose_depth
# from formulas.two_deep import two_deep
# from formulas.three_deep import three_deep
# from formulas.four_deep import four_deep
# from formulas.five_deep import five_deep
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from TL_learning_utils import extract_weights
import time

parser = argparse.ArgumentParser(description='ECG Test')
parser.add_argument('exp', help='experiment dataset', type=str, default='ecg')
parser.add_argument('depth', help='test number', type=int, default=2)
parser.add_argument('save_dir', help='results folder')
parser.add_argument('--quantize', help='quantize weights', action='store_true')
parser.add_argument('--since', help='include since templates', action='store_true')
parser.add_argument('--excl', help='include since templates', action='store_true')
args = parser.parse_args()
print(args)


# GET EXPERIMENT ARGUMENTS
DIR = args.save_dir
EXP = args.exp
DEPTH = args.depth
QUANT = 'one-hot' if args.quantize else False
QT = 'quant' if args.quantize else ''
SINCE = args.since
SI = 'since' if args.since else ''
EXCL = args.excl
TEST = False
EX = 'excl' if args.excl else ''
TRAINING_PROGRESS_FILE_NAME = '{}/{}_{}{}{}{}.txt'.format(DIR, EXP, DEPTH, QT, SI, EX)
WEIGHTS_PROGRESS_FILE = '{}/{}_{}{}{}{}_WEIGHTS.json'.format(DIR, EXP, DEPTH, QT, SI, EX)
CRUISE_CONTROL_CONT = False
ECG_CONT = False

TRAINING_PROGRESS_FILE = open(TRAINING_PROGRESS_FILE_NAME, 'w')
MODEL_VIS_FILE = EXP + '_' + str(DEPTH) + '.png'

# GET PATHS TO DATA
if EXP == 'ecg':
    good_data_files = \
        ['experiments/ecg/hr_traces_len_10/1_nsr_traces_len_10.csv',
         'experiments/ecg/hr_traces_len_10_diff/1_nsr_traces_len_10.csv']
    bad_data_files = \
        ['experiments/ecg/hr_traces_len_10/4_afib_traces_len_10.csv',
         'experiments/ecg/hr_traces_len_10_diff/4_afib_traces_len_10.csv']
    delim = ','
    feature_names = ['hr', 'hr_dx']
elif EXP == 'ecg_cont':
    good_data_files = \
        ['experiments/ecg/hr_traces_len_10/1_nsr_traces_len_10.csv',
         'experiments/ecg/hr_traces_len_10_diff/1_nsr_traces_len_10.csv']
    bad_data_files = \
        ['experiments/ecg/hr_traces_len_10/4_afib_traces_len_10.csv',
         'experiments/ecg/hr_traces_len_10_diff/4_afib_traces_len_10.csv']
    good_lab = np.loadtxt('experiments/ecg/labels_normal_ecg.csv')
    bad_lab = np.loadtxt('experiments/ecg/labels_anomaly_ecg.csv')
    delim = ','
    feature_names = ['hr', 'hr_dx']
    ECG_CONT = True
elif EXP == 'hapt':
    good_data_files = ['experiments/hapt_data/good_hapt_data.csv']
    bad_data_files = ['experiments/hapt_data/bad_hapt_data.csv']
    delim = ' '
    feature_names = ['x']
elif EXP == 'lyft':
    good_data_files = ['experiments/lyft_data/lyft_good_traces_x.csv',
                       'experiments/lyft_data/lyft_good_traces_x_diff.csv',
                       'experiments/lyft_data/lyft_good_traces_y.csv',
                       'experiments/lyft_data/lyft_good_traces_y_diff.csv']
    bad_data_files = ['experiments/lyft_data/bad_traces_x.csv',
                      'experiments/lyft_data/bad_traces_x_diff.csv',
                      'experiments/lyft_data/bad_traces_y.csv',
                      'experiments/lyft_data/bad_traces_y_diff.csv']
    delim = ','
    feature_names = ['x', 'x_diff', 'y', 'y_diff']
elif EXP == 'cruise_cont':
    good_data_files = \
        ['experiments/cruise_control_train/Traces_normal_BIG2.csv']
    bad_data_files = \
        ['experiments/cruise_control_train/Traces_anomaly_BIG2.csv']
    good_lab = np.loadtxt(
        'experiments/cruise_control_train/Labels_normal_BIG2.csv')
    bad_lab = np.loadtxt(
        'experiments/cruise_control_train/Labels_anomaly_BIG2.csv')
    delim = ','
    feature_names = ['v']
    CRUISE_CONTROL_CONT = True
elif EXP == 'cruise':
    good_data_files = \
        ['experiments/cruise_control_train/Traces_normal_BIG2.csv']
    bad_data_files = \
        ['experiments/cruise_control_train/Traces_anomaly_BIG2.csv']
    delim = ','
    feature_names = ['v']
else:
    print('INVALID EXPERIMENT KEY, EXITING')
    sys.exit()

# LOAD DATA
good_data = []
for file in good_data_files:
    good_data.append(np.loadtxt(file, delimiter=delim))
bad_data = []
for file in bad_data_files:
    bad_data.append(np.loadtxt(file, delimiter=delim))

data_size = good_data[0].shape[0]
trace_length = good_data[0].shape[1]

train_size = int(data_size * 0.8)

# PARTITITION TRAIN AND TEST DATA SETS, NORMALIZE
num_features = len(good_data)

good_data_train = []
bad_data_train = []
feature_normalizers = {}
for i in range(num_features):
    gd = good_data[i][0:train_size, :]
    bd = bad_data[i][0:train_size, :]
    train = np.vstack((gd, bd))
    # normalizer = (np.max(train) - np.min(train))
    normalizer = 1
    gd = gd / normalizer
    bd = bd / normalizer
    # APPLY SIGMOID
    gd = 1 / (1 + np.exp(-1 * gd))
    bd = 1 / (1 + np.exp(-1 * bd))
    good_data_train.append(gd)
    bad_data_train.append(bd)
    feature_normalizers[feature_names[i]] = normalizer
print('Initial Feature Normalizer')
print(feature_normalizers)

# NETWORK PARAMS
np.random.seed(42)
tf.random.set_seed(42)
epochs = 5000
epoch_check = int(epochs / 100)
batches = 4
batch_size = int(train_size / 4)  # per class
reduce_lr_after = 100
early_stopping_factor = np.inf  # no early stopping
# early_stopping_factor = 0  # stop after first lr reduction
# early_stopping_factor = 5 if EXCL else 10
# init_lr = 0.003
init_lr = 0.001
reduce_lr_factor = 0.98


# BUILD NETWORK
input_shape = (trace_length, 1)
if DEPTH == 2:
    # formula, inputs, CHOICE_CONFIG, feature_normalizers = \
    #     two_deep(feature_names, input_shape, feature_normalizers,
    #              quantize=QUANT, with_since=SINCE)
    formula, inputs, CHOICE_CONFIG, feature_normalizers = \
        choose_length(feature_names, input_shape, feature_normalizers,
                      L=2, quantize=QUANT, with_since=SINCE, excl=EXCL)
    # formula, inputs, CHOICE_CONFIG, feature_normalizers = \
    #     choose_depth(feature_names, input_shape, feature_normalizers,
    #                   D=2, quantize=QUANT, with_since=SINCE, excl=EXCL)
elif DEPTH == 3:
    # formula, inputs, CHOICE_CONFIG, feature_normalizers = \
    #     three_deep(feature_names, input_shape, feature_normalizers,
    #                quantize=QUANT, with_since=SINCE)
    formula, inputs, CHOICE_CONFIG, feature_normalizers = \
        choose_length(feature_names, input_shape, feature_normalizers,
                      L=3, quantize=QUANT, with_since=SINCE, excl=EXCL)
elif DEPTH == 4:
    # formula, inputs, CHOICE_CONFIG, feature_normalizers = \
    # four_deep(feature_names, input_shape, feature_normalizers,
    #            quantize=QUANT)
    formula, inputs, CHOICE_CONFIG, feature_normalizers = \
        choose_length(feature_names, input_shape, feature_normalizers,
                      L=4, quantize=QUANT, with_since=SINCE, excl=EXCL)
elif DEPTH == 5:
    # formula, inputs, CHOICE_CONFIG, feature_normalizers = \
    # five_deep(feature_names, input_shape, feature_normalizers,
    #            quantize=QUANT)
    formula, inputs, CHOICE_CONFIG, feature_normalizers = \
        choose_length(feature_names, input_shape, feature_normalizers,
                      L=5, quantize=QUANT, with_since=SINCE, excl=EXCL)
elif DEPTH == 6:
    formula, inputs, CHOICE_CONFIG, feature_normalizers = \
        choose_length(feature_names, input_shape, feature_normalizers,
                      L=6, quantize=QUANT, with_since=SINCE, excl=EXCL)
else:
    print('INVALID NETWORK OPTION, EXITING')
    sys.exit()

print('Feature Normalizers')
print(feature_normalizers)
print('Choice Config')
print(CHOICE_CONFIG)


# COMPILE MODEL
if CRUISE_CONTROL_CONT or ECG_CONT:
    out = formula
else:
    out = tf.keras.layers.Activation(tf.keras.activations.tanh)(formula)
model = Model(inputs=inputs, outputs=[out])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=init_lr)
model.compile(optimizer=optimizer, loss='mean_absolute_error')
print(model.summary())
plot_model(model, to_file=MODEL_VIS_FILE, show_shapes=False, show_layer_names=True)

names = [weight.name for layer in model.layers for weight in layer.weights]
original_weights = model.get_weights()


# TRAIN MODEL
save_weights = []
last_loss = np.inf
last_loss_update = -1
best_iter_so_far = -1
train_time_start = time.time()
curr_lr = init_lr
lr_reductions = 0

batch_end_layers = [getattr(layer, 'on_batch_end', None) for layer in model.layers]
print(model.layers)
print(batch_end_layers)

for i in range(epochs):
    idx1 = np.random.choice(train_size, size=train_size, replace=False)
    idx2 = np.random.choice(train_size, size=train_size, replace=False)

    good_epoch_data = []
    bad_epoch_data = []
    for nf in range(num_features):
        good_epoch_data.append(good_data_train[nf][idx1,:])
        bad_epoch_data.append(bad_data_train[nf][idx1, :])

    if CRUISE_CONTROL_CONT or ECG_CONT:
        good_epoch_labels = good_lab[idx1] / normalizer
        bad_epoch_labels = bad_lab[idx2] / normalizer

    batch_loss = []
    for b in range(batches):
        batch_start = b * batch_size
        batch_end = batch_start + batch_size

        features = []
        for nf in range(num_features):
            g_batch = good_epoch_data[nf][batch_start:batch_end, :]
            g_batch = g_batch.reshape((batch_size, trace_length, 1))

            b_batch = bad_epoch_data[nf][batch_start:batch_end, :]
            b_batch = b_batch.reshape((batch_size, trace_length, 1))

            feature_set = np.vstack((g_batch, b_batch))
            features.append(feature_set)

        if CRUISE_CONTROL_CONT or ECG_CONT:
            g_batch_labels = good_epoch_labels[batch_start:batch_end]
            g_batch_labels = g_batch_labels.reshape((batch_size, 1))
            b_batch_labels = bad_epoch_labels[batch_start:batch_end]
            b_batch_labels = b_batch_labels.reshape((batch_size, 1))
            label = np.vstack((g_batch_labels, b_batch_labels))
            # APPLY SIGMOID
            label = 1 / (1 + np.exp(-1 * label))

        else:
            label = np.vstack((np.ones((batch_size, 1)),
                               -1 * np.ones((batch_size, 1))))

        loss = model.train_on_batch(features, label)

        pre_weights = extract_weights(names, model.get_weights(),
                                      CHOICE_CONFIG)

        for layer_num in range(len(batch_end_layers)):
            fn = batch_end_layers[layer_num]
            if callable(fn):
                fn()

        # post_weights = extract_weights(names, model.get_weights(),
        #                               CHOICE_CONFIG)

        batch_loss.append(loss)

    if TEST:
        if i == 100:
            loss = model.evaluate(features, label, verbose=0)
            break

    avg_loss = np.mean(batch_loss)
    if avg_loss < last_loss:
        last_loss = avg_loss
        best_iter_so_far = i
        last_loss_update = i

    else:
        if lr_reductions > early_stopping_factor:
            TRAINING_PROGRESS_FILE.write(
                'Ending training, last model saved at {}\n'.format(last_loss_update))
            break
        if i - last_loss_update > reduce_lr_after:
            last_loss_update = i
            curr_lr *= reduce_lr_factor

            TRAINING_PROGRESS_FILE.write(
                'Recompiling with reduced LR {} at epoch {}\n'.format(
                    curr_lr, last_loss_update))

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=curr_lr)
            model.compile(optimizer=optimizer,
                          loss='mean_absolute_error')

            lr_reductions += 1
            reduce_lr_after = 50  # change to 50 after first reduction

    if i % epoch_check == 0:
        loss = model.evaluate(features, label, verbose=0)
        # save_weights.append({i: {"pre": pre_weights, "post": post_weights}})
        # save_weights.append({i: pre_weights})

        TRAINING_PROGRESS_FILE.write(
            "Epoch {i}, loss {l}\n".format(i=i, l=loss))


train_time_end = time.time()

TRAINING_PROGRESS_FILE.write(
    'TRAINING TIME: {}\n'.format(train_time_end - train_time_start))
print('Training Time: {}'.format(train_time_end - train_time_start))

TRAINING_PROGRESS_FILE.write('WEIGHTS \n')
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()
for name, weight in zip(names, weights):
    TRAINING_PROGRESS_FILE.write(name + ' ' + str(weight))
    TRAINING_PROGRESS_FILE.write('\n')

save_weights.append({i: extract_weights(
    names, model.get_weights(), CHOICE_CONFIG,
    normalizer=feature_normalizers)})

# EVALUATE MODEL
TRAINING_PROGRESS_FILE.write('FINAL EVALUATION \n')
print('FINAL EVALUATION')

good_data_test = []
bad_data_test = []
good_data_test_un = []
bad_data_test_un = []
for i in range(num_features):
    gd = good_data[i][train_size:, :]
    bd = bad_data[i][train_size:, :]
    good_data_test_un.append(gd)
    bad_data_test_un.append(bd)
    gd = gd / feature_normalizers[feature_names[i]]
    bd = bd / feature_normalizers[feature_names[i]]
    # APPLY SIGMOID
    gd = 1 / (1 + np.exp(-1 * gd))
    bd = 1 / (1 + np.exp(-1 * bd))
    gd = gd.reshape(gd.shape[0], trace_length, 1)
    bd = bd.reshape(bd.shape[0], trace_length, 1)
    good_data_test.append(gd)
    bad_data_test.append(bd)

predictions = model.predict(good_data_test, verbose=0)
false_neg = 0
print(predictions[0:5])
for p in predictions:
    final_robs = p
    if final_robs < 0:
        false_neg += 1
good_preds = predictions

predictions = model.predict(bad_data_test, verbose=0)
bad_preds = predictions
false_pos = 0
print(predictions[0:5])
for p in predictions:
    final_robs = p
    if final_robs > 0:
        false_pos += 1

if CRUISE_CONTROL_CONT:
    g = good_lab[train_size:] / normalizer
    g = g.reshape((len(g), 1))
    b = bad_lab[train_size:] / normalizer
    b = b.reshape((len(b), 1))
    test_mae = model.evaluate(np.vstack((good_data_test[0], bad_data_test[0])),
                              np.vstack((g, b)), verbose=0)
    TRAINING_PROGRESS_FILE.write('MAE {}\n'.format(test_mae))

np.savez('{}/{}_{}{}_preds.npz'.format(DIR, EXP, DEPTH, QT),
         good_data=good_data_test, good_data_un=good_data_test_un, good_preds=good_preds,
         bad_data=bad_data_test, bad_data_un=bad_data_test_un, bad_preds=bad_preds)

TRAINING_PROGRESS_FILE.write('# False Negative {}\n'.format(false_neg))
TRAINING_PROGRESS_FILE.write('# False Positive {}\n'.format(false_pos))
TRAINING_PROGRESS_FILE.write('# Total Preds {}\n'.format((len(predictions) * 2)))
mcr = (false_pos + false_neg) / (len(predictions) * 2)
print('MCR {}'.format(mcr))
TRAINING_PROGRESS_FILE.write('MCR {}\n'.format(mcr))

TRAINING_PROGRESS_FILE.close()

with open(WEIGHTS_PROGRESS_FILE, 'w') as f:
    json.dump(save_weights, f)
