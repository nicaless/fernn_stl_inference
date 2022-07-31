import sys

# sys.path.append('/Users/nicole/OSU/GAN_TL')
sys.path.append('/home/nicaless/repos/gan_tl')

import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, RNN
from tensorflow.keras.models import Model
import time
from TLOps import TLEv, TLAlw, TLAtom, TLAnd, TLOR
from TL_learning_utils import TLWeightTransform, extract_weights, \
    TLPass, pos_robustness


parser = argparse.ArgumentParser(description='Hapt Test')
parser.add_argument('test_num', help='test number', type=int, default=1)
parser.add_argument('save_file', help='training progress save file name', default='training_progress')
parser.add_argument('--quantize', help='quantize weights', action='store_true')
parser.add_argument('--rast', help='quantize use xnor quant method', action='store_true')
args = parser.parse_args()
print(args)


TEST_NUM = args.test_num
# QUANTIZE = 'one-hot' if args.quantize else False
QUANTIZE = 'one-hot' if args.quantize else args.rast
DATA_INPUT_FOLDER = 'experiments/hapt_data/'
TRAINING_PROGRESS_FILE = open(DATA_INPUT_FOLDER + args.save_file + '.txt', 'w')
WEIGHTS_FILE_NAME = DATA_INPUT_FOLDER + args.save_file + '_weights.json'
INTERMEDIATE_WEIGHTS_FILE_NAME = DATA_INPUT_FOLDER + '/saved_models/' +\
                                 args.save_file + '_best_weights.json'

# NETWORK PARAMS
np.random.seed(42)
tf.random.set_seed(42)
epochs = 1000
# epoch_check = int(epochs / 10)
epoch_check = int(epochs / 100)
batch_size = 40  # per class
batches = 2
# batch_size = 25  # per class
# batches = 4
reduce_lr_factor = 50
early_stopping_factor = np.inf  # no early stopping
init_lr = 0.003


# READ DATA
good = np.loadtxt(DATA_INPUT_FOLDER + 'good_hapt_data.csv', delimiter=' ')
bad = np.loadtxt(DATA_INPUT_FOLDER + 'bad_hapt_data.csv', delimiter=' ')

# PARTITITION TRAIN AND TEST
train_good = good[0:80, :]
train_bad = bad[0:80, :]

# NORMALIZE
train_X = np.vstack((train_good, train_bad))
normalizer = (np.max(train_X) - np.min(train_X))
train_good = train_good / normalizer
train_bad = train_bad / normalizer


trace_length = train_good.shape[1]
input_shape = (trace_length, 1)

# BUILD NETWORK
'''
For HAPT
1. Alw (x)
2. choose (Alw, Ev) (x)
'''

in1 = Input(input_shape)

# 1. Alw (x)
if TEST_NUM == 1:
    CHOICE_CONFIG = {}
    atom1 = RNN(TLAtom(1), return_sequences=True, name='atom1')(in1)

    alw_output = RNN(TLAlw(trace_length),
                     return_sequences=True, name='alw')(atom1)
    formula = RNN(TLPass(1), name='final_out')(alw_output)

# 2. choose (Alw, Ev) (x)
elif TEST_NUM == 2:
    CHOICE_CONFIG = {'choice1': ['alw', 'ev']}
    atom1 = RNN(TLAtom(1), return_sequences=True, name='atom1')(in1)

    alw = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(atom1)
    ev = RNN(TLEv(trace_length), return_sequences=True, name='ev')(atom1)

    choice_input = concatenate([alw, ev])
    choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                               name='choice1')(choice_input)
    formula = RNN(TLPass(1), name='final_out')(choice)

else:
    print('INVALID TEST, EXITING')
    sys.exit()


# COMPILE MODEL
# out = formula
out = tf.keras.layers.Activation(tf.keras.activations.tanh)(formula)
model = Model(inputs=[in1], outputs=[out])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=init_lr)
model.compile(optimizer=optimizer,
              # loss='binary_crossentropy',
              # loss=pos_robustness
              loss='mean_absolute_error'
              )
names = [weight.name for layer in model.layers for weight in layer.weights]
original_weights = model.get_weights()

# TRAIN MODEL
last_loss = np.inf
last_loss_update = -1
best_iter_so_far = -1
train_time_start = time.time()
curr_lr = init_lr
lr_reductions = 0
for i in range(epochs):
    idx1 = np.random.choice(80, size=80, replace=False)
    idx2 = np.random.choice(80, size=80, replace=False)

    good_data = train_good[idx1, :]

    bad_data = train_bad[idx1, :]

    batch_loss = []
    for b in range(batches):
        batch_start = b * batch_size
        batch_end = batch_start + batch_size

        good_data_batch = good_data[batch_start:batch_end, :]
        bad_data_batch = bad_data[batch_start:batch_end, :]

        good_data_batch = good_data_batch.reshape((batch_size, trace_length, 1))
        bad_data_batch = bad_data_batch.reshape((batch_size, trace_length, 1))

        X = np.vstack((good_data_batch, bad_data_batch))
        lab = np.vstack((np.ones((batch_size, 1)),
                         -1 * np.ones((batch_size, 1))))

        loss = model.train_on_batch([X], lab)
        for layer in model.layers:
            fn = getattr(layer, 'on_batch_end', None)
            if callable(fn):
                fn()
        batch_loss.append(loss)

    # if i == 5:
    #     predictions = model.predict([X, DX], verbose=0)
    #     print(pos_robustness(lab, predictions))
    #     break

    avg_loss = np.mean(batch_loss)
    if avg_loss < last_loss:
        last_loss = avg_loss
        best_iter_so_far = i
        last_loss_update = i

        save_weights = extract_weights(names, model.get_weights(),
                                       CHOICE_CONFIG)
        with open(INTERMEDIATE_WEIGHTS_FILE_NAME, 'w') as f:
            json.dump(save_weights, f)
    else:
        if lr_reductions > early_stopping_factor:
            print('Ending training, last model saved at {}'.format(
                last_loss_update))
            TRAINING_PROGRESS_FILE.write(
                'Ending training, last model saved at {}\n'.format(last_loss_update))
            break
        if i - last_loss_update > reduce_lr_factor:
            last_loss_update = i
            curr_lr *= 0.9

            print(
                'Recompiling with new LR {} at epoch {}'.format(
                    curr_lr, last_loss_update))
            TRAINING_PROGRESS_FILE.write(
                'Recompiling with reduced LR {} at epoch {}\n'.format(
                    curr_lr, last_loss_update))

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=curr_lr)
            model.compile(optimizer=optimizer,
                          loss='mean_absolute_error')

            lr_reductions += 1

    if i % epoch_check == 0:
        loss = model.evaluate([X], lab, verbose=0)

        print("Epoch {i}, loss {l}".format(i=i, l=loss))
        TRAINING_PROGRESS_FILE.write(
            "Epoch {i}, loss {l}\n".format(i=i, l=loss))

train_time_end = time.time()

TRAINING_PROGRESS_FILE.write(
    'Training Time: {}\n'.format(train_time_end - train_time_start))
print('Training Time: {}'.format(train_time_end - train_time_start))

TRAINING_PROGRESS_FILE.write('Best Iteration: {}\n'.format(best_iter_so_far))
print('Best Iteration: {}'.format(best_iter_so_far))

weights = model.get_weights()
TRAINING_PROGRESS_FILE.write('Final Weights: \n')
TRAINING_PROGRESS_FILE.write(str(weights))
TRAINING_PROGRESS_FILE.write('\n')
print(weights)

save_weights = extract_weights(names, weights, CHOICE_CONFIG,
                               normalizer={'atom1': normalizer})
print(save_weights)
with open(WEIGHTS_FILE_NAME, 'w') as f:
    json.dump(save_weights, f)

TRAINING_PROGRESS_FILE.write('Normalizer: {}\n'.format(normalizer))
print('Normalizer: {}'.format(normalizer))


# EVALUATE MODEL
test_good = good[80:, :]
test_bad = bad[80:, :]

test_good = test_good / normalizer
test_bad = test_bad / normalizer

test_good = test_good.reshape(test_good.shape[0], trace_length, 1)

test_bad = test_bad.reshape(test_bad.shape[0], trace_length, 1)

# MCR calculation
# (false positive + false negative / total)
predictions = model.predict([test_good], verbose=0)
print(predictions.shape)
print(predictions[0:5])
false_neg = 0
for p in predictions:
    final_robs = p
    if final_robs < 0:
        false_neg += 1

predictions = model.predict([test_bad], verbose=0)
print(predictions.shape)
print(predictions[0:5])
false_pos = 0
for p in predictions:
    final_robs = p
    if final_robs > 0:
        false_pos += 1

TRAINING_PROGRESS_FILE.write('# False Negative {}\n'.format(false_neg))
TRAINING_PROGRESS_FILE.write('# False Positive {}\n'.format(false_pos))
print(false_neg)
print(false_pos)

print(false_neg + false_pos)
mcr = (false_pos + false_neg) / (test_good.shape[0] + test_bad.shape[0])
print('MCR (Quantized)')
print(mcr)

TRAINING_PROGRESS_FILE.write('# MCR {}\n'.format(mcr))


TRAINING_PROGRESS_FILE.close()
