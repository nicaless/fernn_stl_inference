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


parser = argparse.ArgumentParser(description='ECG Test')
parser.add_argument('test_num', help='test number', type=int, default=1)
parser.add_argument('save_file', help='training progress save file name', default='training_progress')
parser.add_argument('--quantize', help='quantize weights', action='store_true')
parser.add_argument('--rast', help='quantize use xnor quant method', action='store_true')
args = parser.parse_args()
print(args)


TEST_NUM = args.test_num
# QUANTIZE = 'one-hot' if args.quantize else False
QUANTIZE = 'one-hot' if args.quantize else args.rast
DATA_INPUT_FOLDER = 'experiments/ecg/'
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
batch_size = 50  # per class
batches = 2
# batch_size = 25  # per class
# batches = 4
reduce_lr_factor = 50
early_stopping_factor = np.inf  # no early stopping
init_lr = 0.003


# READ DATA
good_file = '1_nsr_traces_len_10.csv'
good = np.loadtxt(DATA_INPUT_FOLDER + 'hr_traces_len_10/' + good_file,
                  delimiter=',')
good_dx = np.loadtxt(DATA_INPUT_FOLDER + 'hr_traces_len_10_diff/' + good_file,
                     delimiter=',')

bad_file = '4_afib_traces_len_10.csv'
bad = np.loadtxt(DATA_INPUT_FOLDER + 'hr_traces_len_10/' + bad_file,
                  delimiter=',')
bad_dx = np.loadtxt(DATA_INPUT_FOLDER + 'hr_traces_len_10_diff/' + bad_file,
                     delimiter=',')

# PARTITITION TRAIN AND TEST
train_good = good[0:100, :]
train_good_dx = good_dx[0:100, :]

train_bad = bad[0:100, :]
train_bad_dx = bad_dx[0:100, :]

# NORMALIZE
train_hr = np.vstack((train_good, train_bad))
normalizer = (np.max(train_hr) - np.min(train_hr))
train_good = train_good / normalizer
train_bad = train_bad / normalizer

train_hr_dx = np.vstack((train_good_dx, train_bad_dx))
normalizer_dx = (np.max(train_hr_dx) - np.min(train_hr_dx))
train_good_dx = train_good_dx / normalizer_dx
train_bad_dx = train_bad_dx / normalizer_dx


trace_length = train_good.shape[1]
input_shape = (trace_length, 1)

# BUILD NETWORK
'''
For ECG
1. choose (Alw, Ev) (hr)
2. choose (Alw, Ev) (hr_dx)
3. choose (Alw, Ev) (choose hr, hr_dx)
4. choose (Alw, Ev) (choose (hr AND hr_dx,  hr, OR hr_dx) )
5. choose (Alw, Ev) (choose hr, hr_dx)  AND choose (Alw, Ev) (choose hr, hr_dx)
6. choose (Alw, Ev) (choose hr, hr_dx) OR choose (Alw, Ev) (choose hr, hr_dx)
7. choose (Alw, Ev) (choose hr, hr_dx)  CHOOSE(AND_OR) choose (Alw, Ev) (choose hr, hr_dx)
'''

in1 = Input(input_shape)
in2 = Input(input_shape)

# 1. choose (Alw, Ev) (hr)
if TEST_NUM == 1:
    CHOICE_CONFIG = {'choice1': ['alw', 'ev']}
    atom_hr = RNN(TLAtom(1), return_sequences=True, name='hr')(in1)

    alw = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(atom_hr)
    ev = RNN(TLEv(trace_length), return_sequences=True, name='ev')(atom_hr)

    choice_input = concatenate([alw, ev])
    choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                            name='choice1')(choice_input)
    formula = RNN(TLPass(1), name='final_out')(choice)


# 2. choose (Alw, Ev) (hr_dx)
elif TEST_NUM == 2:
    CHOICE_CONFIG = {'choice1': ['alw', 'ev']}
    atom_hr_dx = RNN(TLAtom(1), return_sequences=True, name='hr_dx')(in2)

    alw = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(atom_hr_dx)
    ev = RNN(TLEv(trace_length), return_sequences=True, name='ev')(atom_hr_dx)

    choice_input = concatenate([alw, ev])
    choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                               name='choice1')(choice_input)
    formula = RNN(TLPass(1), name='final_out')(choice)

# 3. choose (Alw, Ev) (choose hr, hr_dx)
elif TEST_NUM == 3:
    CHOICE_CONFIG = {'choice1': ['alw', 'ev'],
                     'choice2': ['hr', 'hr_dx']}

    atom_hr = RNN(TLAtom(1), return_sequences=True, name='hr')(in1)
    atom_hr_dx = RNN(TLAtom(1), return_sequences=True, name='hr_dx')(in2)

    atom_choice_input = concatenate([atom_hr, atom_hr_dx])
    atom_choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                    name='choice2')(atom_choice_input)

    alw = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(atom_choice)
    ev = RNN(TLEv(trace_length), return_sequences=True, name='ev')(atom_choice)

    choice_input = concatenate([alw, ev])
    choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                               name='choice1')(choice_input)
    formula = RNN(TLPass(1), name='final_out')(choice)

# 4. choose (Alw, Ev) (choose (hr AND hr_dx,  hr, OR hr_dx) )
elif TEST_NUM == 4:
    CHOICE_CONFIG = {'choice1': ['alw', 'ev'],
                     'choice2': ['and', 'or']}

    atom_hr = RNN(TLAtom(1), return_sequences=True, name='hr')(in1)
    atom_hr_dx = RNN(TLAtom(1), return_sequences=True, name='hr_dx')(in2)

    and_input = concatenate([atom_hr, atom_hr_dx])
    and_output = RNN(TLAnd(1), return_sequences=True, name='and')(and_input)

    or_input = concatenate([atom_hr, atom_hr_dx])
    or_output = RNN(TLAnd(1), return_sequences=True, name='or')(or_input)

    and_or_choice_input = concatenate([and_output, or_output])
    and_or_choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                      name='choice2')(and_or_choice_input)

    alw = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(
        and_or_choice)
    ev = RNN(TLEv(trace_length), return_sequences=True, name='ev')(and_or_choice)

    choice_input = concatenate([alw, ev])
    choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                               name='choice1')(choice_input)
    formula = RNN(TLPass(1), name='final_out')(choice)

# 5. choose (Alw, Ev) (choose hr, hr_dx)  AND choose (Alw, Ev) (choose hr, hr_dx)
elif TEST_NUM == 5:
    CHOICE_CONFIG = {'choice1': ['alw1', 'ev1'],
                     'choice2': ['alw2', 'ev2'],
                     'choice3': ['hr1', 'dx1'],
                     'choice4': ['hr2', 'dx2'],
                     }
    # AND LH
    atom_hr1 = RNN(TLAtom(1), return_sequences=True, name='hr1')(in1)
    atom_hr_dx1 = RNN(TLAtom(1), return_sequences=True, name='hr_dx1')(in2)

    atom_choice_input1 = concatenate([atom_hr1, atom_hr_dx1])
    atom_choice1 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                    name='choice3')(atom_choice_input1)

    alw1 = RNN(TLAlw(trace_length), return_sequences=True, name='alw1')(
        atom_choice1)
    ev1 = RNN(TLEv(trace_length), return_sequences=True, name='ev1')(atom_choice1)

    choice_input1 = concatenate([alw1, ev1])
    choice1 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                               name='choice1')(choice_input1)

    # AND RH
    atom_hr2 = RNN(TLAtom(1), return_sequences=True, name='hr2')(in1)
    atom_hr_dx2 = RNN(TLAtom(1), return_sequences=True, name='hr_dx2')(in2)

    atom_choice_input2 = concatenate([atom_hr2, atom_hr_dx2])
    atom_choice2 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                    name='choice4')(atom_choice_input2)

    alw2 = RNN(TLAlw(trace_length), return_sequences=True, name='alw2')(
        atom_choice2)
    ev2 = RNN(TLEv(trace_length), return_sequences=True, name='ev2')(atom_choice1)

    choice_input2 = concatenate([alw2, ev2])
    choice2 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                               name='choice2')(choice_input1)

    # AND
    and_input = concatenate([choice1, choice2])
    and_output = RNN(TLAnd(1), return_sequences=True, name='and')(and_input)

    formula = RNN(TLPass(1), name='final_out')(and_output)

# 6. choose (Alw, Ev) (choose hr, hr_dx) OR choose (Alw, Ev) (choose hr, hr_dx)
elif TEST_NUM == 6:
    CHOICE_CONFIG = {'choice1': ['alw1', 'ev1'],
                     'choice2': ['alw2', 'ev2'],
                     'choice3': ['hr1', 'dx1'],
                     'choice4': ['hr2', 'dx2'],
                     }
    # OR LH
    atom_hr1 = RNN(TLAtom(1), return_sequences=True, name='hr1')(in1)
    atom_hr_dx1 = RNN(TLAtom(1), return_sequences=True, name='hr_dx1')(in2)

    atom_choice_input1 = concatenate([atom_hr1, atom_hr_dx1])
    atom_choice1 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                     name='choice3')(atom_choice_input1)

    alw1 = RNN(TLAlw(trace_length), return_sequences=True, name='alw1')(
        atom_choice1)
    ev1 = RNN(TLEv(trace_length), return_sequences=True, name='ev1')(
        atom_choice1)

    choice_input1 = concatenate([alw1, ev1])
    choice1 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                name='choice1')(choice_input1)

    # OR RH
    atom_hr2 = RNN(TLAtom(1), return_sequences=True, name='hr2')(in1)
    atom_hr_dx2 = RNN(TLAtom(1), return_sequences=True, name='hr_dx2')(in2)

    atom_choice_input2 = concatenate([atom_hr2, atom_hr_dx2])
    atom_choice2 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                     name='choice4')(atom_choice_input2)

    alw2 = RNN(TLAlw(trace_length), return_sequences=True, name='alw2')(
        atom_choice2)
    ev2 = RNN(TLEv(trace_length), return_sequences=True, name='ev2')(
        atom_choice1)

    choice_input2 = concatenate([alw2, ev2])
    choice2 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                name='choice2')(choice_input1)

    # OR
    or_input = concatenate([choice1, choice2])
    or_output = RNN(TLOR(1), return_sequences=True, name='or')(or_input)

    formula = RNN(TLPass(1), name='final_out')(or_output)

# 7. choose (Alw, Ev) (choose hr, hr_dx)  CHOOSE(AND_OR) choose (Alw, Ev) (choose hr, hr_dx)
elif TEST_NUM == 7:
    CHOICE_CONFIG = {'choice0': ['and', 'or'],
                     'choice1': ['alw1', 'ev1'],
                     'choice2': ['alw2', 'ev2'],
                     'choice3': ['hr1', 'dx1'],
                     'choice4': ['hr2', 'dx2'],
                     }
    # LH
    atom_hr1 = RNN(TLAtom(1), return_sequences=True, name='hr1')(in1)
    atom_hr_dx1 = RNN(TLAtom(1), return_sequences=True, name='hr_dx1')(in2)

    atom_choice_input1 = concatenate([atom_hr1, atom_hr_dx1])
    atom_choice1 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                     name='choice3')(atom_choice_input1)

    alw1 = RNN(TLAlw(trace_length), return_sequences=True, name='alw1')(
        atom_choice1)
    ev1 = RNN(TLEv(trace_length), return_sequences=True, name='ev1')(
        atom_choice1)

    choice_input1 = concatenate([alw1, ev1])
    choice1 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                name='choice1')(choice_input1)

    # RH
    atom_hr2 = RNN(TLAtom(1), return_sequences=True, name='hr2')(in1)
    atom_hr_dx2 = RNN(TLAtom(1), return_sequences=True, name='hr_dx2')(in2)

    atom_choice_input2 = concatenate([atom_hr2, atom_hr_dx2])
    atom_choice2 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                     name='choice4')(atom_choice_input2)

    alw2 = RNN(TLAlw(trace_length), return_sequences=True, name='alw2')(
        atom_choice2)
    ev2 = RNN(TLEv(trace_length), return_sequences=True, name='ev2')(
        atom_choice1)

    choice_input2 = concatenate([alw2, ev2])
    choice2 = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                name='choice2')(choice_input1)

    # AND
    and_input = concatenate([choice1, choice2])
    and_output = RNN(TLAnd(1), return_sequences=True, name='and')(and_input)

    # OR
    or_input = concatenate([choice1, choice2])
    or_output = RNN(TLOR(1), return_sequences=True, name='or')(or_input)

    # AND/OR CHOICE
    and_or_choice_input = concatenate([and_output, or_output])
    and_or_choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                      name='choice0')(and_or_choice_input)

    formula = RNN(TLPass(1), name='final_out')(and_or_choice)
else:
    print('INVALID TEST, EXITING')
    sys.exit()


# COMPILE MODEL
# out = formula
out = tf.keras.layers.Activation(tf.keras.activations.tanh)(formula)
model = Model(inputs=[in1, in2], outputs=[out])

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
    idx1 = np.random.choice(100, size=100, replace=False)
    idx2 = np.random.choice(100, size=100, replace=False)

    good_data = train_good[idx1, :]
    good_data_dx = train_good_dx[idx1, :]

    bad_data = train_bad[idx1, :]
    bad_data_dx = train_bad_dx[idx1, :]

    batch_loss = []
    for b in range(batches):
        batch_start = b * batch_size
        batch_end = batch_start + batch_size
        good_data_batch = good_data[batch_start:batch_end, :]
        good_data_batch_dx = good_data_dx[batch_start:batch_end, :]
        bad_data_batch = bad_data[batch_start:batch_end, :]
        bad_data_batch_dx = bad_data_dx[batch_start:batch_end, :]

        good_data_batch = good_data_batch.reshape((batch_size, trace_length, 1))
        good_data_batch_dx = good_data_batch_dx.reshape((batch_size, trace_length, 1))
        bad_data_batch = bad_data_batch.reshape((batch_size, trace_length, 1))
        bad_data_batch_dx = bad_data_batch_dx.reshape((batch_size, trace_length, 1))

        X = np.vstack((good_data_batch, bad_data_batch))
        DX = np.vstack((good_data_batch_dx, bad_data_batch_dx))
        lab = np.vstack((np.ones((batch_size, 1)),
                         -1 * np.ones((batch_size, 1))))

        loss = model.train_on_batch([X, DX], lab)
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
        loss = model.evaluate([X, DX], lab, verbose=0)

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
                               normalizer={'hr': normalizer,
                                           'hr_dx': normalizer_dx,
                                           'hr1': normalizer,
                                           'hr_dx1': normalizer_dx,
                                           'hr2': normalizer,
                                           'hr_dx2': normalizer_dx
                                           })
print(save_weights)
with open(WEIGHTS_FILE_NAME, 'w') as f:
    json.dump(save_weights, f)

TRAINING_PROGRESS_FILE.write('Normalizer: {}\n'.format(normalizer))
print('Normalizer: {}'.format(normalizer))

TRAINING_PROGRESS_FILE.write('Normalizer DX: {}\n'.format(normalizer_dx))
print('Normalizer DX: {}'.format(normalizer_dx))


# EVALUATE MODEL
test_good = good[100:, :]
test_bad = bad[100:, :]
test_good_dx = good_dx[100:, :]
test_bad_dx = bad_dx[100:, :]

test_good = test_good / normalizer
test_bad = test_bad / normalizer

test_good_dx = test_good_dx / normalizer_dx
test_bad_dx = test_bad_dx / normalizer_dx


test_good = test_good.reshape(test_good.shape[0], trace_length, 1)
test_good_dx = test_good.reshape(test_good_dx.shape[0], trace_length, 1)

test_bad = test_bad.reshape(test_bad.shape[0], trace_length, 1)
test_bad_dx = test_bad_dx.reshape(test_bad_dx.shape[0], trace_length, 1)

# MCR calculation
# (false positive + false negative / total)
predictions = model.predict([test_good, test_good_dx], verbose=0)
print(predictions.shape)
print(predictions[0:5])
false_neg = 0
for p in predictions:
    final_robs = p
    if final_robs < 0:
        false_neg += 1

predictions = model.predict([test_bad, test_bad_dx], verbose=0)
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
