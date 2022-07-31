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


parser = argparse.ArgumentParser(description='Lyft Test')
parser.add_argument('test_num', help='test number', type=int, default=1)
parser.add_argument('save_file', help='training progress save file name', default='training_progress')
parser.add_argument('--quantize', help='quantize weights', action='store_true')
parser.add_argument('--rast', help='quantize use xnor quant method', action='store_true')
args = parser.parse_args()
print(args)


TEST_NUM = args.test_num
# QUANTIZE = 'one-hot' if args.quantize else False
QUANTIZE = 'one-hot' if args.quantize else args.rast
DATA_INPUT_FOLDER = 'experiments/lyft_data/'
TRAINING_PROGRESS_FILE = open(DATA_INPUT_FOLDER + args.save_file + '.txt', 'w')
WEIGHTS_FILE_NAME = DATA_INPUT_FOLDER + args.save_file + '_weights.json'
INTERMEDIATE_WEIGHTS_FILE_NAME = DATA_INPUT_FOLDER + '/saved_models/' +\
                                 args.save_file + '_best_weights.json'

# NETWORK PARAMS
np.random.seed(42)
tf.random.set_seed(42)
# epochs = 1000
epochs = 5000
# epoch_check = int(epochs / 10)
epoch_check = int(epochs / 100)
batch_size = 1000  # per class
batches = 8
# reduce_lr_factor = 50
reduce_lr_factor = 100
# early_stopping_factor = 5
early_stopping_factor = np.inf
init_lr = 0.003
# init_lr = 0.001
# init_lr = 0.0005


# READ DATA
bad_x = np.loadtxt(DATA_INPUT_FOLDER + 'bad_traces_x.csv', delimiter=',')
good_x = np.loadtxt(DATA_INPUT_FOLDER + 'lyft_good_traces_x.csv', delimiter=',')

bad_y = np.loadtxt(DATA_INPUT_FOLDER + 'bad_traces_y.csv', delimiter=',')
good_y = np.loadtxt(DATA_INPUT_FOLDER + 'lyft_good_traces_y.csv', delimiter=',')

# PARTITITION TRAIN AND TEST
train_good_x = good_x[0:8000, :]
train_bad_x = bad_x[0:8000, :]

train_good_y = good_y[0:8000, :]
train_bad_y = bad_y[0:8000, :]

# NORMALIZE DATA
train_X_x = np.vstack((train_good_x, train_bad_x))
normalizer_x = (np.max(train_X_x) - np.min(train_X_x))
train_good_x = train_good_x / normalizer_x
train_bad_x = train_bad_x / normalizer_x

train_X_y = np.vstack((train_good_y, train_bad_y))
normalizer_y = (np.max(train_X_y) - np.min(train_X_y))
train_good_y = train_good_y / normalizer_y
train_bad_y = train_bad_y / normalizer_y

trace_length = good_x.shape[1]
input_shape = (trace_length, 1)


# BUILD NETWORK
'''
For Lyft Test
1. choose (Alw, Ev) (a1.x)
2. choose (Alw, Ev) (choose (a1.x, a1.y)
3. choose (Alw, Ev) (choose (a1.x AND a1.y,  a2.x OR a2.y) )
4. choose (Alw, Ev) (choose (choose (a1.x, a1.y) AND choose (a2.x, a2.y)  , 
                             choose (a1.x, a1.y) OR choose (a1.x, a1.y)) )
'''

in1 = Input(input_shape)
in2 = Input(input_shape)

# 1. choose (Alw, Ev) (a1.x)
if TEST_NUM == 1:
    CHOICE_CONFIG = {'choice1': ['alw', 'ev']}
    atom_x = RNN(TLAtom(1), return_sequences=True, name='x')(in1)

    alw = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(atom_x)
    ev = RNN(TLEv(trace_length), return_sequences=True, name='ev')(atom_x)

    choice_input = concatenate([alw, ev])
    choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                            name='choice1')(choice_input)
    formula = RNN(TLPass(1), name='final_out')(choice)


# 2. choose (Alw, Ev) (choose (a1.x, a1.y)
elif TEST_NUM == 2:
    CHOICE_CONFIG = {'choice1': ['alw', 'ev'],
                     'choice2': ['x', 'y']}
    atom_x = RNN(TLAtom(1), return_sequences=True, name='x')(in1)
    atom_y = RNN(TLAtom(1), return_sequences=True, name='y')(in2)

    atom_choice_input = concatenate([atom_x, atom_y])

    atom_choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                    name='choice2')(atom_choice_input)

    alw = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(atom_choice)
    ev = RNN(TLEv(trace_length), return_sequences=True, name='ev')(atom_choice)

    choice_input = concatenate([alw, ev])
    choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                               name='choice1')(choice_input)
    formula = RNN(TLPass(1), name='final_out')(choice)

# 3. choose (Alw, Ev) (choose (a1.x AND a1.y,  a2.x OR a2.y) )
elif TEST_NUM == 3:
    CHOICE_CONFIG = {'choice1': ['alw', 'ev'],
                     'choice2': ['and', 'or']}

    atom1_x = RNN(TLAtom(1), return_sequences=True, name='x1')(in1)
    atom1_y = RNN(TLAtom(1), return_sequences=True, name='y1')(in2)
    and_input = concatenate([atom1_x, atom1_y])
    and_output = RNN(TLAnd(1), return_sequences=True, name='and')(and_input)

    atom2_x = RNN(TLAtom(1), return_sequences=True, name='x2')(in1)
    atom2_y = RNN(TLAtom(1), return_sequences=True, name='y2')(in2)
    or_input = concatenate([atom2_x, atom2_y])
    or_output = RNN(TLOR(1), return_sequences=True, name='or')(or_input)

    and_or_choice_input = concatenate([and_output, or_output])
    and_or_choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                      name='choice2')(and_or_choice_input)

    alw = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(and_or_choice)
    ev = RNN(TLEv(trace_length), return_sequences=True, name='ev')(and_or_choice)

    choice_input = concatenate([alw, ev])
    choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                name='choice1')(choice_input)
    formula = RNN(TLPass(1), name='final_out')(choice)

# 4. choose (Alw, Ev) (choose (choose (a1.x, a1.y) AND choose (a2.x, a2.y)  ,
#                              choose (a1.x, a1.y) OR choose (a1.x, a1.y)) )
elif TEST_NUM == 4:
    CHOICE_CONFIG = {'choice1': ['alw', 'ev'],
                     'choice2': ['and', 'or'],

                     'choice3_and_lh': ['x1', 'y1'],
                     'choice3_and_rh': ['x2', 'y2'],

                     'choice3_or_lh': ['x3', 'y3'],
                     'choice3_or_rh': ['x4', 'y4'],
                     }

    # AND LH
    atom1_x = RNN(TLAtom(1), return_sequences=True, name='x1')(in1)
    atom1_y = RNN(TLAtom(1), return_sequences=True, name='y1')(in2)

    and_lh_choice_input = concatenate([atom1_x, atom1_y])
    and_lh_choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                      name='choice3_and_lh')(and_lh_choice_input)

    # AND RH
    atom2_x = RNN(TLAtom(1), return_sequences=True, name='x2')(in1)
    atom2_y = RNN(TLAtom(1), return_sequences=True, name='y2')(in2)

    and_rh_choice_input = concatenate([atom2_x, atom2_y])
    and_rh_choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                      name='choice3_and_rh')(and_rh_choice_input)

    # OR LH
    atom3_x = RNN(TLAtom(1), return_sequences=True, name='x3')(in1)
    atom3_y = RNN(TLAtom(1), return_sequences=True, name='y3')(in2)

    or_lh_choice_input = concatenate([atom3_x, atom3_y])
    or_lh_choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                      name='choice3_or_lh')(or_lh_choice_input)

    # OR RH
    atom4_x = RNN(TLAtom(1), return_sequences=True, name='x4')(in1)
    atom4_y = RNN(TLAtom(1), return_sequences=True, name='y4')(in2)

    or_rh_choice_input = concatenate([atom4_x, atom4_y])
    or_rh_choice = TLWeightTransform(1, transform='sum', quantize=QUANTIZE,
                                      name='choice3_or_rh')(or_rh_choice_input)

    # AND
    and_input = concatenate([and_lh_choice, and_rh_choice])
    and_output = RNN(TLAnd(1), return_sequences=True, name='and')(and_input)

    # OR
    or_input = concatenate([or_lh_choice, or_rh_choice])
    or_output = RNN(TLOR(1), return_sequences=True, name='or')(or_input)

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
    idx1 = np.random.choice(8000, size=8000, replace=False)
    idx2 = np.random.choice(8000, size=8000, replace=False)

    good_data_x = train_good_x[idx1, :]
    good_data_y = train_good_x[idx1, :]

    bad_data_x = train_bad_x[idx2, :]
    bad_data_y = train_bad_y[idx2, :]

    batch_loss = []
    for b in range(batches):
        batch_start = b * batch_size
        batch_end = batch_start + batch_size
        good_data_batch_x = good_data_x[batch_start:batch_end, :]
        good_data_batch_y = good_data_y[batch_start:batch_end, :]
        bad_data_batch_x = bad_data_x[batch_start:batch_end, :]
        bad_data_batch_y = bad_data_y[batch_start:batch_end, :]

        good_data_batch_x = good_data_batch_x.reshape((batch_size, trace_length, 1))
        good_data_batch_y = good_data_batch_y.reshape((batch_size, trace_length, 1))
        bad_data_batch_x = bad_data_batch_x.reshape((batch_size, trace_length, 1))
        bad_data_batch_y = bad_data_batch_y.reshape((batch_size, trace_length, 1))

        X = np.vstack((good_data_batch_x, bad_data_batch_x))
        Y = np.vstack((good_data_batch_y, bad_data_batch_y))
        lab = np.vstack((np.ones((batch_size, 1)),
                         -1 * np.ones((batch_size, 1))))

        loss = model.train_on_batch([X, Y], lab)
        for layer in model.layers:
            fn = getattr(layer, 'on_batch_end', None)
            if callable(fn):
                fn()
        batch_loss.append(loss)

    # if i == 5:
    #     predictions = model.predict([X, Y], verbose=0)
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
            curr_lr *= 0.98

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
        loss = model.evaluate([X, Y], lab, verbose=0)

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
                               normalizer={'x': normalizer_x,
                                           'y': normalizer_y,
                                           'x1': normalizer_x,
                                           'y1': normalizer_y,
                                           'x2': normalizer_x,
                                           'y2': normalizer_y,
                                           'x3': normalizer_x,
                                           'y3': normalizer_y,
                                           'x4': normalizer_x,
                                           'y4': normalizer_y})
print(save_weights)
with open(WEIGHTS_FILE_NAME, 'w') as f:
    json.dump(save_weights, f)

TRAINING_PROGRESS_FILE.write('Normalizer: {}\n'.format(normalizer_x))
print('Normalizer X: {}'.format(normalizer_x))

TRAINING_PROGRESS_FILE.write('Normalizer: {}\n'.format(normalizer_y))
print('Normalizer Y: {}'.format(normalizer_y))

# loss = model.evaluate([X, Y], lab, verbose=0)
# TRAINING_PROGRESS_FILE.write("Test loss {a} \n".format(a=loss))
# print("Test Loss {a}".format(a=loss))

# EVALUATE MODEL
test_good_x = good_x[8000:, :]
test_bad_x = bad_x[8000:, :]
test_good_y = good_y[8000:, :]
test_bad_y = bad_y[8000:, :]

test_good_x = test_good_x / normalizer_x
test_bad_x = test_bad_x / normalizer_x

test_good_y = test_good_y / normalizer_y
test_bad_y = test_bad_y / normalizer_y

test_good_x = test_good_x.reshape(test_good_x.shape[0],
                                  trace_length, 1)
test_good_y = test_good_y.reshape(test_good_x.shape[0],
                                  trace_length, 1)

# MCR calculation
# (false positive + false negative / total)
predictions = model.predict([test_good_x, test_good_y], verbose=0)
print(predictions.shape)
print(predictions[0:5])
false_neg = 0
for p in predictions:
    final_robs = p
    if final_robs < 0:
        false_neg += 1

predictions = model.predict([test_bad_x, test_bad_y], verbose=0)
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
mcr = (false_pos + false_neg) / (test_good_x.shape[0] * 2)
print('MCR (Quantized)')
print(mcr)

TRAINING_PROGRESS_FILE.write('# MCR {}\n'.format(mcr))


TRAINING_PROGRESS_FILE.close()


