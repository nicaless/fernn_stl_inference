import sys

sys.path.append('/Users/nicole/OSU/GAN_TL')

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, RNN, Dense
from tensorflow.keras.models import Model
import time
from TLOps import TLEv, TLAlw, TLAtom
from TL_learning_utils import TLWeightTransform, extract_weights


DATA_INPUT_FOLDER = 'experiments/cruise_control_train/'
np.random.seed(42)
tf.random.set_seed(42)

# READ DATA
# bad = np.loadtxt(DATA_INPUT_FOLDER + 'Traces_anomaly_BIG.csv', delimiter=',')
# good = np.loadtxt(DATA_INPUT_FOLDER + 'Traces_normal_BIG.csv', delimiter=',')

bad = np.loadtxt(DATA_INPUT_FOLDER + 'Traces_anomaly_BIG2.csv', delimiter=',')
good = np.loadtxt(DATA_INPUT_FOLDER + 'Traces_normal_BIG2.csv', delimiter=',')

# PARTITITION TRAIN AND TEST
train_good = good[0:800, :]
train_bad = bad[0:800, :]

# NORMALIZE DATA
train_X = np.vstack((train_good, train_bad))
normalizer = (np.max(train_X) - np.min(train_X))
train_good = train_good / normalizer
train_bad = train_bad / normalizer

trace_length = good.shape[1]
input_shape = (trace_length, 1)

# BUILD NETWORK
in1 = Input(input_shape)


# EXP1 BIG ev(a1 | a2 | a3) | alw(a4 | a5 | a6)
# saved_weights = json.load(
#     open(DATA_INPUT_FOLDER + 'saved_weights_exp1_1.json', 'r'))
# atom1 = RNN(TLAtom(1,
#                    w=saved_weights['ev_atom1']['weight'],
#                    b=saved_weights['ev_atom1']['bias']),
#             return_sequences=True, name='atom1')(in1)
# out = RNN(TLEv(trace_length), return_sequences=True, name='ev')(atom1)

# EXP1 BIG2 ev(a1 | a2 | a3) | alw(a4 | a5 | a6)
# saved_weights = json.load(
#     # open(DATA_INPUT_FOLDER + 'saved_weights_exp1_2.json', 'r'))
#     open(DATA_INPUT_FOLDER + '/saved_models/best_model_weights.json', 'r'))
# atom1 = RNN(TLAtom(1,
#                    w=saved_weights['alw_atom4']['weight'],
#                    b=saved_weights['alw_atom4']['bias']),
#             return_sequences=True, name='atom1')(in1)
# out = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(atom1)


# EXP2 BIG ev(a1)
# saved_weights = json.load(
#     open(DATA_INPUT_FOLDER + 'saved_weights_exp2_1.json', 'r'))
# atom1 = RNN(TLAtom(1,
#                    w=saved_weights['ev_atom1']['weight'],
#                    b=saved_weights['ev_atom1']['bias']),
#             return_sequences=True, name='atom1')(in1)
# out = RNN(TLEv(trace_length), return_sequences=True, name='ev')(atom1)

# EXP2 BIG2 alw(a1)
saved_weights = json.load(
    # open(DATA_INPUT_FOLDER + 'saved_weights_exp2_2.json', 'r'))
    open(DATA_INPUT_FOLDER + '/saved_models/best_model_weights.json', 'r'))
atom1 = RNN(TLAtom(1,
                   w=saved_weights['alw_atom4']['weight'],
                   b=saved_weights['alw_atom4']['bias']),
            return_sequences=True, name='atom1')(in1)
out = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(atom1)


# EXP3 BIG alw(a1) | ev(a2)
# saved_weights = json.load(
#     open(DATA_INPUT_FOLDER + 'saved_weights_exp3_1.json', 'r'))
# atom1 = RNN(TLAtom(1,
#                    w=saved_weights['ev_atom1']['weight'],
#                    b=saved_weights['ev_atom1']['bias']),
#             return_sequences=True, name='atom1')(in1)
# out = RNN(TLEv(trace_length), return_sequences=True, name='ev')(atom1)

# EXP3 BIG 2 alw(a1) | ev(a2)
# saved_weights = json.load(
#     open(DATA_INPUT_FOLDER + 'saved_weights_exp3_2.json', 'r'))
#     # open(DATA_INPUT_FOLDER + '/saved_models/best_model_weights.json', 'r'))
# atom1 = RNN(TLAtom(1,
#                    w=saved_weights['alw_atom4']['weight'],
#                    b=saved_weights['alw_atom4']['bias']),
#             return_sequences=True, name='atom1')(in1)
# out = RNN(TLAlw(trace_length), return_sequences=True, name='alw')(atom1)


# EVALUATE MODEL
model = Model(inputs=[in1], outputs=[out])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

test_good = good[800:, :]
test_bad = bad[800:, :]
test_good = test_good / normalizer
test_bad = test_bad / normalizer

test_good_y = np.ones((200, 1))
test_bad_y = np.zeros((200, 1))

# MCR calculation
# (false positive + false negative / total)
predictions = model.predict(test_good, verbose=0)
false_neg = 0
for p in predictions:
    final_robs = p[-1][0]
    if final_robs < 0:
        false_neg += 1

predictions = model.predict(test_bad, verbose=0)
false_pos = 0
for p in predictions:
    final_robs = p[-1][0]
    if final_robs > 0:
        false_pos += 1

print(false_neg)
print(false_pos)

print(false_neg + false_pos)
print('MCR (Quantized)')
print((false_pos + false_neg) / 400)


