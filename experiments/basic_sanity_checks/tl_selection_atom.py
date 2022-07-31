import sys

sys.path.append('/Users/nicole/OSU/GAN_TL')

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, RNN
from tensorflow.keras.models import Model
from TLOps import TLEv, TLAtom
from TL_learning_utils import TLWeightTransform, extract_weights


DATA_INPUT_FOLDER = 'experiments/basic_sanity_checks/'
np.random.seed(42)
tf.random.set_seed(42)

# READ DATA
data1 = np.loadtxt(DATA_INPUT_FOLDER + 'data1_trace.csv', delimiter=',')
data2 = np.loadtxt(DATA_INPUT_FOLDER + 'data2_trace.csv', delimiter=',')
data3 = np.loadtxt(DATA_INPUT_FOLDER + 'data3_trace.csv', delimiter=',')

trace_length = 5
input_shape = (trace_length, 1)

CHOICE_CONFIG = {'choice_atoms': ['atom1', 'atom2', 'atom3']}

# MAKE CHOICE BETWEEN ATOMS THAT FEED INTO AN EVENTUALLY
in1 = Input(input_shape)
in2 = Input(input_shape)
in3 = Input(input_shape)
phi = RNN(TLAtom(1, w=1, b=0), return_sequences=True)(in1)
psi = RNN(TLAtom(1, w=1, b=0), return_sequences=True)(in2)
chi = RNN(TLAtom(1, w=1, b=0), return_sequences=True)(in3)

ev_input = concatenate([phi, psi, chi])
ev_weighted_input = \
    TLWeightTransform(1, w=[0, 0, 1],
                      transform='sum', name='choice_atoms')(ev_input)
ev_output = RNN(TLEv(1), return_sequences=True, name='and')(ev_weighted_input)

model = Model(inputs=[in1, in2, in3], outputs=[ev_output])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

predictions = model.predict([data1, data2, data3], verbose=0)
predictions_save = np.reshape(predictions, (100, 5))

print(data1[0, ])
print(data2[0, ])
print(data3[0, ])
print(predictions_save[0, ])

np.savetxt(DATA_INPUT_FOLDER + 'rnn_predictions_select_atoms.csv',
           predictions_save, delimiter=',', fmt='%.5e')


# MAKE CHOICE TRAINABLE
in1 = Input(input_shape)
in2 = Input(input_shape)
in3 = Input(input_shape)
phi = RNN(TLAtom(1), return_sequences=True, name='atom1')(in1)
psi = RNN(TLAtom(1), return_sequences=True, name='atom2')(in2)
chi = RNN(TLAtom(1), return_sequences=True, name='atom3')(in3)

ev_input = concatenate([phi, psi, chi])
ev_weighted_input = \
    TLWeightTransform(1, quantize='one-hot',
                      transform='sum', name='choice_atoms')(ev_input)
ev_output = RNN(TLEv(1), return_sequences=True, name='and')(ev_weighted_input)

trainable_model = Model(inputs=[in1, in2, in3], outputs=[ev_output])
trainable_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

names = [weight.name for layer in trainable_model.layers for weight in layer.weights]
original_weights = trainable_model.get_weights()

print('Original Weights')
for name, weight in zip(names, original_weights):
    print(name, weight)

print(extract_weights(names, original_weights, CHOICE_CONFIG))

epochs = 100
for i in range(epochs):
    # prepare good samples
    good_data = np.random.uniform(low=0, high=5, size=(100, trace_length))

    # prepare bad examples
    bad_data = np.random.uniform(low=-5, high=0, size=(100, trace_length))

    # reshape (samples, timesteps, features)
    good_data = good_data.reshape((100, trace_length, 1))
    bad_data = bad_data.reshape((100, trace_length, 1))

    X = np.vstack((good_data, bad_data))
    y = np.vstack((np.ones((100, 1)), np.zeros((100, 1))))

    trainable_model.train_on_batch([X, X, X], y)

    if i % 20 == 0:
        loss, acc = trainable_model.evaluate([X, X, X], y, verbose=0)
        print("Epoch {i}, acc {a}  loss {l}".format(i=i, a=acc, l=loss))

weights = trainable_model.get_weights()

print('Original Weights')
for name, weight in zip(names, original_weights):
    print(name, weight)

print('Trained Weights')
for name, weight in zip(names, weights):
    print(name, weight)

print(extract_weights(names, weights, CHOICE_CONFIG))


with open(DATA_INPUT_FOLDER + 'training_results.txt', 'a') as f:
    if np.array_equal(original_weights, weights):
       result = 'Fail: TL Select Atoms - Weights Adjusted After Training'
    else:
       result = 'Pass: TL Select Atoms - Weights Adjusted After Training'

    f.write(result + '\n')
