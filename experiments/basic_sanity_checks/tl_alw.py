import sys

sys.path.append('/Users/nicole/OSU/GAN_TL')

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, RNN
from tensorflow.keras.models import Model
from TLOps import TLAlw, TLAtom


DATA_INPUT_FOLDER = 'experiments/basic_sanity_checks/'
np.random.seed(42)
tf.random.set_seed(42)

# READ DATA
data1 = np.loadtxt(DATA_INPUT_FOLDER + 'data1_trace.csv', delimiter=',')
data2 = np.loadtxt(DATA_INPUT_FOLDER + 'data2_trace.csv', delimiter=',')

trace_length = 5
input_shape = (trace_length, 1)

# MAKE TL EV
in1 = Input(input_shape)
phi = RNN(TLAtom(1, w=1, b=0), return_sequences=True)(in1)
alw_output = RNN(TLAlw(trace_length), return_sequences=True)(phi)
model = Model(inputs=[in1], outputs=[alw_output])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

predictions = model.predict([data1], verbose=0)
predictions_save = np.reshape(predictions, (100, 5))

np.savetxt(DATA_INPUT_FOLDER + 'rnn_predictions_alw.csv',
           predictions_save, delimiter=',', fmt='%.5e')

# MAKE TRAINABLE TL EV
in1 = Input(input_shape)
phi = RNN(TLAtom(1), return_sequences=True)(in1)
alw_output = RNN(TLAlw(trace_length), return_sequences=True)(phi)
trainable_model = Model(inputs=[in1], outputs=[alw_output])
trainable_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

names = [weight.name for layer in model.layers for weight in layer.weights]
original_weights = trainable_model.get_weights()

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

    trainable_model.train_on_batch([X], y)

    if i % 20 == 0:
        loss, acc = trainable_model.evaluate([X], y, verbose=0)
        print("Epoch {i}, acc {a}  loss {l}".format(i=i, a=acc, l=loss))

weights = trainable_model.get_weights()

print('Original Weights')
print(original_weights)

print('Trained Weights')
print(weights)


with open(DATA_INPUT_FOLDER + 'training_results.txt', 'a') as f:
    if np.array_equal(original_weights, weights):
       result = 'Fail: TL Alw - Weights Adjusted After Training'
    else:
       result = 'Pass: TL Alw - Weights Adjusted After Training'

    f.write(result + '\n')

