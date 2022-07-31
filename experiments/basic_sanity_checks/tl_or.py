import sys

sys.path.append('/Users/nicole/OSU/GAN_TL')

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, RNN
from tensorflow.keras.models import Model
from TLOps import TLOR, TLAtom
from TL_learning_utils import TLWeightTransform


DATA_INPUT_FOLDER = 'experiments/basic_sanity_checks/'
np.random.seed(42)
tf.random.set_seed(42)

# READ DATA
data1 = np.loadtxt(DATA_INPUT_FOLDER + 'data1_trace.csv', delimiter=',')
data2 = np.loadtxt(DATA_INPUT_FOLDER + 'data2_trace.csv', delimiter=',')

trace_length = 5
input_shape = (trace_length, 1)

# MAKE TL UNTIL
in1 = Input(input_shape)
in2 = Input(input_shape)
phi = RNN(TLAtom(1, w=1, b=0), return_sequences=True)(in1)
psi = RNN(TLAtom(1, w=1, b=0), return_sequences=True)(in2)
or_input = concatenate([phi, psi])
or_weighted_input = TLWeightTransform(1, w=1)(or_input)  # BOTH GO INTO THE OR
or_output = RNN(TLOR(1), return_sequences=True)(or_weighted_input)
model = Model(inputs=[in1, in2], outputs=[or_output])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


predictions = model.predict([data1, data2], verbose=0)
predictions_save = np.reshape(predictions, (100, 5))

np.savetxt(DATA_INPUT_FOLDER + 'rnn_predictions_or.csv',
           predictions_save, delimiter=',', fmt='%.5e')


# MAKE TRAINABLE TL_UNTIL
in1 = Input(input_shape)
in2 = Input(input_shape)
phi = RNN(TLAtom(1), return_sequences=True)(in1)
psi = RNN(TLAtom(1), return_sequences=True)(in2)
or_input = concatenate([phi, psi])
or_weighted_input = TLWeightTransform(1, quantize=True)(or_input)  # LEARN WHICH GO INTO THE OR
or_output = RNN(TLOR(1), return_sequences=True)(or_weighted_input)
trainable_model = Model(inputs=[in1, in2], outputs=[or_output])
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

    trainable_model.train_on_batch([X, X], y)

    if i % 20 == 0:
        loss, acc = trainable_model.evaluate([X, X], y, verbose=0)
        print("Epoch {i}, acc {a}  loss {l}".format(i=i, a=acc, l=loss))

weights = trainable_model.get_weights()

print('Original Weights')
print(original_weights)

print('Trained Weights')
print(weights)


with open(DATA_INPUT_FOLDER + 'training_results.txt', 'a') as f:
    if np.array_equal(original_weights, weights):
       result = 'Fail: TL OR - Weights Adjusted After Training'
    else:
       result = 'Pass: TL OR - Weights Adjusted After Training'

    f.write(result + '\n')

