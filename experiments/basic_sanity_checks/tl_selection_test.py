import sys

sys.path.append('/Users/nicole/OSU/GAN_TL')

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, RNN, Dense
from tensorflow.keras.models import Model
import time
from TLOps import TLAtom
from TL_learning_utils import TLWeightTransform, TLPass, extract_weights, pos_robustness


np.random.seed(42)
tf.random.set_seed(42)
trace_length = 5
CHOICE_CONFIG = {'choice_atoms': ['pos', 'neg']}
epochs = 100
epoch_check = int(epochs / 10)
# epoch_check = 1
init_lr = 0.001

input_shape = (trace_length, 1)
in1 = Input(input_shape)

# pos_atom = RNN(TLAtom(1, w=1, b=0), return_sequences=True, name='pos')(in1)
# neg_atom = RNN(TLAtom(1, w=-1, b=0), return_sequences=True, name='neg')(in1)
pos_atom = RNN(TLAtom(1), return_sequences=True, name='pos')(in1)
neg_atom = RNN(TLAtom(1), return_sequences=True, name='neg')(in1)

atom_concat = concatenate([pos_atom, neg_atom])
choice = TLWeightTransform(1, transform='sum', quantize='one-hot',
                           name='choice_atoms')(atom_concat)
# choice = TLWeightTransform(1, transform='sum',
#                            name='choice_atoms')(atom_concat)
# out = RNN(TLAtom(1, w=1, b=0), name='final_out')(choice)
# out = RNN(TLAtom(1), name='final_out')(choice)
out = RNN(TLPass(1), name='final_out')(choice)
out = tf.keras.layers.Activation(tf.keras.activations.tanh)(out)

model = Model(inputs=[in1], outputs=[out])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=init_lr)
model.compile(optimizer=optimizer,
              loss=pos_robustness
              # loss='binary_crossentropy',
              )
names = [weight.name for layer in model.layers for weight in layer.weights]
original_weights = model.get_weights()

train_time_start = time.time()
for i in range(epochs):
    # prepare good samples
    good_data = np.random.uniform(low=0, high=5, size=(100, trace_length))

    # prepare bad examples
    bad_data = np.random.uniform(low=-5, high=0, size=(100, trace_length))

    # reshape (samples, timesteps, features)
    good_data = good_data.reshape((100, trace_length, 1))
    bad_data = bad_data.reshape((100, trace_length, 1))

    X = np.vstack((good_data, bad_data))
    y = np.vstack((np.ones((100, 1)),
                   -1 * np.zeros((100, 1))))

    if i % epoch_check == 0:
        with tf.GradientTape() as tape:
            # loss1 = model.train_on_batch([X], y)
            #
            # for layer in model.layers:
            #     fn = getattr(layer, 'on_batch_end', None)
            #     if callable(fn):
            #         fn()

            pred = model(X)
            loss2 = pos_robustness(y, pred)

        # # grads1 = tape.gradient(loss1, model.trainable_variables)
        grads2 = tape.gradient(loss2, model.trainable_variables)
        # print(model.get_weights())
        # print(loss1)
        # # print(grads1)
        # print(loss2)
        print(grads2)

        loss1 = model.evaluate([X], y, verbose=0)
        print("Epoch {i}, loss {l}".format(i=i, l=loss1))
        # continue

    loss = model.train_on_batch([X], y)

    # print("End of Batch Update {}".format(i))
    # for layer in model.layers:
    #     fn = getattr(layer, 'on_batch_end', None)
    #     if callable(fn):
    #         fn()


train_time_end = time.time()
print('Training Time: {}'.format(train_time_end - train_time_start))

weights = model.get_weights()
for name, weight in zip(names, weights):
    print(name, weight)
print(extract_weights(names, weights, CHOICE_CONFIG))

good_data = np.random.uniform(low=0, high=5, size=(50, trace_length))
bad_data = np.random.uniform(low=-5, high=0, size=(50, trace_length))
good_data = good_data.reshape((50, trace_length, 1))
bad_data = bad_data.reshape((50, trace_length, 1))

predictions = model.predict(good_data, verbose=0)
false_neg = 0
for p in predictions:
    final_robs = p[-1]
    if final_robs < 0:
        false_neg += 1

predictions = model.predict(bad_data, verbose=0)
false_pos = 0
for p in predictions:
    final_robs = p[-1]
    if final_robs > 0:
        false_pos += 1

print(false_neg)
print(false_pos)

print(false_neg + false_pos)
print('MCR (Quantized)')
print((false_pos + false_neg) / 100)

