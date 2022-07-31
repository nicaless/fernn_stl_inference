import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, Dense, RNN, SimpleRNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from TLOps import buildUntil, TLAtom, TLAlw, TLEv, TLOR, TLWeightTransform


tf.random.set_seed(42)
np.random.seed(42)

good_uav_dists = pickle.load(open('good_uav_dists.pkl', 'rb'))

n = len(good_uav_dists)
scale = 20
data = []
for idx in range(n):
    uav_dist_data = [good_uav_dists[idx][i] for i in range(scale)]
    data.append(uav_dist_data)
data = np.array(data)

# GENERATE ROBUSTNESS VALUES
true_in_shape = (20, 1)
true_in1 = Input(true_in_shape)
true_atom1 = RNN(TLAtom(1, b=0.65, w=1), return_sequences=True, go_backwards=True)(true_in1)
true_atom2 = RNN(TLAtom(1, b=0, w=-1), return_sequences=True, go_backwards=True)(true_in1)
true_atom3 = RNN(TLAtom(1, b=0, w=1), return_sequences=True, go_backwards=True)(true_in1)
#### OR ####
# true_or_input = concatenate([true_atom1, true_atom2, true_atom3])
# true_or_weights = TLWeightTransform(1, w=1)(true_or_input)
# true_output = RNN(TLOR(1), return_sequences=True)(true_or_weights)

#### ALW ####
true_alw_input = true_atom2
true_alw_weights = TLWeightTransform(1, w=1, transform='sum')(true_alw_input)
true_output = RNN(TLAlw(20), return_sequences=True)(true_alw_weights)

#### EV(ALW) ####
# true_ev_alw_input = true_atom1
# true_ev_alw_weights = TLWeightTransform(1, w=1, transform='sum')(true_ev_alw_input)
# true_output = RNN([TLEv(20), TLAlw(20)], return_sequences=True, go_backwards=True)(true_ev_alw_weights)

#### UNTIL ####
# true_phi = [true_atom1]
# true_psi = [true_atom2]
# true_output = buildUntil(true_phi, true_psi, true_in_shape[0])

true_model = Model(inputs=[true_in1], outputs=[true_output])

print(true_model.summary())
true_model.compile(optimizer='adam',
                   loss='mean_squared_error',
                   metrics=['mean_squared_error'])
plot_model(true_model, to_file='true_model.png', show_shapes=True)

input_data = data.reshape((n, scale, 1))
robs = true_model.predict([input_data], verbose=0)
robs = np.reshape(robs, (n, 20))
np.savetxt('test_robs_good_uav_dists.csv', robs)


# LEARN MODEL STRUCTURE
baseline_model = Sequential()
baseline_model.add(Input((20, 1)))
baseline_model.add(SimpleRNN(5))
baseline_model.add(Dense(20))
baseline_model.compile(optimizer='adam',
                       loss='mean_squared_error',
                       metrics=['mean_squared_error'])

in1 = Input((20, 1))
atom1 = RNN(TLAtom(1), return_sequences=True, go_backwards=True)(in1)
atom2 = RNN(TLAtom(1), return_sequences=True, go_backwards=True)(in1)
atom3 = RNN(TLAtom(1), return_sequences=True, go_backwards=True)(in1)
atom4 = RNN(TLAtom(1), return_sequences=True, go_backwards=True)(in1)
atom5 = RNN(TLAtom(1), return_sequences=True, go_backwards=True)(in1)

#### OR ####
# or_input = concatenate([atom1, atom2, atom3, atom4, atom5])
# or_weights = TLWeightTransform(1, quantize='sign')(or_input)
# # or_weights = TLWeightTransform(1)(or_input)
# final_output = RNN(TLOR(1), return_sequences=True)(or_weights)

#### ALW ####
alw_input = concatenate([atom1, atom2, atom3, atom4, atom5])
alw_weights = TLWeightTransform(1, transform='sum', quantize='one-hot')(alw_input)
# alw_weights = TLWeight(1, transform='sum')(alw_input)
final_output = RNN(TLAlw(20), return_sequences=True)(alw_weights)

#### EV(ALW) ####
# ev_alw_input = concatenate([atom1, atom2, atom3, atom4, atom5])
# ev_alw_weights = TLWeightTransform(1, transform='sum')(ev_alw_input)
# final_output = RNN([TLEv(20), TLAlw(20)], return_sequences=True, go_backwards=True)(ev_alw_weights)

#### UNTIL ####
# phis = [atom1, atom2]
# psis = [atom3, atom4, atom5]
# final_output = buildUntil(phis, psis, true_in_shape[0], learn_weights=True)

model = Model(inputs=[in1], outputs=[final_output])

print(model.summary())
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])
plot_model(model, to_file='trained_model.png', show_shapes=True)
names = [weight.name for layer in model.layers for weight in layer.weights]
original_weights = model.get_weights()

epochs = 500
batch_size = 512
epoch_check = 50
for i in range(epochs):
    choose_ids = np.random.choice(int(n*.8),
                                  size=batch_size, replace=False)
    X = np.array([input_data[idx] for idx in choose_ids])
    y = np.array([robs[idx] for idx in choose_ids])

    model.train_on_batch(X, y)
    baseline_model.train_on_batch(X, y)

    if i % epoch_check == 0:
        _, acc = model.evaluate(X, y, verbose=0)
        _, bacc = baseline_model.evaluate(X, y, verbose=0)
        print("Epoch {i}, mse {a}. baseline mse: {b}".format(i=i, a=acc, b=bacc))

# EVALUATE ON TEST SET
choose_ids = np.random.choice(list(np.arange(int(n*.8), n)),
                              size=batch_size, replace=False)
test_X = np.array([input_data[idx] for idx in choose_ids])
test_y = np.array([robs[idx] for idx in choose_ids])
_, acc = model.evaluate(test_X, test_y, verbose=0)
_, bacc = baseline_model.evaluate(test_X, test_y, verbose=0)
print("Test Set, mse {a}. baseline mse: {b}".format(i=i, a=acc, b=bacc))


trained_weights = model.get_weights()

print('Original Weights')
for name, weight in zip(names, original_weights):
    print(name, weight)

print('Trained Weights')
for name, weight in zip(names, trained_weights):
    print(name, weight)

print('COMPARE ROBUSTNESS VALUES TO BREACH')

choose_ids = np.random.choice(list(np.arange(int(n*.8), n)),
                              size=5, replace=False)
test_X = np.array([input_data[idx] for idx in choose_ids])
predictions = model.predict(test_X)
true_model_vals = true_model.predict(test_X)
test_X = np.reshape(test_X, (5, 20))
np.savetxt('test_data.csv', test_X)

print('Trained Model Predictions')
for p in predictions:
    print(p.T)

print('True Model Predictions')
for p in true_model_vals:
    print(p.T)

print('Starting up MATLAB, Getting Robustness Values from Breach')
mat_command = '/Applications/MATLAB_R2020b.app/bin/matlab '
matlab_args = '-nodesktop -nosplash -r "eval_rnn;exit;"'
os.system(mat_command + matlab_args)
