# from data_sampling import *
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, Dense, RNN, SimpleRNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from TLOps import buildUntil, TLAtom, TLOR, TLAlw, TLEv, TLNext, TLUntil, TLWeightTransform

tf.random.set_seed(42)

# PARAMETERS
test_req_stl = 'testSpecUntil'  # must match what is selected in `eval_rnn.m`
n = 5  # number of random signals to choose from for testing

# TESTS
scale = 10


def test_TLAtom(input_shape, learn_weights=False):
    model = Sequential()
    model.add(Input(input_shape))
    if learn_weights:
        # model.add(RNN(TLAtom(1), go_backwards=True, return_sequences=True))
        model.add(RNN(TLAtom(1), return_sequences=True))
    else:
        # model.add(RNN(TLAtom(1, b=0.65, w=1),
        #               go_backwards=True, return_sequences=True))
        # model.add(RNN(TLAtom(1, b=0.65, w=1), return_sequences=True))

        # model.add(RNN(TLAtom(1, b=0.65, w=1), return_sequences=True))
        model.add(RNN(TLAtom(1, b=0, w=1), return_sequences=True))
    return model


def test_TLAtom2d(input_shape, learn_weights=False):
    model = Sequential()
    model.add(Input(input_shape))
    if learn_weights:
        model.add(RNN(TLAtom(2), go_backwards=True, return_sequences=True))
        model.add(RNN(TLOR(1), return_sequences=True))
    else:
        model.add(RNN(TLAtom(2, b=[0.65, 0], w=[1, -1]),
                      go_backwards=True, return_sequences=True))
        model.add(RNN(TLOR(1), return_sequences=True))
    return model


def test_TLOR(input_shape, learn_weights=False):
    in1 = Input(input_shape)
    if learn_weights:
        # atom1 = RNN(TLAtom(1), go_backwards=True, return_sequences=True)(in1)
        # atom2 = RNN(TLAtom(1), go_backwards=True, return_sequences=True)(in1)
        atom1 = RNN(TLAtom(1), return_sequences=True)(in1)
        atom2 = RNN(TLAtom(1), return_sequences=True)(in1)
        or_input = concatenate([atom1, atom2])
        # or_weighted_input = TLWeightTransform(1)(or_input)
        or_weighted_input = TLWeightTransform(1, quantize=True)(or_input)
    else:
        # atom1 = RNN(TLAtom(1, b=0.65, w=1), go_backwards=True,
        #             return_sequences=True)(in1)
        # atom2 = RNN(TLAtom(1, b=0, w=-1), go_backwards=True,
        #             return_sequences=True)(in1)
        # atom1 = RNN(TLAtom(1, b=0.65, w=1), return_sequences=True)(in1)
        # atom2 = RNN(TLAtom(1, b=0, w=-1), return_sequences=True)(in1)
        atom1 = RNN(TLAtom(1, b=-0.5, w=1), return_sequences=True)(in1)
        atom2 = RNN(TLAtom(1, b=0, w=1), return_sequences=True)(in1)
        or_input = concatenate([atom1, atom2])
        or_weighted_input = TLWeightTransform(1, w=1)(or_input)

    final_output = RNN(TLOR(1), return_sequences=True)(or_weighted_input)
    model = Model(inputs=[in1], outputs=[final_output])
    return model


def test_TLNext(input_shape, learn_weights=False):
    model = Sequential()
    model.add(Input(input_shape))
    if learn_weights:
        model.add(RNN(TLAtom(1), return_sequences=True))
    else:
        model.add(RNN(TLAtom(1, b=0, w=1), return_sequences=True))

    model.add(RNN(TLNext(10), return_sequences=True))
    return model


def test_TLAlw(input_shape, learn_weights=False):
    model = Sequential()
    model.add(Input(input_shape))
    if learn_weights:
        # model.add(RNN(TLAtom(1), return_sequences=True, go_backwards=True))
        model.add(RNN(TLAtom(1), return_sequences=True))
    else:
        # model.add(RNN(TLAtom(1, b=0, w=-1), return_sequences=True,
        #               go_backwards=True))
        # model.add(RNN(TLAtom(1, b=0, w=-1), return_sequences=True))
        model.add(RNN(TLAtom(1, b=0, w=1), return_sequences=True))
    model.add(RNN(TLAlw(10), return_sequences=True))
    return model


# def test_TLAlwBounded(input_shape, learn_weights=False):
#     model = Sequential()
#     model.add(Input(input_shape))
#     if learn_weights:
#         model.add(RNN(TLAtom(1), return_sequences=True, go_backwards=True))
#     else:
#         model.add(RNN(TLAtom(1, b=0, w=-1), return_sequences=True,
#                       go_backwards=True))
#     model.add(RNN(TLAlw(10, start=0, end=1), return_sequences=True))
#     return model


def test_TLEv(input_shape, learn_weights=False):
    model = Sequential()
    model.add(Input(input_shape))
    if learn_weights:
        # model.add(RNN(TLAtom(1), return_sequences=True, go_backwards=True))
        model.add(RNN(TLAtom(1), return_sequences=True))
    else:
        # model.add(RNN(TLAtom(1, b=0, w=-1), return_sequences=True,
        #               go_backwards=True))
        # model.add(RNN(TLAtom(1, b=0, w=-1), return_sequences=True))
        model.add(RNN(TLAtom(1, b=0, w=1), return_sequences=True))
    model.add(RNN(TLEv(10), return_sequences=True))
    return model


# def test_TLEvBounded(input_shape, learn_weights=False):
#     model = Sequential()
#     model.add(Input(input_shape))
#     if learn_weights:
#         model.add(RNN(TLAtom(1), return_sequences=True, go_backwards=True))
#     else:
#         model.add(RNN(TLAtom(1, b=0, w=-1), return_sequences=True,
#                       go_backwards=True))
#     model.add(RNN(TLEv(10, start=0, end=1), return_sequences=True))
#     return model


def test_TLEvAlw(input_shape, learn_weights=False):
    model = Sequential()
    model.add(Input(input_shape))
    if learn_weights:
        # model.add(RNN(TLAtom(1), return_sequences=True, go_backwards=True))
        model.add(RNN(TLAtom(1), return_sequences=True))
    else:
        # model.add(RNN(TLAtom(1, b=0, w=-1), return_sequences=True,
        #               go_backwards=True))
        # model.add(RNN(TLAtom(1, b=0, w=-1), return_sequences=True))
        model.add(RNN(TLAtom(1, b=0, w=1), return_sequences=True))
    model.add(RNN([TLEv(10), TLAlw(10)], return_sequences=True))
    return model


def test_TLUntil(input_shape, learn_weights=False):
    in1 = Input(input_shape)
    # phi = RNN(TLAtom(1, b=5, w=-1), return_sequences=True)(in1)
    # psi = RNN(TLAtom(1, b=3, w=-1), return_sequences=True)(in1)

    in2 = Input(input_shape)
    phi = RNN(TLAtom(1, w=1, b=0), return_sequences=True)(in1)
    psi = RNN(TLAtom(1, w=1, b=0), return_sequences=True)(in2)

    # until_output = buildUntil([phi], [psi], scale, learn_weights=learn_weights)
    until_input = concatenate([phi, psi])
    # until_output = RNN(TLUntil(10), return_sequences=True)(until_input)
    # until_output = RNN(TLUntil(6), return_sequences=True, return_state=True)(until_input)
    until_output = RNN(TLUntil(4), return_sequences=True, return_state=True)(until_input)

    # model = Model(inputs=[in1], outputs=[until_output])
    model = Model(inputs=[in1, in2], outputs=[until_output])
    return model


def test(model, trainable_model):
    print('Comparing Robustness Values with Breach')
    print(model.summary())
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, weight in zip(names, weights):
        print(name, weight)

    # good_data, good_y = sample_good_data(n=n, train=False, scale=scale)
    # good_data = np.array([
    #     [1 for i in range(10)],
    #     [4 for i in range(10)],
    #     [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
    #     [-1 for i in range(10)],
    #     [(-1)**i * 4 for i in range(10)],
    #     [-1 for i in range(5)] + [1 for i in range(5)],
    #     [4, 4, 4, 4, 4, 7, 7, 7, 7, 7],
    #     [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    #     [(-1) ** i * 6 for i in range(10)],
    #     [4, -1, 4.1, -0.5, 4.3, -3, 4, -4, 4, -4],
    #     [7, 6, 7.1, 5.9, 1, 7, 6.9, 6.1, 7, 6]
    # ])
    good_data = np.array([
        [1 for i in range(6)],
        [4 for i in range(6)],
        [4, 4, 4, 1, 1, 1],
        [-1 for i in range(6)],
        [(-1) ** i * 4 for i in range(6)],
        [-1 for i in range(3)] + [1 for i in range(3)],
        [4, 4, 4, 7, 7, 7],
        [7, 7, 7, 7, 7, 7],
        [(-1) ** i * 6 for i in range(6)],
        [4, -1, 4.1, -0.5, 4.3, -3],
        [7, 2, 7.1, 5.9, 1, 7],
        [4, -1, -2, -0.5, 4.3, -3],
    ])
    scale = 6

    # np.savetxt('test_data.csv', good_data)
    good_data = good_data.reshape((12, scale, 1))

    print('RNN Robustness Values')
    x = np.array([[1, 2, 3, 4]])
    y = np.array([[5.3, 4.3, 3, -1]])

    # predictions = model.predict([good_data], verbose=0)
    predictions = model.predict([x, y], verbose=0)
    # print("plot model")
    # plot_model(model, to_file='model.png', show_shapes=True)

    # for p in predictions:
    #     print(p.T)
    for p, s in predictions:
        print(p)
    for p, s in predictions:
        print(s)


    # print('Starting up MATLAB, Getting Robustness Values from Breach')
    # mat_command = '/Applications/MATLAB_R2021b.app/bin/matlab '
    # # matlab_args = '-nodesktop -nosplash -r "eval_rnn;exit;"'
    # matlab_args = '-nodesktop -nosplash -r "eval_until;exit;"'
    # os.system(mat_command + matlab_args)

    # print('Training')
    # model = trainable_model
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    original_weights = model.get_weights()
    #
    # epochs = 30
    # for i in range(epochs):
    #     # prepare good samples
    #     good_data, good_y = sample_good_data(n=100)
    #
    #     # prepare bad examples
    #     bad_data, bad_y = sample_bad_data(n=100)
    #
    #     # reshape (samples, timesteps, features)
    #     good_data = good_data.reshape((100, 10, 1))
    #     bad_data = bad_data.reshape((100, 10, 1))
    #
    #     X, y = np.vstack((good_data, bad_data)), np.vstack((good_y, bad_y))
    #
    #     model.train_on_batch([X], y)
    #
    #     if i % 10 == 0:
    #         loss, acc = model.evaluate(X, y, verbose=0)
    #         print("Epoch {i}, acc {a}  loss {l}".format(i=i, a=acc, l=loss))
    #
    # names = [weight.name for layer in model.layers for weight in layer.weights]
    # weights = model.get_weights()

    print('Original Weights')
    for name, weight in zip(names, original_weights):
        print(name, weight)

    # print('Trained Weights')
    # for name, weight in zip(names, weights):
    #     print(name, weight)

    # print('Trained Model Predictions')
    # good_data, good_y = sample_good_data(n=1, train=False, scale=scale)
    # good_data = good_data.reshape((1, scale, 1))
    # print(good_data)
    # predictions = model.predict([good_data], verbose=0)
    # for p in predictions:
    #     print(p.T)


if __name__ == "__main__":
    # input_shape = (10, 1)
    # input_shape = (6, 1)
    input_shape = (4, 1)
    if test_req_stl == 'testSpecAtom':
        model = test_TLAtom(input_shape)
        trainable_model = test_TLAtom(input_shape, learn_weights=True)
    # if test_req_stl == 'testSpecAtom2d':
    #     model = test_TLAtom2d(input_shape)
    #     trainable_model = test_TLAtom2d(input_shape, learn_weights=True)
    elif test_req_stl == 'testSpecOr':
        model = test_TLOR(input_shape)
        trainable_model = test_TLOR(input_shape, learn_weights=True)
    elif test_req_stl == 'testSpecAlways':
        model = test_TLAlw(input_shape)
        trainable_model = test_TLAlw(input_shape, learn_weights=True)
    # elif test_req_stl == 'testSpecAlwBounded':
    #     model = test_TLAlwBounded(input_shape)
    #     trainable_model = test_TLAlwBounded(input_shape, learn_weights=True)
    elif test_req_stl == 'testSpecEv':
        model = test_TLEv(input_shape)
        trainable_model = test_TLEv(input_shape, learn_weights=True)
    # elif test_req_stl == 'testSpecEvBounded':
    #     model = test_TLEvBounded(input_shape)
    #     trainable_model = test_TLEvBounded(input_shape, learn_weights=True)
    elif test_req_stl == 'testSpecEvAlw':
        model = test_TLEvAlw(input_shape)
        trainable_model = test_TLEvAlw(input_shape, learn_weights=True)
    elif test_req_stl == 'testSpecUntil':
        model = test_TLUntil(input_shape)
        trainable_model = test_TLUntil(input_shape, learn_weights=True)
    elif test_req_stl == 'testSpecNext':
        model = test_TLNext(input_shape)
        trainable_model = test_TLNext(input_shape, learn_weights=True)
    else:
        print('INVALID REQUIREMENT')
    test(model, trainable_model)
