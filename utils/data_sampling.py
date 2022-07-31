import math
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

stdx_db = pickle.load(open('hapt_stdx_db.pkl', 'rb'))

good_uav_dists = pickle.load(open('good_uav_dists.pkl', 'rb'))
bad_uav_dists = pickle.load(open('bad_uav_dists.pkl', 'rb'))

pitch_data = pd.read_csv('all_pitch_traces.csv', header=None)
pitch_data_train = pitch_data.loc[0:int(.8*(len(pitch_data)))]
pitch_data_test = pitch_data.loc[int(.8*(len(pitch_data))):len(pitch_data)]


def sample_good_data(n=10000, scale=20, dataset='hapt', train=True):
    data = []
    if dataset == 'hapt':
        start = 0 if train else int(len(stdx_db['dynamic']) * .8)
        end = int(len(stdx_db['dynamic']) * .8) if train else len(stdx_db['dynamic'])

        for i in range(n):
            idx = np.random.choice(list(range(start, end)))
            trace = stdx_db['dynamic'][idx]
            trace = trace[:scale]
            data.append(trace)
    elif dataset == 'pitch':
        data = pitch_data_train.sample(n=n, replace=True)
    elif dataset == 'pitch_test':
        data = pitch_data_test.sample(n=n, replace=True)
    else:
        start = 0 if train else int(len(good_uav_dists) * .9)
        end = int(len(good_uav_dists) * .9) if train else len(good_uav_dists)
        for i in range(n):
            idx = np.random.choice(list(range(start, end)))
            uav_dist_data = [good_uav_dists[idx][i] for i in range(scale)]
            data.append(uav_dist_data)

    return np.array(data), np.ones((n, 1))


def sample_bad_data(n=10000, scale=20, dataset='hapt', train=True):
    data = []
    if dataset == 'hapt':
        start = 0 if train else int(len(stdx_db['static']) * .8)
        end = int(len(stdx_db['static']) * .8) if train else len(stdx_db['static'])

        for i in range(n):
            idx = np.random.choice(list(range(start, end)))
            data.append(stdx_db['static'][idx])
    elif dataset == 'pitch':
        rand_8ve = []
        for i in range(n):
            rand_8ve.append(np.random.choice([-3, 2, -1, 1, 2, 3]))
        data = pitch_data_train.sample(n=n, replace=True)
        return np.array(data) + 12 * np.reshape(rand_8ve, (n, 1)), np.zeros((n, 1))
    elif dataset == 'pitch_test':
        rand_8ve = []
        for i in range(n):
            rand_8ve.append(np.random.choice([-3, 2, -1, 1, 2, 3]))
        data = pitch_data_test.sample(n=n, replace=True)
        return np.array(data) + 12 * np.reshape(rand_8ve, (n, 1)), np.zeros((n, 1))
    else:
        start = 0 if train else int(len(bad_uav_dists) * .9)
        end = int(len(bad_uav_dists) * .9) if train else len(bad_uav_dists)
        for i in range(n):
            idx = np.random.choice(list(range(start, end)))
            uav_dist_data = [bad_uav_dists[idx][i] for i in range(scale)]
            data.append(uav_dist_data)
    return np.array(data), np.zeros((n, 1))


def latent_inputs(n, latent_dim=3):
    input = np.random.randn(latent_dim * n)

    # reshape into a batch of inputs for the network
    input = input.reshape(n, latent_dim)

    return input


# data = np.load('uniform_test_data_good.npz')
# x = []
# for l in range(20):
#     pos1 = data['pos1'][l]
#     pos2 = data['pos2'][l]
#     y = []
#     for idx in range(0, 20):
#         dist = math.sqrt((pos2[0][idx] - pos1[0][idx])**2 + (pos2[1][idx] - pos1[1][idx])**2 + (pos2[2][idx] - pos1[2][idx])**2)
#         y.append(dist)
#     x.append(y)
#
# data = np.load('uniform_test_data_bad.npz')
# a = []
# for l in range(20):
#     pos1 = data['pos1'][l]
#     pos2 = data['pos2'][l]
#     b = []
#     for idx in range(0, 20):
#         dist = math.sqrt((pos2[0][idx] - pos1[0][idx])**2 + (pos2[1][idx] - pos1[1][idx])**2 + (pos2[2][idx] - pos1[2][idx])**2)
#         b.append(dist)
#     a.append(b)
#
# for t in x:
#     plt.plot(t, alpha=0.5, color='blue')
#
# for t in a:
#     plt.plot(t, alpha=0.5, color='red')
#
# plt.xlim([0, 21])
#
# plt.show()
#
#
# good_data, good_y = sample_good_data(n=100, dataset='pitch')
# bad_data, bad_y = sample_bad_data(n=100, dataset='pitch')
# np.savetxt('good_pitch_data.csv', good_data)
# np.savetxt('bad_pitch_data.csv', bad_data)


# for t in good_data:
#     plt.plot(t, alpha=0.5, color='blue')

# for t in bad_data:
#     plt.plot(t, alpha=0.5, color='red')

# # plt.xlim([0, 20])
# plt.xlabel('time')
# plt.ylabel('pitch')

# plt.savefig('pitch_data_example.png')
