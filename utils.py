import numpy as np


def np_to_list_tuples(x, y):
    return [(x[i], y[i]) for i in range(x.shape[0])]


def one_hot_encode(x):
    # presently assumes continuity; simplest version
    one_hot_array = np.zeros((len(x), 1 + max(x) - min(x)))
    for i in range(len(x)):
        one_hot_array[i][x[i]] = 1
    return one_hot_array
