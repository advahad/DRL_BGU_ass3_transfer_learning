import numpy as np


def pad_and_reshape(origin_size, max_size):
    pad_to_add = np.zeros(max_size - len(origin_size))
    concatenated = np.concatenate((origin_size, pad_to_add), axis=0)
    concatenated = concatenated.reshape([1, max_size])
    return concatenated
