import numpy as np

class Bin:
    def __init__(self, low, high, mean):
        self.low = low
        self.high = high
        self.mean = mean

    def contains_value(self, value):
        return self.low <= value < self.high


class DiscreteBins:
    def __init__(self, discrete_arr):
        self.bins_list = create_bins(discrete_arr)

    def cont_to_discrete(self, continuous_value):
        for bin in self.bins_list:
            if bin.contains_value(continuous_value):
                return bin.mean
        else:
            return self.bins_list[0].mean

    def discrete_to_cont(self, index):
        return self.bins_list[index].mean


def create_bins(discrete_arr):
    bins = []
    for idx in range(0, len(discrete_arr) - 2, 2):
        lower_range = discrete_arr[idx]
        higher_range = discrete_arr[idx + 2]
        mean = discrete_arr[idx + 1]

        bin = Bin(lower_range, higher_range, mean)
        bins.append(bin)
    return bins
