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
        self.bins_list = self.create_bins(discrete_arr)

    def cont_to_discrete(self, continuous_value):
        mean = self.get_bin_index_and_mean(continuous_value)[1]
        return mean

    def get_bin_index(self, continuous_value):
        idx = self.get_bin_index_and_mean(continuous_value)[0]
        return idx

    def discrete_to_cont(self, index):
        return self.bins_list[index].mean

    def get_bin_index_and_mean(self, continuous_value):
        for idx, bin in enumerate(self.bins_list):
            if bin.contains_value(continuous_value):
                return idx, bin.mean
        else:
            print("get default bin")
            return self.bins_list[0].mean

    @staticmethod
    def create_bins(discrete_arr):
        bins = []
        for idx in range(0, len(discrete_arr) - 2, 2):
            lower_range = discrete_arr[idx]
            higher_range = discrete_arr[idx + 2]
            mean = discrete_arr[idx + 1]

            bin = Bin(lower_range, higher_range, mean)
            bins.append(bin)
        return bins
