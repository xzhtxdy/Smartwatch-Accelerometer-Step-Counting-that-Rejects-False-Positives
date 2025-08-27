import numpy as np
from .low_pass_filter import butter_lowpass
import os
from .read_data import *

__all__ = ['make_windows',
           'read_data_for_one_subject',
           'read_data_for_one_activity',
           "data_filter"]


def check_package_number(f, filepath):
    start, end, length = f[0, 0], f[-1, 0], len(f)
    if end < start:
        end = end + 65536
    if end - start - length + 1 == 0:
        return True
    else:
        for i in range(len(f) - 1):
            if f[i, 0] < f[0, 0]:
                f[i:, 0] = f[i:, 0] + 65536
            if f[i + 1, 0] - f[i, 0] > 1:
                print(f[i + 1, 0], f[i, 0])
        print('{}丢失了{}个数据'.format(filepath, f[-1, 0] - f[0, 0] - len(f) + 1))
        return False


def gap_fill(f):
    start_id = []
    step_id = []
    for i in range(len(f) - 1):
        if f[i, 0] < f[0, 0]:
            f[i:, 0] = f[i:, 0] + 65536
        if f[i + 1, 0] - f[i, 0] > 1:
            start_id.append(i)
            step_id.append(int(f[i + 1, 0] - f[i, 0]))
    for (i, j) in zip(start_id, step_id):
        step = (f[i + 1, :] - f[i, :]) / j
        for k in range(1, j):
            f = np.insert(f, i + k, f[i, :] + step * k, axis=0)
    assert f[-1, 0] - f[0, 0] - len(f) + 1 == 0
    print("gap filled linearly")
    return f


def data_filter(array, n, cutoff_f, frequency):
    data_filtered = np.empty(shape=array.shape)
    for col in range(array.shape[1]):
        data_col = array[:, col]
        data_col_filtered = butter_lowpass(n, cutoff_f, frequency, data_col)
        data_filtered[:, col] = data_col_filtered
    return data_filtered


def make_windows(array, win_len, stride, dim):
    """
    A generator to make windows from the array
    :param array:
    :param win_len:
    :param stride:
    :param dim:
    :return:
    """
    r = np.arange(len(array))
    s = r[::stride]
    z = list(zip(s, s + win_len))
    for a in z:
        yield array[a[0]:a[1], :dim]
