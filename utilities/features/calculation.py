import numpy as np
from scipy import stats

def mean_absolute_value(data_win):
    return np.mean(abs(data_win), axis=0)


def waveform_length(data_win,  win_size):
    waveform_length = [0] * data_win.shape[1]
    for i in range(win_size - 1):
        waveform_length = waveform_length + abs(data_win[i + 1, :] - data_win[i, :])
    return np.array(waveform_length)


def zero_crossings(data_win, win_size, delta=0.01):
    col_num = data_win.shape[1]
    zero_crossing = [0] * col_num
    for col in range(col_num):
        for i in range(win_size - 1):
            if ((data_win[i][col] * data_win[i + 1][col] < 0) and (
                    abs(data_win[i + 1][col] - data_win[i][col]) > delta)):
                zero_crossing[col] += 1
    return np.array(zero_crossing)


def slope_sign_changes(data_win, win_size, delta=0.01):
    col_num = data_win.shape[1]
    slope_sign_changes = [0] * col_num
    for col in range(col_num):
        for i in range(1, win_size - 1):
            if ((((data_win[i][col] > data_win[i - 1][col]) and (data_win[i][col] > data_win[i + 1][col]))
                 or ((data_win[i][col] < data_win[i - 1][col]) and (data_win[i][col] < data_win[i + 1][col])))
                    and (abs(data_win[i][col] - data_win[i + 1][col]) >= delta or abs(
                        data_win[i][col] - data_win[i - 1][col]) >= delta)):
                slope_sign_changes[col] += 1
    return np.array(slope_sign_changes)


# 均方根值 root mean square
def root_mean_square(data_win):
    return np.sqrt(np.mean(np.square(data_win), axis=0))


def mean(data_win):
    return np.mean(data_win, axis=0)


def std(data_win):
    return np.std(data_win, axis=0, ddof=1)


def skewness(data_win):
    return stats.skew(data_win)


def kurtosis(data_win):
    return stats.kurtosis(data_win)


def median_abs_deviation(data_win):
    return stats.median_abs_deviation(data_win)


def interquartile_range(data_win):
    return stats.iqr(data_win, axis=0)


def waveform_factor(data_win, col_num):
    root_mean_square = np.sqrt(np.mean(np.square(data_win), axis=0))
    mav = np.mean(abs(data_win), axis=0)
    waveform_factor = [0] * col_num
    for col in range(col_num):
        waveform_factor[col] = root_mean_square[col] / mav[col]
    return waveform_factor


