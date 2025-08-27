from scipy import signal
import numpy as np


def butter_lowpass(order, cutoff, fs, data):
    """
    同MATLAB中的butter：数字低通巴特沃斯滤波器，得到滤波器系数
    :param cutoff: 截止频率(Hz)
    :param fs: 采样频率(Hz)
    :param order: 滤波器阶数
    :return: b分子，a分母
    数字低通巴特沃斯滤波过程：对一维数据滤波（只需调用这一个函数即可滤波）
    :param data: 一维数组，二维行/列数组，一维list(下面的return与之对应)
    """
    # nyq = 0.5 * fs                  # 信号频率
    # normal_cutoff = cutoff / nyq    # 归一化截止频率=截止频率/信号频率，即MATLAB中butter 的 Wn
    Wn = 2 * cutoff / fs
    b, a = signal.butter(order, Wn, 'low', analog=False)   # 原本butter只能输出一个值，但用2个变量接收也无报错
    # y = signal.lfilter(b, a, data)
    y = signal.filtfilt(b, a, data)
    return y  # Filter requirements


def butter_lowpass_filter(data, cutoff, fs, order=3):
    """
    数字低通巴特沃斯滤波过程：对一维数据滤波（只需调用这一个函数即可滤波）
    :param data: 一维数组，二维行/列数组，一维list(下面的return与之对应)
    :param cutoff:
    :param fs:
    :param order:
    :return: 一维数组，二维行/列数组，一维数组
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y  # Filter requirements.


def smooth(a, WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))
