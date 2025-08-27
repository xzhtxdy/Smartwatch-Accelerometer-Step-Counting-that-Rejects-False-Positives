import numpy as np
import pandas as pd
from scipy import signal
from ..config import data_config
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.signal import butter, filtfilt
def normalize(array):
    sc = StandardScaler()
    array = sc.fit_transform(array)
    return array, sc

def norm_of_vector(acc, offset=0):
    norm = np.zeros((acc.shape[0], 1))
    for i in range(acc.shape[0]):
        norm[i] = (acc[i][0] ** 2 + acc[i][1] ** 2 + acc[i][2] ** 2)
    return norm - offset

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    对数据应用带通滤波器。

    参数:
    - data: 要过滤的数据，假设是n行1列的NumPy数组。
    - lowcut: 低频截止频率。
    - highcut: 高频截止频率。
    - fs: 采样频率。
    - order: 滤波器的阶数，默认为5。

    返回值:
    - 过滤后的数据。
    """
    nyq = 0.5 * fs  # Nyquist频率
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0, padlen=0)
    return filtered_data

def filter(array, filter_type, filter_order, filter_cutoff):
    """
    Filters data with a Butterworth filter.
    :param array: data to be filtered
    :param filter_type: 'lowpass' or 'highpass'
    :param filter_order: 1, 2, 3, 4, etc.
    :param filter_cutoff: cutoff frequency in Hz
    :return: filtered data
    """
    fs = int(data_config['sampling_frequency'])
    wn = filter_cutoff / (fs / 2)
    if filter_type == "lowpass":
        b, a = signal.butter(filter_order, wn, btype='lowpass', analog=False)
    elif filter_type == "highpass":
        b, a = signal.butter(filter_order, wn, btype='highpass', analog=False)
    else:
        raise ValueError("filter_type must be 'lowpass' or 'highpass'")
    return signal.filtfilt(b, a, array, axis=0)

def moving_average_filter(data, window_size):
    """
    使用简单的移动平均滤波器对数据进行平滑处理。

    参数:
    - data: 一维numpy数组，输入数据。
    - window_size: 整数，移动平均的窗口大小。

    返回值:
    - 平滑后的数据，numpy数组。
    """
    # 使用NumPy的convolve函数实现移动平均滤波
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data

def fft(array, fs):
    """
    Performs a Fast Fourier Transform on data.
    :param array: data to be transformed
    :param fs: sampling frequency
    :return: frequency and magnitude arrays
    """
    l = array.shape[0]
    N = np.power(2, np.ceil(np.log2(l)))
    fre = np.fft.fftfreq(int(N), 1 / fs)
    mag= np.abs(np.fft.fft(array, int(N), axis=0))
    mag = mag[:int(N / 2), :]
    fre = fre[:int(N / 2)]
    return fre, mag


def integral(array):
    """
    Calculates the integral of data.
    :param array: data to be integrated
    :return: integrated data
    """
    return np.cumsum(array - np.mean(array, axis=0), axis=0)


def concatenate(data):
    """
    Concatenates data into one array.
    :param data: dict of data
    :return: concatenated data
    """
    return np.concatenate([data[key] for key in data.keys()], axis=0)


def encode_labels_integers(labels, categories=None):
    """
    Encodes labels as integers.
    :param labels: labels to be encoded
    :return: encoded labels
    """
    if categories is None:
        categories = data_config['labels']
    return np.vectorize(categories.index)(labels)


def encode_labels_one_hot(labels):
    """
    Encodes labels as one-hot vectors.
    :param labels: series-like, labels to be encoded
    :return: encoded labels
    """
    enc = OneHotEncoder()
    enc.fit([["left"],
             ["right"],
             ["leftshift"],
             ["rightshift"]])
    return enc.transform(labels.reshape(-1, 1)).toarray()

def fill_df(df: pd.DataFrame):
    """
    Fill the missing values in the dataframe.
    :param df:
    :return:
    """
    end_index = df.index[-1]
    # The indexes of the missing values.
    new_df = pd.DataFrame(index=range(df.index[0], end_index + 1))
    new_df = new_df.join(df)
    return new_df
