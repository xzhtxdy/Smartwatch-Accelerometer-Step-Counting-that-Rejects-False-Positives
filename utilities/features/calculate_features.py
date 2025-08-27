import warnings
import numpy as np
from numpy.fft import rfft
from scipy import stats
from scipy.fftpack import fft
from scipy.stats import entropy

warnings.filterwarnings("ignore")


# 时域特征

def __mean_absolute_value(data_win):
    return np.mean(abs(data_win), axis=0)


def __waveform_length(data_win,  win_size):
    waveform_length = [0] * data_win.shape[1]
    for i in range(win_size - 1):
        waveform_length = waveform_length + abs(data_win[i + 1, :] - data_win[i, :])
    return np.array(waveform_length)



def __zero_crossings(data_win, win_size, delta=0.01):
    col_num = data_win.shape[1]
    zero_crossing = [0] * col_num
    for col in range(col_num):
        for i in range(win_size - 1):
            if ((data_win[i][col] * data_win[i + 1][col] < 0) and (
                    abs(data_win[i + 1][col] - data_win[i][col]) > delta)):
                zero_crossing[col] += 1
    return np.array(zero_crossing)


def __slope_sign_changes(data_win, win_size, delta=0.01):
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
def __root_mean_square(data_win):
    return np.sqrt(np.mean(np.square(data_win), axis=0))


def __mean(data_win):
    return np.mean(data_win, axis=0)


def __std(data_win):
    return np.std(data_win, axis=0, ddof=1)


def __skewness(data_win):
    return stats.skew(data_win)


def __kurtosis(data_win):
    return stats.kurtosis(data_win)


def __median_abs_deviation(data_win):
    return stats.median_abs_deviation(data_win)


def __interquartile_range(data_win):
    return stats.iqr(data_win, axis=0)


def __waveform_factor(data_win, col_num):
    root_mean_square = np.sqrt(np.mean(np.square(data_win), axis=0))
    mav = np.mean(abs(data_win), axis=0)
    waveform_factor = [0] * col_num
    for col in range(col_num):
        waveform_factor[col] = root_mean_square[col] / mav[col]
    return waveform_factor


# 频域特征

def __spectral_peak(data_win,  win_size):
    N = data_win.shape[0]  # N=win_size
    mean = np.mean(data_win, axis=0)
    data_win = data_win - mean
    spectral = rfft(data_win, axis=0)
    abs_spectral = np.abs(spectral) / N
    abs_spectral_half = abs_spectral[:win_size // 2, :]
    index_spectral_peak = np.argmax(abs_spectral_half, axis=0)
    spectral_peak_frequency = np.max(abs_spectral_half, axis=0)

    return index_spectral_peak, spectral_peak_frequency

def __energy(data_win):
    return np.sum(np.square(data_win), axis=0)

def __entropy(data_win):
    # 使用每列的概率分布计算熵
    entropies = [entropy(np.histogram(data_win[:, col], bins=10, density=True)[0]) for col in range(data_win.shape[1])]
    return np.array(entropies)

def __correlation_coefficients(data_win):
    corr_matrix = np.corrcoef(data_win.T)  # 计算相关系数矩阵
    corr_coeffs = []
    # 获取上三角部分的相关系数（避免重复计算）
    for i in range(data_win.shape[1]):
        for j in range(i + 1, data_win.shape[1]):
            corr_coeffs.append(corr_matrix[i, j])
    return np.array(corr_coeffs)
def __fft_bins(data_win):
    N = data_win.shape[0]
    fft_values = np.fft.fft(data_win, axis=0)  # 计算FFT
    fft_magnitude = np.abs(fft_values)[:N // 2, :]  # 取频谱的前一半
    return np.mean(fft_magnitude, axis=0)  # 计算每个通道的FFT均值

# def __wavelet_coefficients(data_win, wavelet='db4', level=3):
#     coeffs = [pywt.wavedec(data_win[:, col], wavelet, level=level) for col in range(data_win.shape[1])]
#     # 提取近似和细节系数
#     wavelet_features = np.array([np.concatenate([c[0] for c in coeffs], axis=0)])  # 近似系数
#     return np.mean(wavelet_features, axis=0)  # 取均值作为特征
def __spectral_entropy(data_win, win_size):
    # 计算FFT
    N = data_win.shape[0]
    fft_values = np.fft.fft(data_win, axis=0)  # 对每个通道做FFT
    fft_magnitude = np.abs(fft_values)[:N // 2, :]  # 取正频率部分
    power_spectrum = np.square(fft_magnitude)  # 功率谱

    # 归一化功率谱
    power_spectrum_sum = np.sum(power_spectrum, axis=0)
    normalized_power_spectrum = power_spectrum / power_spectrum_sum

    # 计算谱熵
    entropy = -np.sum(normalized_power_spectrum * np.log2(normalized_power_spectrum + 1e-10), axis=0)  # 避免log(0)
    return entropy
def __power_ratio(data_win, win_size, low_freq=0, mid_freq=10, samplerate=25):
    N = data_win.shape[0]
    fft_values = np.fft.fft(data_win, axis=0)  # 计算FFT
    fft_magnitude = np.abs(fft_values)[:N // 2, :]  # 取正频率部分
    power_spectrum = np.square(fft_magnitude)  # 功率谱

    # 计算频带的功率
    freqs = np.fft.fftfreq(N, 1/samplerate)[:N // 2]  # 频率对应
    low_freq_range = np.logical_and(freqs >= low_freq, freqs < 4)  # 0-4 Hz
    mid_freq_range = np.logical_and(freqs >= 4, freqs < 10)  # 4-10 Hz

    # 计算每个频带的功率
    low_band_power = np.sum(power_spectrum[low_freq_range, :], axis=0)
    mid_band_power = np.sum(power_spectrum[mid_freq_range, :], axis=0)

    # 计算功率比
    total_power = low_band_power + mid_band_power
    low_power_ratio = low_band_power / total_power
    mid_power_ratio = mid_band_power / total_power

    return low_power_ratio,mid_power_ratio


def triaxial_cross_correlation(data_win):
    # data_win 是一个 shape 为 (win_size, 3) 的数组，表示三轴数据
    # data_win[:, 0] -> x 轴数据
    # data_win[:, 1] -> y 轴数据
    # data_win[:, 2] -> z 轴数据

    # 计算 x, y, z 轴之间的交叉相关
    Cxy = np.corrcoef(data_win[:, 0], data_win[:, 1])[0, 1]  # x 与 y 轴的相关性
    Cxz = np.corrcoef(data_win[:, 0], data_win[:, 2])[0, 1]  # x 与 z 轴的相关性
    Cyz = np.corrcoef(data_win[:, 1], data_win[:, 2])[0, 1]  # y 与 z 轴的相关性

    # 返回三对轴之间的交叉相关系数
    return Cxy, Cxz, Cyz


def maximum_cross_correlation(data_win):
    """
    计算三轴信号之间交叉相关的最大值
    """
    Cxy, Cxz, Cyz = triaxial_cross_correlation(data_win)

    # 计算最大交叉相关值
    max_cross_correlation = np.max([abs(Cxy), abs(Cxz), abs(Cyz)])

    return max_cross_correlation
def get_features(data_win):
    win_size= data_win.shape[0]

    mav = __mean_absolute_value(data_win)
    rms = __root_mean_square(data_win)
    mean = __mean(data_win)

    # waveform_length = __waveform_length(data_win,  win_size)
    waveform_length = __waveform_length(data_win, win_size)
    zero_crossings = __zero_crossings(data_win,  win_size)
    slope_sign_changes = __slope_sign_changes(data_win,  win_size)

    std = __std(data_win)
    energy = __energy(data_win)
    entropy_values = __entropy(data_win)
    # correlation_coeffs = __correlation_coefficients(data_win)
    spectral_entropy = __spectral_entropy(data_win, win_size)
    # Power Ratio of Different Frequency Bands (Updated for LF: 0-4 Hz and MF: 4-10 Hz)
    low_power_ratio, mid_power_ratio = __power_ratio(data_win, win_size, low_freq=0, mid_freq=10, samplerate=25)
    # Triaxial Cross-Correlation
    Cxy, Cxz, Cyz = triaxial_cross_correlation(data_win)
    # 计算最大交叉相关值
    max_cross_correlation_value = maximum_cross_correlation(data_win)

    skewness = __skewness(data_win)
    kurtosis = __kurtosis(data_win)
    median_abs_deviation = __median_abs_deviation(data_win)
    inter_quartile_range = __interquartile_range(data_win)

    features = np.concatenate((mav, rms, mean,
                               zero_crossings, slope_sign_changes,
                                kurtosis, spectral_entropy, [Cxy], [Cxz], [Cyz],
                              ))

    return features

def get_features2(data_win):
    win_size, col_num = data_win.shape

    mav_features = []
    rms_features = []
    mean_features = []
    waveform_length_mean_features = []
    zero_crossings_features = []
    slope_sign_changes_features = []
    std_features = []
    skewness_features = []
    kurtosis_features = []
    median_abs_deviation_features = []

    for col in range(col_num):
        data_col = data_win[:, col]
        data_col_2d = data_col.reshape(-1, 1)
        mav = __mean_absolute_value(data_col_2d)
        rms = __root_mean_square(data_col_2d)
        mean = __mean(data_col_2d)
        waveform_length = __waveform_length(data_col_2d, win_size)
        zero_crossings = __zero_crossings(data_col_2d, win_size)
        slope_sign_changes = __slope_sign_changes(data_col_2d, win_size)
        std = __std(data_col_2d)
        skewness = __skewness(data_col_2d)
        kurtosis = __kurtosis(data_col_2d)
        median_abs_deviation = __median_abs_deviation(data_col_2d)

        mav_features.append(mav)
        rms_features.append(rms)
        mean_features.append(mean)
        waveform_length_mean_features.append(waveform_length)
        zero_crossings_features.append(zero_crossings)
        slope_sign_changes_features.append(slope_sign_changes)
        std_features.append(std)
        skewness_features.append(skewness)
        kurtosis_features.append(kurtosis)
        median_abs_deviation_features.append(median_abs_deviation)

    # 将所有列的特征连接起来
    features = np.concatenate((mav_features, rms_features, mean_features, waveform_length_mean_features, zero_crossings_features, slope_sign_changes_features, std_features))
    features = features.reshape(1, -1)

    return features