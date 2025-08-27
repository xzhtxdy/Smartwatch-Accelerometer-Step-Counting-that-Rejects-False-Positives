import numpy as np
import scipy
def check_mean(flattened_peaks_list,array,f_array,co):
    filtered_list = []
    flattened_peaks_list_con=np.concatenate(flattened_peaks_list)
    array_flatten = array.flatten()  # 确保array是一维的
    for n in flattened_peaks_list_con:
        if (array_flatten[n] - f_array[n])/f_array[n] > co:
             filtered_list.append(n)
    return np.array(filtered_list)  # 返回一个NumPy数组

def find_peaks(array, method=None, thre=0.1,**kwargs):
    """
    Finds peaks in an array.
    :param array: array to find peaks in
    :return: indices of peaks
    """
    peaks_list = []
    if method == "cwt":
        for i in range(array.shape[1]):
            peaks = array.find_peaks_cwt(array[:, i], kwargs['width'])
            peaks_list.append(peaks)
    elif method == "argrelextrema":
        for i in range(array.shape[1]):
            peaks = scipy.signal.argrelextrema(array[:, i], np.greater,order=kwargs['order'])
            # print(peaks)
            peaks = peaks[0]
            peaks_list.append(peaks)
    elif method == "pingding":
        signal = array.flatten()
        peaks_list = []
        plateau_start = None

        for i in range(1, len(signal) - 1):
            if signal[i] > thre and signal[i - 1] <= thre:
                # Start of a new plateau
                plateau_start = i
            elif signal[i] > thre and signal[i + 1] <= thre:
                # End of a plateau
                if plateau_start is not None:
                    # Calculate the midpoint of the plateau as the peak position
                    peak_index = (plateau_start + i) // 2
                    peaks_list.append(peak_index)
                    plateau_start = None  # Reset for the next plateau




    else:
        for i in range(array.shape[1]):
            peaks, _ = signal.find_peaks(array[:, i])
            peaks_list.append(peaks)
    return peaks_list
