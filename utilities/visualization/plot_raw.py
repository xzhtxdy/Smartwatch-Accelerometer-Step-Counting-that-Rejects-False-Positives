import matplotlib.pyplot as plt
import pandas as pd
from utilities.config import general_config, data_config, plot_config
import numpy as np


def plot_raw_data(data, title=None, ylabel=None, save_path=None):
    """
    Plot raw data and save the plot to the specified path.
    """
    x = np.arange(data.shape[0]) / data_config['sampling_frequency']
    data = np.array(data)
    if data.shape[1] == 1:
        fig = plt.figure(figsize=plot_config['figsize'])
        plt.plot(data)
    else:
        fig, axes = plt.subplots(data.shape[1], 1, figsize=plot_config['figsize'], sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(x, data[:, i])
            if ylabel:
                ax.set_ylabel(ylabel[i], fontdict=plot_config['font_label'])
    if title:
        fig.suptitle(title, fontdict=plot_config['font_title'])
    else:
        fig.suptitle("Raw data", fontdict=plot_config['font_title'])
    if save_path:
        fig.savefig(save_path)
    plt.xlabel("Time(s)", fontdict=plot_config['font_label'])
    return fig

def plot_frequency_domain(fre, mag, title="Frequency Domain", ylabel=None, save_path=None):
    """
    Plot frequency domain of data and save the plot to the specified path.
    """
    fig = plt.figure(figsize=plot_config['figsize'])
    if mag.shape[1] == 1:
        plt.plot(fre, mag)
    else:
        for i in range(mag.shape[1]):
            ax = fig.add_subplot(mag.shape[1], 1, i+1)
            ax.plot(fre, mag[:, i], label=i)
    fig.suptitle(title, fontdict=plot_config['font_title'])
    if save_path:
        fig.savefig(save_path)
    plt.xlabel("Frequency(Hz)", fontdict=plot_config['font_label'])
    return fig

def plot_raw_data_with_peaks(array, peaks, title=None, ylabel=None, save_path=None):
    """
    Plot raw data and save the plot to the specified path.
    """
    fig = plot_raw_data(array, title, ylabel, save_path)
    if array.shape[1] == 1:
        fig.axes[0].plot(peaks[0], array[peaks[0]], "o")
    else:
        for i, ax in enumerate(fig.axes):
            ax.plot(peaks[i], array[peaks[i], i], "o")
    return fig