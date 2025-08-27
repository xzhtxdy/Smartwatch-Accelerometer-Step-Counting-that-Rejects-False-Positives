import matplotlib
matplotlib.use("TkAgg")  # 确保交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from time import sleep
from utilities.data import reader
from utilities.data.preprocessing import norm_of_vector
from utilities.features.extract_features import extract_single_features
from utilities.labeling import find_peaks
from utilities.evaluate_classifier import predict_with_bagging_model, load_bagging_model
from utilities import generate_dataset, parameters
import pandas as pd

def demo_visualization(interval=1.5):

    # 参数
    project_path = parameters["project_dir"]
    data_dir = parameters["data_dir"]
    window_size = parameters["win_len"]
    stride = parameters["win_stride"]

    # 数据
    path = os.path.join(project_path, 'data', 'walking_data', 'test.txt')
    data = reader.read_file(path, **{"delimiter": '\t', "header": None}).iloc[:, 0:3].to_numpy()
    norm = norm_of_vector(data)

    # 模型加载
    loaded_model = load_bagging_model(os.path.join(project_path, 'result', 'wsbocsvm_model.pkl'))

    # 步数记录
    total_peaks = 0
    step_counts = [0]
    time_axis = [0]

    # 打开交互模式
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    # 设置全屏
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()  # Qt5Agg / QtAgg
    except Exception:
        try:
            manager.window.state('zoomed')  # TkAgg
        except Exception:
            pass

    num_frames = math.ceil((len(norm) - window_size) / stride) + 1

    for frame_i in range(num_frames):
        i = frame_i * stride
        segment = norm[i:i + window_size]
        segment_raw = data[i:i + window_size]

        # 特征提取
        pd_Features = extract_single_features(segment_raw)

        # 分类
        pred = predict_with_bagging_model(loaded_model, pd_Features)

        # 峰值检测
        peaks = ([],)
        if pred == 1:
            mean_val = np.mean(segment)
            order = 7 if mean_val < 1.2 else 4
            peaks = find_peaks(segment, method='argrelextrema', order=order)
            total_peaks += len(peaks[0])

        step_counts.append(total_peaks)
        time_axis.append(i + window_size)
        print(total_peaks)

        # === 清空子图 ===
        for ax in axs:
            ax.clear()

        # (1) 原始信号 + 当前窗口
        axs[0].plot(data[:, 0], label="ax", color="r", alpha=0.6)
        axs[0].plot(data[:, 1], label="ay", color="g", alpha=0.6)
        axs[0].plot(data[:, 2], label="az", color="b", alpha=0.6)
        axs[0].axvspan(i, i + window_size, color="yellow", alpha=0.8)
        axs[0].set_title(f"Sliding Window on Raw 3-axis Signal ({i} to {i + window_size})", fontsize=16)
        axs[0].legend()

        # (2) 特征
        pd_Features = pd.DataFrame(pd_Features)
        pd_Features.iloc[0].plot(kind="bar", ax=axs[1])
        axs[1].set_title("Extracted Features for Current Window", fontsize=16)

        # (3) Walking / Non-Walking + 峰值
        if pred == 1 and len(peaks[0]) > 0:
            axs[2].plot(segment, color="black", label="Magnitude")
            axs[2].plot(peaks[0], segment[peaks[0]], "ro", label="Detected Peaks")
            axs[2].set_title(f"[Prediction result: Walking] Step counts = {len(peaks[0])}", fontsize=16)
            axs[2].legend()
        else:
            axs[2].plot(segment, color="black", label="Magnitude")
            axs[2].set_title("[Prediction result: Non-Walking] Step counts = 0", fontsize=16, color='red')
            axs[2].legend()

        # (4) 步数累积
        axs[3].plot(time_axis, step_counts, marker="o")
        axs[3].set_title("Accumulated Step Counts Over Time", fontsize=16)
        axs[3].set_xlabel("Time Index")
        axs[3].set_ylabel("Steps")
        # 在点上标注数值
        for x, y in zip(time_axis, step_counts):
            axs[3].text(x, y, str(y), ha='center', va='bottom', fontsize=15)
        plt.tight_layout()
        fig.canvas.draw()
        plt.pause(interval)

    # 保持最终图像
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    demo_visualization(interval=2)
