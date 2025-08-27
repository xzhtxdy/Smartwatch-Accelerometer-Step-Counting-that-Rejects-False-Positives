import matplotlib.pyplot as plt
from utilities.config import plot_config


def plot_change_along_epoches(accuracy, loss):
    fig, axes = plt.subplots(2, 1, figsize=plot_config["figsize"], sharex=True)
    axes[0].plot(accuracy)
    axes[0].set_ylabel("Accuracy")
    axes[1].plot(loss)
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    return fig
