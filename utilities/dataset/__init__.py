import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, target, window_size, stride):
        self.data = torch.tensor(data, dtype=torch.float32, device='cuda')
        self.target = torch.tensor(target, dtype=torch.float32, device='cuda')
        self.window_size = window_size
        self.stride = stride
        self.start_idx = np.arange(0, self.data.shape[0] - self.window_size, self.stride)

    def __len__(self):
        return len(self.start_idx)

    def __getitem__(self, index):
        idx = self.start_idx[index]
        window_data = self.data[idx:idx + self.window_size].T
        return window_data, self.target[idx + self.window_size]
