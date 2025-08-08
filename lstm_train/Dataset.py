import numpy as np
import torch
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    def __init__(self, X_path, mask_path, y_path):
        self.X = np.load(X_path, mmap_mode='r')
        self.mask = np.load(mask_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.length = min(self.X.shape[0], self.mask.shape[0], self.y.shape[0])

        if not (self.X.shape[0] == self.mask.shape[0] == self.y.shape[0]):
            print(f"[WARN] Length mismatch: X={self.X.shape[0]}, mask={self.mask.shape[0]}, y={self.y.shape[0]} → use length={self.length}")

    def __len__(self):
        # return self.X.shape[0]
        return self.length

    def __getitem__(self, idx):
        # 원하는 인덱스 데이터만 numpy→torch로 변환해서 반환
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.mask[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )
