import numpy as np
import torch
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    def __init__(self, X_path, mask_path, y_path):
        # np.load with mmap_mode='r' → RAM에 전체를 올리지 않고 인덱스 접근시만 로딩
        self.X = np.load(X_path, mmap_mode='r')
        self.mask = np.load(mask_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # 원하는 인덱스 데이터만 numpy→torch로 변환해서 반환
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.mask[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )
