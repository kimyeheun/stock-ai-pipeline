import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_preprocessing import DataPreProcessingConfig  # 경로에 따라 조정


def plot_label_distribution(y: np.ndarray):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, palette="Set2", hue=None)
    plt.title("Label Distribution")
    plt.xlabel("Label (0 = Hold, 1 = Buy, 2 = Sell)")
    plt.ylabel("Count")
    plt.xticks([0, 1, 2])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = DataPreProcessingConfig
    X_path = config.X_PATH
    X = np.load(X_path)
    print(f"Loaded features shape: {X.shape}")  # (N, T, F)
    y_path = config.Y_PATH

    # 라벨 데이터 불러오기
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"{y_path} 파일이 존재하지 않습니다. 먼저 데이터 전처리를 실행하세요.")

    y = np.load(y_path)
    print(f"Loaded labels shape: {y.shape}")

    # 라벨 분포 시각화
    plot_label_distribution(y)
