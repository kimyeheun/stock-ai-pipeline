import os

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data_collection.Config import DataPreProcessingConfig


def load_data(x_path, y_path):
    return np.load(x_path), np.load(y_path)

def save_chunked(X, y, mask, chunk_size=100_000, save_dir="./data/split_data2"):
    os.makedirs(save_dir, exist_ok=True)
    N = X.shape[0]
    num_chunks = (N + chunk_size - 1) // chunk_size

    for i in tqdm(range(num_chunks), decs="save chunks"):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, N)
        np.save(os.path.join(save_dir, f"X_{i:03d}.npy"), X[start:end])
        np.save(os.path.join(save_dir, f"y_{i:03d}.npy"), y[start:end])
        np.save(os.path.join(save_dir, f"mask_{i:03d}.npy"), mask[start:end])
        print(f"Saved chunk {i+1}/{num_chunks}: [{start}:{end}]")

def mask_features(X, all_features, selected_features):
    idx = [all_features.index(f) for f in selected_features]
    return X[:, :, idx]

def process_and_save_chunks(X_raw, y, feature_list, scaler_path, chunk_size=100_000, save_dir="./data/split_data"):
    os.makedirs(save_dir, exist_ok=True)
    N, T, F = X_raw.shape

    # 1. partial_fit 전체 통계량 계산
    scaler = StandardScaler()
    for i in tqdm(range(0, N, chunk_size), desc="Scaler partial_fit"):
        X_chunk = mask_features(X_raw[i:i+chunk_size], feature_list, feature_list)
        X_chunk = X_chunk.astype(np.float32)
        X_chunk = np.nan_to_num(X_chunk)
        X_flat = X_chunk.reshape(-1, F)
        scaler.partial_fit(X_flat)
    joblib.dump(scaler, scaler_path)
    print("Scaler saved.")

    # 2. transform + mask + 저장
    for i in tqdm(range(0, N, chunk_size), desc="Save chunks"):
        start = i
        end = min(i+chunk_size, N)
        X_chunk = mask_features(X_raw[start:end], feature_list, feature_list).astype(np.float32)
        mask_chunk = mask_features(~np.isnan(X_raw[start:end]), feature_list, feature_list).astype(np.float32)
        X_chunk = np.nan_to_num(X_chunk)
        X_flat = X_chunk.reshape(-1, F)
        X_scaled = scaler.transform(X_flat).reshape(-1, T, F)
        np.save(os.path.join(save_dir, f"X_{i//chunk_size:03d}.npy"), X_scaled[:end-start])
        np.save(os.path.join(save_dir, f"y_{i//chunk_size:03d}.npy"), y[start:end])
        np.save(os.path.join(save_dir, f"mask_{i//chunk_size:03d}.npy"), mask_chunk[:end-start])


def main():
    # 1. 데이터 로드
    X_raw, y = load_data(DataPreProcessingConfig.X_PATH, DataPreProcessingConfig.Y_PATH)
    # 마스킹 + chunk 단위 저장
    process_and_save_chunks(X_raw, y,
                            DataPreProcessingConfig.ALL_FEATURES,
                            DataPreProcessingConfig.SCALER_PATH,
                            chunk_size=100_000, save_dir=DataPreProcessingConfig.SPLIT_PATH)
    print("All chunks saved.")


if __name__=="__main__":
    main()
