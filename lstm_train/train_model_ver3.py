# NOTE: train model for chunked DataSet

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from Dataset import NpyDataset
from Model import MaskAwareLSTM
from lstm_train.Config import TrainingConfig


def split_file_data(x_files, y_files, test_size=0.2, shuffle=True):
    N = len(x_files)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    split = int(N * (1 - test_size))
    train_idx, val_idx = indices[:split], indices[split:]
    x_train = [x_files[i] for i in train_idx]
    y_train = [y_files[i] for i in train_idx]
    x_val = [x_files[i] for i in val_idx]
    y_val = [y_files[i] for i in val_idx]
    return x_train, x_val, y_train, y_val

def initialize_model(input_dim, output_dim, device, hidden_dim=64):
    model = MaskAwareLSTM(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    return model, criterion, optimizer

def train_and_validate_model(model, criterion, optimizer, epochs, device, clip_grad_norm=1.0):
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        total_loss, total_correct, total_count = 0, 0, 0
        for x_path, mask_path, y_path in zip(X_train, mask_train, y_train):
            X = np.load(x_path, mmap_mode='r')
            y = np.load(y_path, mmap_mode='r')
            mask = np.load(mask_path, mmap_mode='r')
            if not (X.shape[0] == y.shape[0] == mask.shape[0]):
                print(f"Skipping chunk: {x_path}, mismatch shape")
                continue

            train_dataset = NpyDataset(x_path, mask_path, y_path)
            train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)

            for xb, mask, yb in train_loader:
                xb, mask, yb = xb.to(device), mask.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb, mask)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                total_correct += (out.argmax(1) == yb).sum().item()
                total_count += xb.size(0)
        train_acc = total_correct / total_count
        avg_loss = total_loss / total_count

        # Validation
        model.eval()
        val_loss, val_correct, val_count = 0, 0, 0
        with torch.no_grad():
            for x_path, mask_path, y_path in zip(X_val, mask_val, y_val):
                X = np.load(x_path, mmap_mode='r')
                y = np.load(y_path, mmap_mode='r')
                mask = np.load(mask_path, mmap_mode='r')
                if not (X.shape[0] == y.shape[0] == mask.shape[0]):
                    print(f"Skipping chunk: {x_path}, mismatch shape")
                    continue

                val_dataset = NpyDataset(x_path, mask_path, y_path)
                val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)
                for xb, mask, yb in val_loader:
                    xb, mask, yb = xb.to(device), mask.to(device), yb.to(device)
                    out = model(xb, mask)
                    loss = criterion(out, yb)
                    val_loss += loss.item() * xb.size(0)
                    val_correct += (out.argmax(1) == yb).sum().item()
                    val_count += xb.size(0)
        val_acc = val_correct / val_count
        val_loss = val_loss / val_count

        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        wandb.log({"train/loss": avg_loss, "train/acc": train_acc, "val/loss": val_loss, "val/acc": val_acc, "epoch": epoch+1})

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # 1. Chunk 파일 리스트 수집
    file_indices = sorted(set(
        int(fname.split('_')[1].split('.')[0])
        for fname in os.listdir(TrainingConfig.SPLIT_PATH)
        if fname.startswith("X_")
    ))
    X_files = [os.path.join(TrainingConfig.SPLIT_PATH, f"X_{i:03d}.npy") for i in file_indices]
    y_files = [os.path.join(TrainingConfig.SPLIT_PATH, f"y_{i:03d}.npy") for i in file_indices]
    mask_files = [os.path.join(TrainingConfig.SPLIT_PATH, f"mask_{i:03d}.npy") for i in file_indices]

    if len(X_files) != len(y_files) or len(X_files) != len(mask_files):
        raise ValueError("X/y/mask 파일 개수 불일치")

    X_train, X_val, y_train, y_val = split_file_data(X_files, y_files, test_size=TrainingConfig.TEST_SIZE, shuffle=TrainingConfig.SHUFFLE_DATA)
    mask_train, mask_val, _, _ = split_file_data(mask_files, y_files, test_size=TrainingConfig.TEST_SIZE, shuffle=TrainingConfig.SHUFFLE_DATA)
    print(f"Train chunks: {len(X_train)}, Val chunks: {len(X_val)}")

    # 2. 데이터 모양 확인
    x_sample = np.load(X_train[0], mmap_mode='r')
    y_sample = np.load(y_train[0], mmap_mode='r')
    N, T, F = x_sample.shape # (데이터 개수, 윈도우 크기, 피쳐 개수)
    output_dim = len(np.unique(y_sample)) # 0, 1, 2 (3개)

    # 3. Model Init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, optimizer = initialize_model(F, output_dim, device)

    # 8. wandb
    wandb.init(
        project="Stock_LSTM",
        config={
            "epochs": TrainingConfig.EPOCHS,
            "batch_size": TrainingConfig.BATCH_SIZE,
            "learning_rate": TrainingConfig.LEARNING_RATE,
            "clip_grad_norm": TrainingConfig.CLIP_GRAD_NORM,
            "model": "MaskAwareLSTM",
            "input_dim": F,
            "output_dim": output_dim,
            "selected_features": TrainingConfig.ALL_FEATURES,
        }
    )

    # 9. Train
    train_and_validate_model(model, criterion, optimizer, TrainingConfig.EPOCHS, device, TrainingConfig.CLIP_GRAD_NORM)

    # 10. Save
    save_model(model, TrainingConfig.MODEL_PATH)
