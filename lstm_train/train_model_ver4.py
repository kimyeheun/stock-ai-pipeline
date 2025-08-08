# NOTE: Developed train model

import os
import wandb

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from lstm_train.Config import TrainingConfig
from lstm_train.Dataset import NpyDataset
from lstm_train.Model_ver2 import MaskAwareTCNTransformer


def split_file_data_triplet(x_files, y_files, mask_files, test_size=0.2, shuffle=True, seed=42):
     N = len(x_files)
     indices = np.arange(N)
     if shuffle:
         rng = np.random.default_rng(seed)
         rng.shuffle(indices)
     split = int(N * (1 - test_size))
     train_idx, val_idx = indices[:split], indices[split:]
     take = lambda lst, idx: [lst[i] for i in idx]
     return (
         take(x_files, train_idx), take(x_files, val_idx),
         take(y_files, train_idx), take(y_files, val_idx),
         take(mask_files, train_idx), take(mask_files, val_idx),
     )

def compute_class_weights(y_files, device):
    all_labels = []
    for y_path in y_files:
        y = np.load(y_path, mmap_mode='r')
        all_labels.append(y)
    all_labels = np.concatenate(all_labels)
    classes, counts = np.unique(all_labels, return_counts=True)
    total = counts.sum()
    weights = total / (len(classes) * counts)  # 역비율
    return torch.tensor(weights, dtype=torch.float32).to(device)

def initialize_model(input_dim, output_dim, device, hidden_dim=64, class_weights=None):
    model = MaskAwareTCNTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        tcn_channels=128,
        tcn_kernel_size=5,
        tcn_drop=0.1,
        tcn_dilations=(1, 2, 4, 8),
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        transformer_drop=0.1,
        mlp_hidden=128,
        mlp_drop=0.2,
    ).to(device)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    return model, criterion, optimizer

def train_and_validate_model(model, criterion, optimizer, epochs, device, clip_grad_norm=1.0):
    for epoch in tqdm(range(epochs), desc="Epoch"):
        # ---------- Training ----------
        model.train()
        total_loss, total_correct, total_count = 0, 0, 0
        for x_path, mask_path, y_path in zip(X_train, mask_train, y_train):
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

        # ---------- Validation ----------
        model.eval()
        val_loss, val_correct, val_count = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_path, mask_path, y_path in zip(X_val, mask_val, y_val):
                val_dataset = NpyDataset(x_path, mask_path, y_path)
                val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)
                for xb, mask, yb in val_loader:
                    xb, mask, yb = xb.to(device), mask.to(device), yb.to(device)
                    out = model(xb, mask)
                    loss = criterion(out, yb)
                    val_loss += loss.item() * xb.size(0)
                    preds = out.argmax(1)
                    val_correct += (preds == yb).sum().item()
                    val_count += xb.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(yb.cpu().numpy())

        val_acc = val_correct / val_count
        val_loss = val_loss / val_count

        # ---------- Metrics ----------
        report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
        f1_sell = report.get("2", {}).get("f1-score", 0.0)  # Sell 클래스 F1

        print(f"Epoch {epoch+1}: "
              f"train_loss={avg_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
              f"f1_sell={f1_sell:.4f}")
        print(classification_report(all_labels, all_preds, digits=4))

        wandb.log({
            "train/loss": avg_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/f1_sell": f1_sell,  # Sell 클래스 F1 기록
            "epoch": epoch+1
        })

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

    X_train, X_val, y_train, y_val, mask_train, mask_val = split_file_data_triplet(
        X_files, y_files, mask_files,
        test_size = TrainingConfig.TEST_SIZE,
        shuffle = TrainingConfig.SHUFFLE_DATA,
        seed = 42,
    )
    print(f"Train chunks: {len(X_train)}, Val chunks: {len(X_val)}")

    # 2. 데이터 모양 확인
    x_sample = np.load(X_train[0], mmap_mode='r')
    y_sample = np.load(y_train[0], mmap_mode='r')
    N, T, F = x_sample.shape # (데이터 개수, 윈도우 크기, 피쳐 개수)
    output_dim = len(np.unique(y_sample)) # 0, 1, 2 (3개)
    print(f"Data Shape: {x_sample.shape}, Output Shape: {output_dim}")

    # 3. wandb init
    wandb.init(
        project="Stock_LSTM",
        config={
            "epochs": TrainingConfig.EPOCHS,
            "batch_size": TrainingConfig.BATCH_SIZE,
            "learning_rate": TrainingConfig.LEARNING_RATE,
            "clip_grad_norm": TrainingConfig.CLIP_GRAD_NORM,
            "model": "MaskAwareTCNTransformer",
            "input_dim": F,
            "output_dim": output_dim,
            "selected_features": TrainingConfig.ALL_FEATURES,
        }
    )

    # 4. Model Init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weights(y_files, device)
    model, criterion, optimizer = initialize_model(F, output_dim, device, class_weights=class_weights)
    print(f"Device: {device}")

    # 5. Train
    train_and_validate_model(model, criterion, optimizer, TrainingConfig.EPOCHS, device, TrainingConfig.CLIP_GRAD_NORM)

    # 6. Save
    save_model(model, TrainingConfig.MODEL_PATH)
