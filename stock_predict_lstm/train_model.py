import joblib
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.preprocessing import StandardScaler

from Model import MaskAwareLSTM
from stock_predict_lstm.Config import TrainingConfigBefore as TrainingConfig


def load_data(x_path, y_path):
    return np.load(x_path), np.load(y_path)

def mask_features(X, all_features, selected_features):
    idx = [all_features.index(f) for f in selected_features]
    return X[:, :, idx]

def split_data(X, y, test_size=0.2, shuffle=True):
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    split = int(N * (1 - test_size))
    train_idx, val_idx = indices[:split], indices[split:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    # mask 생성: NaN 위치를 0, 사용 가능한 feature는 1
    train_mask = ~np.isnan(X_train)
    val_mask = ~np.isnan(X_val)
    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(train_mask, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(val_mask, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    )

def initialize_model(input_dim, output_dim, device, hidden_dim=64):
    model = MaskAwareLSTM(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    return model, criterion, optimizer

def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, clip_grad_norm=1.0):
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_count = 0, 0, 0
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


# --- 메인 실행부 ---
if __name__ == "__main__":
    # 1. Load Data
    X, y = load_data(TrainingConfig.X_PATH, TrainingConfig.Y_PATH)
    X = mask_features(X, TrainingConfig.ALL_FEATURES, TrainingConfig.ALL_FEATURES)

    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

    # 4. Save scaler
    joblib.dump(scaler, TrainingConfig.SCALER_PATH)

    # Determine input and output dimensions from result
    input_dim = X.shape[2]
    output_dim = len(np.unique(y))  # Assuming labels are 0, 1, 2 for Buy, Sell, Hold
    print(np.unique(y))  # y의 유니크값 확인
    print(f"y.min={y.min()}, y.max={y.max()}, output_dim={output_dim}")

    # 2. Split Data
    X_train, X_val, y_train, y_val = split_data(
        X, y,
        test_size=TrainingConfig.TEST_SIZE,
        shuffle=TrainingConfig.SHUFFLE_DATA
    )
    print("학습셋 라벨 분포:", np.bincount(y_train.astype(int)))

    # 3. Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=TrainingConfig.BATCH_SIZE
    )

    # 4. Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, optimizer = initialize_model(input_dim, output_dim, device)

    # 4.5. Initialize wandb
    wandb.init(
        project="lstm-stock-classifier-100",
        config={
            "epochs": TrainingConfig.EPOCHS,
            "batch_size": TrainingConfig.BATCH_SIZE,
            "learning_rate": TrainingConfig.LEARNING_RATE,
            "clip_grad_norm": TrainingConfig.CLIP_GRAD_NORM,
            "model": "MaskAwareLSTM",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "selected_features": TrainingConfig.ALL_FEATURES,
        }
    )

    # 5. Train Model
    train_and_validate_model(
        model,
        train_loader, val_loader,
        criterion, optimizer,
        epochs=TrainingConfig.EPOCHS,
        device=device,
        clip_grad_norm=TrainingConfig.CLIP_GRAD_NORM
    )

    # 6. Save Model
    save_model(model, TrainingConfig.MODEL_PATH)
