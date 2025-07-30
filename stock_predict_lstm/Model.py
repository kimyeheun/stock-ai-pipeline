import torch
import torch.nn as nn

class MaskedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # buy_offset, sell_offset
        )

    def forward(self, x, mask):
        x = x * mask  # 마스킹 적용
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # 마지막 timestep 기준

class MaskAwareLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim * 2, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )
    def forward(self, x, mask):
        # x: [B, T, F], mask: [B, T, F]
        # NaN 처리: NaN은 0으로 대체 (학습에서 mask가 0이면 무시됨)
        x_proc = torch.nan_to_num(x, nan=0.0)
        lstm_in = torch.cat([x_proc, mask], dim=2)  # [B, T, F*2]
        lstm_out, _ = self.lstm(lstm_in)
        output = self.head(lstm_out[:, -1])  # <-- 여기!
        return output
