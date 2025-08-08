import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTE: MaskedLSTM
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

# NOTE: MaskAwareLSTM
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

# NOTE: TCN MODEL

class Chomp1d(nn.Module):
    """Causal padding 보정: Conv1d의 right padding을 잘라내 인과성 유지"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: [B, C, L]
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN의 기본 residual 블록: Dilated Conv → GELU → Dropout × 2 + Residual"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal right padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        # x: [B, C, L]
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.drop2(out)

        res = self.downsample(x)
        y = out + res  # [B, C, L]
        # LayerNorm은 [B, L, C]에 적용하므로 transpose
        y = y.transpose(1, 2)  # [B, L, C]
        y = self.norm(y)
        y = y.transpose(1, 2)  # [B, C, L]
        return y


class TCN(nn.Module):
    """다중 dilation 스택"""
    def __init__(self, in_ch, channels, kernel_size=5, dropout=0.1, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers = []
        prev = in_ch
        for d in dilations:
            layers.append(TemporalBlock(prev, channels, kernel_size, d, dropout))
            prev = channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, L]
        return self.net(x)  # [B, C, L]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # [1, T, D] broadcast


class MaskAwareTCNTransformer(nn.Module):
    """
    Input [B, T, F] (+ mask [B, T, F])
      → Masked input
      → TCN(Conv1d with dilation, causal)
      → TransformerEncoder (batch_first)
      → Masked Attention Pooling (time)
      → FC → output_dim
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        tcn_channels: int = 128,
        tcn_kernel_size: int = 5,
        tcn_drop: float = 0.1,
        tcn_dilations=(1, 2, 4, 8),
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        transformer_drop: float = 0.1,
        mlp_hidden: int = 128,
        mlp_drop: float = 0.2,
    ):
        super().__init__()
        # --- 입력 전처리 ---
        # 마스크 반영: x * mask 사용 (NaN→0)
        # Conv1d 입력 채널: 피처 수(F). TCN 전에 선형 확장 없이 바로 진행.
        self.tcn = TCN(
            in_ch=input_dim,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_drop,
            dilations=tcn_dilations
        )

        # TCN 출력[C=tcn_channels] → d_model로 투영 후 Transformer
        self.proj = nn.Linear(tcn_channels, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=transformer_drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- 시간축 어텐션 풀링 (마스크 인지) ---
        self.time_attn = nn.Linear(d_model, 1)

        # --- 분류 헤드 ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(mlp_hidden, output_dim),
        )

    @staticmethod
    def _build_time_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        mask: [B, T, F] with 1/0
        returns: src_key_padding_mask [B, T] (True = pad/ignore)
        """
        # 시점 중 하나라도 관측된 피처가 있으면 유효
        time_valid = (mask.sum(dim=2) > 0)  # [B, T] bool
        # Transformer는 True가 pad(무시), False가 keep
        pad_mask = ~time_valid
        return pad_mask

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, valid_mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        scores: [B, T, 1], valid_mask: [B, T] (True=valid)
        """
        # 무효 위치는 -inf로 가려서 softmax에서 0이 되게 함
        minus_inf = torch.finfo(scores.dtype).min
        masked = scores.squeeze(-1).masked_fill(~valid_mask, minus_inf)  # [B, T]
        attn = F.softmax(masked, dim=dim)  # [B, T]
        return attn.unsqueeze(-1)  # [B, T, 1]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, T, F]
        mask: [B, T, F] (1.0=observed, 0.0=missing)
        """
        # 1) 입력 마스크 적용 + NaN 제거
        x = torch.nan_to_num(x, nan=0.0)
        x = x * mask  # feature-level masking

        B, T, F = x.shape

        # 2) TCN (Conv1d는 [B, C, L])
        x_c = x.transpose(1, 2)  # [B, F, T]
        tcn_out = self.tcn(x_c)  # [B, C, T]
        tcn_out = tcn_out.transpose(1, 2)  # [B, T, C]

        # 3) 선형투영 + 위치인코딩
        h = self.proj(tcn_out)  # [B, T, d_model]
        h = self.posenc(h)      # [B, T, d_model]

        # 4) Transformer 인코더 (시점 pad mask 사용)
        key_padding_mask = self._build_time_key_padding_mask(mask).to(h.device)  # [B, T]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # [B, T, d_model]

        # 5) 시간축 어텐션 풀링 (유효 시점만)
        valid_time = ~key_padding_mask  # [B, T] True=valid
        scores = self.time_attn(h)      # [B, T, 1]
        attn = self._masked_softmax(scores, valid_time)  # [B, T, 1]
        pooled = (h * attn).sum(dim=1)  # [B, d_model]

        # 6) 분류
        logits = self.head(pooled)  # [B, output_dim]
        return logits
