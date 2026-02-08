"""
저혈압 조기 예측 모델: 1D-CNN + LSTM.
"""
import torch
import torch.nn as nn


class HypotensionModel(nn.Module):
    """
    1D-CNN + LSTM 기반 저혈압 이진 분류 모델.
    입력: (batch, num_features) 또는 (batch, seq_len, num_features).
    """

    def __init__(
        self,
        in_dim: int,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        # 1D-CNN: (B, 1, F) -> (B, conv_channels, F)
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # LSTM 입력: (B, F, conv_channels) -> seq_len=F, input_size=conv_channels
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) or (B, T, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, F)
        # (B, 1, F) -> conv -> (B, C, F)
        x = self.conv(x)
        # (B, C, F) -> (B, F, C) for LSTM (seq_len=F, input_size=C)
        x = x.permute(0, 2, 1)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (B, lstm_hidden)
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


class HypoNet(nn.Module):
    """저혈압 이진 분류용 MLP (기존 호환)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
