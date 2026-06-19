"""Self-attention sequence model: learns temporal structure directly from raw windows
instead of relying on hand-engineered rolling-window features (the LSTM's alternative)."""

import torch
from torch import nn

from battery_rul.config import (
    ATTENTION_D_MODEL,
    ATTENTION_DIM_FEEDFORWARD,
    ATTENTION_DROPOUT,
    ATTENTION_NHEAD,
    ATTENTION_NUM_LAYERS,
    WINDOW_SIZE,
)


class BatteryAttention(nn.Module):
    """Transformer encoder over a feature-history window, predicting SOH from the final step."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = ATTENTION_D_MODEL,
        nhead: int = ATTENTION_NHEAD,
        num_layers: int = ATTENTION_NUM_LAYERS,
        dim_feedforward: int = ATTENTION_DIM_FEEDFORWARD,
        dropout: float = ATTENTION_DROPOUT,
        max_seq_len: int = WINDOW_SIZE,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        x = self.encoder(x)
        return self.head(x[:, -1, :])
