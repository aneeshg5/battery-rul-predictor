"""DNN architectures: the paper's exact baseline and a configurable upgraded version."""

import torch
from torch import nn

from battery_rul.config import HIDDEN_LAYERS


class PaperDNN(nn.Module):
    """Exact replica of the paper's 2-hidden-layer DNN: ReLU then Sigmoid, linear output."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden1, hidden2 = HIDDEN_LAYERS
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.Sigmoid(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpgradedDNN(nn.Module):
    """Configurable DNN: Linear -> BatchNorm1d -> activation -> Dropout per hidden layer,
    with a residual add whenever a block's input and output dims match."""

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activation = nn.ReLU()
        in_dim = input_dim
        for size in layer_sizes:
            self.linears.append(nn.Linear(in_dim, size))
            self.norms.append(nn.BatchNorm1d(size))
            self.dropouts.append(nn.Dropout(dropout))
            in_dim = size
        self.output = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, norm, dropout in zip(self.linears, self.norms, self.dropouts, strict=True):
            residual = x
            x = dropout(self.activation(norm(linear(x))))
            if x.shape == residual.shape:
                x = x + residual
        return self.output(x)
