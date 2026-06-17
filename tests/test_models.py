import torch

from battery_rul.models.dnn import PaperDNN, UpgradedDNN
from battery_rul.models.lstm import BatteryLSTM


def test_paper_dnn_forward_shape() -> None:
    model = PaperDNN(input_dim=5)
    out = model(torch.randn(32, 5))
    assert out.shape == (32, 1)


def test_upgraded_dnn_forward_shape_various_configs() -> None:
    for layer_sizes in ([16], [32, 32], [8, 16, 8]):
        model = UpgradedDNN(input_dim=9, layer_sizes=layer_sizes)
        out = model(torch.randn(4, 9))
        assert out.shape == (4, 1)


def test_upgraded_dnn_residual_connection_shape() -> None:
    model = UpgradedDNN(input_dim=9, layer_sizes=[9, 9])
    out = model(torch.randn(4, 9))
    assert out.shape == (4, 1)


def test_lstm_forward_shape() -> None:
    model = BatteryLSTM(input_size=9)
    out = model(torch.randn(32, 50, 9))
    assert out.shape == (32, 1)


def test_parameter_counts_reasonable() -> None:
    paper_params = sum(p.numel() for p in PaperDNN(input_dim=5).parameters())
    upgraded_params = sum(
        p.numel() for p in UpgradedDNN(input_dim=9, layer_sizes=[32, 32]).parameters()
    )
    lstm_params = sum(p.numel() for p in BatteryLSTM(input_size=9).parameters())

    assert 0 < paper_params < 1_000
    assert 0 < upgraded_params < 100_000
    assert 0 < lstm_params < 1_000_000
