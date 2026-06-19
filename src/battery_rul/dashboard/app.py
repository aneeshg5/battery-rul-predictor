"""Plotly Dash dashboard: voltage trace, SOH gauge, SOH overlay, training curve, model table.

All charts read from precomputed parquet files in data/processed/predictions/ — no model
inference happens at request time (see scripts/precompute_predictions.py).
"""

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from battery_rul.config import PAPER_BASELINE_RMSE, PREDICTIONS_DIR

APPROACH_BATTERIES = {1: ["RW9"], 2: ["RW10", "RW11", "RW12"]}
APPROACH_MODELS = {
    1: [{"label": "Paper DNN", "value": "paper_dnn"}],
    2: [
        {"label": "Paper DNN", "value": "paper_dnn"},
        {"label": "Upgraded DNN", "value": "upgraded_dnn"},
        {"label": "LSTM", "value": "lstm"},
        {"label": "Attention", "value": "attention"},
        {"label": "LightGBM", "value": "lightgbm"},
    ],
}
MODEL_LABELS = {
    "paper_dnn": "Paper DNN (ours)",
    "upgraded_dnn": "Upgraded DNN (ours)",
    "lstm": "LSTM (ours)",
    "attention": "Attention (ours)",
    "lightgbm": "LightGBM (ours)",
}
MODELS_WITHOUT_TRAINING_CURVE = {"lightgbm"}
GAUGE_STEPS = [
    {"range": [0, 80], "color": "#f8d7da"},
    {"range": [80, 90], "color": "#fff3cd"},
    {"range": [90, 100], "color": "#d4edda"},
]


def load_predictions(model: str, approach: int, battery: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{model}_approach{approach}_{battery}.parquet"
    return pd.read_parquet(path)


def load_history(model: str, approach: int) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{model}_approach{approach}_history.parquet"
    return pd.read_parquet(path)


def build_comparison_rows() -> list[tuple[str, float]]:
    """Avg RMSE (Approach 2) per model, ours computed from precomputed predictions."""
    rows = list(PAPER_BASELINE_RMSE.items())
    for model in ("paper_dnn", "upgraded_dnn", "lstm", "attention", "lightgbm"):
        rmses = []
        for battery in APPROACH_BATTERIES[2]:
            df = load_predictions(model, 2, battery)
            rmses.append(((df["soh_actual"] - df["soh_predicted"]) ** 2).mean() ** 0.5)
        rows.append((MODEL_LABELS[model], sum(rmses) / len(rmses)))
    return rows


def build_comparison_table() -> html.Table:
    rows = build_comparison_rows()
    best_rmse = min(rmse for _, rmse in rows)
    header = html.Tr([html.Th("Model"), html.Th("Avg RMSE (Approach 2)")])
    body = []
    for name, rmse in rows:
        style = {"fontWeight": "bold", "color": "#155724"} if rmse == best_rmse else {}
        body.append(
            html.Tr([html.Td(name, style=style), html.Td(f"{rmse * 100:.2f}%", style=style)])
        )
    return html.Table([header, *body], className="comparison-table")


def soh_gauge_color(soh_percent: float) -> str:
    if soh_percent < 80:
        return "#dc3545"
    if soh_percent < 90:
        return "#ffc107"
    return "#28a745"


def make_voltage_figure(df: pd.DataFrame, battery: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=df["absolute_time_raw"], y=df["voltage_raw"], mode="lines", name="Voltage")
    )
    fig.update_layout(
        title=f"Voltage vs Time — {battery}",
        xaxis_title="Absolute time (s)",
        yaxis_title="Voltage (V)",
    )
    return fig


def make_soh_figure(df: pd.DataFrame, battery: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["absolute_time_raw"], y=df["soh_actual"] * 100, mode="lines", name="Actual SOH"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["absolute_time_raw"],
            y=df["soh_predicted"] * 100,
            mode="lines",
            name="Predicted SOH",
        )
    )
    fig.update_layout(
        title=f"SOH vs Time — {battery}", xaxis_title="Absolute time (s)", yaxis_title="SOH (%)"
    )
    return fig


def make_gauge_figure(soh_percent: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=soh_percent,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": soh_gauge_color(soh_percent)},
                "steps": GAUGE_STEPS,
            },
            title={"text": "Latest Predicted SOH"},
        )
    )
    fig.update_layout(height=300)
    return fig


def make_training_curve_figure(history: pd.DataFrame, model: str, approach: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["epoch"], y=history["train_rmse"], mode="lines+markers", name="Train RMSE"
        )
    )
    fig.add_trace(
        go.Scatter(x=history["epoch"], y=history["val_rmse"], mode="lines+markers", name="Val RMSE")
    )
    fig.update_layout(
        title=f"Training Curve — {model} (Approach {approach})",
        xaxis_title="Epoch",
        yaxis_title="RMSE",
    )
    return fig


app = Dash(__name__)
app.title = "Battery RUL Predictor"

app.layout = html.Div(
    [
        html.H1("Battery RUL Predictor — Live SOH Dashboard"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Approach"),
                        dcc.RadioItems(
                            id="approach-radio",
                            options=[
                                {"label": "1 (intra-battery)", "value": 1},
                                {"label": "2 (cross-battery)", "value": 2},
                            ],
                            value=2,
                        ),
                        html.Label("Model"),
                        dcc.Dropdown(id="model-dropdown"),
                        html.Label("Battery"),
                        dcc.Dropdown(id="battery-dropdown"),
                    ],
                    className="sidebar",
                ),
                html.Div(
                    [
                        dcc.Graph(id="voltage-graph"),
                        dcc.Graph(id="gauge-graph"),
                        dcc.Graph(id="soh-graph"),
                        dcc.Graph(id="training-curve-graph"),
                        html.H3("Model Comparison"),
                        build_comparison_table(),
                    ],
                    className="main-panel",
                ),
            ],
            className="content",
        ),
    ]
)


@app.callback(
    Output("model-dropdown", "options"),
    Output("model-dropdown", "value"),
    Input("approach-radio", "value"),
)
def update_model_options(approach: int) -> tuple[list[dict], str]:
    options = APPROACH_MODELS[approach]
    return options, options[0]["value"]


@app.callback(
    Output("battery-dropdown", "options"),
    Output("battery-dropdown", "value"),
    Input("approach-radio", "value"),
)
def update_battery_options(approach: int) -> tuple[list[dict], str]:
    batteries = APPROACH_BATTERIES[approach]
    return [{"label": b, "value": b} for b in batteries], batteries[0]


@app.callback(
    Output("voltage-graph", "figure"),
    Output("gauge-graph", "figure"),
    Output("soh-graph", "figure"),
    Output("training-curve-graph", "figure"),
    Input("approach-radio", "value"),
    Input("model-dropdown", "value"),
    Input("battery-dropdown", "value"),
)
def update_charts(
    approach: int, model: str, battery: str
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    if not model or not battery:
        empty = go.Figure()
        return empty, empty, empty, empty
    df = load_predictions(model, approach, battery)
    latest_soh_percent = float(df["soh_predicted"].iloc[-1] * 100)
    # SOH from the quantile-based formula can fall outside [0, 100] on noisy samples
    # (see CHANGELOG.md Phase 2); clamp only the gauge display, not the line charts.
    gauge_percent = min(max(latest_soh_percent, 0.0), 100.0)
    if model in MODELS_WITHOUT_TRAINING_CURVE:
        training_curve_figure = go.Figure().update_layout(
            title=f"Training Curve — {model} (boosting rounds, not epochs; no per-round log)"
        )
    else:
        training_curve_figure = make_training_curve_figure(
            load_history(model, approach), model, approach
        )
    return (
        make_voltage_figure(df, battery),
        make_gauge_figure(gauge_percent),
        make_soh_figure(df, battery),
        training_curve_figure,
    )
