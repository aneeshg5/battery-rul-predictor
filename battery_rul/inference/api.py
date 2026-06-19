"""FastAPI REST endpoint for real-time SOH inference.

uvicorn battery_rul.inference.api:app --reload --port 8000
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from battery_rul.config import (
    MODEL_PATH,
    MODEL_VERSION,
    SCALER_PATH,
    TEST_BATTERIES,
    TRAIN_BATTERY,
    UPGRADED_DNN_RMSE,
)
from battery_rul.inference.predictor import Predictor

logger = logging.getLogger(__name__)

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _predictor
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        _predictor = Predictor(MODEL_PATH, SCALER_PATH)
    else:
        logger.warning("Model artifact not found at %s; /predict will return 503", MODEL_PATH)
    yield


app = FastAPI(title="Battery RUL Predictor API", lifespan=lifespan)


def get_predictor() -> Predictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return _predictor


class PredictRequest(BaseModel):
    voltage_history: list[float] = Field(min_length=2)
    current: float
    temperature: float
    step_type: Literal["charge", "discharge", "rest"]


class PredictResponse(BaseModel):
    soh: float
    soh_percent: float
    rul_estimate: str
    confidence: str
    model_version: str


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "architecture": "UpgradedDNN",
        "rmse": UPGRADED_DNN_RMSE,
        "training_battery": TRAIN_BATTERY,
        "test_batteries": TEST_BATTERIES,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest, predictor: Predictor = Depends(get_predictor)  # noqa: B008
) -> PredictResponse:
    result = predictor.predict(
        request.voltage_history, request.current, request.temperature, request.step_type
    )
    soh = float(result["soh"])
    return PredictResponse(
        soh=soh,
        soh_percent=soh * 100,
        rul_estimate=str(result["rul_estimate"]),
        confidence=str(result["confidence"]),
        model_version=MODEL_VERSION,
    )
