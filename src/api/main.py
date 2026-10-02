"""
FastAPI app for SentinelML fraud detection.
No /content/drive paths here on purpose — this file must run unchanged
inside Docker later. All paths are relative / handled by FraudPredictor
(with MODEL_PATH env var override).
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.schemas.transaction import TransactionInput
from src.schemas.prediction import PredictionOutput
from src.models.predictor import FraudPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinelml")

# TODO: 0.5 is a PLACEHOLDER. Needs tuning from Phase 1's PR curve
# (precision/recall tradeoff) once that analysis is revisited — not final.
DEFAULT_THRESHOLD = 0.5

app = FastAPI(title="SentinelML Fraud Detection API", version="0.1.0")

# Loaded ONCE at import time (module-level singleton) — not per-request.
predictor = FraudPredictor()


def _extract_probability(raw_output):
    """
    Defensive adapter: FraudPredictor.predict() might return a plain
    float/numpy number, OR a dict (e.g. {"probability": 0.02} or
    {"fraud_probability": 0.02}) depending on how Phase 2 implemented it.

    This normalizes either shape into a single float so /predict doesn't
    crash. TODO: once predictor.py's exact return type is confirmed,
    this adapter can be simplified/removed.
    """
    if isinstance(raw_output, dict):
        for key in ("probability", "fraud_probability", "proba", "score"):
            if key in raw_output:
                return float(raw_output[key])
        raise ValueError(
            f"predict() returned a dict with unexpected keys: {list(raw_output.keys())}"
        )
    # Handles plain float, numpy.float64, etc.
    return float(raw_output)


@app.get("/health")
def health():
    """
    Honest health check — never returns 'healthy' if the model
    actually failed to load.
    """
    if not predictor.is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "error": predictor.get_load_error(),
            },
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": DEFAULT_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {predictor.get_load_error()}",
        )

    features = transaction.to_feature_array()

    try:
        raw_output = predictor.predict(features)
        probability = _extract_probability(raw_output)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from e

    return PredictionOutput(
        fraud_probability=probability,
        is_fraud=bool(probability >= DEFAULT_THRESHOLD),
        threshold_used=DEFAULT_THRESHOLD,
        # TODO: hardcoded for now — pull real version from model.joblib
        # metadata once that's added, instead of a string literal.
        model_version="v1-lightgbm",
    )
