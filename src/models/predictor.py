"""
Inference wrapper around the trained LightGBM fraud model.
Loads the model once and exposes a predict() method.
No hardcoded absolute paths — resolves relative to this file's location,
so the same code works in Colab AND inside a Docker container.
"""

import os
from pathlib import Path

import joblib
import pandas as pd

# Path resolution logic:
# This file lives at:        <project_root>/src/models/predictor.py
# The model lives at:        <project_root>/models/model.joblib
# So we go up 2 levels from this file (models/ -> src/ -> project_root)
# then into models/model.joblib.
#
# An env var override is included because in Docker, the project root
# might be mounted differently — this keeps the code portable without
# ever hardcoding /content/drive/... inside src/.
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "model.joblib"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH))

# Threshold is a module-level constant, not buried in logic.
# NOTE: 0.5 is a placeholder. Given ~0.17% fraud rate, this MUST be
# re-tuned using the PR curve from Phase 1 before production use.
DEFAULT_THRESHOLD = 0.5

MODEL_VERSION = "lgbm_v1"

# Must match X_train.columns order from Phase 1 training EXACTLY.
# Passing a DataFrame with these names (instead of a raw list) to
# predict_proba() eliminates the "X does not have valid feature names"
# warning AND removes a silent column-order-mismatch risk.
FEATURE_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount",
]


class FraudPredictor:
    """
    Loads the model lazily and caches it in memory.
    One instance should be created at API startup and reused
    across requests — NOT reloaded per request (I/O + deserialization
    cost would wreck latency on every call).
    """

    def __init__(self, model_path: Path = MODEL_PATH, threshold: float = DEFAULT_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self._load_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self._load_error = f"Model file not found at: {self.model_path}"
            return
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:  # noqa: BLE001 — intentionally broad: any load failure should degrade gracefully
            self._load_error = f"Failed to load model: {e}"

    def is_ready(self) -> bool:
        """Used by /health in Phase 3 — must reflect TRUE model state, no faking."""
        return self.model is not None

    def get_load_error(self) -> str | None:
        return self._load_error

    def predict(self, features: list[float]) -> dict:
        if not self.is_ready():
            raise RuntimeError(f"Model not loaded: {self._load_error}")

        if len(features) != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Expected {len(FEATURE_COLUMNS)} features, got {len(features)}"
            )

        # DataFrame with named columns, in training order — prevents the
        # sklearn "X does not have valid feature names" warning and guards
        # against silent column-order mismatches.
        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        proba = float(self.model.predict_proba(X)[0][1])  # prob of class=1 (fraud)
        return {
            "fraud_probability": proba,
            "is_fraud": proba >= self.threshold,
            "threshold_used": self.threshold,
            "model_version": MODEL_VERSION,
        }
