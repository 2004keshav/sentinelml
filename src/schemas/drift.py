"""Request/response schemas for the /drift-check endpoint."""
from pydantic import BaseModel, ConfigDict

# Must match FEATURE_COLUMNS in src/monitoring/drift.py exactly.
_FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class DriftRow(BaseModel):
    """
    One raw feature row for drift checking. Explicit fields (not a
    generic dict) so extra='forbid' actually rejects unexpected keys.
    """
    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class DriftCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[DriftRow]


class FeatureDrift(BaseModel):
    psi: float
    status: str


class DriftCheckResponse(BaseModel):
    n_live_rows: int
    n_reference_rows: int
    overall_psi: float
    overall_status: str
    per_feature: dict[str, FeatureDrift]
