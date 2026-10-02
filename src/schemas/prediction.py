"""
Output schema returned by the fraud detection API.
"""

from pydantic import BaseModel, Field


class PredictionOutput(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    threshold_used: float = Field(..., description="Decision threshold applied, for auditability")
    model_version: str = Field(..., description="Identifier of the model that produced this prediction")
