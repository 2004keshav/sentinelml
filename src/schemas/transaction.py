"""
Input schema for a single credit card transaction.
Matches the 30 features used to train the LightGBM model:
Time, V1...V28 (PCA components), Amount.
"""

from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",        # reject unknown fields instead of silently ignoring them
        str_strip_whitespace=True,
    )

    Time: float = Field(..., description="Seconds elapsed since first transaction in dataset")
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
    Amount: float = Field(..., ge=0, description="Transaction amount, must be non-negative")

    def to_feature_array(self) -> list[float]:
        """
        Returns features in the EXACT column order the model was trained on.
        This order must match X_train.columns from Phase 1 training.
        """
        return [
            self.Time, self.V1, self.V2, self.V3, self.V4, self.V5, self.V6,
            self.V7, self.V8, self.V9, self.V10, self.V11, self.V12, self.V13,
            self.V14, self.V15, self.V16, self.V17, self.V18, self.V19, self.V20,
            self.V21, self.V22, self.V23, self.V24, self.V25, self.V26, self.V27,
            self.V28, self.Amount,
        ]
