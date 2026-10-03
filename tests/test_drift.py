"""
Tests for the drift monitoring module and /drift-check endpoint.
Uses ONLY real rows from data/reference_sample.csv - no synthetic data,
consistent with the project's testing philosophy.
"""
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.monitoring.drift import DriftDetector

REFERENCE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "reference_sample.csv"
)

client = TestClient(app)


@pytest.fixture
def reference_df():
    return pd.read_csv(REFERENCE_PATH)


def test_drift_detector_self_comparison_is_stable(reference_df):
    """Reference data compared against a random half of itself should show ~no drift."""
    detector = DriftDetector()
    live_sample = reference_df.sample(frac=0.5, random_state=42)
    report = detector.compute_drift_report(live_sample)
    assert report["overall_status"] == "stable"
    assert report["overall_psi"] < 0.1


def test_drift_detector_flags_real_shifted_slice(reference_df):
    """
    A genuinely skewed real slice (top 20% by Amount) should show
    higher PSI on the Amount feature than a random slice does.
    """
    detector = DriftDetector()
    extreme_slice = reference_df.sort_values("Amount").tail(
        int(len(reference_df) * 0.2)
    )
    report = detector.compute_drift_report(extreme_slice)
    assert report["per_feature"]["Amount"]["psi"] > 0.1


def test_drift_check_endpoint_with_real_rows(reference_df):
    """Hits /drift-check with real reference rows, expects a valid report back."""
    sample_rows = reference_df.sample(n=50, random_state=1).to_dict(orient="records")
    response = client.post("/drift-check", json={"rows": sample_rows})
    assert response.status_code == 200
    body = response.json()
    assert body["n_live_rows"] == 50
    assert "overall_psi" in body
    assert "per_feature" in body


def test_drift_check_endpoint_rejects_extra_fields(reference_df):
    """extra='forbid' should reject rows with unexpected keys, same as TransactionInput."""
    bad_row = reference_df.sample(n=1, random_state=1).to_dict(orient="records")[0]
    bad_row["unexpected_field"] = 123
    response = client.post("/drift-check", json={"rows": [bad_row]})
    assert response.status_code == 422
