import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.main import app

client = TestClient(app)

def test_health_returns_200_when_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True

def test_predict_with_real_reference_row():
    df = pd.read_csv("data/reference_sample.csv")
    row = df.iloc[0].to_dict()
    row.pop("Class", None)

    response = client.post("/predict", json=row)
    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
