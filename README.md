# SentinelML — Real-Time Fraud Detection API with Drift Monitoring & CI/CD

A production-style fraud detection system built on the Kaggle Credit Card Fraud dataset (284,807 transactions, ~0.17% fraud rate). The project covers the full lifecycle: model training, a served API, automated testing, CI/CD, data drift monitoring, and cloud deployment.

**Live API:** https://sentinelml-i9dd.onrender.com
*(Hosted on Render's free tier — the instance spins down after ~15 minutes of inactivity, so the first request after idle time may take 30–50 seconds to respond while it wakes up.)*

---

## Overview

Fraud detection datasets are extremely imbalanced (here, fraud is ~0.17% of transactions), which makes naive accuracy metrics meaningless and makes honest failure handling in a serving system non-negotiable. This project is built around two principles:

1. **The model has to actually work on the minority class** — evaluated with PR-AUC, Precision, Recall, and F1, not accuracy.
2. **The API must never lie about its own health.** Every endpoint fails honestly (HTTP 503 with a real error message) if the model or monitoring component isn't loaded — there is no fallback to a fake "success" response.

## Architecture

```
Client
  │
  ▼
FastAPI app (src/api/main.py)
  ├── /health        → reports true load status of model + drift detector
  ├── /predict        → validates input (Pydantic, extra="forbid") → LightGBM → fraud probability
  └── /drift-check     → compares a batch of live rows against a real reference sample (PSI)
  │
  ▼
Docker image → GitHub Actions CI (lint → test → build) → Render (cloud deploy)
```

**Key design choices:**
- **Pydantic schemas with `extra="forbid"`** on all inputs — malformed or unexpected fields are rejected at the API boundary, not silently dropped.
- **Drift detection uses PSI (Population Stability Index)** computed against a real 5,000-row sample of the training data (`data/reference_sample.csv`) — never synthetic or randomly generated baseline data.
- **Graceful degradation, not silent failure.** The model and drift detector are loaded once at startup; if loading fails, the relevant endpoints return `503` with the actual error, instead of crashing the whole app or returning a fake healthy status.

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Returns true load status of the model. `200` if ready, `503` with error detail if not. |
| `/predict` | POST | Accepts a transaction (30 features: `Time`, `V1`–`V28`, `Amount`), returns fraud probability, binary flag, and threshold used. |
| `/drift-check` | POST | Accepts a batch of raw feature rows, returns per-feature and overall PSI scores against the reference baseline. |

Example `/predict` request body:
```json
{
  "Time": 406.0, "V1": -2.31, "V2": 1.95, "...": "...",
  "V28": -0.14, "Amount": 0.0
}
```

## Model

- **Algorithm:** LightGBM classifier
- **Evaluation (held-out test set):** PR-AUC **0.6087**, Precision **0.633**, Recall **0.7041**, F1 **0.6667**
- **Validation sanity check:** on the full set of 492 known-fraud transactions in the dataset, the model correctly flags 463 (94.1%) at the current threshold — included as a qualitative check, not a substitute for the held-out test metrics above, since it includes rows the model saw during training.

## Tech Stack

- **Model:** LightGBM, scikit-learn
- **API:** FastAPI, Pydantic v2, Uvicorn
- **Testing:** pytest (6 tests — API health/predict, drift self-comparison, drift endpoint, schema rejection)
- **CI/CD:** GitHub Actions (ruff lint → pytest → Docker build, all on every push to `main`)
- **Containerization:** Docker
- **Deployment:** Render (free tier)
- **Data handling:** pandas, NumPy

## Running Locally

```bash
git clone https://github.com/2004keshav/sentinelml.git
cd sentinelml
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Running with Docker

```bash
docker build -t sentinelml .
docker run -p 8000:8000 sentinelml
```

### Running tests

```bash
pytest
ruff check .
```

## CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow (`.github/workflows/ci.yml`) with three stages, run in order:

1. **Lint** — `ruff` checks code style and catches common errors
2. **Test** — full `pytest` suite (6 tests covering API and drift monitoring)
3. **Docker build** — confirms the image builds cleanly from the Dockerfile

All three stages must pass before a deploy is considered valid.

## Known Limitations

Being transparent about what's *not* done, rather than overstating scope:

- **Decision threshold (0.5) is a placeholder** — it has not yet been tuned against the precision/recall tradeoff from Phase 1's PR curve. This is next on the list.
- **Drift checks on very small batches (e.g. a single row) are not statistically meaningful** — PSI is designed for comparing distributions, so meaningful drift checks require a reasonably sized batch of live data.
- **CI validates that the Docker image builds, but does not run it** — a runtime-only failure (a missing system library, in one case during this project) can pass CI and only surface after deployment. `/health` honestly reporting `503` in that situation is what caught it.
- **Free-tier hosting** means the live demo sleeps after inactivity; not representative of production latency.

## Project Structure

```
sentinelml/
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── data/
│   ├── reference_sample.csv      # real 5,000-row sample of training data
│   └── baseline_stats.json
├── models/
│   └── model.joblib
├── src/
│   ├── api/main.py
│   ├── models/predictor.py
│   ├── monitoring/drift.py
│   └── schemas/
│       ├── transaction.py
│       ├── prediction.py
│       └── drift.py
└── tests/
    ├── test_api.py
    └── test_drift.py
```
