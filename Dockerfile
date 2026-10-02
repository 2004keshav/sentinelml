FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/model.joblib ./models/model.joblib
COPY data/reference_sample.csv ./data/reference_sample.csv
COPY data/baseline_stats.json ./data/baseline_stats.json

ENV MODEL_PATH=/app/models/model.joblib

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
