FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the packages the API actually needs at runtime
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir \
    fastapi>=0.115 \
    "uvicorn[standard]>=0.32" \
    pandas==2.3.3 \
    numpy==2.3.5 \
    scikit-learn==1.8.0 \
    xgboost==3.1.3 \
    lightgbm==4.6.0 \
    joblib>=1.4 \
    pyyaml>=6.0 \
    pyarrow>=12.0

# Copy source code — use PYTHONPATH instead of pip install -e .
COPY src/ src/
COPY config.yaml .
COPY data/processed/dataset.parquet data/processed/dataset.parquet
COPY models/checkpoints/ models/checkpoints/
COPY api/ api/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
