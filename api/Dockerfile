FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY api/requirements.txt api/requirements.txt
COPY config.yaml .
COPY data/processed/dataset.parquet data/processed/dataset.parquet
COPY models/checkpoints/ models/checkpoints/

RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
