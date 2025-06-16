# RC Beam Fire Performance Prediction

Machine learning pipeline for predicting the fire resistance performance of
reinforced concrete (RC) beams strengthened with fibre-reinforced polymers (FRP).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Place the dataset Excel file at `data/raw/dataset.xlsx`. The expected sheet
name and column mapping are configured in `config.yaml`.

## Pipeline stages

1. **Scan** — `python scripts/01_scan.py` — EDA summary and visualisations
2. **Preprocess** — `python scripts/02_preprocess.py` — clean and persist data
