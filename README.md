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

1. **Scan** — `python scripts/01_scan.py`
2. **Preprocess** — `python scripts/02_preprocess.py`
3. **Train classifier** — `python scripts/05_train_hybrid.py`
4. **Train regressor** — `python scripts/06_train_frt.py`
5. **Case analysis** — `python scripts/12_case_analyzer.py --config config.yaml --beam I1_B1`
6. **Dashboard** — `streamlit run src/rcbeam_fire/dashboard/app.py`

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- Schema validation (`test_schema.py`)
- Classification/regression metric thresholds (`test_metrics.py`)
- Case analyzer payload structure (`test_case_analyzer.py`)
- Dashboard resource loading (`test_dashboard.py`)

See `docs/TESTING_PLAN.md` for the expansion roadmap.
