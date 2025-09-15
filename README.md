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
3. **Train classifier** — `python scripts/05_train_hybrid.py` — XGBoost + LDAM hybrid
4. **Train regressor** — `python scripts/06_train_frt.py` — LightGBM fire resistance time model
5. **Case analysis** — `python scripts/12_case_analyzer.py --config config.yaml --beam I1_B1`

## Cross-validation & ablation

```bash
python scripts/05_train_hybrid.py --cv          # grouped 5-fold CV
python scripts/05_train_hybrid.py --ablate       # objective/resampler grid search
python scripts/05_train_hybrid.py --bootstrap 1000  # bootstrap CIs
```

## Case analysis

The case analyzer generates JSON reports with SHAP-informed recommendations:

```bash
python scripts/12_case_analyzer.py --config config.yaml --beam I1_B1
```

## Model adapters

`rcbeam_fire.utils.model_adapters` provides `ClassifierAdapter` and `RegressorAdapter`
wrappers for standardised inference across blend and single-model packs.
