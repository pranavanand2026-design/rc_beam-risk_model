# RC Beam Fire Performance Prediction

A machine learning pipeline for predicting the fire resistance performance of
reinforced concrete (RC) beams strengthened with fibre-reinforced polymers (FRP).

The system classifies beams into failure modes (No Failure / Strength Failure /
Deflection Failure) and estimates fire resistance time (FRT) in minutes, then
generates actionable design recommendations to improve fire safety.

## Quick start

```bash
git clone https://github.com/<your-org>/rc_beam.git
cd rc_beam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at `data/raw/dataset.xlsx`, then run the full pipeline:

```bash
python scripts/run_pipeline.py
```

Or run stages individually:

```bash
python scripts/01_scan.py           # EDA: summary stats and visualisations
python scripts/02_preprocess.py     # Clean and persist data
python scripts/05_train_hybrid.py   # Train XGBoost + LDAM hybrid classifier
python scripts/06_train_frt.py      # Train LightGBM fire resistance regressor
python scripts/12_case_analyzer.py --config config.yaml --beam I1_B1
```

## Dashboard

Launch the Streamlit design studio for interactive exploration:

```bash
streamlit run src/rcbeam_fire/dashboard/app.py
```

## Cross-validation & ablation

```bash
python scripts/05_train_hybrid.py --cv              # grouped 5-fold CV
python scripts/05_train_hybrid.py --ablate           # objective/resampler grid
python scripts/05_train_hybrid.py --bootstrap 1000   # bootstrap confidence intervals
```

## Testing

```bash
pytest tests/ -v
```

| Suite | Coverage |
|-------|----------|
| `test_schema.py` | Processed dataset column presence and physical range checks |
| `test_metrics.py` | Classification macro-F1 and regression R² acceptance thresholds |
| `test_case_analyzer.py` | Case analyzer JSON payload structure and boundary inputs |
| `test_dashboard.py` | Dashboard resource-loading smoke test |

See `docs/TESTING_PLAN.md` for the expansion roadmap.

## Project structure

```
rc_beam/
├── config.yaml              # Pipeline configuration
├── requirements.txt         # Python dependencies
├── pyrightconfig.json       # IDE type-checking hints
├── src/rcbeam_fire/         # Core library
│   ├── config.py            # YAML config loader
│   ├── data/                # scan.py, preprocess.py
│   ├── models/              # blend.py (classifier), frt.py (regressor)
│   ├── analysis/            # insight.py, eurocode.py
│   ├── utils/               # io.py, model_adapters.py
│   └── dashboard/           # app.py (Streamlit)
├── scripts/                 # CLI entry points (01–12 + run_pipeline.py)
├── tests/                   # pytest suite
├── data/                    # raw/ and processed/
├── models/                  # checkpoints/ (gitignored)
├── outputs/                 # figs/ and tables/ (gitignored)
├── experiments/             # Archived trial scripts
└── docs/                    # Report, figures, presentation
```

## Configuration

All pipeline settings live in `config.yaml`:

- **paths** — directories for raw data, processed data, outputs, and models
- **data** — dataset filename, sheet name, and target column
- **analysis** — exposure/margin defaults, adjustable features, and bounds

## Model artefacts

Trained models are excluded from version control. Regenerate them by running
the pipeline scripts. See `models/README.md` for details.

## Hand-over checklist

- [ ] Clone the repo and install dependencies
- [ ] Place `dataset.xlsx` in `data/raw/`
- [ ] Run `python scripts/run_pipeline.py` to generate all artefacts
- [ ] Launch the dashboard with `streamlit run src/rcbeam_fire/dashboard/app.py`
- [ ] Run `pytest tests/ -v` to verify acceptance thresholds
