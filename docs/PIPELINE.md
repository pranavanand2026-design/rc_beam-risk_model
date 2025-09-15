# RC Beam Fire Pipeline Overview

This repository is organised around the workflow described in *RC_Beam.pdf*. Each stage lives in a dedicated module so that scripts, notebooks, and the dashboard can reuse the same logic.

## 1. Data inspection
- Module: `rcbeam_fire.data.scan.scan_dataset`
- CLI: `python scripts/01_scan.py`
- Output: summary tables/plots under `outputs/tables` and `outputs/figs`

## 2. Preprocessing
- Module: `rcbeam_fire.data.preprocess.run_preprocess`
- CLI: `python scripts/02_preprocess.py`
- Responsibilities: load Excel, normalise column names, coerce numerics, create `target`, drop unlabeled rows, write `data/processed/dataset.(parquet|csv)`

## 3. Failure-mode classifier (XGB + LDAM)
- Module: `rcbeam_fire.models.blend.train_hybrid_classifier`
- CLI: `python scripts/05_train_hybrid.py`
- Steps: targeted SMOTE for baseline leg, LDAM-DRW leg with Borderline SMOTE, blend tuning, threshold optimisation, metric reporting, serialization to `models/checkpoints/blend_xgb_ldam_nophys.joblib`

## 4. Fire-resistance regressor
- Module: `rcbeam_fire.models.frt.train_frt_regressor`
- CLI: `python scripts/06_train_frt.py`
- Output: LightGBM/XGBoost regressor pack saved to `models/checkpoints/frt_regressor.joblib`, metrics and diagnostic plots under `outputs`

## 5. Case-level insight engine
- Module: `rcbeam_fire.analysis.insight.build_case_action_plan`
- Used by: `scripts/12_case_analyzer.py`, dashboard
- Produces: scenario summary, ranked recommendations, safe-case reference stats

## 6. Dashboard
- Module: `rcbeam_fire.dashboard.app`
- Launch: `streamlit run src/rcbeam_fire/dashboard/app.py`
- Features: beam selection, scenario inputs, probability bars, action plan, editable playground

## 7. Experiments archive
Legacy notebooks/scripts are preserved in `experiments/` as trials (`trial*_*.py`). They are not part of the primary pipeline but remain for reference.

### Running scripts
All scripts default to `config.yaml`; override with `--config` if needed. Each script prepends `src/` to `sys.path` so you can run them directly from the project root.
