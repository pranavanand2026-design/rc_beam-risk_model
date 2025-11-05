"""
Train a binary Safe/Unsafe gate model for RC-beam fire performance.

Usage:
  python -m scripts.train_gate --config config.yaml

Reads:
  - config.yaml → paths.processed (dataset location), analysis.exposure_minutes, analysis.margin_minutes
  - data/processed/dataset.(parquet|csv)

Outputs:
  - models/checkpoints/frt_safety_gate.joblib
"""

from __future__ import annotations
import os, argparse, json, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- IO / CONFIG ----------------
def load_cfg(p: str) -> dict:
    import yaml
    with open(p, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def load_processed(paths_cfg: dict) -> pd.DataFrame:
    parq = os.path.join(paths_cfg["processed"], "dataset.parquet")
    csv_ = os.path.join(paths_cfg["processed"], "dataset.csv")
    return pd.read_parquet(parq) if os.path.exists(parq) else pd.read_csv(csv_)

# ---------------- FEATURE SELECTION (numeric only) ----------------
# Columns we should never use as predictors
HARD_EXCLUDE = {
    # identifiers / labels
    "BN", "target", "LimitState", "Limit State", "Limit state",
    "failure_mode", "LimitState ", "Limit State ", "Limit state ",
    # outcomes measured AFTER the fire test (leakage for a safety gate)
    "FR",          # fire resistance time (ground-truth used to make the label)
    "F_to_EF",     # 'Failure' vs 'End of Fire' flag
    "F/EF",        # original name that may still appear in some sheets
    "df"           # deflection at failure
}

def build_numeric_features(df: pd.DataFrame) -> List[str]:
    num_cols = set(df.select_dtypes(include=[np.number]).columns)
    leak_lower = {"fr", "f_to_ef", "f/ef", "df"}
    keep = []
    for c in num_cols:
        if c in HARD_EXCLUDE: 
            continue
        if c.lower() in leak_lower:
            continue
        keep.append(c)
    return sorted(keep)

# ---------------- THRESHOLD TUNING ----------------
def tune_threshold_by_f1(y_true: np.ndarray, p_safe: np.ndarray) -> float:
    """Return probability threshold that maximizes F1 on validation."""
    prec, rec, thr = precision_recall_curve(y_true, p_safe)
    f1 = np.where((prec+rec) > 0, 2*prec*rec/(prec+rec), 0.0)
    if f1.size == 0 or thr.size == 0:
        return 0.5
    # precision_recall_curve returns thresholds aligned to all but last point
    best_idx = int(np.argmax(f1[:-1])) if f1.shape[0] > thr.shape[0] else int(np.argmax(f1))
    t = float(np.clip(thr[min(best_idx, len(thr)-1)], 0.05, 0.95))
    return t

# ---------------- MAIN TRAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    analysis = cfg.get("analysis", {})
    EXPOSURE = float(analysis.get("exposure_minutes", 90))
    MARGIN   = float(analysis.get("margin_minutes", 10))
    RANDOM_STATE = int(cfg.get("seed", 42))

    # 1) Load data
    df = load_processed(paths)
    if "FR" not in df.columns:
        raise ValueError("Dataset must contain column 'FR' (fire resistance time, minutes).")

    # 2) Build binary target: Safe if FR >= exposure + margin
    threshold_minutes = EXPOSURE + MARGIN
    df = df.copy()
    df["safe"] = (pd.to_numeric(df["FR"], errors="coerce") >= threshold_minutes).astype(int)

    # 3) Features: numeric-only, exclude IDs/labels
    features = build_numeric_features(df)
    if not features:
        raise ValueError("No numeric features found for gate training after exclusions.")

    # Drop any remaining rows with NaNs in features or target
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df["safe"].astype(int)
    mask = ~X.isna().any(axis=1)
    X, y = X.loc[mask], y.loc[mask]

    # Defensive: drop exact duplicate rows (features + label) before splitting
    Xy = X.copy()
    Xy["__y"] = y.values
    Xy = Xy.drop_duplicates()
    y = Xy["__y"].astype(int)
    X = Xy.drop(columns="__y")

    for leak_name in ["FR", "F_to_EF", "F/EF", "df"]:
        if leak_name in X.columns:
            print(f"⚠️  Leak feature still present and will be dropped: {leak_name}")
            X = X.drop(columns=[leak_name], errors="ignore")

    # 4) Split & scale (scaler saved in pack)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # 5) Train LightGBM with class balance + probability calibration
    base = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=25,
        random_state=RANDOM_STATE
    )
    clf_cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf_cal.fit(X_train_s, y_train)

    # 6) Validate
    p_safe_val = clf_cal.predict_proba(X_val_s)[:, 1]
    y_hat_val  = (p_safe_val >= 0.5).astype(int)
    auc = roc_auc_score(y_val, p_safe_val)
    print("=== Safe/Unsafe Gate (FR-based) ===")
    print(classification_report(y_val, y_hat_val, digits=3))
    print(f"AUC: {auc:.3f}")

    # 7) Tune operating threshold on validation by F1
    tuned_thr = tune_threshold_by_f1(y_val.values, p_safe_val)
    print(f"Tuned decision threshold (by F1): {tuned_thr:.3f}")

    # 8) Save pack
    models_dir = os.path.join(paths.get("models", "models"), "checkpoints")
    ensure_dirs(models_dir)
    pack_path = os.path.join(models_dir, "frt_safety_gate.joblib")
    artifact = {
        "model": clf_cal,
        "scaler": scaler,
        "features": features,
        "threshold": float(tuned_thr),
        "exposure": float(EXPOSURE),
        "margin": float(MARGIN),
        "minutes_threshold": float(threshold_minutes)
    }
    joblib.dump(artifact, pack_path)
    print(f"\nSaved calibrated gate model → {pack_path}")

if __name__ == "__main__":
    main()