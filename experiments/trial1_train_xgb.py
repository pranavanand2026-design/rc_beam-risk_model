from __future__ import annotations
import os, argparse, yaml, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# -----------------------
# Utilities
# -----------------------
def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def load_processed(paths: dict) -> pd.DataFrame:
    parq = os.path.join(paths["processed"], "dataset.parquet")
    csv_ = os.path.join(paths["processed"], "dataset.csv")
    if os.path.exists(parq):
        return pd.read_parquet(parq)
    return pd.read_csv(csv_)

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def save_confusion_matrix(cm: np.ndarray, labels: list, out_path: str, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=20)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.title(title)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=9)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# -----------------------
# Main: SMOTE-balanced training
# -----------------------
def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    paths = cfg["paths"]
    outputs_dir = paths["outputs"]
    figs_dir = os.path.join(outputs_dir, "figs")
    tabs_dir = os.path.join(outputs_dir, "tables")
    models_dir = os.path.join(paths.get("models", "models"), "checkpoints")
    ensure_dirs(figs_dir, tabs_dir, models_dir)

    # 1) Load data
    df = load_processed(paths)

    # 2) Safe (leak-free) features only
    safe_features = [
        "L","Ac","Cc","As","Af","tins","hi","fc","fy","Es","fu",
        "Efrp","Tg","kins","rinscins","Ld","LR"
    ]
    feat_cols = [c for c in safe_features if c in df.columns]
    X = df[feat_cols].copy()

    # 3) Clean & encode target
    y_raw = df["target"].astype(str).str.strip().replace({
        "0": "No Failure",
        "1": "Strength Failure",
        "2": "Deflection Failure",
        "3": "Other",
    })
    valid_classes = ["No Failure", "Strength Failure", "Deflection Failure"]
    mask = y_raw.isin(valid_classes)
    X = X[mask]
    y_raw = y_raw[mask]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)  # integers 0..K-1
    classes_ = list(le.classes_)

    # 4) Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) SMOTE on TRAIN ONLY (no leakage)
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    # (Optional) keep modest class weights even after SMOTE (can comment out)
    cls, counts = np.unique(y_train_bal, return_counts=True)
    inv_freq = counts.sum() / (len(cls) * counts)
    class_weight_map = {c: w for c, w in zip(cls, inv_freq)}
    sample_weight = np.vectorize(class_weight_map.get)(y_train_bal)

    # 6) Model
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes_),
        n_estimators=700,
        learning_rate=0.06,
        max_depth=6,
        min_child_weight=12,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )

    # 7) Fit (validate on untouched val set)
    model.fit(
        X_train_bal, y_train_bal,
        sample_weight=sample_weight,      # comment this line to test “SMOTE only”
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 8) Evaluate
    y_pred_enc = model.predict(X_val)
    y_pred = le.inverse_transform(y_pred_enc)
    y_val_names = le.inverse_transform(y_val)

    labels_sorted = list(classes_)
    acc  = accuracy_score(y_val_names, y_pred)
    bacc = balanced_accuracy_score(y_val_names, y_pred)
    cm   = confusion_matrix(y_val_names, y_pred, labels=labels_sorted)
    report = classification_report(y_val_names, y_pred, labels=labels_sorted, output_dict=True)

    # 9) Save metrics/plots (distinct filenames for easy comparison)
    metrics_path = os.path.join(tabs_dir, "xgb_metrics_smote.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "labels": labels_sorted,
            "class_counts_val": {lbl: int((y_val_names == lbl).sum()) for lbl in labels_sorted},
            "training_counts_after_smote": {int(k): int(v) for k, v in zip(cls, counts)},
            "label_mapping": {i: name for i, name in enumerate(classes_)},
        }, f, indent=2)

    pd.DataFrame(report).to_csv(os.path.join(tabs_dir, "xgb_classification_report_smote.csv"))

    cm_path = os.path.join(figs_dir, "xgb_confusion_matrix_smote.png")
    save_confusion_matrix(cm, labels_sorted, cm_path, title="XGB Confusion Matrix (SMOTE)")

    # 10) Save model
    model_path = os.path.join(models_dir, "xgb_model_smote.json")
    model.save_model(model_path)

    print("✓ SMOTE-balanced training complete.")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bacc:.4f}")
    print(f"  Model saved → {model_path}")
    print(f"  Metrics → {metrics_path}")
    print(f"  Confusion → {cm_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()
    main(args.config)