from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_classification_report_meets_thresholds():
    report_path = Path("outputs/tables/blend_xgb_ldam_nophys_report.csv")
    assert report_path.exists(), "Classification report missing; run training pipeline."
    report = pd.read_csv(report_path, index_col=0)
    macro_f1 = float(report.loc["f1-score", "macro avg"])
    balanced_acc = float(report.loc["recall", "macro avg"])
    assert macro_f1 >= 0.80, f"Macro F1 too low: {macro_f1:.3f}"
    assert balanced_acc >= 0.80, f"Balanced accuracy too low: {balanced_acc:.3f}"


def test_regression_metrics_meet_thresholds():
    metrics_path = Path("outputs/tables/frt_raw_metrics.json")
    assert metrics_path.exists(), "Regression metrics missing; run training pipeline."
    metrics = json.loads(metrics_path.read_text())
    valid = metrics["valid"]
    assert valid["R2"] >= 0.90, f"Validation R2 too low: {valid['R2']:.3f}"
    assert valid["MAE"] <= 12.0, f"Validation MAE too high: {valid['MAE']:.2f}"
