from __future__ import annotations

from typing import Dict, List

SAFE_FEATURES = [
    "L",
    "Ac",
    "Cc",
    "As",
    "Af",
    "tins",
    "hi",
    "fc",
    "fy",
    "Es",
    "fu",
    "Efrp",
    "Tg",
    "kins",
    "rinscins",
    "Ld",
    "LR",
]

MONOTONE_FEATURE_SIGNS = {
    "Cc": 1,
    "tins": 1,
    "hi": 1,
    "LR": -1,
}

RESAMPLERS = {"targeted": "targeted", "borderline": "borderline", "smoteenn": "smoteenn"}
OBJECTIVE_MODES = {"ldam_drw", "logit_adjusted", "focal"}


def build_monotone_constraints(feature_names: List[str]) -> List[int]:
    return [MONOTONE_FEATURE_SIGNS.get(name, 0) for name in feature_names]


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prune_collinear(X: pd.DataFrame, thresh: float = 0.97) -> List[str]:
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > thresh)]
    return [c for c in X.columns if c not in drop_cols]


def prepare_data(
    df: pd.DataFrame,
    do_prune: bool = True,
    return_meta: bool = False,
):
    feat_cols = [c for c in SAFE_FEATURES if c in df.columns]
    X = df[feat_cols].copy()
    y_raw = df["target"].astype(str).str.strip().replace(
        {"0": "No Failure", "1": "Strength Failure", "2": "Deflection Failure", "3": "Other"}
    )
    valid = ["No Failure", "Strength Failure", "Deflection Failure"]
    mask = y_raw.isin(valid)
    X, y_raw = X[mask], y_raw[mask]
    dropped = int(len(df) - len(X))
    if do_prune:
        keep = prune_collinear(X, 0.97)
        X = X[keep]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    if return_meta:
        before_feats = [c for c in feat_cols]
        after_feats = list(X.columns)
        dropped_feats = [c for c in before_feats if c not in after_feats]
        class_counts = {cls: int((y == idx).sum()) for idx, cls in enumerate(le.classes_)}
        meta = {
            "feature_list_before": before_feats,
            "feature_list_after": after_feats,
            "dropped_features": dropped_feats,
            "dropped_rows_non_target": dropped,
            "class_counts": class_counts,
            "prune_threshold": 0.97 if do_prune else None,
        }
        return X, y, le, list(le.classes_), meta
    return X, y, le, list(le.classes_)
