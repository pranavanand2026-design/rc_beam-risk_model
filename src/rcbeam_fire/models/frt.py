from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..config import load_config
from ..utils.io import ensure_dirs, load_processed_dataset

RAW_FEATURES: List[str] = [
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


def _make_lgbm():
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        return None


def _make_xgb():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=900,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
        )
    except Exception:
        return None


def _make_ridge():
    from sklearn.linear_model import Ridge

    return Ridge(alpha=1.0, random_state=42)


def _reg_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _prepare_matrix(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({c: pd.to_numeric(df.get(c, np.nan), errors="coerce") for c in RAW_FEATURES})
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    return X.loc[mask], y.loc[mask]
