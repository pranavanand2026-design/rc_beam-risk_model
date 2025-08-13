from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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


def _plot_importance(model, feature_names: List[str], out_png: Path, title: str = "Feature importance"):
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.array(model.feature_importances_, dtype=float)
            order = np.argsort(imp)[::-1][:20]
            plt.figure(figsize=(7, 6))
            plt.barh(np.array(feature_names)[order][::-1], imp[order][::-1])
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception:
        pass


def _plot_parity(y_true, y_pred, out_png: Path, title: str = "Predicted vs Actual FRT (min)"):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual FRT (min)")
    plt.ylabel("Predicted FRT (min)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_residuals(y_true, y_pred, out_png: Path):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, s=10, alpha=0.7)
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Predicted FRT (min)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual plot")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_error_summary(train_metrics: Dict[str, float], valid_metrics: Dict[str, float], out_png: Path):
    labels = ["MAE", "RMSE"]
    train_vals = [train_metrics.get("MAE", 0.0), train_metrics.get("RMSE", 0.0)]
    valid_vals = [valid_metrics.get("MAE", 0.0), valid_metrics.get("RMSE", 0.0)]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(5, 4))
    plt.bar(x - width / 2, train_vals, width, label="Train")
    plt.bar(x + width / 2, valid_vals, width, label="Valid")
    plt.xticks(x, labels)
    plt.ylabel("Minutes")
    plt.title("Error summary")
    plt.legend()
    for idx, val in enumerate(train_vals):
        plt.text(x[idx] - width / 2, val + 0.5, f"{val:.1f}", ha="center", va="bottom")
    for idx, val in enumerate(valid_vals):
        plt.text(x[idx] + width / 2, val + 0.5, f"{val:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


@dataclass
class FRTResult:
    model_name: str
    metrics_train: Dict[str, float]
    metrics_valid: Dict[str, float]
    model_path: Path
    metrics_path: Path
    parity_plot: Path
    importance_plot: Path
    residual_plot: Path
    error_plot: Path


def _prepare_matrix(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({c: pd.to_numeric(df.get(c, np.nan), errors="coerce") for c in RAW_FEATURES})
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    return X.loc[mask], y.loc[mask]


def train_frt_regressor(
    config_path: str | Path | None = None,
    target_col: str = "FR",
) -> FRTResult:
    cfg = load_config(config_path)
    paths_cfg = cfg["paths"]
    outs_dir = Path(paths_cfg["outputs"]) / "tables"
    models_dir = Path(paths_cfg.get("models", "models")) / "checkpoints"
    figs_dir = outs_dir.parent / "figs"
    ensure_dirs(outs_dir, models_dir, figs_dir)

    df = load_processed_dataset(paths_cfg)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X, y = _prepare_matrix(df, target_col)

    try:
        bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=bins)
    except Exception:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = _make_lgbm() or _make_xgb() or _make_ridge()
    model_name = model.__class__.__name__
    model.fit(X_tr, y_tr)

    y_hat_tr = model.predict(X_tr)
    y_hat_val = model.predict(X_val)
    metrics_train = _reg_metrics(y_tr, y_hat_tr)
    metrics_valid = _reg_metrics(y_val, y_hat_val)

    import joblib

    pack = {"model": model, "features": RAW_FEATURES, "target": target_col}
    model_path = models_dir / "frt_regressor.joblib"
    joblib.dump(pack, model_path)

    metrics_path = outs_dir / "frt_raw_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"model": model_name, "train": metrics_train, "valid": metrics_valid}, f, indent=2)

    importance_plot = figs_dir / "frt_raw_feature_importance.png"
    parity_plot = figs_dir / "frt_raw_parity.png"
    residual_plot = figs_dir / "frt_raw_residuals.png"
    error_plot = figs_dir / "frt_error_summary.png"
    _plot_importance(model, RAW_FEATURES, importance_plot)
    _plot_parity(y_val, y_hat_val, parity_plot)
    _plot_residuals(y_val, y_hat_val, residual_plot)
    _plot_error_summary(metrics_train, metrics_valid, error_plot)

    print("\n=== FRT REGRESSOR (RAW-ONLY) ===")
    print(f"Model: {model_name}")
    print(f"Train → MAE={metrics_train['MAE']:.2f}  RMSE={metrics_train['RMSE']:.2f}  R2={metrics_train['R2']:.3f}")
    print(f"Valid → MAE={metrics_valid['MAE']:.2f}  RMSE={metrics_valid['RMSE']:.2f}  R2={metrics_valid['R2']:.3f}")
    print(f"Saved model → {model_path}")
    print(f"Metrics     → {metrics_path}")
    print(f"Parity plot → {parity_plot}")
    print(f"Residuals   → {residual_plot}")
    print(f"Error bars  → {error_plot}")
    print(f"Importance  → {importance_plot}")

    return FRTResult(
        model_name=model_name,
        metrics_train=metrics_train,
        metrics_valid=metrics_valid,
        model_path=model_path,
        metrics_path=metrics_path,
        parity_plot=parity_plot,
        importance_plot=importance_plot,
        residual_plot=residual_plot,
        error_plot=error_plot,
    )


__all__ = ["train_frt_regressor", "FRTResult", "RAW_FEATURES"]
