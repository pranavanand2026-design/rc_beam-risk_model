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


try:
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE
    from imblearn.combine import SMOTEENN
except Exception as exc:  # pragma: no cover
    raise ImportError("imblearn is required for the hybrid classifier") from exc


def targeted_smote(X, y, classes_: List[str], strength_ratio=1.0, nofail_ratio=0.7, random_state=42):
    _, counts = np.unique(y, return_counts=True)
    majority = int(counts.max())
    name_to_idx = {name: i for i, name in enumerate(classes_)}
    strat = {}
    if "Strength Failure" in name_to_idx:
        strat[name_to_idx["Strength Failure"]] = int(majority * strength_ratio)
    if "No Failure" in name_to_idx:
        target_nf = int(majority * nofail_ratio)
        cur_nf = counts[name_to_idx["No Failure"]]
        strat[name_to_idx["No Failure"]] = max(target_nf, int(cur_nf))
    if not strat:
        return X, y
    sm = SMOTE(random_state=random_state, sampling_strategy=strat, k_neighbors=5)
    return sm.fit_resample(X, y)


def resample_with_strategy(X, y, classes, strategy: str, random_state: int = 42):
    strategy = strategy.lower()
    if strategy == "targeted":
        return targeted_smote(X, y, classes, strength_ratio=1.0, nofail_ratio=0.7, random_state=random_state)
    if strategy == "borderline":
        sampler = BorderlineSMOTE(kind="borderline-1", k_neighbors=5, random_state=random_state)
        return sampler.fit_resample(X, y)
    if strategy == "smoteenn":
        sampler = SMOTEENN(random_state=random_state)
        return sampler.fit_resample(X, y)
    raise ValueError(f"Unknown resampler strategy: {strategy}")


from typing import Optional, Tuple

try:
    from xgboost import DMatrix, XGBClassifier, train as xgb_train
except Exception:  # pragma: no cover - optional dependency
    DMatrix = None
    XGBClassifier = None
    xgb_train = None


def make_xgb(num_classes: int, monotone: Optional[List[int]] = None):
    if XGBClassifier is None:
        raise ImportError("xgboost is required for the hybrid classifier.")
    mono = None
    if monotone is not None:
        mono = "(" + ",".join(str(int(m)) for m in monotone) + ")"
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=650,
        learning_rate=0.06,
        max_depth=6,
        min_child_weight=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        reg_alpha=0.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        monotone_constraints=mono,
    )


import os
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass
class HybridResult:
    name: str
    acc: float
    bacc: float
    macro_f1: float
    report_csv: Path
    cm_png: Path
    model_path: Path
    thresholds: Dict[str, float]
    roc_png: Path
    pr_png: Path
    topk_png: Path
    metrics_json: Path
    data_stats_json: Optional[Path] = None
    summary_json: Optional[Path] = None
    bootstrap_json: Optional[Path] = None


def save_confusion_matrix(
    cm: np.ndarray, labels: List[str], out_path: Path, title: str = "Confusion", normalize: bool = False
):
    """Save a confusion matrix with an accessible colormap and annotations.

    Args:
        cm: integer confusion matrix (rows=true, cols=pred)
        labels: class names in order
        out_path: PNG path
        title: plot title
        normalize: if True, annotate by row-normalised values
    """
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        data = np.divide(data, row_sums)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(data, interpolation="nearest", cmap="Blues")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=20)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.title(title)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count" if not normalize else "Proportion", rotation=270, labelpad=15)

    # Annotate cells
    thresh = data.max() / 2.0 if data.size else 0.0
    for (i, j), _ in np.ndenumerate(data):
        val = data[i, j]
        txt = f"{val:.2f}" if normalize else f"{int(cm[i, j])}"
        plt.text(
            j,
            i,
            txt,
            ha="center",
            va="center",
            fontsize=9,
            color="white" if val > thresh else "black",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


from sklearn.preprocessing import StandardScaler
from ..config import load_config
from ..utils.io import ensure_dirs, load_processed_dataset


class LDAMDRWXGB:
    """XGBoost helper with LDAM/DRW, logit-adjusted, or focal objectives."""

    def __init__(
        self,
        classes: List[str],
        random_state: int = 42,
        mode: str = "ldam_drw",
        resampler: str = "borderline",
        focal_gamma: float = 2.0,
    ):
        if xgb_train is None or DMatrix is None:
            raise ImportError("xgboost is required for LDAMDRWXGB.")
        self.classes = classes
        self.scaler: Optional[StandardScaler] = None
        self.booster = None
        self.random_state = random_state
        self.thresholds: Dict[str, float] = {}
        self.mode = mode if mode in OBJECTIVE_MODES else "ldam_drw"
        self.resampler = resampler if resampler in RESAMPLERS else "borderline"
        self.focal_gamma = focal_gamma
        self.log_priors: Optional[np.ndarray] = None
        self.params_: Optional[Dict[str, float]] = None
        self.margins_: Optional[List[float]] = None
        self.class_weights_: Optional[List[float]] = None
        self.resampled_counts_: Optional[Dict[str, int]] = None

    @staticmethod
    def _class_balanced_weights(y, beta=0.999):
        from collections import Counter

        c = Counter(y)
        cls = sorted(c.keys())
        eff = np.array([1.0 - beta ** c[i] for i in cls], float)
        eff[eff <= 0] = 1e-8
        w = (1.0 - beta) / eff
        w = w / w.mean()
        out = np.zeros(max(cls) + 1, float)
        for i, k in enumerate(cls):
            out[k] = w[i]
        return out

    @staticmethod
    def _compute_margins_ldam(y, num_classes, counts, max_m=0.5):
        n = np.array([counts.get(c, 1) for c in range(num_classes)], float)
        m = 1.0 / (np.power(n, 0.25) + 1e-12)
        return m * (max_m / m.max())

    @staticmethod
    def _ldam_softmax_obj(margins, num_classes):
        def _obj(preds, dtrain):
            y = dtrain.get_label().astype(int)
            N, K = y.shape[0], num_classes
            logits = preds.reshape(N, K)
            logits[np.arange(N), y] -= margins[y]
            m = logits.max(axis=1, keepdims=True)
            p = np.exp(logits - m)
            p = p / p.sum(axis=1, keepdims=True)
            oh = np.zeros_like(p)
            oh[np.arange(N), y] = 1.0
            grad = (p - oh).reshape(-1)
            hess = (p * (1.0 - p)).reshape(-1)
            return grad, hess

        return _obj

    def _focal_softmax_obj(self, num_classes: int, gamma: float):
        def _obj(preds, dtrain):
            y = dtrain.get_label().astype(int)
            logits = preds.reshape(-1, num_classes)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            p = exp / exp.sum(axis=1, keepdims=True)
            pt = p[np.arange(p.shape[0]), y]
            pt = np.clip(pt, 1e-12, 1.0)
            factor = (1.0 - pt) ** gamma

            grad = p.copy()
            grad[np.arange(p.shape[0]), y] -= 1.0
            grad *= factor[:, None]

            log_pt = np.log(pt)
            aux = gamma * (1.0 - pt) ** (gamma - 1.0) * pt * log_pt
            grad[np.arange(p.shape[0]), y] -= aux
            grad += aux[:, None] * p

            hess = factor[:, None] * (p * (1.0 - p))
            return grad.reshape(-1), hess.reshape(-1)

        return _obj

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X.values) if self.scaler is not None else X.values
        if self.mode == "logit_adjusted" and self.log_priors is not None:
            logits = self.booster.predict(DMatrix(Xs), output_margin=True).reshape(-1, len(self.classes))
            logits = logits - self.log_priors
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            return exp / exp.sum(axis=1, keepdims=True)
        return self.booster.predict(DMatrix(Xs))

    def fit(self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray):
        from collections import Counter as Cn

        K = len(self.classes)
        self.scaler = StandardScaler().fit(X_tr.values)
        Xs_tr = self.scaler.transform(X_tr.values)
        Xs_val = self.scaler.transform(X_val.values)

        cnt = Cn(y_tr)
        total = sum(cnt.values())
        priors = np.array([cnt.get(i, 1) / total for i in range(K)], dtype=float)
        self.log_priors = np.log(np.clip(priors, 1e-12, 1.0))

        Xs_tr_res, y_tr_res = resample_with_strategy(
            Xs_tr, y_tr, self.classes, self.resampler, random_state=self.random_state
        )
        counts_res = {self.classes[int(c)]: int((y_tr_res == c).sum()) for c in np.unique(y_tr_res)}
        for cname in self.classes:
            counts_res.setdefault(cname, 0)
        self.resampled_counts_ = counts_res

        margins = self._compute_margins_ldam(y_tr_res, K, Cn(y_tr_res), max_m=0.5) if self.mode == "ldam_drw" else np.zeros(K)
        cbw = self._class_balanced_weights(y_tr_res, beta=0.999)
        self.margins_ = margins.tolist()
        self.class_weights_ = cbw.tolist()
        w_train = np.array([cbw[c] for c in y_tr_res], float)

        dtrain = DMatrix(Xs_tr_res, label=y_tr_res, weight=w_train)
        dval = DMatrix(Xs_val, label=y_val)

        params = {
            "num_class": K,
            "eta": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "nthread": -1,
            "seed": self.random_state,
        }
        self.params_ = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in params.items()}

        if self.mode == "ldam_drw":
            obj_fn = self._ldam_softmax_obj(margins, K)
            booster = xgb_train(
                params,
                dtrain,
                num_boost_round=200,
                obj=obj_fn,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            booster = xgb_train(
                params,
                dtrain,
                num_boost_round=300,
                obj=obj_fn,
                evals=[(dtrain, "train_cb"), (dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
                xgb_model=booster,
            )
        elif self.mode == "focal":
            booster = xgb_train(
                params,
                dtrain,
                num_boost_round=400,
                obj=self._focal_softmax_obj(K, self.focal_gamma),
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
        else:  # logit adjusted
            booster = xgb_train(
                params,
                dtrain,
                num_boost_round=400,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
        self.booster = booster
