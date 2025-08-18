from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE
    from imblearn.combine import SMOTEENN
except Exception as exc:  # pragma: no cover
    raise ImportError("imblearn is required for the hybrid classifier") from exc
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    auc,
)
from sklearn.model_selection import train_test_split
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # fallback for older scikit-learn
    from sklearn.model_selection import GroupKFold as StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.isotonic import IsotonicRegression

from ..config import load_config
from ..utils.io import ensure_dirs, load_processed_dataset

try:
    from xgboost import DMatrix, XGBClassifier, train as xgb_train
except Exception:  # pragma: no cover - optional dependency
    DMatrix = None
    XGBClassifier = None
    xgb_train = None

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


def fit_isotonic_calibrators(
    proba: np.ndarray, y_true: np.ndarray, classes: List[str]
) -> Tuple[np.ndarray, Dict[str, IsotonicRegression]]:
    calibrators: Dict[str, IsotonicRegression] = {}
    calibrated_cols = []
    for idx, cname in enumerate(classes):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(proba[:, idx], (y_true == idx).astype(int))
        calibrators[cname] = iso
        calibrated_cols.append(iso.predict(proba[:, idx]))
    calibrated = np.column_stack(calibrated_cols)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    zero_mask = row_sums[:, 0] <= 0
    calibrated[~zero_mask] = calibrated[~zero_mask] / row_sums[~zero_mask]
    if np.any(zero_mask):
        calibrated[zero_mask] = 1.0 / len(classes)
    return calibrated, calibrators


def apply_calibrators(
    proba: np.ndarray, calibrators: Dict[str, IsotonicRegression], classes: List[str]
) -> np.ndarray:
    if not calibrators:
        return proba
    calibrated_cols = []
    for idx, cname in enumerate(classes):
        iso = calibrators.get(cname)
        if iso is not None:
            calibrated_cols.append(iso.predict(proba[:, idx]))
        else:
            calibrated_cols.append(proba[:, idx])
    calibrated = np.column_stack(calibrated_cols)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    zero_mask = row_sums[:, 0] <= 0
    calibrated[~zero_mask] = calibrated[~zero_mask] / row_sums[~zero_mask]
    if np.any(zero_mask):
        calibrated[zero_mask] = 1.0 / len(classes)
    return calibrated


def prune_collinear(X: pd.DataFrame, thresh: float = 0.97) -> List[str]:
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > thresh)]
    return [c for c in X.columns if c not in drop_cols]


def prepare_data(
    df: pd.DataFrame,
    do_prune: bool = True,
    return_meta: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, List[str]]:
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


def plot_multiclass_roc(
    y_true: np.ndarray, proba: np.ndarray, classes: List[str], out_path: Path
) -> Tuple[Optional[float], Dict[str, float]]:
    if len(classes) <= 1:
        return None, {}
    try:
        y_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    except Exception:
        return None, {}

    plt.figure(figsize=(6, 5))
    macro_auc = None
    try:
        macro_auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except Exception:
        pass

    per_class_auc: Dict[str, float] = {}
    for idx, cname in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, idx], proba[:, idx])
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{cname} (AUC={auc_val:.3f})")
            per_class_auc[cname] = float(auc_val)
        except ValueError:
            continue

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    if macro_auc is not None:
        plt.title(f"ROC curve (macro AUC={macro_auc:.3f})")
    else:
        plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return macro_auc, per_class_auc


def plot_multiclass_pr(
    y_true: np.ndarray, proba: np.ndarray, classes: List[str], out_path: Path
) -> Tuple[Optional[float], Dict[str, float]]:
    if len(classes) <= 1:
        return None, {}
    try:
        y_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    except Exception:
        return None, {}

    plt.figure(figsize=(6, 5))
    macro_ap = None
    try:
        macro_ap = average_precision_score(y_true, proba, average="macro")
    except Exception:
        try:
            macro_ap = average_precision_score(y_bin, proba, average="macro")
        except Exception:
            macro_ap = None

    per_class_ap: Dict[str, float] = {}
    for idx, cname in enumerate(classes):
        try:
            precision, recall, _ = precision_recall_curve(y_bin[:, idx], proba[:, idx])
            ap = auc(recall, precision)
            plt.plot(recall, precision, label=f"{cname} (AP={ap:.3f})")
            per_class_ap[cname] = float(ap)
        except ValueError:
            continue

    baseline = y_bin.mean(axis=0).mean() if np.any(y_bin) else 0.0
    plt.hlines(baseline, 0, 1, colors="gray", linestyles="--", label="Baseline")
    if macro_ap is not None:
        plt.title(f"Precision-Recall curve (macro AP={macro_ap:.3f})")
    else:
        plt.title("Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return macro_ap, per_class_ap


def plot_topk_accuracy(
    y_true: np.ndarray, proba: np.ndarray, classes: List[str], out_path: Path, ks: Tuple[int, ...] = (1, 2, 3)
) -> Dict[int, float]:
    ks = tuple(k for k in ks if 0 < k <= proba.shape[1])
    if not ks:
        return {}
    sorted_idx = np.argsort(proba, axis=1)[:, ::-1]
    scores: Dict[int, float] = {}
    for k in ks:
        hits = np.any(sorted_idx[:, :k] == y_true[:, None], axis=1)
        scores[k] = float(np.mean(hits))

    plt.figure(figsize=(5, 4))
    labels = [f"Top-{k}" for k in ks]
    values = [scores[k] for k in ks]
    plt.bar(labels, values, color="#1f77b4")
    upper = min(1.05, max(values) + 0.1)
    plt.ylim(0, upper)
    for i, val in enumerate(values):
        plt.text(i, val + 0.02, f"{val:.2f}", ha="center", va="bottom")
    plt.ylabel("Accuracy")
    plt.title("Top-K accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return scores


def _expected_calibration_error(probs: np.ndarray, true: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)
    if total == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc = true[mask].mean()
        ece += np.abs(acc - conf) * mask.sum()
    return ece / total


def compute_calibration_stats(
    y_true: np.ndarray, proba: np.ndarray, classes: List[str], n_bins: int = 10
) -> Tuple[List[Tuple[str, np.ndarray, np.ndarray]], Dict[str, float], Dict[str, float]]:
    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    ece_per_class: Dict[str, float] = {}
    brier_per_class: Dict[str, float] = {}
    for idx, cname in enumerate(classes):
        probs = proba[:, idx]
        labels = (y_true == idx).astype(int)
        if probs.size == 0:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="uniform")
        except ValueError:
            # calibration_curve may fail if all labels are identical; skip plotting but record metrics
            frac_pos, mean_pred = np.array([]), np.array([])
        curves.append((cname, mean_pred, frac_pos))
        ece_val = _expected_calibration_error(probs, labels, n_bins=n_bins)
        brier_val = float(np.mean((probs - labels) ** 2))
        ece_per_class[cname] = float(ece_val) if np.isfinite(ece_val) else float("nan")
        brier_per_class[cname] = brier_val
    return curves, ece_per_class, brier_per_class


def plot_reliability_diagram(
    curves: List[Tuple[str, np.ndarray, np.ndarray]], out_path: Path, title: str = "Reliability Diagram"
) -> None:
    plt.figure(figsize=(5.5, 4.5))
    plotted = False
    for cname, mean_pred, frac_pos in curves:
        if mean_pred.size == 0 or frac_pos.size == 0:
            continue
        plotted = True
        plt.plot(mean_pred, frac_pos, marker="o", label=cname)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_probability_histograms(
    proba: np.ndarray, classes: List[str], out_path: Path, bins: int = 10
) -> None:
    n_classes = len(classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(4.0 * n_classes, 3.2), sharey=True)
    if n_classes == 1:
        axes = [axes]
    for idx, cname in enumerate(classes):
        ax = axes[idx]
        ax.hist(proba[:, idx], bins=np.linspace(0.0, 1.0, bins + 1), color="#1f77b4", edgecolor="white")
        ax.set_title(cname)
        ax.set_xlabel("Probability")
        if idx == 0:
            ax.set_ylabel("Count")
        ax.set_xlim(0.0, 1.0)
    fig.suptitle("Post-calibration probability distribution", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_weight_sweep(tuning_rows: List[Dict], out_path: Path) -> None:
    if not tuning_rows:
        return
    frame = pd.DataFrame(tuning_rows)
    if frame.empty:
        return
    frame = frame.sort_values("w_xgb")
    plt.figure(figsize=(5.5, 4.0))
    plt.plot(frame["w_xgb"], frame["macro_f1"], marker="o", color="#2ca02c")
    for _, row in frame.iterrows():
        txt_parts = []
        for cname, val in row.get("thresholds", {}).items():
            txt_parts.append(f"{cname.split()[0]}={val:.2f}")
        if not txt_parts:
            continue
        plt.annotate(
            ", ".join(txt_parts),
            xy=(row["w_xgb"], row["macro_f1"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("XGB weight in blend (w)")
    plt.ylabel("Macro F1")
    plt.title("Blend weight vs macro F1")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_threshold_sweep(tuning_rows: List[Dict], classes: List[str], out_path: Path) -> None:
    if not tuning_rows:
        return
    frame = pd.DataFrame(tuning_rows)
    if frame.empty or "thresholds" not in frame.columns:
        return
    frame = frame.sort_values("w_xgb")
    long_records: List[Dict[str, float]] = []
    for _, row in frame.iterrows():
        thresholds = row.get("thresholds", {})
        for cname in classes:
            if cname not in thresholds:
                continue
            long_records.append(
                {
                    "class": cname,
                    "w_xgb": float(row["w_xgb"]),
                    "threshold": float(thresholds[cname]),
                }
            )
    if not long_records:
        return
    long_df = pd.DataFrame(long_records)
    plt.figure(figsize=(5.5, 4.0))
    for cname, grp in long_df.groupby("class"):
        grp = grp.sort_values("w_xgb")
        plt.plot(grp["w_xgb"], grp["threshold"], marker="o", label=cname)
    plt.xlabel("XGB weight in blend (w)")
    plt.ylabel("Probability threshold")
    plt.title("Per-class threshold tuning")
    plt.ylim(0.2, 0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _summarise_array(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": None, "std": None, "p2_5": None, "p97_5": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p2_5": float(np.percentile(arr, 2.5)),
        "p97_5": float(np.percentile(arr, 97.5)),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: List[str],
    thresholds: Dict[str, float],
    n_samples: int = 1000,
    random_state: int = 42,
) -> Dict[str, Dict]:
    if n_samples <= 0 or proba.size == 0:
        return {}
    rng = np.random.default_rng(random_state)
    global_metrics = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    per_class_metrics: Dict[str, Dict[str, List[float]]] = {
        cls: {"precision": [], "recall": [], "f1": []} for cls in classes
    }

    for _ in range(n_samples):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_bs = y_true[idx]
        proba_bs = proba[idx]
        y_pred_bs, report_bs = evaluate_probs_with_thresholds(proba_bs, y_bs, classes, thresholds)
        global_metrics["accuracy"].append(float(accuracy_score(y_bs, y_pred_bs)))
        global_metrics["balanced_accuracy"].append(float(balanced_accuracy_score(y_bs, y_pred_bs)))
        global_metrics["macro_f1"].append(float(f1_score(y_bs, y_pred_bs, average="macro")))

        for cls in classes:
            cls_report = report_bs.get(cls, {})
            per_class_metrics[cls]["precision"].append(float(cls_report.get("precision", 0.0)))
            per_class_metrics[cls]["recall"].append(float(cls_report.get("recall", 0.0)))
            per_class_metrics[cls]["f1"].append(float(cls_report.get("f1-score", 0.0)))

    summary = {
        "global": {metric: _summarise_array(vals) for metric, vals in global_metrics.items()},
        "per_class": {
            cls: {metric: _summarise_array(vals) for metric, vals in metrics.items()}
            for cls, metrics in per_class_metrics.items()
        },
        "config": {
            "samples": n_samples,
            "random_state": random_state,
        },
    }
    return summary


def evaluate_probs_with_thresholds(
    proba: np.ndarray, y_true: np.ndarray, classes: List[str], th: Dict[str, float]
) -> Tuple[np.ndarray, Dict]:
    pred_idx = proba.argmax(axis=1).copy()
    for ci, cname in enumerate(classes):
        t = th.get(cname)
        if t is None:
            continue
        pred_idx[proba[:, ci] >= t] = ci
    report = classification_report(y_true, pred_idx, target_names=classes, output_dict=True, zero_division=0)
    return pred_idx, report


def tune_thresholds(proba: np.ndarray, y_true: np.ndarray, classes: List[str]) -> Tuple[Dict[str, float], float]:
    cand = {c: np.linspace(0.25, 0.60, 8) for c in classes}
    cand.pop("Deflection Failure", None)
    keys = list(cand.keys())
    best_f1, best = -1.0, {}

    def recurse(i, cur):
        nonlocal best_f1, best
        if i == len(keys):
            th = cur.copy()
            pred = proba.argmax(axis=1).copy()
            for ci, cname in enumerate(classes):
                t = th.get(cname)
                if t is not None:
                    pred[proba[:, ci] >= t] = ci
            f1 = f1_score(y_true, pred, average="macro")
            if f1 > best_f1:
                best_f1, best = f1, th.copy()
            return
        k = keys[i]
        for t in cand[k]:
            cur[k] = float(t)
            recurse(i + 1, cur)

    recurse(0, {})
    return best, best_f1


def blend_probs(p_a: np.ndarray, p_b: np.ndarray, w: float) -> np.ndarray:
    return w * p_a + (1.0 - w) * p_b


def train_hybrid_classifier(
    config_path: str | os.PathLike | None = None,
    name: str = "blend_xgb_ldam_nophys",
    objective_mode: str = "ldam_drw",
    resampler: str = "targeted",
    focal_gamma: float = 2.0,
    bootstrap_samples: int = 0,
    bootstrap_random_state: int = 42,
) -> HybridResult:
    cfg = load_config(config_path)
    paths_cfg = cfg["paths"]
    outs_dir = Path(paths_cfg["outputs"])
    figs_dir = outs_dir / "figs"
    tabs_dir = outs_dir / "tables"
    models_dir = Path(paths_cfg.get("models", "models")) / "checkpoints"
    ensure_dirs(figs_dir, tabs_dir, models_dir)

    df = load_processed_dataset(paths_cfg)
    do_prune = bool(cfg.get("data", {}).get("prune_collinear", True))
    X, y, le, classes, meta = prepare_data(df, do_prune=do_prune, return_meta=True)
    monotone = build_monotone_constraints(list(X.columns))
    objective_mode = objective_mode if objective_mode in OBJECTIVE_MODES else "ldam_drw"
    resampler = resampler if resampler in RESAMPLERS else "targeted"

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    total_rows = int(len(y))
    y_val_arr = np.asarray(y_val, dtype=int)

    def _class_count_map(arr: np.ndarray) -> Dict[str, int]:
        counts = {cls: 0 for cls in classes}
        uniq, cnts = np.unique(arr, return_counts=True)
        for u, c in zip(uniq, cnts):
            counts[classes[int(u)]] = int(c)
        return counts

    def _convert_for_json(value):
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    split_counts = {
        "train": _class_count_map(y_tr),
        "validation": _class_count_map(y_val),
        "total": meta["class_counts"],
    }
    split_payload = {
        "total_rows": total_rows,
        "feature_count_before": len(meta["feature_list_before"]),
        "feature_count_after": len(meta["feature_list_after"]),
        "feature_list_before": meta["feature_list_before"],
        "feature_list_after": meta["feature_list_after"],
        "dropped_features": meta["dropped_features"],
        "dropped_rows_non_target": meta["dropped_rows_non_target"],
        "class_counts": split_counts,
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratified": True,
            "grouped": False,
        },
    }

    Xb, yb = resample_with_strategy(X_tr, y_tr, classes, resampler, random_state=42)
    resampled_counts = _class_count_map(yb)
    split_payload["resampled_train_counts"] = resampled_counts
    split_payload["resampler"] = resampler
    xgb_leg = make_xgb(num_classes=len(classes), monotone=monotone)
    xgb_leg.fit(Xb, yb)
    proba_xgb = xgb_leg.predict_proba(X_val)

    ldam_leg = LDAMDRWXGB(
        classes=classes,
        random_state=42,
        mode=objective_mode,
        resampler=resampler,
        focal_gamma=focal_gamma,
    )
    ldam_leg.fit(X_tr, y_tr, X_val, y_val)
    proba_ldam = ldam_leg.predict_proba(X_val)
    if getattr(ldam_leg, "resampled_counts_", None):
        split_payload["resampled_train_counts_ldam"] = {
            cls: int(ldam_leg.resampled_counts_.get(cls, 0)) for cls in classes
        }

    weights = np.linspace(0.50, 0.90, 9)
    best = {"macro_f1": -1.0}
    tuning_rows = []

    for w in weights:
        p_blend = blend_probs(proba_xgb, proba_ldam, w)
        p_cal, calibrators = fit_isotonic_calibrators(p_blend, y_val, classes)
        th, _ = tune_thresholds(p_cal, y_val, classes)
        y_pred_idx, rep = evaluate_probs_with_thresholds(p_cal, y_val, classes, th)
        macro = f1_score(y_val, y_pred_idx, average="macro")
        bacc = balanced_accuracy_score(y_val, y_pred_idx)
        acc = accuracy_score(y_val, y_pred_idx)
        tuning_rows.append({"w_xgb": w, "macro_f1": macro, "bacc": bacc, "acc": acc, "thresholds": th})
        if macro > best["macro_f1"]:
            best.update(
                dict(
                    w_xgb=w,
                    macro_f1=macro,
                    bacc=bacc,
                    acc=acc,
                    thresholds=th,
                    calibrators=calibrators,
                    proba=p_blend.copy(),
                    report=rep,
                    y_pred=y_pred_idx.copy(),
                )
            )

    pd.DataFrame(tuning_rows).to_csv(tabs_dir / "blend_xgb_ldam_tuning.csv", index=False)

    # Recompute calibrated probabilities with best configuration
    proba_blend_best = blend_probs(proba_xgb, proba_ldam, best["w_xgb"])
    _, ece_per_class_pre, brier_per_class_pre = compute_calibration_stats(y_val_arr, proba_blend_best, classes)
    ece_macro_pre = float(np.nanmean(list(ece_per_class_pre.values()))) if ece_per_class_pre else float("nan")
    brier_macro_pre = float(np.nanmean(list(brier_per_class_pre.values()))) if brier_per_class_pre else float("nan")
    proba_cal_best, calibrators_final = fit_isotonic_calibrators(proba_blend_best, y_val, classes)
    best["calibrators"] = calibrators_final
    th_final, _ = tune_thresholds(proba_cal_best, y_val, classes)
    best["thresholds"] = th_final
    y_pred, rep = evaluate_probs_with_thresholds(proba_cal_best, y_val, classes, th_final)
    best["acc"] = float(accuracy_score(y_val, y_pred))
    best["bacc"] = float(balanced_accuracy_score(y_val, y_pred))
    best["macro_f1"] = float(f1_score(y_val, y_pred, average="macro"))
    best["report"] = rep
    best["y_pred"] = y_pred
    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(classes))))

    rep_csv = tabs_dir / f"{name}_report.csv"
    pd.DataFrame(rep).to_csv(rep_csv)

    cm_png = figs_dir / f"{name}_confusion.png"
    save_confusion_matrix(cm, classes, cm_png, f"{name} Confusion")
    cm_norm_png = figs_dir / f"{name}_confusion_norm.png"
    save_confusion_matrix(cm, classes, cm_norm_png, f"{name} Confusion (row-normalised)", normalize=True)

    roc_png = figs_dir / f"{name}_roc.png"
    pr_png = figs_dir / f"{name}_pr_curve.png"
    topk_png = figs_dir / f"{name}_topk.png"
    macro_auc_plot, per_class_auc_plot = plot_multiclass_roc(y_val_arr, proba_cal_best, classes, roc_png)
    macro_ap_plot, per_class_ap_plot = plot_multiclass_pr(y_val_arr, proba_cal_best, classes, pr_png)
    k_candidates = tuple(dict.fromkeys([1, min(2, len(classes)), min(3, len(classes))]))
    topk_scores = plot_topk_accuracy(y_val_arr, proba_cal_best, classes, topk_png, ks=k_candidates)

    roc_auc_per_class = {cls: (float(per_class_auc_plot.get(cls)) if per_class_auc_plot.get(cls) is not None else None) for cls in classes}
    average_precision_per_class = {
        cls: (float(per_class_ap_plot.get(cls)) if per_class_ap_plot.get(cls) is not None else None) for cls in classes
    }

    try:
        macro_auc_score = float(roc_auc_score(y_val_arr, proba_cal_best, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc_score = float(macro_auc_plot) if macro_auc_plot is not None else None

    try:
        y_bin = label_binarize(y_val_arr, classes=np.arange(len(classes)))
        macro_ap_score = float(average_precision_score(y_bin, proba_cal_best, average="macro"))
    except Exception:
        macro_ap_score = float(macro_ap_plot) if macro_ap_plot is not None else None

    curves, ece_per_class, brier_per_class = compute_calibration_stats(y_val_arr, proba_cal_best, classes)
    reliability_png = figs_dir / f"{name}_reliability.png"
    plot_reliability_diagram(curves, reliability_png, title=f"{name} reliability")
    calib_hist_png = figs_dir / f"{name}_calibration_hist.png"
    plot_probability_histograms(proba_cal_best, classes, calib_hist_png)

    weight_sweep_png = figs_dir / f"{name}_weight_sweep.png"
    plot_weight_sweep(tuning_rows, weight_sweep_png)
    threshold_sweep_png = figs_dir / f"{name}_threshold_sweep.png"
    plot_threshold_sweep(tuning_rows, classes, threshold_sweep_png)

    ece_macro = float(np.nanmean(list(ece_per_class.values()))) if ece_per_class else float("nan")
    brier_macro = float(np.nanmean(list(brier_per_class.values()))) if brier_per_class else float("nan")

    data_stats_json = tabs_dir / f"{name}_data_stats.json"
    with open(data_stats_json, "w") as fh:
        json.dump(split_payload, fh, indent=2)

    bootstrap_json = None
    if bootstrap_samples and bootstrap_samples > 0:
        bs_summary = bootstrap_metrics(
            y_val_arr,
            proba_cal_best,
            classes,
            best["thresholds"],
            n_samples=int(bootstrap_samples),
            random_state=int(bootstrap_random_state),
        )
        bootstrap_json = tabs_dir / f"{name}_bootstrap.json"
        with open(bootstrap_json, "w") as fh:
            json.dump(bs_summary, fh, indent=2)
        split_payload["bootstrap"] = {
            "samples": int(bootstrap_samples),
            "random_state": int(bootstrap_random_state),
            "summary_file": str(bootstrap_json),
        }

    metrics_json = tabs_dir / f"{name}_metrics.json"
    metrics_payload = {
        "accuracy": _convert_for_json(best["acc"]),
        "balanced_accuracy": _convert_for_json(best["bacc"]),
        "macro_f1": _convert_for_json(best["macro_f1"]),
        "macro_roc_auc": _convert_for_json(macro_auc_score),
        "macro_average_precision": _convert_for_json(macro_ap_score),
        "roc_auc_per_class": {k: _convert_for_json(v) for k, v in roc_auc_per_class.items()},
        "average_precision_per_class": {k: _convert_for_json(v) for k, v in average_precision_per_class.items()},
        "top_k_accuracy": {f"top_{k}": _convert_for_json(v) for k, v in topk_scores.items()},
        "blend_weight": _convert_for_json(best["w_xgb"]),
        "probability_thresholds": {k: _convert_for_json(v) for k, v in best["thresholds"].items()},
        "ece_macro": _convert_for_json(ece_macro),
        "brier_macro": _convert_for_json(brier_macro),
        "ece_per_class": {k: _convert_for_json(v) for k, v in ece_per_class.items()},
        "brier_per_class": {k: _convert_for_json(v) for k, v in brier_per_class.items()},
        "ece_macro_pre": _convert_for_json(ece_macro_pre),
        "brier_macro_pre": _convert_for_json(brier_macro_pre),
        "ece_per_class_pre": {k: _convert_for_json(v) for k, v in ece_per_class_pre.items()},
        "brier_per_class_pre": {k: _convert_for_json(v) for k, v in brier_per_class_pre.items()},
    }
    if bootstrap_json is not None:
        metrics_payload["bootstrap_summary_file"] = str(bootstrap_json)
    with open(metrics_json, "w") as fh:
        json.dump(metrics_payload, fh, indent=2)

    xgb_params = {k: _convert_for_json(v) for k, v in xgb_leg.get_params().items()}
    ldam_params = {k: _convert_for_json(v) for k, v in (getattr(ldam_leg, "params_", {}) or {}).items()}
    ldam_margins = [ _convert_for_json(v) for v in (getattr(ldam_leg, "margins_", []) or []) ]
    ldam_class_weights = [ _convert_for_json(v) for v in (getattr(ldam_leg, "class_weights_", []) or []) ]
    ldam_resampled_counts = {
        cls: _convert_for_json(val) for cls, val in (getattr(ldam_leg, "resampled_counts_", {}) or {}).items()
    }
    threshold_grid = [float(v) for v in np.linspace(0.25, 0.60, 8)]

    model_path = models_dir / f"{name}.joblib"
    summary_json = tabs_dir / f"{name}_summary.json"
    summary_payload = {
        "model_name": name,
        "dataset_stats_file": str(data_stats_json),
        "metrics_file": str(metrics_json),
        "classification_report": str(rep_csv),
        "tuning_table": str(tabs_dir / "blend_xgb_ldam_tuning.csv"),
        "features": {
            "before": meta["feature_list_before"],
            "after": meta["feature_list_after"],
            "dropped": meta["dropped_features"],
        },
        "dataset": split_payload,
        "imbalance": {
            "strategy": resampler,
            "resampled_train_counts": resampled_counts,
            "ldam_resampled_counts": ldam_resampled_counts,
            "targeted_smote_params": {"strength_ratio": 1.0, "no_failure_ratio": 0.7, "k_neighbors": 5, "random_state": 42},
        },
        "xgb_params": xgb_params,
        "ldam_params": {
            "mode": ldam_leg.mode,
            "resampler": ldam_leg.resampler,
            "focal_gamma": _convert_for_json(ldam_leg.focal_gamma),
            "booster_params": ldam_params,
            "margins": ldam_margins,
            "class_weights": ldam_class_weights,
        },
        "blend": {
            "weight": _convert_for_json(best["w_xgb"]),
            "thresholds": {k: _convert_for_json(v) for k, v in best["thresholds"].items()},
            "weight_candidates": [float(w) for w in weights],
            "threshold_grid_candidates": threshold_grid,
        },
        "calibration": {
            "type": "per-class isotonic",
            "ece_macro_pre": _convert_for_json(ece_macro_pre),
            "ece_macro_post": _convert_for_json(ece_macro),
            "brier_macro_pre": _convert_for_json(brier_macro_pre),
            "brier_macro_post": _convert_for_json(brier_macro),
            "ece_per_class_post": {k: _convert_for_json(v) for k, v in ece_per_class.items()},
            "brier_per_class_post": {k: _convert_for_json(v) for k, v in brier_per_class.items()},
            "ece_per_class_pre": {k: _convert_for_json(v) for k, v in ece_per_class_pre.items()},
            "brier_per_class_pre": {k: _convert_for_json(v) for k, v in brier_per_class_pre.items()},
            "figures": {
                "reliability": str(reliability_png),
                "probability_hist": str(calib_hist_png),
            },
        },
        "performance": {
            "accuracy": _convert_for_json(best["acc"]),
            "balanced_accuracy": _convert_for_json(best["bacc"]),
            "macro_f1": _convert_for_json(best["macro_f1"]),
            "macro_roc_auc": _convert_for_json(macro_auc_score),
            "macro_average_precision": _convert_for_json(macro_ap_score),
            "roc_auc_per_class": {k: _convert_for_json(v) for k, v in roc_auc_per_class.items()},
            "average_precision_per_class": {k: _convert_for_json(v) for k, v in average_precision_per_class.items()},
            "top_k_accuracy": {f"top_{k}": _convert_for_json(v) for k, v in topk_scores.items()},
        },
        "bootstrap": {
            "samples": int(bootstrap_samples) if bootstrap_samples else 0,
            "random_state": int(bootstrap_random_state),
            "summary_file": str(bootstrap_json) if bootstrap_json else None,
        },
        "figures": {
            "confusion": str(cm_png),
            "confusion_normalised": str(cm_norm_png),
            "roc": str(roc_png),
            "pr": str(pr_png),
            "topk": str(topk_png),
            "weight_sweep": str(weight_sweep_png),
            "threshold_sweep": str(threshold_sweep_png),
        },
        "artifacts": {
            "model_pack": str(model_path),
        },
    }
    with open(summary_json, "w") as fh:
        json.dump(summary_payload, fh, indent=2)

    import joblib

    pack = {
        "blend_name": name,
        "classes": classes,
        "features": list(X.columns),
        "w_xgb": float(best["w_xgb"]),
        "thresholds": {k: float(v) for k, v in best["thresholds"].items()},
        "xgb_leg": {"type": "xgb_smote_nophys", "model": xgb_leg},
        "ldam_leg": {"type": "ldam_drw_xgb_nophys", "scaler": ldam_leg.scaler, "booster": ldam_leg.booster},
        "calibrators": best.get("calibrators", {}),
        "monotone_constraints": monotone,
    }
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(pack, model_path)

    return HybridResult(
        name=name,
        acc=float(best["acc"]),
        bacc=float(best["bacc"]),
        macro_f1=float(best["macro_f1"]),
        report_csv=rep_csv,
        cm_png=cm_png,
        model_path=model_path,
        thresholds={k: float(v) for k, v in best["thresholds"].items()},
        roc_png=roc_png,
        pr_png=pr_png,
        topk_png=topk_png,
        metrics_json=metrics_json,
        data_stats_json=data_stats_json,
        summary_json=summary_json,
        bootstrap_json=bootstrap_json,
    )


def _prepare_Xy_groups(df: pd.DataFrame, do_prune: bool = True) -> Tuple[pd.DataFrame, np.ndarray, List[str], pd.Series]:
    feat_cols = [c for c in SAFE_FEATURES if c in df.columns]
    X = df[feat_cols].copy()
    y_raw = df["target"].astype(str).str.strip().replace(
        {"0": "No Failure", "1": "Strength Failure", "2": "Deflection Failure", "3": "Other"}
    )
    valid = ["No Failure", "Strength Failure", "Deflection Failure"]
    mask = y_raw.isin(valid)
    X, y_raw = X[mask], y_raw[mask]
    groups = df.loc[mask, "BN"].astype(str) if "BN" in df.columns else pd.Series(["all"] * len(X), index=X.index)
    if do_prune:
        keep = prune_collinear(X, 0.97)
        X = X[keep]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    return X.reset_index(drop=True), y, classes, groups.reset_index(drop=True)


def cv_train_and_eval(
    config_path: str | os.PathLike | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    name: str = "blend_xgb_ldam_nophys",
    objective_mode: str = "ldam_drw",
    resampler: str = "targeted",
    focal_gamma: float = 2.0,
) -> Tuple[Path, Dict, Path]:
    """Grouped CV with nested threshold tuning and fold leaderboard output."""
    cfg = load_config(config_path)
    paths_cfg = cfg["paths"]
    outs_dir = Path(paths_cfg["outputs"]) / "tables"
    figs_dir = Path(paths_cfg["outputs"]) / "figs"
    ensure_dirs(outs_dir, figs_dir)

    df = load_processed_dataset(paths_cfg)
    do_prune = bool(cfg.get("data", {}).get("prune_collinear", True))
    X, y, classes, groups = _prepare_Xy_groups(df, do_prune=do_prune)
    monotone = build_monotone_constraints(list(X.columns))
    objective_mode = objective_mode if objective_mode in OBJECTIVE_MODES else "ldam_drw"
    resampler = resampler if resampler in RESAMPLERS else "targeted"
    monotone = build_monotone_constraints(list(X.columns))

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows: List[Dict] = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # inner split for threshold tuning
        X_tr_in, X_val_in, y_tr_in, y_val_in = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=random_state, stratify=y_tr
        )

        # train legs on inner train
        Xb_in, yb_in = resample_with_strategy(X_tr_in, y_tr_in, classes, resampler, random_state=random_state)
        xgb_leg = make_xgb(num_classes=len(classes), monotone=monotone)
        xgb_leg.fit(Xb_in, yb_in)
        proba_xgb_val = xgb_leg.predict_proba(X_val_in)

        ldam_leg = LDAMDRWXGB(
            classes=classes,
            random_state=random_state,
            mode=objective_mode,
            resampler=resampler,
            focal_gamma=focal_gamma,
        )
        ldam_leg.fit(X_tr_in, y_tr_in, X_val_in, y_val_in)
        proba_ldam_val = ldam_leg.predict_proba(X_val_in)

        best = {"macro_f1": -1.0}
        for w in np.linspace(0.50, 0.90, 9):
            p_blend = blend_probs(proba_xgb_val, proba_ldam_val, w)
            p_cal, calibrators = fit_isotonic_calibrators(p_blend, y_val_in, classes)
            th, _ = tune_thresholds(p_cal, y_val_in, classes)
            pred_idx, _ = evaluate_probs_with_thresholds(p_cal, y_val_in, classes, th)
            macro = f1_score(y_val_in, pred_idx, average="macro")
            if macro > best["macro_f1"]:
                best.update(dict(
                    w_xgb=float(w),
                    thresholds={k: float(v) for k, v in th.items()},
                    macro_f1=float(macro),
                    calibrators=calibrators,
                ))

        # ensure thresholds/calibrators align with best weight
        p_blend_best_val = blend_probs(proba_xgb_val, proba_ldam_val, best["w_xgb"])
        p_cal_best_val, calibrators_best_val = fit_isotonic_calibrators(p_blend_best_val, y_val_in, classes)
        th_best_val, _ = tune_thresholds(p_cal_best_val, y_val_in, classes)
        best["calibrators"] = calibrators_best_val
        best["thresholds"] = {k: float(v) for k, v in th_best_val.items()}

        # retrain on outer train
        Xb, yb = resample_with_strategy(X_tr, y_tr, classes, resampler, random_state=random_state)
        xgb_full = make_xgb(num_classes=len(classes), monotone=monotone)
        xgb_full.fit(Xb, yb)
        proba_xgb_te = xgb_full.predict_proba(X_te)

        ldam_full = LDAMDRWXGB(
            classes=classes,
            random_state=random_state,
            mode=objective_mode,
            resampler=resampler,
            focal_gamma=focal_gamma,
        )
        ldam_full.fit(X_tr, y_tr, X_te, y_te)
        proba_ldam_te = ldam_full.predict_proba(X_te)

        proba_te = blend_probs(proba_xgb_te, proba_ldam_te, best["w_xgb"])
        proba_te = apply_calibrators(proba_te, best.get("calibrators", {}), classes)
        pred_idx_te, _ = evaluate_probs_with_thresholds(proba_te, y_te, classes, best["thresholds"])
        acc = accuracy_score(y_te, pred_idx_te)
        bacc = balanced_accuracy_score(y_te, pred_idx_te)
        macro = f1_score(y_te, pred_idx_te, average="macro")
        fold_rows.append({
            "fold": fold,
            "acc": float(acc),
            "bacc": float(bacc),
            "macro_f1": float(macro),
            "w_xgb": best["w_xgb"],
            "thresholds": best["thresholds"],
            "objective_mode": objective_mode,
            "resampler": resampler,
        })

    df_rows = pd.DataFrame(fold_rows)
    agg = df_rows[["acc", "bacc", "macro_f1"]].agg(["mean", "std"]).reset_index()
    leaderboard = outs_dir / f"cv_leaderboard_{name}.csv"
    with leaderboard.open("w") as fh:
        df_rows.to_csv(fh, index=False)
        fh.write("\n")
        agg.to_csv(fh, index=False)

    print("\n=== GROUPED CV (BN) RESULTS ===")
    print(df_rows)
    print("\nAggregate (mean/std):\n", agg)
    summary = {
        "objective_mode": objective_mode,
        "resampler": resampler,
        "macro_f1_mean": float(agg.loc[agg["index"] == "mean", "macro_f1"].values[0]),
        "macro_f1_std": float(agg.loc[agg["index"] == "std", "macro_f1"].values[0]),
        "bacc_mean": float(agg.loc[agg["index"] == "mean", "bacc"].values[0]),
        "bacc_std": float(agg.loc[agg["index"] == "std", "bacc"].values[0]),
        "acc_mean": float(agg.loc[agg["index"] == "mean", "acc"].values[0]),
        "acc_std": float(agg.loc[agg["index"] == "std", "acc"].values[0]),
        "n_splits": int(n_splits),
        "random_state": int(random_state),
    }
    summary_json = outs_dir / f"cv_summary_{name}.json"
    with summary_json.open("w") as fh:
        json.dump(summary, fh, indent=2)
    return leaderboard, summary, summary_json


def run_ablation_grid(
    config_path: str | os.PathLike | None = None,
    objective_modes: Optional[List[str]] = None,
    resamplers: Optional[List[str]] = None,
    focal_gamma: float = 2.0,
    n_splits: int = 5,
    random_state: int = 42,
    name: str = "blend_xgb_ldam_nophys",
) -> Path:
    cfg = load_config(config_path)
    outs_dir = Path(cfg["paths"]["outputs"]) / "tables"
    ensure_dirs(outs_dir)

    objective_modes = objective_modes or ["ldam_drw", "logit_adjusted", "focal"]
    resamplers = resamplers or ["targeted", "borderline", "smoteenn"]

    rows = []
    for obj in objective_modes:
        for res in resamplers:
            try:
                leaderboard, summary = cv_train_and_eval(
                    config_path=config_path,
                    n_splits=n_splits,
                    random_state=random_state,
                    name=f"{name}_{obj}_{res}",
                    objective_mode=obj,
                    resampler=res,
                    focal_gamma=focal_gamma,
                )
                summary["leaderboard"] = str(leaderboard)
                rows.append(summary)
            except Exception as exc:  # pragma: no cover
                rows.append(
                    {
                        "objective_mode": obj,
                        "resampler": res,
                        "error": str(exc),
                    }
                )

    df = pd.DataFrame(rows)
    out_path = outs_dir / "ablation_results.csv"
    df.to_csv(out_path, index=False)
    print("\nAblation grid saved →", out_path)
    return out_path


__all__ = [
    "train_hybrid_classifier",
    "HybridResult",
    "SAFE_FEATURES",
    "cv_train_and_eval",
    "run_ablation_grid",
]

# Cross-validation support added
# Ablation grid support added
