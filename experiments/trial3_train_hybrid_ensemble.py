from __future__ import annotations
import os, json, argparse, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier

# Resampling choices
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# Optional libs
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    import shap
except Exception:
    shap = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------- IO / CONFIG -------------------------
def load_cfg(p: str) -> dict:
    import yaml
    with open(p, "r") as f:
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

# ------------------------- FEATURES -------------------------
SAFE_FEATURES = [
    "L","Ac","Cc","As","Af","tins","hi","fc","fy","Es","fu",
    "Efrp","Tg","kins","rinscins","Ld","LR"
]

def add_physics_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    eps = 1e-9
    def sdiv(a,b):
        b = np.where(np.abs(b) > eps, b, np.nan)
        out = a / b
        return np.nan_to_num(out, nan=0.0)

    if {"As","Ac"}.issubset(X.columns): X["rho_s_As_over_Ac"] = sdiv(X["As"], X["Ac"])
    if {"Af","Ac"}.issubset(X.columns): X["rho_f_Af_over_Ac"] = sdiv(X["Af"], X["Ac"])
    if {"Af","As"}.issubset(X.columns): X["Af_over_As"]       = sdiv(X["Af"], X["As"])
    if {"Ld","L"}.issubset(X.columns):  X["Ld_over_L"]        = sdiv(X["Ld"], X["L"])
    if {"LR","fc"}.issubset(X.columns): X["LR_times_fc"]      = X["LR"] * X["fc"]
    if {"Efrp","Af","Es","As"}.issubset(X.columns):
        X["EfrpAf_over_EsAs"] = sdiv(X["Efrp"]*X["Af"], X["Es"]*X["As"])
    if {"tins","kins","Tg"}.issubset(X.columns):
        X["thermal_index"] = sdiv(X["tins"]*X["kins"], X["Tg"])
    return X

def prune_collinear(X: pd.DataFrame, thresh: float = 0.97) -> List[str]:
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > thresh)]
    return [c for c in X.columns if c not in drop_cols]

# ------------------------- DATA -------------------------
def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, List[str]]:
    feat_cols = [c for c in SAFE_FEATURES if c in df.columns]
    X = df[feat_cols].copy()

    y_raw = df["target"].astype(str).str.strip().replace({
        "0": "No Failure",
        "1": "Strength Failure",
        "2": "Deflection Failure",
        "3": "Other",
    })
    valid = ["No Failure","Strength Failure","Deflection Failure"]
    mask = y_raw.isin(valid)
    X, y_raw = X[mask], y_raw[mask]

    X = add_physics_features(X)
    keep = prune_collinear(X, 0.97)
    X = X[keep]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le, list(le.classes_)

# ------------------------- CLASS WEIGHTS -------------------------
def make_sample_weights(y: np.ndarray) -> np.ndarray:
    # inverse-frequency weights normalized to mean=1
    uniq, cnt = np.unique(y, return_counts=True)
    freq = {u:c for u,c in zip(uniq, cnt)}
    inv = {u: 1.0 / c for u,c in freq.items()}
    w = np.array([inv[i] for i in y], dtype=float)
    return w / w.mean()

# ------------------------- RESAMPLING -------------------------
def resample_train(X, y, classes_: List[str], method: str = "smotetomek",
                   strength_ratio=1.0, nofail_ratio=0.7, random_state=42):
    """
    method in {"smotetomek","smote","adasyn","none"}
    """
    _, counts = np.unique(y, return_counts=True)
    majority = int(counts.max())
    name_to_idx = {name:i for i,name in enumerate(classes_)}

    strat = {}
    if "Strength Failure" in name_to_idx:
        strat[name_to_idx["Strength Failure"]] = int(majority * strength_ratio)
    if "No Failure" in name_to_idx:
        target_nf = int(majority * nofail_ratio)
        cur_nf = counts[name_to_idx["No Failure"]]
        strat[name_to_idx["No Failure"]] = max(target_nf, int(cur_nf))
    if method == "none" or not strat:
        return X, y

    if method == "smotetomek":
        sampler = SMOTETomek(random_state=random_state, sampling_strategy=strat)
    elif method == "smote":
        sampler = SMOTE(random_state=random_state, sampling_strategy=strat, k_neighbors=5)
    elif method == "adasyn":
        sampler = ADASYN(random_state=random_state, sampling_strategy=strat, n_neighbors=5)
    else:
        sampler = SMOTETomek(random_state=random_state, sampling_strategy=strat)
    return sampler.fit_resample(X, y)

# ------------------------- MODELS -------------------------
@dataclass
class CandidateResult:
    name: str
    acc: float
    bacc: float
    macro_f1: float
    report_csv: str
    cm_png: str
    model_path: str
    thresholds: Dict[str, float]

def make_lgbm(num_classes: int):
    if LGBMClassifier is None:
        return None
    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=800,
        learning_rate=0.06,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1
    )

def make_xgb(num_classes: int):
    if XGBClassifier is None:
        return None
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=700,
        learning_rate=0.06,
        max_depth=6,
        min_child_weight=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        reg_alpha=0.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    )

def make_brf(num_classes: int):
    return RandomForestClassifier(
        n_estimators=1200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

# ------------------------- EVAL / PLOTS -------------------------
def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title="Confusion"):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=20)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=9)
    plt.title(title); plt.colorbar(); plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def evaluate_probs_with_thresholds(proba: np.ndarray, y_true: np.ndarray, classes: List[str], th: Dict[str,float]) -> Tuple[np.ndarray, Dict]:
    pred_idx = proba.argmax(axis=1).copy()
    for ci, cname in enumerate(classes):
        t = th.get(cname, None)
        if t is None: continue
        mask = proba[:, ci] >= t
        pred_idx[mask] = ci
    report = classification_report(y_true, pred_idx, target_names=classes, output_dict=True)
    return pred_idx, report

def tune_thresholds(proba: np.ndarray, y_true: np.ndarray, classes: List[str]) -> Tuple[Dict[str,float], float]:
    grid = {c: np.linspace(0.20, 0.60, 9) for c in classes}
    if "Deflection Failure" in grid: grid.pop("Deflection Failure", None)
    keys = list(grid.keys())

    best_f1, best = -1.0, {}
    def recurse(i, cur):
        nonlocal best_f1, best
        if i == len(keys):
            th = cur.copy()
            pred = proba.argmax(axis=1).copy()
            for ci, cname in enumerate(classes):
                t = th.get(cname, None)
                if t is not None:
                    pred[proba[:, ci] >= t] = ci
            f1 = f1_score(y_true, pred, average="macro")
            if f1 > best_f1:
                best_f1, best = f1, th.copy()
            return
        k = keys[i]
        for t in grid[k]:
            cur[k] = float(t)
            recurse(i+1, cur)
    recurse(0, {})
    return best, best_f1

# ------------------------- SHAP / EXPLAIN -------------------------
def explain_model(model, X_val: pd.DataFrame, classes: List[str], figs_dir: str, prefix: str):
    if shap is None:
        return
    try:
        explainer = shap.TreeExplainer(model)
        Xs = X_val.sample(min(2000, len(X_val)), random_state=42)
        sv = explainer.shap_values(Xs)
        plt.figure()
        shap.summary_plot(sv, Xs, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(figs_dir, f"{prefix}_shap_summary.png"), dpi=200); plt.close()
    except Exception:
        pass

# ------------------------- TRAIN / SELECT -------------------------
@dataclass
class FitPaths:
    tabs_dir: str
    figs_dir: str
    models_dir: str

@dataclass
class ModelPack:
    name: str
    model: object
    classes: List[str]
    features: List[str]
    thresholds: Dict[str, float]

def train_one(name: str, model, X_tr, y_tr, X_val, y_val, classes: List[str], paths: FitPaths,
              use_class_weights: bool=False) -> CandidateResult:
    import joblib

    # per-sample weights for class imbalance (inverse frequency)
    sample_w = make_sample_weights(y_tr) if use_class_weights else None

    model.fit(X_tr, y_tr, sample_weight=sample_w)

    if hasattr(model, "predict_proba"):
        proba_val = model.predict_proba(X_val)
    else:
        preds = model.predict(X_val)
        proba_val = np.zeros((len(preds), len(classes)))
        for i, p in enumerate(preds):
            proba_val[i, p] = 1.0

    th, _ = tune_thresholds(proba_val, y_val, classes)
    y_pred_idx, report_dict = evaluate_probs_with_thresholds(proba_val, y_val, classes, th)

    acc  = accuracy_score(y_val, y_pred_idx)
    bacc = balanced_accuracy_score(y_val, y_pred_idx)
    macro_f1 = f1_score(y_val, y_pred_idx, average="macro")
    cm = confusion_matrix(y_val, y_pred_idx, labels=list(range(len(classes))))

    rep_csv = os.path.join(paths.tabs_dir, f"{name}_classification_report.csv")
    pd.DataFrame(report_dict).to_csv(rep_csv)
    cm_png = os.path.join(paths.figs_dir, f"{name}_confusion.png")
    save_confusion_matrix(cm, classes, cm_png, f"{name} Confusion")

    mpath = os.path.join(paths.models_dir, f"{name}.joblib")
    joblib.dump({"model": model, "classes": classes, "features": list(X_tr.columns), "thresholds": th}, mpath)

    return CandidateResult(
        name=name, acc=acc, bacc=bacc, macro_f1=macro_f1,
        report_csv=rep_csv, cm_png=cm_png, model_path=mpath, thresholds={k:float(v) for k,v in th.items()}
    )

def soft_vote_top2(packs: List[ModelPack], X_val: pd.DataFrame) -> np.ndarray:
    probas = []
    for p in packs:
        if hasattr(p.model, "predict_proba"):
            probas.append(p.model.predict_proba(X_val))
        else:
            preds = p.model.predict(X_val)
            P = np.zeros((len(preds), len(p.classes)))
            for i, y in enumerate(preds): P[i, y] = 1.0
            probas.append(P)
    return np.mean(probas, axis=0)

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    paths_cfg = cfg["paths"]
    outs = paths_cfg["outputs"]
    figs_dir = os.path.join(outs, "figs")
    tabs_dir = os.path.join(outs, "tables")
    models_dir = os.path.join(paths_cfg.get("models","models"), "checkpoints")
    ensure_dirs(figs_dir, tabs_dir, models_dir)
    fpaths = FitPaths(tabs_dir=tabs_dir, figs_dir=figs_dir, models_dir=models_dir)

    # RESAMPLER selection from config (default smotetomek)
    resampler = cfg.get("train", {}).get("resampler", "smotetomek")
    strength_ratio = cfg.get("train", {}).get("strength_ratio", 1.0)
    nofail_ratio   = cfg.get("train", {}).get("nofail_ratio", 0.7)
    use_weights    = cfg.get("train", {}).get("use_class_weights", True)
    do_softvote    = cfg.get("train", {}).get("soft_vote_top2", True)

    # data
    df = load_processed(paths_cfg)
    X, y, le, classes = prepare_data(df)

    # split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # resample (train only)
    Xb, yb = resample_train(X_tr, y_tr, classes, method=resampler,
                            strength_ratio=strength_ratio, nofail_ratio=nofail_ratio, random_state=42)

    candidates: List[CandidateResult] = []
    trained_packs: List[ModelPack] = []

    # LightGBM + weights
    if LGBMClassifier is not None:
        lgbm = make_lgbm(num_classes=len(classes))
        r = train_one("hybrid_lgbm_"+resampler, lgbm, Xb, yb, X_val, y_val, classes, fpaths, use_class_weights=use_weights)
        candidates.append(r)
        import joblib
        trained_packs.append(ModelPack(r.name, joblib.load(r.model_path)["model"], classes, list(X.columns), r.thresholds))

    # XGBoost + weights
    if XGBClassifier is not None:
        xgb = make_xgb(num_classes=len(classes))
        r = train_one("hybrid_xgb_"+resampler, xgb, Xb, yb, X_val, y_val, classes, fpaths, use_class_weights=use_weights)
        candidates.append(r)
        import joblib
        trained_packs.append(ModelPack(r.name, joblib.load(r.model_path)["model"], classes, list(X.columns), r.thresholds))

    # Balanced RF (no weights passed, uses class_weight)
    brf = make_brf(num_classes=len(classes))
    r = train_one("hybrid_brf", brf, X_tr, y_tr, X_val, y_val, classes, fpaths, use_class_weights=False)
    candidates.append(r)
    import joblib
    trained_packs.append(ModelPack(r.name, joblib.load(r.model_path)["model"], classes, list(X.columns), r.thresholds))

    # leaderboard
    lb = pd.DataFrame([{
        "name": c.name, "macro_f1": c.macro_f1, "bacc": c.bacc, "acc": c.acc,
        "report": c.report_csv, "cm": c.cm_png, "model": c.model_path, "thresholds": c.thresholds
    } for c in candidates]).sort_values(by=["macro_f1","bacc"], ascending=False)
    lb_path = os.path.join(tabs_dir, f"hybrid_{resampler}_leaderboard.csv")
    lb.to_csv(lb_path, index=False)

    # winner by macro-F1
    best = candidates[np.argmax([c.macro_f1 for c in candidates])]

    # optional: soft-vote of top-2
    if do_softvote and len(candidates) >= 2:
        top2 = lb.head(2)["name"].tolist()
        packs2 = [p for p in trained_packs if p.name in top2]
        P = soft_vote_top2(packs2, X_val)
        th, _ = tune_thresholds(P, y_val, classes)
        y_pred_idx, report_dict = evaluate_probs_with_thresholds(P, y_val, classes, th)

        acc  = accuracy_score(y_val, y_pred_idx)
        bacc = balanced_accuracy_score(y_val, y_pred_idx)
        macro_f1 = f1_score(y_val, y_pred_idx, average="macro")
        cm = confusion_matrix(y_val, y_pred_idx, labels=list(range(len(classes))))

        # save artifacts
        rep_csv = os.path.join(tabs_dir, f"hybrid_softvote_{resampler}_classification_report.csv")
        pd.DataFrame(report_dict).to_csv(rep_csv)
        cm_png = os.path.join(figs_dir, f"hybrid_softvote_{resampler}_confusion.png")
        save_confusion_matrix(cm, classes, cm_png, f"SoftVote({top2[0]}+{top2[1]}) Confusion")

        # decide if soft-vote beats best
        if macro_f1 > best.macro_f1 or (macro_f1 == best.macro_f1 and bacc > best.bacc):
            # pack a pseudo-model: store probs via averaging at inference (document in README)
            # here we just report it as winner; for deployment, keep top single winner unless you want to persist an averaging wrapper.
            best = CandidateResult(
                name=f"hybrid_softvote_{resampler}",
                acc=acc, bacc=bacc, macro_f1=macro_f1,
                report_csv=rep_csv, cm_png=cm_png,
                model_path="(soft-vote of top-2)", thresholds=th
            )

    # SHAP for best single-model if possible
    try:
        if "softvote" not in best.name:
            import joblib
            pack = joblib.load(best.model_path)
            explain_model(pack["model"], X_val, classes, figs_dir, prefix=f"{best.name}")
    except Exception:
        pass

    print("\n=== HYBRID ENSEMBLE RESULTS (sorted by macro-F1) ===")
    print(lb[["name","macro_f1","bacc","acc"]].to_string(index=False))
    print(f"\nWinner: {best.name}")
    print(f"  Macro-F1={best.macro_f1:.4f}  BAcc={best.bacc:.4f}  Acc={best.acc:.4f}")
    print(f"  Model → {best.model_path}")
    print(f"  Report → {best.report_csv}")
    print(f"  Confusion → {best.cm_png}")
    print(f"  Leaderboard → {lb_path}")
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()
    main(args.config)