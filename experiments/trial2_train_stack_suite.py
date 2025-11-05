# scripts/04_train_stack.py
from __future__ import annotations
import os, argparse, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional libs (graceful skip if missing)
try:
    from xgboost import XGBClassifier, DMatrix, train as xgb_train
    import xgboost as xgb
except Exception:
    XGBClassifier, xgb_train, xgb = None, None, None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from imblearn.over_sampling import BorderlineSMOTE
except Exception:
    BorderlineSMOTE = None

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
# RAW ONLY (no physics ratios)
SAFE_FEATURES = [
    "L","Ac","Cc","As","Af","tins","hi","fc","fy","Es","fu",
    "Efrp","Tg","kins","rinscins","Ld","LR"
]

def prune_collinear(X: pd.DataFrame, thresh: float = 0.97) -> List[str]:
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > thresh)]
    return [c for c in X.columns if c not in drop_cols]

# ------------------------- DATA -------------------------
def prepare_data(df: pd.DataFrame, do_prune: bool = True) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, List[str]]:
    # raw features only
    feat_cols = [c for c in SAFE_FEATURES if c in df.columns]
    X = df[feat_cols].copy()

    # labels
    y_raw = df["target"].astype(str).str.strip().replace({
        "0": "No Failure",
        "1": "Strength Failure",
        "2": "Deflection Failure",
        "3": "Other",
    })
    valid = ["No Failure","Strength Failure","Deflection Failure"]
    mask = y_raw.isin(valid)
    X, y_raw = X[mask], y_raw[mask]

    # optional collinearity prune
    if do_prune:
        keep = prune_collinear(X, 0.97)
        X = X[keep]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le, list(le.classes_)

# ------------------------- RESAMPLING -------------------------
def targeted_smote(X, y, classes_: List[str], strength_ratio=1.0, nofail_ratio=0.7, random_state=42):
    # up-sample Strength to majority; No Failure to a fraction; leave Deflection
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
    if not strat:
        return X, y
    sm = SMOTE(random_state=random_state, sampling_strategy=strat, k_neighbors=5)
    Xb, yb = sm.fit_resample(X, y)
    return Xb, yb

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

def make_xgb(num_classes: int):
    if XGBClassifier is None:
        return None
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
    )

def make_lgbm(num_classes: int):
    if LGBMClassifier is None:
        return None
    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=750,
        learning_rate=0.06,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

def make_cat(num_classes: int):
    if CatBoostClassifier is None:
        return None
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=750,
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
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

# ------------------------- THRESHOLDS / EVAL -------------------------
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
    report = classification_report(y_true, pred_idx, target_names=classes, output_dict=True, zero_division=0)
    return pred_idx, report

def tune_thresholds(proba: np.ndarray, y_true: np.ndarray, classes: List[str]) -> Tuple[Dict[str,float], float]:
    # tune only minority thresholds
    cand = {c: np.linspace(0.25, 0.60, 8) for c in classes}
    if "Deflection Failure" in cand: cand.pop("Deflection Failure", None)
    keys = list(cand.keys())
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
        for t in cand[k]:
            cur[k] = float(t)
            recurse(i+1, cur)
    recurse(0, {})
    return best, best_f1

# ------------------------- FRIEND MODEL: LDAM-DRW (XGB), RAW ONLY -------------------------
class LDAMDRWXGB:
    """XGBoost + LDAM margins + DRW weights + BorderlineSMOTE + StandardScaler, using RAW features only."""
    def __init__(self, classes: List[str], random_state: int = 42):
        self.classes = classes
        self.scaler: Optional[StandardScaler] = None
        self.booster = None
        self.thresholds: Dict[str,float] = {}
        self.random_state = random_state

    @staticmethod
    def _class_balanced_weights(y, beta=0.999):
        from collections import Counter
        c = Counter(y); cls = sorted(c.keys())
        eff = np.array([1.0 - beta**c[i] for i in cls], float); eff[eff<=0]=1e-8
        w = (1.0 - beta) / eff; w = w / w.mean()
        out = np.zeros(max(cls)+1, float)
        for i,k in enumerate(cls): out[k] = w[i]
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
            p = np.exp(logits - m); p = p / p.sum(axis=1, keepdims=True)
            oh = np.zeros_like(p); oh[np.arange(N), y] = 1.0
            grad = (p - oh)
            hess = p * (1.0 - p)
            return grad, hess
        return _obj

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X.values) if self.scaler is not None else X.values
        return self.booster.predict(DMatrix(Xs))

    def fit(self, X_tr: pd.DataFrame, y_tr: np.ndarray,
            X_val: pd.DataFrame, y_val: np.ndarray):
        assert xgb_train is not None, "xgboost is required for LDAMDRWXGB"
        K = len(self.classes)
        # scale
        self.scaler = StandardScaler().fit(X_tr.values)
        Xs_tr = self.scaler.transform(X_tr.values)
        Xs_val = self.scaler.transform(X_val.values)
        # BorderlineSMOTE on train
        if BorderlineSMOTE is None:
            raise ImportError("imblearn BorderlineSMOTE not installed")
        from collections import Counter as Cn
        cnt = Cn(y_tr); maj = max(cnt.values())
        ratios = {"Deflection Failure": 1.0, "Strength Failure": 0.8, "No Failure": 0.6}
        by_idx = {i: ratios.get(c, 1.0) for i,c in enumerate(self.classes)}
        target_counts = {c: max(cnt[c], int(round(maj * by_idx[c]))) for c in cnt.keys()}
        bsmote = BorderlineSMOTE(kind='borderline-1', k_neighbors=5, random_state=42,
                                 sampling_strategy=target_counts)
        Xs_tr_res, y_tr_res = bsmote.fit_resample(Xs_tr, y_tr)
        # LDAM margins + DRW
        margins = self._compute_margins_ldam(y_tr_res, K, Cn(y_tr_res), max_m=0.5)
        cbw = self._class_balanced_weights(y_tr_res, beta=0.999)
        w_train = np.array([cbw[c] for c in y_tr_res], float)
        dtrain = DMatrix(Xs_tr_res, label=y_tr_res, weight=w_train)
        dval   = DMatrix(Xs_val,    label=y_val)
        params = {
            'num_class': K,
            'eta': 0.05, 'max_depth': 5,
            'subsample': 0.9, 'colsample_bytree': 0.9,
            'lambda': 1.0, 'objective': 'multi:softprob',
            'eval_metric': 'mlogloss', 'nthread': -1, 'seed': 42
        }
        booster = xgb_train(params, dtrain, num_boost_round=200,
                            obj=self._ldam_softmax_obj(margins, K),
                            evals=[(dtrain,'train'),(dval,'val')],
                            early_stopping_rounds=50, verbose_eval=False)
        booster = xgb_train(params, dtrain, num_boost_round=300,
                            obj=self._ldam_softmax_obj(margins, K),
                            evals=[(dtrain,'train_cb'),(dval,'val')],
                            early_stopping_rounds=50, verbose_eval=False,
                            xgb_model=booster)
        self.booster = booster
        # tune thresholds on validation
        proba_val = self.predict_proba(X_val)
        th, _ = tune_thresholds(proba_val, y_val, self.classes)
        self.thresholds = th

# ------------------------- EVAL / SAVE -------------------------
@dataclass
class FitPaths:
    tabs_dir: str
    figs_dir: str
    models_dir: str

def save_conf_and_pack(name: str, proba_val: np.ndarray, y_val: np.ndarray, classes: List[str],
                       th: Dict[str, float], X_cols: List[str], model_obj,
                       paths: FitPaths) -> "CandidateResult":
    import joblib
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
    joblib.dump({"model": model_obj, "classes": classes, "features": X_cols, "thresholds": th}, mpath)
    return CandidateResult(
        name=name, acc=acc, bacc=bacc, macro_f1=macro_f1,
        report_csv=rep_csv, cm_png=cm_png, model_path=mpath, thresholds={k: float(v) for k, v in th.items()}
    )

def train_and_eval_single(name: str, model, X_train, y_train, X_val, y_val, classes: List[str], paths: FitPaths) -> CandidateResult:
    if isinstance(model, LDAMDRWXGB):
        model.fit(X_train, y_train, X_val, y_val)
        proba_val = model.predict_proba(X_val)
        th, _ = tune_thresholds(proba_val, y_val, classes)
        model.thresholds = th
        return save_conf_and_pack(name, proba_val, y_val, classes, th, list(X_train.columns), model, paths)
    else:
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            proba_val = model.predict_proba(X_val)
        else:
            preds = model.predict(X_val)
            proba_val = np.zeros((len(preds), len(classes)))
            for i, p in enumerate(preds):
                proba_val[i, p] = 1.0
        th, _ = tune_thresholds(proba_val, y_val, classes)
        return save_conf_and_pack(name, proba_val, y_val, classes, th, list(X_train.columns), model, paths)

# ------------------------- MAIN -------------------------
def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    paths_cfg = cfg["paths"]
    outs = paths_cfg["outputs"]
    figs_dir = os.path.join(outs, "figs")
    tabs_dir = os.path.join(outs, "tables")
    models_dir = os.path.join(paths_cfg.get("models","models"), "checkpoints")
    ensure_dirs(figs_dir, tabs_dir, models_dir)
    fpaths = FitPaths(tabs_dir=tabs_dir, figs_dir=figs_dir, models_dir=models_dir)

    # data
    df = load_processed(paths_cfg)
    # RAW features only; keep prune True by default
    do_prune = bool(cfg.get("data", {}).get("prune_collinear", True))
    X, y, le, classes = prepare_data(df, do_prune=do_prune)

    # split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates: List[CandidateResult] = []

    # === Targeted SMOTE for your suite
    Xb, yb = targeted_smote(X_tr, y_tr, classes, strength_ratio=1.0, nofail_ratio=0.7, random_state=42)

    # LGBM
    if LGBMClassifier is not None:
        lgbm = make_lgbm(num_classes=len(classes))
        if lgbm is not None:
            candidates.append(train_and_eval_single("suite_lgbm_smote_nophys", lgbm, Xb, yb, X_val, y_val, classes, fpaths))

    # XGB
    if XGBClassifier is not None:
        xgbm = make_xgb(num_classes=len(classes))
        if xgbm is not None:
            candidates.append(train_and_eval_single("suite_xgb_smote_nophys", xgbm, Xb, yb, X_val, y_val, classes, fpaths))

    # CatBoost
    if CatBoostClassifier is not None:
        cat = make_cat(num_classes=len(classes))
        if cat is not None:
            candidates.append(train_and_eval_single("suite_cat_smote_nophys", cat, Xb, yb, X_val, y_val, classes, fpaths))

    # Balanced RF (no SMOTE)
    brf = make_brf(num_classes=len(classes))
    candidates.append(train_and_eval_single("suite_brf_balanced_nophys", brf, X_tr, y_tr, X_val, y_val, classes, fpaths))

    # Friend’s LDAM-DRW (BorderlineSMOTE) on RAW features
    if xgb_train is not None and BorderlineSMOTE is not None:
        ldam_model = LDAMDRWXGB(classes=classes, random_state=42)
        candidates.append(train_and_eval_single("suite_ldam_drw_xgb_nophys", ldam_model, X_tr, y_tr, X_val, y_val, classes, fpaths))

    # leaderboard
    lb = pd.DataFrame([{
        "name": c.name, "macro_f1": c.macro_f1, "bacc": c.bacc, "acc": c.acc,
        "report": c.report_csv, "cm": c.cm_png, "model": c.model_path,
        "thresholds": c.thresholds
    } for c in candidates]).sort_values(by=["macro_f1","bacc"], ascending=False)
    lb_path = os.path.join(tabs_dir, "suite_leaderboard.csv")
    lb.to_csv(lb_path, index=False)

    # winner
    best = candidates[np.argmax([c.macro_f1 for c in candidates])]

    print("\n=== MODEL SUITE RESULTS (sorted by macro-F1) ===")
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