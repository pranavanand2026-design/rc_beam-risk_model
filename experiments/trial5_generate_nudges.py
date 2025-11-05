# scripts/10_generate_nudges.py  (Nudge v3, SHAP-free, surrogate-aware)
from __future__ import annotations
import os, argparse, json, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- IO / CFG ----------------
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

# -------------- FEATURES (match training logic) --------------
SAFE_FEATURES = [
    "L","Ac","Cc","As","Af","tins","hi","fc","fy","Es","fu",
    "Efrp","Tg","kins","rinscins","Ld","LR"
]

def _sdiv(a, b, eps=1e-9):
    b = np.where(np.abs(b) > eps, b, np.nan)
    return np.nan_to_num(a / b, nan=0.0)

def add_physics_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if {"As","Ac"}.issubset(X.columns): X["rho_s_As_over_Ac"] = _sdiv(X["As"], X["Ac"])
    if {"Af","Ac"}.issubset(X.columns): X["rho_f_Af_over_Ac"] = _sdiv(X["Af"], X["Ac"])
    if {"Af","As"}.issubset(X.columns): X["Af_over_As"]       = _sdiv(X["Af"], X["As"])
    if {"Ld","L"}.issubset(X.columns):  X["Ld_over_L"]        = _sdiv(X["Ld"], X["L"])
    if {"LR","fc"}.issubset(X.columns): X["LR_times_fc"]      = X["LR"] * X["fc"]
    if {"Efrp","Af","Es","As"}.issubset(X.columns):
        X["EfrpAf_over_EsAs"] = _sdiv(X["Efrp"]*X["Af"], X["Es"]*X["As"])
    if {"tins","kins","Tg"}.issubset(X.columns):
        X["thermal_index"] = _sdiv(X["tins"]*X["kins"], X["Tg"])
    return X

def prepare_for_model(raw_row: pd.Series, feature_order: List[str]) -> pd.DataFrame:
    base = pd.DataFrame([raw_row.reindex(SAFE_FEATURES).to_dict()])
    full = add_physics_features(base)
    for f in feature_order:
        if f not in full.columns:
            full[f] = 0.0
    return full[feature_order].copy()

# ---------- PROBS + THRESHOLDS + TEMPERATURE -----------
def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / max(T, 1e-8)
    z = z - z.max()
    ez = np.exp(z)
    return ez / np.sum(ez)

def predict_with_thresholds(model, X: pd.DataFrame, classes: List[str], thresholds: Dict[str,float],
                            temperature: float = 1.0) -> Tuple[np.ndarray, int, str]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        pred = model.predict(X)[0]
        proba = np.zeros(len(classes)); proba[pred] = 1.0
    if temperature > 1.0:
        p = np.clip(proba, 1e-6, 1-1e-6)
        z = np.log(p)
        proba = softmax(z, T=temperature)
    pred_idx = int(np.argmax(proba))
    for ci, cname in enumerate(classes):
        t = thresholds.get(cname, None)
        if t is not None and proba[ci] >= t:
            pred_idx = ci
    return proba, pred_idx, classes[pred_idx]

# ---------------- SURROGATE REGRESSOR -------------------
def build_surrogate(df_all: pd.DataFrame, feature_order: List[str]) -> Tuple[object, Dict[str,float]]:
    """
    Build a small regression head to predict a continuous 'No-Failure index':
      No Failure=1.0, Strength=0.4, Deflection=0.2
    Uses LightGBM if available, else Ridge.
    """
    # map labels to names if needed
    y_raw = df_all["target"].astype(str).str.strip().replace({
        "0":"No Failure", "1":"Strength Failure", "2":"Deflection Failure", "3":"Other"
    })
    keep = y_raw.isin(["No Failure","Strength Failure","Deflection Failure"])
    df = df_all.loc[keep].copy().reset_index(drop=True)

    # build X in model feature space
    base = df[[c for c in SAFE_FEATURES if c in df.columns]].copy()
    full = add_physics_features(base)
    for f in feature_order:
        if f not in full.columns:
            full[f] = 0.0
    X = full[feature_order].astype(float)

    # target
    mapping = {"No Failure":1.0, "Strength Failure":0.4, "Deflection Failure":0.2}
    y = y_raw.loc[keep].map(mapping).astype(float).values

    # model
    try:
        from lightgbm import LGBMRegressor
        reg = LGBMRegressor(
            n_estimators=400, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9,
            min_child_samples=25, random_state=42, n_jobs=-1
        )
    except Exception:
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0, random_state=42) if hasattr(Ridge, "__call__") else Ridge(alpha=1.0)

    reg.fit(X, y)

    # basic stats for step sizing (std per feature)
    stats = {f: float(max(X[f].std(ddof=0), 1e-9)) for f in feature_order}
    return reg, stats

# ---------------- NUDGE POLICIES ------------------------
@dataclass
class Nudge:
    feature: str
    kind: str          # "percent" or "abs"
    step: float        # base step
    steps_max: int
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    prefer_dir: Optional[str] = None  # "up"/"down"/None

DEFAULT_NUDGES: List[Nudge] = [
    Nudge("As",   "percent", 0.10, 3, clamp_min=0.0, prefer_dir="up"),      # +10/20/30%
    Nudge("Af",   "percent", 0.10, 3, clamp_min=0.0, prefer_dir="up"),      # +10/20/30%
    Nudge("tins", "abs",     10.0, 2, clamp_min=0.0, prefer_dir="up"),      # +10/20 mm
    Nudge("Cc",   "abs",     10.0, 2, clamp_min=0.0, prefer_dir="up"),      # +10/20 mm
    Nudge("LR",   "percent",-0.10, 2, clamp_min=0.0, clamp_max=1.0, prefer_dir="down"),  # -10/-20%
]

def clamp(v, lo, hi):
    if lo is not None: v = max(v, lo)
    if hi is not None: v = min(v, hi)
    return v

def apply_nudge(row: pd.Series, n: Nudge, k: int) -> pd.Series:
    r = row.copy()
    if n.kind == "percent":
        factor = (1.0 + n.step) ** k
        r[n.feature] = r[n.feature] * factor
    else:
        r[n.feature] = r[n.feature] + n.step * k
    r[n.feature] = clamp(float(r[n.feature]), n.clamp_min, n.clamp_max)
    return r

# --------------- EVALUATE ONE ROW -----------------------
def eval_row(model, classes, thresholds, feature_order, surrogate, feat_std,
             raw_row: pd.Series, temperature: float) -> Dict:
    # classifier baseline
    X0 = prepare_for_model(raw_row, feature_order)
    p0, _, _ = predict_with_thresholds(model, X0, classes, thresholds, temperature)
    idx_no = classes.index("No Failure") if "No Failure" in classes else 0
    base_no = float(p0[idx_no])
    # surrogate baseline
    s0 = float(surrogate.predict(X0)[0])
    return {"PNo": base_no, "S": s0, "proba": {classes[i]: float(p0[i]) for i in range(len(classes))}}

def try_nudge_series(model, classes, thresholds, feature_order,
                     surrogate, feat_std, raw_row: pd.Series,
                     n: Nudge, temperature: float) -> Dict:
    base = eval_row(model, classes, thresholds, feature_order, surrogate, feat_std, raw_row, temperature)
    results = []
    # prefer_dir first
    for k in range(1, n.steps_max+1):
        cand = apply_nudge(raw_row, n, k)
        Xc = prepare_for_model(cand, feature_order)
        pc, _, _ = predict_with_thresholds(model, Xc, classes, thresholds, temperature)
        sc = float(surrogate.predict(Xc)[0])
        idx_no = classes.index("No Failure")
        results.append({
            "feature": n.feature, "k": k, "new_value": float(cand[n.feature]),
            "PNo": float(pc[idx_no]), "S": sc,
            "dPNo": float(pc[idx_no] - base["PNo"]), "dS": float(sc - base["S"])
        })
    # quick opposite check (k=1)
    opp = Nudge(n.feature, n.kind, -n.step, 1, n.clamp_min, n.clamp_max)
    cand = apply_nudge(raw_row, opp, 1)
    Xc = prepare_for_model(cand, feature_order)
    pc, _, _ = predict_with_thresholds(model, Xc, classes, thresholds, temperature)
    sc = float(surrogate.predict(Xc)[0])
    idx_no = classes.index("No Failure")
    results.append({
        "feature": n.feature, "k": -1, "new_value": float(cand[n.feature]),
        "PNo": float(pc[idx_no]), "S": sc,
        "dPNo": float(pc[idx_no] - base["PNo"]), "dS": float(sc - base["S"])
    })
    # rank by surrogate first (smoother), tie-break by dPNo
    best = sorted(results, key=lambda r: (r["dS"], r["dPNo"]), reverse=True)[0]
    best["baseline_PNo"] = base["PNo"]; best["baseline_S"] = base["S"]
    return best

def two_feature_combo(model, classes, thresholds, feature_order,
                      surrogate, feat_std, raw_row: pd.Series,
                      ranked_df: pd.DataFrame, temperature: float, topk: int = 2) -> Dict:
    base = eval_row(model, classes, thresholds, feature_order, surrogate, feat_std, raw_row, temperature)
    r = raw_row.copy()
    chosen = []
    for i in range(min(topk, len(ranked_df))):
        ri = ranked_df.iloc[i]
        if "new_value" not in ri: continue
        feat = ri["feature"]; newv = float(ri["new_value"])
        r[feat] = newv; chosen.append(feat)
    Xc = prepare_for_model(r, feature_order)
    pc, pred_idx, pred_lab = predict_with_thresholds(model, Xc, classes, thresholds, temperature)
    sc = float(surrogate.predict(Xc)[0])
    idx_no = classes.index("No Failure")
    return {
        "applied": chosen,
        "PNo_before": base["PNo"], "PNo_after": float(pc[idx_no]),
        "S_before": base["S"], "S_after": sc,
        "proba_after": {classes[i]: float(pc[i]) for i in range(len(classes))},
        "pred_after": pred_lab
    }

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_pack", default=None)
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--bn", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    outs = paths["outputs"]
    figs_dir = os.path.join(outs, "figs")
    tabs_dir = os.path.join(outs, "tables")
    ensure_dirs(figs_dir, tabs_dir)

    # locate model pack (winner)
    model_pack = args.model_pack
    if model_pack is None:
        cands = [os.path.join(tabs_dir, fn) for fn in sorted(os.listdir(tabs_dir)) if "leaderboard" in fn and fn.endswith(".csv")]
        if not cands:
            raise FileNotFoundError("No leaderboard found. Provide --model_pack.")
        lb = pd.read_csv(cands[-1]).sort_values(by=["macro_f1","bacc"], ascending=False)
        model_pack = lb.iloc[0]["model"]
        print(f"[info] Using winner from: {os.path.basename(cands[-1])} → {model_pack}")

    pack = joblib.load(model_pack)
    model = pack["model"]; classes = pack["classes"]; feature_order = pack["features"]
    thresholds = pack.get("thresholds", {})

    # dataset
    df_all = load_processed(paths)
    for c in SAFE_FEATURES:
        if c not in df_all.columns:
            raise ValueError(f"Dataset missing required column: {c}")

    # choose row
    df = df_all.copy()
    sel = None
    if args.bn is not None and "BN" in df.columns:
        hit = df[df["BN"].astype(str)==str(args.bn)]
        if len(hit)>0: sel = hit.iloc[0]
    if sel is None:
        if args.idx is not None and 0 <= args.idx < len(df):
            sel = df.iloc[args.idx]
        else:
            sel = df.sample(1, random_state=42).iloc[0]

    # surrogate regressor (built from full dataset in model feature space)
    # keep only the 3 classes
    y_raw = df_all["target"].astype(str).str.strip().replace({
        "0":"No Failure","1":"Strength Failure","2":"Deflection Failure","3":"Other"
    })
    keep = y_raw.isin(["No Failure","Strength Failure","Deflection Failure"])
    reg, feat_std = build_surrogate(df_all.loc[keep].copy(), feature_order)

    # baseline
    X0 = prepare_for_model(sel, feature_order)
    p0, _, pred_lab = predict_with_thresholds(model, X0, classes, thresholds, args.temperature)
    idx_no = classes.index("No Failure")
    base = {
        "BN": sel.get("BN", None),
        "pred": pred_lab,
        "proba": {classes[i]: float(p0[i]) for i in range(len(classes))}
    }
    s0 = float(reg.predict(X0)[0])

    # run nudges
    ranked = []
    for n in DEFAULT_NUDGES:
        try:
            best = try_nudge_series(model, classes, thresholds, feature_order, reg, feat_std, sel, n, args.temperature)
            ranked.append(best)
        except Exception as e:
            ranked.append({"feature": n.feature, "error": str(e), "dPNo": -1e9, "dS": -1e9})
    ranked_df = pd.DataFrame(ranked).sort_values(by=["dS","dPNo"], ascending=False)

    combo = two_feature_combo(model, classes, thresholds, feature_order, reg, feat_std, sel, ranked_df, args.temperature)

    # save artifacts
    tag = f"nudges_{base['BN']}" if base["BN"] is not None else "nudges_random"
    out_csv = os.path.join(tabs_dir, f"{tag}.csv")
    out_json = os.path.join(tabs_dir, f"{tag}.json")
    ranked_df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "beam_id": base["BN"],
            "baseline_pred": base["pred"],
            "baseline_proba": base["proba"],
            "baseline_surrogate": s0,
            "nudge_ranking_csv": out_csv,
            "combo_result": combo,
            "thresholds": thresholds,
            "temperature": args.temperature
        }, f, indent=2)

    # pretty print
    print("\n=== BASELINE ===")
    print(f"Beam: {base['BN']}")
    for k,v in base["proba"].items():
        print(f"  P({k}) = {v:.3f}")
    print(f"Surrogate score (No-Failure index): {s0:.3f}")
    print(f"→ Predicted: {base['pred']}  (thresholded, T={args.temperature})")

    print("\n=== NUDGE RANKING (by ΔSurrogate then ΔP(No-Failure)) ===")
    show = ranked_df.head(5)
    for _, r in show.iterrows():
        if "dS" in r and "dPNo" in r:
            direction = "↑" if r["k"]>0 else "↓"
            print(f" {r['feature']:>10s} {direction} → new={r['new_value']:.4f}  ΔS={r['dS']:+.3f}  ΔPNo={r['dPNo']:+.3f}")

    print("\n=== TWO-NUDGE COMBO (what-if) ===")
    if "error" in combo:
        print(" combo failed:", combo["error"])
    else:
        print(f" Applied: {combo['applied']}")
        print(f"  P(NoFailure) before: {combo['PNo_before']:.3f} → after: {combo['PNo_after']:.3f}")
        print(f"  Surrogate before: {combo['S_before']:.3f} → after: {combo['S_after']:.3f}")
        print("  Probs after:")
        for k,v in combo["proba_after"].items():
            print(f"   {k:>18s}: {v:.3f}")
        print(f"  → New predicted: {combo['pred_after']}")

    print(f"\nArtifacts:\n  Nudge table → {out_csv}\n  Pack JSON   → {out_json}\nDone.")

if __name__ == "__main__":
    main()