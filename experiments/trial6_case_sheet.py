# scripts/11_case_sheet_v2.py
from __future__ import annotations
import os, argparse, json, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    return full[feature_order].astype(float).copy()

# -------------- HELPERS ----------------
def pick_winner_from_leaderboard(tables_dir: str) -> str:
    cands = [os.path.join(tables_dir, fn) for fn in sorted(os.listdir(tables_dir))
             if "leaderboard" in fn and fn.endswith(".csv")]
    if not cands:
        raise FileNotFoundError("No leaderboard found in outputs/tables. Provide --model_pack.")
    lb = pd.read_csv(cands[-1]).sort_values(by=["macro_f1","bacc"], ascending=False)
    return lb.iloc[0]["model"]

def percentile_bounds(df: pd.DataFrame, cols: List[str], p_lo: float=0.01, p_hi: float=0.99) -> Dict[str, Tuple[float,float]]:
    b = {}
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) == 0: continue
            b[c] = (float(s.quantile(p_lo)), float(s.quantile(p_hi)))
    return b

# -------------- EXPLAINABILITY ----------------
def try_shap_local(model, Xrow: pd.DataFrame) -> Optional[Dict]:
    """Return dict with 'mode_shap': np.ndarray or list per class; 'frt_shap': np.ndarray if available."""
    out = {"mode_shap": None, "frt_shap": None}
    try:
        import shap
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xrow)
        out["mode_shap"] = sv  # list per class (multiclass) or array
    except Exception:
        pass
    return out

def try_shap_local_reg(reg, Xrow: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        import shap
        expl = shap.TreeExplainer(reg)
        sv = expl.shap_values(Xrow)
        if isinstance(sv, list):  # some regressors return list of one
            sv = sv[0]
        return np.array(sv)[0]
    except Exception:
        return None

def finite_diff_signs_clf(model, Xrow: pd.DataFrame, classes: List[str], feature_order: List[str], frac: float=0.02):
    base = model.predict_proba(Xrow)[0]
    idx = {c:i for i,c in enumerate(classes)}
    effects = {c:{} for c in classes}
    for f in feature_order:
        step = max(abs(float(Xrow[f].iloc[0]))*frac, 1e-3)
        x = Xrow.copy(); x[f] = float(Xrow[f].iloc[0]) + step
        p = model.predict_proba(x)[0]
        for c in classes:
            s = np.sign(p[idx[c]] - base[idx[c]])
            if s != 0: effects[c][f] = int(s)  # +1/-1
    return effects

def finite_diff_signs_reg(reg, Xrow: pd.DataFrame, feature_order: List[str], frac: float=0.02):
    base = float(reg.predict(Xrow)[0])
    eff = {}
    for f in feature_order:
        step = max(abs(float(Xrow[f].iloc[0]))*frac, 1e-3)
        x = Xrow.copy(); x[f] = float(Xrow[f].iloc[0]) + step
        up = float(reg.predict(x)[0])
        s = np.sign(up - base)
        if s != 0: eff[f] = int(s)
    return eff

# -------------- RECOMMENDATION ENGINE ----------------
# Templates keyed by predicted mode; each template line is (category, suggestion, feature tags)
MODE_TEMPLATES = {
    "Deflection Failure": [
        ("Geometry",          "Increase concrete cover (Cc) by ~10–20 mm to delay bar heating and stiffness loss.", ["Cc"]),
        ("Thermal Protection","Thicken insulation (tins) or improve conductivity spec to slow heat ingress.", ["tins","thermal_index","kins"]),
        ("Reinforcement",     "Increase steel ratio ρs = As/Ac (e.g., As +5–15%) to raise stiffness.", ["rho_s_As_over_Ac","As","Ac"]),
        ("FRP System",        "Increase FRP contribution (Af or Efrp) for stiffness/strength retention at elevated T.", ["Af","Efrp","EfrpAf_over_EsAs"]),
        ("Loads",             "Reduce load ratio (LR) or revise load path/reserve factors.", ["LR","Ld_over_L"]),
    ],
    "Strength Failure": [
        ("Materials",         "Verify reduced strengths at temperature (fc, fy) and consider higher grade or protection.", ["fc","fy"]),
        ("FRP System",        "Increase FRP area/grade (Af, Efrp) and ensure anchorage/debonding resistance.", ["Af","Efrp"]),
        ("Geometry",          "Increase effective depth / reinforcement to raise moment capacity.", ["As","rho_s_As_over_Ac"]),
        ("Thermal Protection","Increase cover (Cc) and insulation (tins) to retain residual capacity longer.", ["Cc","tins"]),
        ("Loads",             "Lower LR or modify load distribution/continuity if feasible.", ["LR"]),
    ],
    "No Failure": [
        ("Documentation",     "Maintain configuration; document margin vs required exposure and assumptions.", []),
        ("Robustness",        "Check variability (±10–20 min FRT error band) and detail for robustness.", []),
    ]
}

# Domain monotonic hints (for quality gating of suggestions)
# Meaning: +1 means increasing feature usually increases FRT / reduces risk; -1 opposite.
DOMAIN_TENDENCY = {
    "Cc": +1, "tins": +1, "As": +1, "Af": +1, "LR": -1,
    "rho_s_As_over_Ac": +1, "rho_f_Af_over_Ac": +1, "EfrpAf_over_EsAs": +1,
    "thermal_index": +1, "Ld_over_L": -1
}

def rank_recommendations(
    pred_mode: str,
    frt_gap_min: float,
    top_mode_features: List[str],
    top_frt_features: List[str],
    row_values: Dict[str,float]
) -> List[Dict]:
    """
    Build a prioritized list of recommendations using:
    - scenario gap (how urgent),
    - overlap between top drivers and domain tendencies,
    - current row values (e.g., very low Cc/tins gets higher priority).
    """
    items = MODE_TEMPLATES.get(pred_mode, [])
    # score each item
    recs = []
    for cat, text, tags in items:
        # base score from scenario urgency
        score = 1.0 + max(0.0, -frt_gap_min) / 20.0  # bigger shortfall → higher score
        # add credit if any tag appears in top SHAP/FD lists
        if any(t in top_mode_features for t in tags): score += 0.5
        if any(t in top_frt_features  for t in tags): score += 0.5
        # small boost if current value suggests room for improvement (heuristic thresholds)
        for t in tags:
            v = row_values.get(t, None)
            if v is None: continue
            if t == "Cc" and v < 30: score += 0.4
            if t == "tins" and v < 20: score += 0.3
            if t == "LR" and v > 0.6: score += 0.4
            if t == "As" and v < 400: score += 0.2
        recs.append({"category": cat, "suggestion": text, "score": round(score, 3), "tags": tags})
    # sort by score descending
    recs.sort(key=lambda r: r["score"], reverse=True)
    return recs

# -------------- PLOT ----------------
def plot_case_summary(probs: Dict[str,float], frt: float, exposure: float, margin: float, out_png: str, title: str):
    labs = list(probs.keys())
    vals = [probs[k] for k in labs]
    req = exposure + margin

    fig = plt.figure(figsize=(7.8,4.2))
    # left: class probs
    ax1 = fig.add_subplot(1,2,1)
    ax1.barh(labs[::-1], vals[::-1])
    ax1.set_xlim(0,1)
    ax1.set_title("Failure mode probabilities")
    for i,v in enumerate(vals[::-1]):
        ax1.text(min(v+0.01, 0.98), i, f"{v:.2f}", va="center", fontsize=9)

    # right: FRT vs requirement
    ax2 = fig.add_subplot(1,2,2)
    bars = ax2.bar(["Pred FRT","Req (+margin)"], [frt, req])
    ax2.set_ylabel("minutes"); ax2.set_title("FRT vs scenario")
    colors = ["#4CAF50" if frt >= req else "#F44336", "#607D8B"]
    for b,c in zip(bars, colors): b.set_color(c)
    for i,v in enumerate([frt, req]):
        ax2.text(i, v+1, f"{v:.0f} min", ha="center", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)

# -------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_pack", default=None, help="Classifier pack (.joblib). Defaults to leaderboard winner.")
    ap.add_argument("--frt_pack", default=None, help="FRT regressor pack (.joblib). Defaults to models/checkpoints/frt_regressor.joblib")
    ap.add_argument("--idx", type=int, default=None, help="Row index to analyze (or use --bn).")
    ap.add_argument("--bn", default=None, help="Beam name (BN) to analyze if present.")
    ap.add_argument("--exposure", type=float, default=90.0, help="Scenario exposure time (min).")
    ap.add_argument("--margin", type=float, default=10.0, help="Safety margin (min).")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = cfg["paths"]
    outs_dir = os.path.join(paths["outputs"], "tables")
    figs_dir = os.path.join(paths["outputs"], "figs")
    ensure_dirs(outs_dir, figs_dir)

    # Load dataset
    df_all = load_processed(paths)

    # Load classifier (winner)
    if args.model_pack is None:
        args.model_pack = pick_winner_from_leaderboard(outs_dir)
        print(f"[info] Using winner → {args.model_pack}")
    cpack = joblib.load(args.model_pack)
    clf = cpack["model"]; classes = cpack["classes"]; feat_order = cpack["features"]
    thresholds = cpack.get("thresholds", {})

    # Load FRT regressor
    frt_pack = args.frt_pack or os.path.join(paths.get("models","models"), "checkpoints", "frt_regressor.joblib")
    rpack = joblib.load(frt_pack)
    reg = rpack["model"]; reg_feats = rpack["features"]  # should match feat_order ideally

    # diagnostics for feature drift
    clf_only = [f for f in feat_order if f not in reg_feats]
    reg_only = [f for f in reg_feats if f not in feat_order]
    if clf_only or reg_only:
        print("[diag] Feature drift detected:",
              f"\n  In classifier only: {clf_only}",
              f"\n  In regressor only: {reg_only}")

    # Select case
    sel = None
    if args.bn is not None and "BN" in df_all.columns:
        hit = df_all[df_all["BN"].astype(str) == str(args.bn)]
        if len(hit) > 0: sel = hit.iloc[0]
    if sel is None:
        if args.idx is not None and 0 <= args.idx < len(df_all):
            sel = df_all.iloc[args.idx]
        else:
            sel = df_all.sample(1, random_state=42).iloc[0]
    bn = sel.get("BN", f"row_{sel.name}")

    # Build model-specific input rows
    Xrow_clf = prepare_for_model(sel, feat_order)
    Xrow_reg = prepare_for_model(sel, reg_feats)

    # Build feature space for model + OOD bounds
    base_feats_all = df_all[[c for c in SAFE_FEATURES if c in df_all.columns]].copy()
    feats_all = add_physics_features(base_feats_all)
    for f in feat_order:
        if f not in feats_all.columns:
            feats_all[f] = 0.0
    bounds = percentile_bounds(feats_all[feat_order], feat_order)

    # Xrow = prepare_for_model(sel, feat_order)

    # Classifier prediction with thresholds
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xrow_clf)[0]
    else:
        pred = clf.predict(Xrow_clf)[0]
        proba = np.zeros(len(classes)); proba[pred] = 1.0
    pred_idx = int(np.argmax(proba))
    for ci, cname in enumerate(classes):
        t = thresholds.get(cname, None)
        if t is not None and proba[ci] >= t:
            pred_idx = ci
    pred_lab = classes[pred_idx]
    probs_dict = {classes[i]: float(proba[i]) for i in range(len(classes))}

    # FRT prediction + verdict
    frt_pred = float(reg.predict(Xrow_reg)[0])
    req = float(args.exposure + args.margin)
    frt_gap = frt_pred - req
    verdict = "Meets scenario (with margin)" if frt_pred >= req else f"At Risk ({frt_gap:.1f} min short)"

    # Explainability: SHAP if available, else finite diffs
    top_mode_feats, top_frt_feats = [], []
    # Class SHAP
    try:
        import shap  # noqa
        # multiclass → shap_values = list[class] of (1, n_features)
        expl = try_shap_local(clf, Xrow_clf)
        sv = expl.get("mode_shap", None)
        if isinstance(sv, list):  # multiclass list
            # pick predicted class
            vec = np.array(sv[pred_idx])[0]
            order = np.argsort(np.abs(vec))[::-1][:10]
            top_mode_feats = [feat_order[i] for i in order]
        elif isinstance(sv, np.ndarray):
            vec = sv[0]
            order = np.argsort(np.abs(vec))[::-1][:10]
            top_mode_feats = [feat_order[i] for i in order]
    except Exception:
        eff = finite_diff_signs_clf(clf, Xrow_clf, classes, feat_order, frac=0.02)
        # choose features that *increase* predicted class probability
        inc = eff.get(pred_lab, {})
        top_mode_feats = [f for f, s in sorted(inc.items(), key=lambda kv: -abs(kv[1])) if s > 0][:10]

    # FRT SHAP
    try:
        import shap  # noqa
        rsv = try_shap_local_reg(reg, Xrow_reg)
        if rsv is not None:
            order = np.argsort(np.abs(rsv))[::-1][:10]
            top_frt_feats = [reg_feats[i] for i in order if i < len(reg_feats)]
    except Exception:
        effr = finite_diff_signs_reg(reg, Xrow_reg, reg_feats, frac=0.02)
        # choose features that increase FRT
        top_frt_feats = [f for f, s in sorted(effr.items(), key=lambda kv: -abs(kv[1])) if s > 0][:10]

    # OOD flags
    ood = []
    for f in feat_order:
        val = float(Xrow_clf[f].iloc[0])
        lo, hi = bounds.get(f, (None, None))
        if lo is None: continue
        if val < lo or val > hi:
            ood.append({"feature": f, "value": val, "p01": lo, "p99": hi})

    # Row raw values for heuristics
    row_vals = {}
    # Prefer classifier-prepared row for shared engineered features; fallback to reg row if not present
    for k in ["Cc","tins","As","Af","LR","rho_s_As_over_Ac","EfrpAf_over_EsAs","Ld_over_L","thermal_index"]:
        if k in Xrow_clf.columns:
            row_vals[k] = float(Xrow_clf[k].iloc[0])
        elif k in Xrow_reg.columns:
            row_vals[k] = float(Xrow_reg[k].iloc[0])
        elif k in feats_all.columns:
            row_vals[k] = float(feats_all[k].mean())

    # Build recommendation set (ranked)
    recs = rank_recommendations(pred_lab, frt_gap, top_mode_feats, top_frt_feats, row_vals)

    # Save drivers CSV (just to be transparent)
    drivers_df = pd.DataFrame({
        "top_mode_features": pd.Series(top_mode_feats),
        "top_frt_features": pd.Series(top_frt_feats)
    })
    drivers_csv = os.path.join(outs_dir, f"case_{bn}_drivers_v2.csv")
    drivers_df.to_csv(drivers_csv, index=False)

    # Build JSON case
    case_json = {
        "beam_id": str(bn),
        "mode_probs": probs_dict,
        "pred_mode": pred_lab,
        "frt_minutes": frt_pred,
        "scenario": {"exposure": float(args.exposure), "margin": float(args.margin),
                     "threshold": req, "gap_minutes": float(frt_gap),
                     "verdict": verdict},
        "top_mode_features": top_mode_feats,
        "top_frt_features": top_frt_feats,
        "recommendations": recs,
        "ood_flags": ood
    }
    case_path = os.path.join(outs_dir, f"case_{bn}_v2.json")
    with open(case_path, "w") as f:
        json.dump(case_json, f, indent=2)

    # Plot summary
    fig_path = os.path.join(figs_dir, f"case_{bn}_summary_v2.png")
    title = f"Beam {bn}: {pred_lab} | FRT={frt_pred:.0f} min → {verdict}"
    plot_case_summary(probs_dict, frt_pred, args.exposure, args.margin, fig_path, title)

    # Console summary
    print("\n=== CASE INSIGHT SHEET (v2) ===")
    print(f"Beam: {bn}")
    for k,v in probs_dict.items():
        print(f"  P({k}) = {v:.3f}")
    print(f"FRT = {frt_pred:.1f} min | Requirement = {args.exposure}+{args.margin} = {args.exposure+args.margin:.1f} min")
    print(f"Verdict: {verdict}")
    if top_mode_feats:
        print("Top drivers (mode):", ", ".join(top_mode_feats[:6]))
    if top_frt_feats:
        print("Top drivers (FRT): ", ", ".join(top_frt_feats[:6]))
    print("Top recommendations:")
    for r in recs[:5]:
        print(f"  • [{r['category']}] {r['suggestion']}  (score={r['score']})")
    print(f"Drivers CSV → {drivers_csv}")
    print(f"Case JSON  → {case_path}")
    print(f"Summary PNG→ {fig_path}")
    if ood:
        print(f"[warn] {len(ood)} feature(s) out of training range (p1–p99). Treat with caution.")

if __name__ == "__main__":
    main()