from __future__ import annotations

"""
12_case_analyzer.py
~~~~~~~~~~~~~~~~~~~
Delivers the interpretability/reporting layer promised in the RC_Beam system
design (Section 2.2.4, component `12_case_analyzer.py`). Responsibilities:

* Satisfies FR-03/FR-07 by generating machine-readable case summaries
  (`outputs/tables/case_*.json`) and console narratives for client reviews.
* Implements the “Interpretation & Dashboard layer” from Section 2.2.2 by tying
  classifier/regressor outputs to SHAP-informed feature rationales.
* Supports user stories US-02/US-03 by producing exportable insights and action
  plans derived from the adjustable envelopes described in the analysis config.
"""

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from rcbeam_fire.analysis.insight import DOMAIN_TENDENCY, build_case_action_plan
from rcbeam_fire.analysis.eurocode import evaluate_compliance, pick_requirement
from rcbeam_fire.config import load_config
from rcbeam_fire.utils.io import ensure_dirs, load_processed_dataset
from rcbeam_fire.utils.model_adapters import ClassifierAdapter, RegressorAdapter

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def shap_topk(model, X_row: pd.DataFrame, k: int = 5) -> List[str]:
    """
    Return top-k feature names using SHAP if available, otherwise fall back to
    feature importances or raw order.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_row)
        if isinstance(sv, list):
            contrib = np.sum([np.abs(np.asarray(s_i, dtype=float)) for s_i in sv], axis=0)[0]
        else:
            contrib = np.abs(np.asarray(sv, dtype=float))[0]
        contrib = np.asarray(contrib, dtype=float).ravel()
        if not np.all(np.isfinite(contrib)):
            raise ValueError("non-finite SHAP contributions")
        order = np.argsort(contrib)[::-1]
        picks: List[str] = []
        for idx in order:
            if len(picks) >= k:
                break
            idx_int = int(idx)
            if 0 <= idx_int < len(X_row.columns):
                picks.append(str(X_row.columns[idx_int]))
        if picks:
            return picks
        raise ValueError("empty SHAP ranking")
    except Exception:
        try:
            fi = getattr(model, "feature_importances_", None)
            if fi is None:
                return list(X_row.columns)[:k]
            order = np.argsort(fi)[::-1][:k]
            return [X_row.columns[i] for i in order if i < len(X_row.columns)]
        except Exception:
            return list(X_row.columns)[:k]


def _parse_bounds(cfg_section) -> Dict[str, Dict[str, float]]:
    parsed: Dict[str, Dict[str, float]] = {}
    if not isinstance(cfg_section, dict):
        return parsed
    for feat, entry in cfg_section.items():
        if not isinstance(entry, dict):
            continue
        norm: Dict[str, float] = {}
        for key, value in entry.items():
            if key in {"min", "max", "step"}:
                try:
                    norm[key] = float(value)
                except (TypeError, ValueError):
                    continue
            elif key == "scale":
                norm[key] = str(value)
        if norm:
            parsed[str(feat)] = norm
    return parsed


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_case(
    df: pd.DataFrame,
    clf: ClassifierAdapter,
    frt: RegressorAdapter,
    row_idx: int,
    exposure: float,
    margin: float,
    adjustable_features: List[str],
    eurocode_requirement=None,
    eurocode_alias=None,
    bounds: Dict[str, Dict[str, float]] | None = None,
) -> Dict:
    row = df.loc[row_idx]
    row_df = row.to_frame().T

    pred_idx, proba_arr = clf.predict_label(row_df)
    classes = clf.classes
    probs = proba_arr[0]
    pred_mode = classes[int(pred_idx[0])]

    frt_minutes = float(frt.predict(row_df)[0])
    threshold = float(exposure + margin)
    gap = frt_minutes - threshold

    # Feature drivers
    base_model = clf.pack["xgb_leg"]["model"] if getattr(clf, "is_blend", False) else clf.pack["model"]
    X_clf = row_df[clf.features].copy()
    for col in X_clf.columns:
        X_clf[col] = pd.to_numeric(X_clf[col], errors="coerce")
    top_mode = shap_topk(base_model, X_clf, k=5)
    try:
        X_frt = row_df[frt.features].copy()
        for col in X_frt.columns:
            X_frt[col] = pd.to_numeric(X_frt[col], errors="coerce")
        top_frt = shap_topk(frt.model, X_frt, k=8)
    except Exception:
        top_frt = list(frt.features)[:8]

    scenario = {
        "exposure": float(exposure),
        "margin": float(margin),
        "threshold": threshold,
        "gap_minutes": float(gap),
    }

    eurocode_info = None
    if eurocode_requirement is not None:
        verdict_ec, criteria_ec, adjustments_ec = evaluate_compliance(
            row=row,
            requirement=eurocode_requirement,
            feature_alias=eurocode_alias,
        )
        eurocode_info = {
            "target_minutes": float(eurocode_requirement.minutes),
            "verdict": verdict_ec,
            "criteria": criteria_ec,
            "actions": [
                {
                    "feature": adj.feature,
                    "current": adj.current,
                    "target": adj.target,
                    "delta": adj.delta,
                    "unit": adj.unit,
                    "rationale": adj.rationale,
                }
                for adj in adjustments_ec
            ],
        }

    action_plan = build_case_action_plan(
        df=df,
        row=row,
        clf=clf,
        frt=frt,
        scenario=scenario,
        base_proba=probs,
        pred_mode=pred_mode,
        frt_pred=frt_minutes,
        top_mode=top_mode,
        top_frt=top_frt,
        adjustable_features=adjustable_features,
        domain_tendency=DOMAIN_TENDENCY,
        bounds=bounds,
    )

    rec_payload = [asdict(rec) for rec in action_plan.recommendations]

    return {
        "beam_id": str(row["BN"]),
        "pred_mode": pred_mode,
        "mode_probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
        "frt_minutes": round(frt_minutes, 2),
        "limit_state": {
            "demand_minutes": threshold,
            "resistance_minutes": frt_minutes,
            "margin_minutes": float(gap),
            "status": "safe" if gap >= 0 else "at_risk",
        },
        "scenario": {
            **scenario,
            "verdict": action_plan.scenario.verdict,
            "prob_no_failure": action_plan.scenario.prob_no_failure,
        },
        "eurocode": eurocode_info,
        "top_mode_features": [str(x) for x in top_mode],
        "top_frt_features": [str(x) for x in top_frt],
        "action_plan": {
            "recommendations": rec_payload,
            "notes": action_plan.notes,
            "reference_stats": action_plan.references,
            "combination": asdict(action_plan.combination) if action_plan.combination else None,
        },
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Case-level analysis for RC beam fire performance.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--idx", type=int, help="Row index to analyse")
    parser.add_argument("--beam", type=str, help="Beam ID (BN) to analyse")
    parser.add_argument("--exposure", type=float, help="Fire exposure duration (minutes)")
    parser.add_argument("--margin", type=float, help="Safety margin to add (minutes)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    outputs_dir = Path(paths["outputs"])
    tables_dir = outputs_dir / "tables"
    ensure_dirs(tables_dir)

    df = load_processed_dataset(paths)

    models_dir = Path(paths.get("models", "models")) / "checkpoints"
    prefer = [
        "blend_xgb_ldam_nophys.joblib",
        "suite_lgbm_smote.joblib",
        "suite_xgb_smote.joblib",
    ]
    clf_pack_path = next(((models_dir / fn) for fn in prefer if (models_dir / fn).exists()), None)
    if clf_pack_path is None:
        raise FileNotFoundError("No classifier pack found in models/checkpoints/.")

    frt_path = models_dir / "frt_regressor.joblib"
    if not frt_path.exists():
        raise FileNotFoundError("Missing frt_regressor.joblib in models/checkpoints/.")

    import joblib

    clf = ClassifierAdapter(joblib.load(clf_pack_path))
    frt = RegressorAdapter(joblib.load(frt_path))

    analysis_cfg = cfg.get("analysis", {})
    exposure = args.exposure if args.exposure is not None else float(analysis_cfg.get("exposure_minutes", 90))
    margin = args.margin if args.margin is not None else float(analysis_cfg.get("margin_minutes", 10))
    adjustable = analysis_cfg.get("adjustable_features", ["Cc", "tins", "hi", "LR", "As", "Af"])
    eurocode_section = analysis_cfg.get("eurocode") if isinstance(analysis_cfg, dict) else None
    eurocode_alias = analysis_cfg.get("eurocode_alias") if isinstance(analysis_cfg, dict) else None
    bounds_cfg = _parse_bounds(analysis_cfg.get("bounds") if isinstance(analysis_cfg, dict) else None)
    try:
        eurocode_requirement = pick_requirement(exposure + margin, eurocode_section)
    except Exception:
        eurocode_requirement = None

    if args.beam is not None:
        matches = df.index[df["BN"].astype(str) == str(args.beam)].tolist()
        if not matches:
            raise ValueError(f"Beam ID '{args.beam}' not found in dataset.")
        row_idx = matches[0]
    else:
        row_idx = int(args.idx if args.idx is not None else 0)

    result = analyze_case(
        df=df,
        clf=clf,
        frt=frt,
        row_idx=row_idx,
        exposure=exposure,
        margin=margin,
        adjustable_features=adjustable,
        eurocode_requirement=eurocode_requirement,
        eurocode_alias=eurocode_alias,
        bounds=bounds_cfg,
    )

    base_name = f"case_{result['beam_id']}"
    with open(tables_dir / f"{base_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== CASE ANALYSIS → BN={result['beam_id']} ===")
    print(f" Predicted failure mode: {result['pred_mode']}")
    print(" Mode probabilities:")
    for name, val in result["mode_probs"].items():
        print(f"   {name:>18}: {val:.3f}")
    print(f" FRT prediction: {result['frt_minutes']:.1f} min")
    print(f" Scenario verdict: {result['scenario']['verdict']} (threshold {result['scenario']['threshold']:.1f} min)")
    ls = result.get("limit_state")
    if ls:
        print(
            f" Limit-state check (Ed ≤ Rd): Ed={ls['demand_minutes']:.1f} min vs Rd={ls['resistance_minutes']:.1f} min"
            f" (Δ {ls['margin_minutes']:+.1f} min)"
        )

    euro_info = result.get("eurocode")
    if euro_info:
        minutes = euro_info.get("target_minutes")
        verdict = euro_info.get("verdict", "Unknown")
        label = f"R{minutes:.0f}" if minutes is not None else "R?"
        print(f"\n Eurocode envelope check ({label}): {verdict}")
        for crit in euro_info.get("criteria", []):
            desc = crit.get("description", crit.get("criterion", ""))
            status = crit.get("status", "unknown")
            actual = crit.get("actual")
            required = crit.get("required")
            delta = crit.get("delta")
            actual_txt = "–" if actual is None else f"{actual:.2f}"
            required_txt = f"{required:.2f}" if required is not None else "–"
            delta_txt = "" if delta is None else f" (Δ {delta:+.2f})"
            print(f"   [{status.upper():7}] {desc}: {actual_txt} vs {required_txt}{delta_txt}")
        actions = euro_info.get("actions") or []
        if actions:
            print("  Remedial steps to regain compliance:")
            for act in actions:
                delta = act["delta"]
                direction = "Increase" if delta >= 0 else "Reduce"
                unit = act.get("unit") or ""
                print(
                    f"   • {direction} {act['feature']} to ~{act['target']:.2f}{unit if unit else ''}"
                    f" (Δ {delta:+.2f}{unit if unit else ''}) – {act['rationale']}"
                )

    recs = result["action_plan"]["recommendations"]
    if recs:
        print("\n Recommended adjustments:")
        for rec in recs:
            delta_prob = rec["expected_delta_prob"]
            delta_frt = rec["expected_delta_frt"]
            print(
                f"  - {rec['action']}"
                f" → ΔFRT {delta_frt:+.1f} min, ΔP(No Failure) {delta_prob:+.3f}"
                f", predicted mode → {rec['new_mode']}"
                f" | Rationale: {rec['rationale']}"
            )
    else:
        print("\n No single-parameter adjustment within reference envelopes improved the scenario.")

    combo = result["action_plan"].get("combination")
    if combo:
        print("\n Combined plan:")
        actions = combo.get("actions", [])
        for action in actions:
            print(f"  • {action}")
        print(
            f"   → Expected FRT {combo['expected_frt']:.1f} min"
            f" (Δ {combo['expected_delta_frt']:+.1f}), "
            f"P(No Failure) {combo['expected_prob_no_fail']:.3f}"
            f" (Δ {combo['expected_delta_prob']:+.3f}), "
            f"predicted mode → {combo['predicted_mode']}"
        )

    notes = list(result["action_plan"]["notes"])
    if recs and all(rec["new_mode"] != "No Failure" for rec in recs):
        if not (combo and combo.get("predicted_mode") == "No Failure"):
            notes.append("Single-parameter adjustments still classify the beam as failing; combine modifications or revisit the exposure target to clear the safety margin.")

    if notes:
        print("\n Notes:")
        for note in notes:
            print(f"  • {note}")


if __name__ == "__main__":
    main()

