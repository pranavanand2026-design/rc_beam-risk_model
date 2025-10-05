from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rcbeam_fire.analysis.insight import DOMAIN_TENDENCY, build_case_action_plan
from rcbeam_fire.config import DEFAULT_CONFIG_PATH, load_config
from rcbeam_fire.utils.io import load_processed_dataset
from rcbeam_fire.utils.model_adapters import ClassifierAdapter, RegressorAdapter


# Small epsilon for numeric step bounds in the design playground
SMALL_EPS = 1e-6


def _parse_bounds_cfg(analysis_cfg: Dict) -> Dict[str, Dict[str, float]]:
    parsed: Dict[str, Dict[str, float]] = {}
    if not isinstance(analysis_cfg, dict):
        return parsed
    bounds = analysis_cfg.get("bounds")
    if not isinstance(bounds, dict):
        return parsed
    for feat, entry in bounds.items():
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


def load_resources(config_path: Path = DEFAULT_CONFIG_PATH):
    cfg = load_config(config_path)
    df = load_processed_dataset(cfg["paths"])

    models_dir = Path(cfg["paths"].get("models", "models")) / "checkpoints"
    prefer = ["blend_xgb_ldam_nophys.joblib", "suite_lgbm_smote.joblib", "suite_xgb_smote.joblib"]
    clf_pack_path = next((models_dir / fn for fn in prefer if (models_dir / fn).exists()), None)
    if clf_pack_path is None:
        raise FileNotFoundError("Classifier pack not found in models/checkpoints/.")

    frt_path = models_dir / "frt_regressor.joblib"
    if not frt_path.exists():
        raise FileNotFoundError("Missing frt_regressor.joblib in models/checkpoints/.")

    clf = ClassifierAdapter(joblib.load(clf_pack_path))
    frt = RegressorAdapter(joblib.load(frt_path))

    analysis_cfg = cfg.get("analysis", {})
    adjustable = analysis_cfg.get("adjustable_features", ["Cc", "tins", "hi", "LR", "As", "Af"])
    bounds_cfg = _parse_bounds_cfg(analysis_cfg)

    return cfg, df, clf, frt, adjustable, bounds_cfg


def compute_feature_ranges(df: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, float]]:
    ranges: Dict[str, Dict[str, float]] = {}
    for feat in features:
        if feat not in df.columns:
            continue
        col = pd.to_numeric(df[feat], errors="coerce").dropna()
        if col.empty:
            continue
        ranges[feat] = {
            "min": float(col.quantile(0.05)),
            "max": float(col.quantile(0.95)),
            "mean": float(col.mean()),
            "step": float(max((col.quantile(0.75) - col.quantile(0.25)) / 20.0, 0.1)),
        }
    return ranges


def predict_case(row: pd.Series, clf: ClassifierAdapter, frt: RegressorAdapter):
    row_df = row.to_frame().T
    pred_idx, proba = clf.predict_label(row_df)
    classes = clf.classes
    probs = proba[0]
    pred_mode = classes[int(pred_idx[0])]
    frt_minutes = float(frt.predict(row_df)[0])
    return pred_mode, probs, frt_minutes


def render_recommendations(recs: List[Dict]):
    if not recs:
        st.success("Beam already satisfies the scenario; document current configuration.")
        return

    rec_df = pd.DataFrame(recs).rename(
        columns={
            "action": "Recommended change",
            "expected_delta_frt": "Δ FRT (min)",
            "expected_delta_prob": "Δ P(No Failure)",
            "expected_frt": "New FRT (min)",
            "expected_prob_no_fail": "New P(No Failure)",
            "rationale": "Why it helps",
        }
    )
    rec_df["Δ FRT (min)"] = rec_df["Δ FRT (min)"].map(lambda x: f"{x:+.1f}")
    rec_df["Δ P(No Failure)"] = rec_df["Δ P(No Failure)"].map(lambda x: f"{x:+.3f}")
    rec_df["New FRT (min)"] = rec_df["New FRT (min)"].map(lambda x: f"{x:.1f}")
    rec_df["New P(No Failure)"] = rec_df["New P(No Failure)"].map(lambda x: f"{x:.3f}")
    st.dataframe(
        rec_df[
            [
                "Recommended change",
                "Δ FRT (min)",
                "Δ P(No Failure)",
                "New FRT (min)",
                "New P(No Failure)",
                "Why it helps",
            ]
        ],
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="RC Beam Fire Design Studio", layout="wide")
    st.title("🔥 RC Beam Fire Design Studio")

    cfg, df, clf, frt, adjustable, bounds_cfg = load_resources()
    feature_ranges = compute_feature_ranges(df, adjustable)

    id_col = "BN"
    analysis_cfg = cfg.get("analysis", {})
    default_exposure = float(analysis_cfg.get("exposure_minutes", 90))
    default_margin = float(analysis_cfg.get("margin_minutes", 10))

    sidebar = st.sidebar
    sidebar.header("Scenario")

    # Input mode: choose a dataset row or enter parameters manually
    input_mode = sidebar.radio("Input mode", ("Dataset row", "Manual"), index=0)
    show_truth = False
    if input_mode == "Dataset row":
        beam_id = sidebar.selectbox("Beam ID", sorted(df[id_col].astype(str).unique()))
        row_series = df[df[id_col].astype(str) == str(beam_id)].iloc[0]
        row_df = pd.DataFrame(row_series).T
        show_truth = sidebar.checkbox("Compare with ground truth", value=True)
    else:
        # Build manual row for the union of classifier/regressor features
        combined_features = list(dict.fromkeys(list(clf.features) + list(getattr(frt, "features", []))))
        defaults: Dict[str, float] = {}
        for f in combined_features:
            if f in df.columns:
                col = pd.to_numeric(df[f], errors="coerce").dropna()
                defaults[f] = float(col.median()) if not col.empty else 0.0
            else:
                defaults[f] = 0.0
        sidebar.markdown("Enter parameters (manual mode)")
        values: Dict[str, float] = {}
        for f in combined_features:
            values[f] = sidebar.number_input(f, value=float(defaults[f]), format="%.3f")
        row_df = pd.DataFrame([values])
        row_series = row_df.iloc[0]

    exposure = sidebar.number_input("Fire exposure (minutes)", min_value=0.0, value=default_exposure, step=5.0)
    margin = sidebar.number_input("Safety margin (minutes)", min_value=0.0, value=default_margin, step=5.0)
    threshold = exposure + margin

    pred_mode, probs, frt_minutes = predict_case(row_series, clf, frt)
    gap = frt_minutes - threshold
    prob_series = pd.Series(probs, index=clf.classes)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Mode", pred_mode)
    col2.metric("No-Failure Probability", f"{prob_series.get('No Failure', np.nan):.3f}")
    col3.metric("Predicted FRT (min)", f"{frt_minutes:.1f}", delta=f"{gap:+.1f} vs requirement")

    st.write(
        f"**Scenario threshold:** {threshold:.1f} min  |  **Verdict:** "
        f"{'✅ Meets requirement' if gap >= 0 else '⚠️ At risk'}"
    )

    # Optional: show ground truth for dataset rows
    if input_mode == "Dataset row" and show_truth:
        gt_mode = None
        if "LimitState" in row_series.index:
            gt_mode = str(row_series["LimitState"]) if pd.notna(row_series["LimitState"]) else None
        if gt_mode is None and "target" in row_series.index:
            gt_mode = str(row_series["target"]) if pd.notna(row_series["target"]) else None
        try:
            gt_frt = float(pd.to_numeric(row_series.get("FR", np.nan), errors="coerce"))
        except Exception:
            gt_frt = np.nan
        err_frt = None if np.isnan(gt_frt) else (frt_minutes - gt_frt)

        c1, c2, c3 = st.columns(3)
        c1.metric("Actual Mode (dataset)", gt_mode or "—")
        c2.metric("Actual FRT (min)", f"{gt_frt:.1f}" if gt_frt==gt_frt else "—")
        c3.metric("ΔFRT pred-actual (min)", f"{err_frt:+.1f}" if err_frt is not None else "—")

    with st.expander("Beam parameters", expanded=False):
        st.dataframe(row_df, use_container_width=True)

    st.subheader("Class probabilities")
    st.bar_chart(prob_series)

    st.subheader("Action plan")
    scenario_dict = {"exposure": exposure, "margin": margin, "threshold": threshold, "gap_minutes": gap}
    action_plan = build_case_action_plan(
        df=df,
        row=row_series,
        clf=clf,
        frt=frt,
        scenario=scenario_dict,
        base_proba=probs,
        pred_mode=pred_mode,
        frt_pred=frt_minutes,
        top_mode=[],
        top_frt=[],
        adjustable_features=adjustable,
        domain_tendency=DOMAIN_TENDENCY,
        bounds=bounds_cfg,
    )
    render_recommendations([asdict(rec) for rec in action_plan.recommendations])

    if action_plan.notes:
        st.info(" ".join(action_plan.notes))

    st.subheader("Design playground")
    edited_row = row_series.copy()
    columns = st.columns(len(adjustable) if adjustable else 1)

    for idx, feature in enumerate(adjustable):
        if feature not in row_series.index:
            continue
        stats = feature_ranges.get(
            feature, {"min": float(row_series[feature]) * 0.5, "max": float(row_series[feature]) * 1.5, "step": 1.0}
        )
        bound_entry = bounds_cfg.get(feature)
        current = float(row_series[feature])
        if bound_entry:
            scale = bound_entry.get("scale")
            if scale == "x":
                min_factor = bound_entry.get("min")
                max_factor = bound_entry.get("max")
                step_factor = bound_entry.get("step")
                if min_factor is not None:
                    stats["min"] = max(stats["min"], current * min_factor)
                if max_factor is not None:
                    stats["max"] = min(stats["max"], current * max_factor)
                if step_factor is not None:
                    stats["step"] = max(abs(current) * step_factor, SMALL_EPS)
            else:
                min_bound = bound_entry.get("min")
                max_bound = bound_entry.get("max")
                step_bound = bound_entry.get("step")
                if min_bound is not None:
                    stats["min"] = max(stats["min"], min_bound)
                if max_bound is not None:
                    stats["max"] = min(stats["max"], max_bound)
                if step_bound is not None:
                    stats["step"] = max(step_bound, SMALL_EPS)
        if stats["min"] > stats["max"]:
            stats["min"], stats["max"] = stats["max"], stats["min"]
        with columns[idx % len(columns)]:
            # Clip default value into [min, max] to satisfy Streamlit constraints
            min_v = float(stats["min"]) 
            max_v = float(stats["max"]) 
            current_clipped = float(current)
            if current_clipped < min_v:
                current_clipped = min_v
            if current_clipped > max_v:
                current_clipped = max_v

            step = max(float(stats.get("step", 1.0)), SMALL_EPS)
            value = st.number_input(
                feature,
                value=current_clipped,
                min_value=min_v,
                max_value=max_v,
                step=step,
                format="%.2f" if step < 1 else "%.1f",
                key=f"edit_{feature}",
            )
            edited_row[feature] = value

    if st.button("Recalculate with edits"):
        new_mode, new_probs, new_frt = predict_case(edited_row, clf, frt)
        new_gap = new_frt - threshold
        st.write(f"**New prediction:** {new_mode}")
        st.write(f"**New FRT:** {new_frt:.1f} min ({new_gap:+.1f} vs requirement)")
        st.bar_chart(pd.Series(new_probs, index=clf.classes))

        new_plan = build_case_action_plan(
            df=df,
            row=edited_row,
            clf=clf,
            frt=frt,
            scenario=scenario_dict,
            base_proba=new_probs,
            pred_mode=new_mode,
            frt_pred=new_frt,
            top_mode=[],
            top_frt=[],
            adjustable_features=adjustable,
            domain_tendency=DOMAIN_TENDENCY,
            bounds=bounds_cfg,
        )

        render_recommendations([asdict(rec) for rec in new_plan.recommendations])


if __name__ == "__main__":
    main()
