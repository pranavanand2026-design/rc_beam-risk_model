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

    return cfg, df, clf, frt, adjustable


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
            "expected_delta_frt": "\u0394 FRT (min)",
            "expected_delta_prob": "\u0394 P(No Failure)",
            "expected_frt": "New FRT (min)",
            "expected_prob_no_fail": "New P(No Failure)",
            "rationale": "Why it helps",
        }
    )
    rec_df["\u0394 FRT (min)"] = rec_df["\u0394 FRT (min)"].map(lambda x: f"{x:+.1f}")
    rec_df["\u0394 P(No Failure)"] = rec_df["\u0394 P(No Failure)"].map(lambda x: f"{x:+.3f}")
    rec_df["New FRT (min)"] = rec_df["New FRT (min)"].map(lambda x: f"{x:.1f}")
    rec_df["New P(No Failure)"] = rec_df["New P(No Failure)"].map(lambda x: f"{x:.3f}")
    st.dataframe(
        rec_df[
            [
                "Recommended change",
                "\u0394 FRT (min)",
                "\u0394 P(No Failure)",
                "New FRT (min)",
                "New P(No Failure)",
                "Why it helps",
            ]
        ],
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="RC Beam Fire Design Studio", layout="wide")
    st.title("RC Beam Fire Design Studio")

    cfg, df, clf, frt, adjustable = load_resources()
    feature_ranges = compute_feature_ranges(df, adjustable)

    id_col = "BN"
    analysis_cfg = cfg.get("analysis", {})
    default_exposure = float(analysis_cfg.get("exposure_minutes", 90))
    default_margin = float(analysis_cfg.get("margin_minutes", 10))

    sidebar = st.sidebar
    sidebar.header("Scenario")
    beam_id = sidebar.selectbox("Beam ID", sorted(df[id_col].astype(str).unique()))
    row_series = df[df[id_col].astype(str) == str(beam_id)].iloc[0]
    row_df = pd.DataFrame(row_series).T

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
    )
    render_recommendations([asdict(rec) for rec in action_plan.recommendations])

    if action_plan.notes:
        st.info(" ".join(action_plan.notes))


if __name__ == "__main__":
    main()
