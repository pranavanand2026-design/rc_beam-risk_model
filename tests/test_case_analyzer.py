from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def analysis_defaults(cfg):
    analysis = cfg.get("analysis", {})
    exposure = float(analysis.get("exposure_minutes", 90))
    margin = float(analysis.get("margin_minutes", 10))
    adjustable = analysis.get("adjustable_features", ["Cc", "tins", "hi", "LR", "As", "Af"])
    return exposure, margin, adjustable


def test_case_analyzer_returns_expected_payload(case_module, processed_df, model_bundle, analysis_defaults):
    clf, frt = model_bundle
    exposure, margin, adjustable = analysis_defaults
    result = case_module.analyze_case(
        df=processed_df,
        clf=clf,
        frt=frt,
        row_idx=0,
        exposure=exposure,
        margin=margin,
        adjustable_features=adjustable,
    )
    assert {"beam_id", "pred_mode", "mode_probs", "frt_minutes", "scenario"}.issubset(result.keys())
    assert isinstance(result["mode_probs"], dict)
    assert result["scenario"]["threshold"] == pytest.approx(exposure + margin, rel=1e-6)


def test_case_analyzer_boundary_cover(case_module, processed_df, model_bundle, analysis_defaults):
    clf, frt = model_bundle
    exposure, margin, adjustable = analysis_defaults
    cover = pd.to_numeric(processed_df["Cc"], errors="coerce")
    idx = cover.sub(25.0).abs().idxmin()
    result = case_module.analyze_case(
        df=processed_df,
        clf=clf,
        frt=frt,
        row_idx=int(idx),
        exposure=exposure,
        margin=margin,
        adjustable_features=adjustable,
    )
    assert "action_plan" in result
    assert isinstance(result["action_plan"], dict)


def test_case_analyzer_handles_modified_row(case_module, processed_df, model_bundle, analysis_defaults):
    clf, frt = model_bundle
    exposure, margin, adjustable = analysis_defaults
    custom = processed_df.iloc[0].copy()
    custom["BN"] = "PYTEST_SYNTH"
    custom["LR"] = max(custom.get("LR", 0), 95)
    df_aug = pd.concat([processed_df, pd.DataFrame([custom])], ignore_index=True)
    result = case_module.analyze_case(
        df=df_aug,
        clf=clf,
        frt=frt,
        row_idx=len(df_aug) - 1,
        exposure=exposure,
        margin=margin,
        adjustable_features=adjustable,
    )
    assert result["beam_id"] == "PYTEST_SYNTH"
    assert result["scenario"]["gap_minutes"] <= result["scenario"]["threshold"]
