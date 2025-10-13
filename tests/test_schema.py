from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "BN",
    "target",
    "LimitState",
    "L",
    "Ac",
    "Cc",
    "As",
    "Af",
    "tins",
    "hi",
    "fc",
    "fy",
    "Ld",
    "LR",
    "FR",
}


def test_processed_dataset_has_expected_columns(processed_df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(processed_df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"


def test_target_column_is_populated(processed_df: pd.DataFrame):
    assert processed_df["target"].notna().all()
    assert processed_df["target"].str.strip().ne("").all()


def test_physical_ranges(processed_df: pd.DataFrame, cfg):
    upper_limits = {"Cc": 120, "tins": 250, "hi": 150, "LR": 200, "As": 5000, "Af": 5000}
    for col in cfg.get("analysis", {}).get("bounds", {}).keys():
        if col not in processed_df:
            continue
        series = pd.to_numeric(processed_df[col], errors="coerce").dropna()
        assert (series >= -5).all(), f"{col} contains implausible negative values"
        limit = upper_limits.get(col)
        if limit is not None:
            assert (series <= limit).all(), f"{col} has values above practical limits"


def test_boundary_cover_samples_exist(processed_df: pd.DataFrame, cfg):
    cover = pd.to_numeric(processed_df["Cc"], errors="coerce")
    assert cover.notna().all()
    spread = cover.max() - cover.min()
    assert spread >= 10, "Cover data does not span enough range for boundary tests"


def test_no_negative_fire_resistance(processed_df: pd.DataFrame):
    fr = pd.to_numeric(processed_df["FR"], errors="coerce").dropna()
    assert (fr >= 0).all(), "Fire resistance time should not be negative"
