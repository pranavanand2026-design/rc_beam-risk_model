from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.model_adapters import ClassifierAdapter, RegressorAdapter

# ---------------------------------------------------------------------------
# Domain metadata
# ---------------------------------------------------------------------------

FEATURE_LABELS: Dict[str, str] = {
    "Cc": "Concrete cover",
    "tins": "Soffit insulation thickness",
    "hi": "Side insulation depth",
    "As": "Steel reinforcement area",
    "Af": "FRP area",
    "LR": "Load ratio",
    "Ld": "Applied load",
    "kins": "Insulation thermal conductivity",
}

FEATURE_UNITS: Dict[str, str] = {
    "Cc": "mm",
    "tins": "mm",
    "hi": "mm",
    "As": "mm\u00b2",
    "Af": "mm\u00b2",
    "LR": "%",
    "Ld": "kN",
    "kins": "W/mK",
}

# +1 -> increasing the feature is typically beneficial for fire performance,
# -1 -> decreasing is beneficial.
DOMAIN_TENDENCY: Dict[str, int] = {
    "Cc": +1,
    "tins": +1,
    "hi": +1,
    "As": +1,
    "Af": +1,
    "LR": -1,
    "Ld": -1,
    "kins": -1,
}

SMALL_EPS = 1e-6


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _bounds_for_feature(
    bounds: Optional[Dict[str, Dict[str, float]]],
    feature: str,
    current: float,
) -> Dict[str, float]:
    if not bounds:
        return {}
    entry = bounds.get(feature)
    if not entry:
        return {}

    info: Dict[str, float] = {"current": current}
    scale = entry.get("scale")
    if scale == "x":
        min_factor = entry.get("min")
        max_factor = entry.get("max")
        step_factor = entry.get("step")
        if min_factor is not None:
            info["min"] = current * float(min_factor)
            info["min_factor"] = float(min_factor)
        if max_factor is not None:
            info["max"] = current * float(max_factor)
            info["max_factor"] = float(max_factor)
        if step_factor is not None:
            info["step_factor"] = float(step_factor)
            info["step"] = abs(current) * float(step_factor)
        info["scale"] = "x"
    else:
        min_raw = entry.get("min")
        max_raw = entry.get("max")
        step_raw = entry.get("step")
        use_percent = False
        if abs(current) > 1.5:
            if max_raw is not None and abs(float(max_raw)) <= 1.0:
                use_percent = True
            elif max_raw is None and min_raw is not None and abs(float(min_raw)) <= 1.0:
                use_percent = True
            elif step_raw is not None and abs(float(step_raw)) <= 1.0 and (max_raw is None or abs(float(max_raw)) <= 1.0):
                use_percent = True
        scale_factor = 100.0 if use_percent else 1.0
        if min_raw is not None:
            info["min"] = float(min_raw) * scale_factor
        if max_raw is not None:
            info["max"] = float(max_raw) * scale_factor
        if step_raw is not None:
            info["step"] = float(step_raw) * scale_factor
    return info


def _clamp_to_bounds(candidate: float, info: Dict[str, float]) -> float:
    if not info:
        return float(candidate)
    value = float(candidate)
    if info.get("min") is not None:
        value = max(value, info["min"])
    if info.get("max") is not None:
        value = min(value, info["max"])
    return value

def _safe_subset(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["target"].astype(str).str.strip().eq("No Failure")
    return df.loc[mask].copy()


def compute_reference_stats(
    df: pd.DataFrame, features: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    """
    Returns percentiles for features derived from the No-Failure subset.
    """
    safe_df = _safe_subset(df)
    stats: Dict[str, Dict[str, float]] = {}

    for feat in features:
        if feat not in safe_df.columns:
            continue
        col = pd.to_numeric(safe_df[feat], errors="coerce").dropna()
        if col.empty:
            continue
        stats[feat] = {
            "median": float(col.quantile(0.50)),
            "p25": float(col.quantile(0.25)),
            "p75": float(col.quantile(0.75)),
            "p90": float(col.quantile(0.90)),
            "min": float(col.min()),
            "max": float(col.max()),
        }
    return stats


def _friendly_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def _unit(feature: str) -> str:
    return FEATURE_UNITS.get(feature, "")
