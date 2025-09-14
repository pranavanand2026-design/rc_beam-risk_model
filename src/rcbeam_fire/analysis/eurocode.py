"""
Helpers to approximate Eurocode EN 1992-1-2 fire design envelopes.

We keep the rule-set intentionally lightweight: a handful of lookups capture
minimum cover, insulation thickness and maximum load ratio for common fire
ratings (R60/R90/R120). Values come from the design guidance discussed in the
project report and serve as defaults – teams can override them via
`analysis.eurocode` in the config file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Default envelope inspired by EN 1992-1-2 tables for simply supported beams.
# Teams can override these via config if their project uses different classes.
DEFAULT_REQUIREMENTS: Dict[str, Dict[str, float]] = {
    "r60": {
        "minutes": 60,
        "min_cover_mm": 30.0,
        "min_insulation_mm": 60.0,
        "max_load_ratio": 55.0,
    },
    "r90": {
        "minutes": 90,
        "min_cover_mm": 40.0,
        "min_insulation_mm": 90.0,
        "max_load_ratio": 50.0,
    },
    "r120": {
        "minutes": 120,
        "min_cover_mm": 50.0,
        "min_insulation_mm": 110.0,
        "max_load_ratio": 45.0,
    },
}


@dataclass(frozen=True)
class EurocodeRequirement:
    minutes: float
    min_cover_mm: float
    min_insulation_mm: float
    max_load_ratio: float

    @staticmethod
    def from_mapping(mapping: Mapping[str, float]) -> "EurocodeRequirement":
        return EurocodeRequirement(
            minutes=float(mapping["minutes"]),
            min_cover_mm=float(mapping["min_cover_mm"]),
            min_insulation_mm=float(mapping["min_insulation_mm"]),
            max_load_ratio=float(mapping["max_load_ratio"]),
        )


def _normalise_requirements(cfg_section: Optional[Mapping[str, Mapping[str, float]]]) -> List[EurocodeRequirement]:
    data = dict(DEFAULT_REQUIREMENTS)
    if cfg_section:
        for key, req in cfg_section.items():
            if not isinstance(req, Mapping):
                continue
            data[key] = {
                "minutes": float(req.get("minutes", req.get("rating", 0))),
                "min_cover_mm": float(req.get("min_cover_mm", req.get("cover", 0))),
                "min_insulation_mm": float(req.get("min_insulation_mm", req.get("insulation", 0))),
                "max_load_ratio": float(req.get("max_load_ratio", req.get("load_ratio", 1))),
            }
    reqs = [EurocodeRequirement.from_mapping(v) for v in data.values()]
    reqs.sort(key=lambda r: r.minutes)
    return reqs


def pick_requirement(threshold_minutes: float, cfg_section: Optional[Mapping[str, Mapping[str, float]]] = None) -> EurocodeRequirement:
    """
    Return the Eurocode requirement that covers the desired exposure threshold.
    """
    reqs = _normalise_requirements(cfg_section)
    for req in reqs:
        if req.minutes >= threshold_minutes:
            return req
    return reqs[-1]


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class EurocodeAdjustment:
    feature: str
    current: float
    target: float
    delta: float
    unit: str
    rationale: str


def evaluate_compliance(
    row: pd.Series,
    requirement: EurocodeRequirement,
    feature_alias: Optional[Mapping[str, Iterable[str]]] = None,
) -> Tuple[str, List[Dict[str, float]], List[EurocodeAdjustment]]:
    """
    Compare the beam row against Eurocode-inspired envelopes.

    Parameters
    ----------
    row : pd.Series
        Beam record.
    requirement : EurocodeRequirement
        Target envelope derived from exposure threshold.
    feature_alias : mapping
        Optional mapping for feature lookups. Defaults cover Cc (cover), hi/tins (insulation) and LR (load ratio).

    Returns
    -------
    verdict : str
        'Compliant', 'Marginal' or 'Non-compliant'.
    criteria : list of dict
        Per-criterion breakdown with actual vs. required values.
    adjustments : list[EurocodeAdjustment]
        Minimum adjustments required to satisfy failed criteria (one per failing feature where possible).
    """
    alias = feature_alias or {
        "cover": ("Cc",),
        "insulation": ("hi", "tins"),
        "load_ratio": ("LR",),
    }

    def _pick_value(keys: Iterable[str]) -> float:
        for key in keys:
            if key in row.index:
                val = _to_float(row[key])
                if np.isfinite(val):
                    return val
        return float("nan")

    cover_val = _pick_value(alias.get("cover", []))
    insulation_val = _pick_value(alias.get("insulation", []))
    load_ratio_val = _pick_value(alias.get("load_ratio", []))

    criteria: List[Dict[str, float]] = []

    def _status_min(actual: float, required: float, unit: str) -> Dict[str, float]:
        if not np.isfinite(actual):
            status = "unknown"
            delta = float("nan")
        else:
            delta = actual - required
            status = "pass" if delta >= -1e-6 else "fail"
        return {
            "criterion": f"min_{unit}",
            "actual": float(actual) if np.isfinite(actual) else None,
            "required": float(required),
            "delta": float(delta) if np.isfinite(delta) else None,
            "status": status,
        }

    def _status_max(actual: float, required: float, display_actual: float, display_required: float) -> Dict[str, float]:
        if not np.isfinite(actual):
            status = "unknown"
            delta = float("nan")
        else:
            delta = required - actual
            status = "pass" if actual <= required + 1e-6 else "fail"
        return {
            "criterion": "max_load_ratio",
            "actual": float(display_actual) if np.isfinite(display_actual) else None,
            "required": float(display_required) if np.isfinite(display_required) else None,
            "delta": float(display_required - display_actual) if np.isfinite(display_required) and np.isfinite(display_actual) else None,
            "status": status,
        }

    criteria.append(
        {
            **_status_min(cover_val, requirement.min_cover_mm, "cover_mm"),
            "description": f"Concrete cover ≥ {requirement.min_cover_mm:.0f} mm",
        }
    )
    criteria.append(
        {
            **_status_min(insulation_val, requirement.min_insulation_mm, "insulation_mm"),
            "description": f"Insulation depth ≥ {requirement.min_insulation_mm:.0f} mm",
        }
    )
    lr_actual = load_ratio_val
    lr_required = requirement.max_load_ratio
    is_percent_actual = np.isfinite(lr_actual) and lr_actual > 1.5
    is_percent_required = np.isfinite(lr_required) and lr_required > 1.5
    lr_actual_norm = (lr_actual / 100.0) if is_percent_actual else lr_actual
    lr_required_norm = (lr_required / 100.0) if is_percent_required else lr_required
    display_as_percent = is_percent_actual or is_percent_required
    if display_as_percent:
        lr_actual_display = lr_actual if np.isfinite(lr_actual) else float("nan")
        lr_required_display = lr_required if np.isfinite(lr_required) else float("nan")
        lr_desc = f"Load ratio ≤ {lr_required_display:.0f} %"
    else:
        lr_actual_display = lr_actual_norm if np.isfinite(lr_actual_norm) else float("nan")
        lr_required_display = lr_required_norm if np.isfinite(lr_required_norm) else float("nan")
        lr_desc = f"Load ratio ≤ {lr_required_norm:.2f}"
    criteria.append(
        {
            **_status_max(lr_actual_norm, lr_required_norm, lr_actual_display, lr_required_display),
            "description": lr_desc,
        }
    )

    failed = [c for c in criteria if c["status"] == "fail"]
    unknown = [c for c in criteria if c["status"] == "unknown"]

    if failed:
        verdict = "Non-compliant"
    elif unknown:
        verdict = "Marginal"
    else:
        verdict = "Compliant"

    adjustments: List[EurocodeAdjustment] = []

    def _first_existing(keys: Sequence[str]) -> Optional[str]:
        for key in keys:
            if key in row.index and np.isfinite(_to_float(row[key])):
                return key
        return None

    # Cover adjustment
    if failed and any(c["criterion"] == "min_cover_mm" and c["status"] == "fail" for c in criteria):
        feat = _first_existing(alias.get("cover", []))
        if feat:
            current = _to_float(row[feat])
            target = max(current, requirement.min_cover_mm)
            adjustments.append(
                EurocodeAdjustment(
                    feature=feat,
                    current=current,
                    target=target,
                    delta=target - current,
                    unit="mm",
                    rationale=f"Increase concrete cover to meet Eurocode R{int(requirement.minutes)} minimum ({requirement.min_cover_mm:.0f} mm).",
                )
            )

    # Insulation adjustment (side or soffit)
    if failed and any(c["criterion"] == "min_insulation_mm" and c["status"] == "fail" for c in criteria):
        for feat in alias.get("insulation", []):
            if feat not in row.index:
                continue
            current = _to_float(row[feat])
            if not np.isfinite(current):
                continue
            target = max(current, requirement.min_insulation_mm)
            adjustments.append(
                EurocodeAdjustment(
                    feature=feat,
                    current=current,
                    target=target,
                    delta=target - current,
                    unit="mm",
                    rationale=f"Increase insulation depth to Eurocode R{int(requirement.minutes)} minimum ({requirement.min_insulation_mm:.0f} mm).",
                )
            )
            # include both hi and tins if present; no break to surface each channel

    # Load ratio adjustment
    if failed and any(c["criterion"] == "max_load_ratio" and c["status"] == "fail" for c in criteria):
        feat = _first_existing(alias.get("load_ratio", []))
        if feat:
            current = _to_float(row[feat])
            if np.isfinite(current):
                target = min(current, requirement.max_load_ratio)
                adjustments.append(
                    EurocodeAdjustment(
                        feature=feat,
                        current=current,
                        target=target,
                        delta=target - current,
                        unit="%",
                        rationale=f"Reduce load ratio to Eurocode R{int(requirement.minutes)} cap ({requirement.max_load_ratio:.0f}%).",
                    )
                )

    return verdict, criteria, adjustments


__all__ = [
    "EurocodeRequirement",
    "pick_requirement",
    "evaluate_compliance",
    "EurocodeAdjustment",
]

