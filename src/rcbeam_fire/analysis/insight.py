from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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
    "As": "mm²",
    "Af": "mm²",
    "LR": "%",
    "Ld": "kN",
    "kins": "W/mK",
}

# +1 → increasing the feature is typically beneficial for fire performance,
# -1 → decreasing is beneficial.
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


# ---------------------------------------------------------------------------
# Recommendation dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSummary:
    exposure: float
    margin: float
    threshold: float
    frt_pred: float
    gap_minutes: float
    pred_mode: str
    prob_no_failure: float

    @property
    def verdict(self) -> str:
        if self.gap_minutes >= 0:
            return f"Meets scenario (+{self.gap_minutes:.1f} min margin)"
        return f"At risk ({self.gap_minutes:.1f} min short)"


@dataclass
class Recommendation:
    feature: str
    title: str
    action: str
    current: float
    target: float
    delta_value: float
    expected_frt: float
    expected_delta_frt: float
    expected_prob_no_fail: float
    expected_delta_prob: float
    new_mode: str
    rationale: str
    priority: float
    unit: str


@dataclass
class CombinationRecommendation:
    features: List[str]
    actions: List[str]
    expected_frt: float
    expected_delta_frt: float
    expected_prob_no_fail: float
    expected_delta_prob: float
    predicted_mode: str


@dataclass
class ActionPlan:
    scenario: ScenarioSummary
    recommendations: List[Recommendation]
    references: Dict[str, Dict[str, float]]
    notes: List[str]
    combination: Optional[CombinationRecommendation] = None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _choose_target_value(
    feature: str,
    current: float,
    stats: Dict[str, float],
    direction: int,
    bounds: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[float]:
    """Pick a target value guided by safe-case percentiles and bounded envelopes."""
    info = _bounds_for_feature(bounds, feature, current)
    candidates: List[float] = []

    def _add_candidate(raw_val: Optional[float]) -> None:
        if raw_val is None:
            return
        cand = _clamp_to_bounds(float(raw_val), info)
        if direction > 0 and cand > current + SMALL_EPS:
            candidates.append(cand)
        elif direction < 0 and cand < current - SMALL_EPS:
            candidates.append(cand)

    if direction > 0:
        for key in ("p75", "p90", "max"):
            _add_candidate(stats.get(key))
    elif direction < 0:
        for key in ("p25", "median", "min"):
            _add_candidate(stats.get(key))

    if direction > 0 and info.get("max") is not None:
        _add_candidate(info["max"])
    if direction < 0 and info.get("min") is not None:
        _add_candidate(info["min"])

    if not candidates and info.get("step"):
        step = info["step"]
        if direction > 0:
            _add_candidate(current + step)
        elif direction < 0:
            _add_candidate(current - step)

    if not candidates:
        # modest fallback nudging by 5% (or absolute 5 units) within bounds
        delta = max(abs(current) * 0.05, 5.0)
        if direction > 0:
            _add_candidate(current + delta)
        elif direction < 0:
            _add_candidate(current - delta)

    if direction > 0:
        return min(candidates, default=None)
    if direction < 0:
        return max(candidates, default=None)
    return None


def _format_action(direction: int, feature: str, delta: float, current: float, target: float) -> str:
    label = _friendly_label(feature)
    unit = _unit(feature)
    if direction >= 0:
        trend = "Increase"
    else:
        trend = "Reduce"
    if unit:
        return f"{trend} {label} from {current:.2f} {unit} to ~{target:.2f} {unit}"
    return f"{trend} {label} from {current:.2f} to ~{target:.2f}"


def _feature_flag(feature: str, top_list: Sequence[str]) -> bool:
    return any(str(feature) == str(item) for item in top_list)


def _compute_priority(
    delta_frt: float,
    delta_prob: float,
    gap_minutes: float,
    driver_flags: Tuple[bool, bool],
) -> float:
    score = 0.0
    if delta_frt > 0:
        score += min(delta_frt / max(1.0, abs(gap_minutes) if gap_minutes < 0 else delta_frt), 2.0)
    if delta_prob > 0:
        score += delta_prob * 4.0
    if driver_flags[0]:
        score += 0.4
    if driver_flags[1]:
        score += 0.4
    return score


def _best_combination(
    row: pd.Series,
    recommendations: Sequence[Recommendation],
    clf: ClassifierAdapter,
    frt: RegressorAdapter,
    base_prob: float,
    base_frt: float,
    bounds: Optional[Dict[str, Dict[str, float]]] = None,
    scenario_threshold: Optional[float] = None,
    max_size: int = 4,
) -> Optional[CombinationRecommendation]:
    if len(recommendations) < 2:
        return None

    idx_no_failure = None
    if "No Failure" in clf.classes:
        idx_no_failure = int(clf.classes.index("No Failure"))

    best_combo: Optional[CombinationRecommendation] = None
    best_score: Tuple[float, float, float, float, float, float] = (
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
    )

    threshold = scenario_threshold if scenario_threshold is not None else base_frt

    for size in range(2, min(max_size, len(recommendations)) + 1):
        for combo in combinations(recommendations, size):
            mutant = row.copy()
            for rec in combo:
                mutant[rec.feature] = rec.target
            mutant_df = mutant.to_frame().T
            pred_idx, proba = clf.predict_label(mutant_df)
            predicted_mode = clf.classes[int(pred_idx[0])]
            prob_no_fail = float(proba[0, idx_no_failure]) if idx_no_failure is not None else 0.0
            frt_new = float(frt.predict(mutant_df)[0])
            delta_prob = prob_no_fail - base_prob
            delta_frt = frt_new - base_frt

            if delta_prob <= 0 and delta_frt <= 0:
                continue

            margin = frt_new - threshold
            meets_threshold = margin >= -1e-6
            score_primary = 0
            if predicted_mode == "No Failure" and meets_threshold:
                score_primary = 3
            elif predicted_mode == "No Failure":
                score_primary = 2
            elif meets_threshold:
                score_primary = 1

            total_rel_change = 0.0
            for rec in combo:
                info = _bounds_for_feature(bounds, rec.feature, rec.current)
                range_span = None
                if info.get("max") is not None and info.get("min") is not None:
                    range_span = max(info["max"] - info["min"], SMALL_EPS)
                elif info.get("scale") == "x":
                    min_factor = info.get("min_factor", 1.0)
                    max_factor = info.get("max_factor", 1.0)
                    range_span = max(abs(rec.current) * (max_factor - min_factor), SMALL_EPS)
                if not range_span or range_span <= SMALL_EPS:
                    range_span = max(abs(rec.current), 1.0)
                total_rel_change += abs(rec.target - rec.current) / range_span

            score = (
                float(score_primary),
                float(delta_prob),
                -abs(margin) if score_primary >= 2 else float(margin),
                -float(total_rel_change),
                -float(len(combo)),
                float(delta_frt),
            )

            if score > best_score:
                best_score = score
                best_combo = CombinationRecommendation(
                    features=[rec.feature for rec in combo],
                    actions=[rec.action for rec in combo],
                    expected_frt=frt_new,
                    expected_delta_frt=delta_frt,
                    expected_prob_no_fail=prob_no_fail,
                    expected_delta_prob=delta_prob,
                    predicted_mode=predicted_mode,
                )

    return best_combo


def build_case_action_plan(
    df: pd.DataFrame,
    row: pd.Series,
    clf: ClassifierAdapter,
    frt: RegressorAdapter,
    scenario: Dict[str, float],
    base_proba: np.ndarray,
    pred_mode: str,
    frt_pred: float,
    top_mode: Sequence[str],
    top_frt: Sequence[str],
    adjustable_features: Iterable[str],
    max_actions: int = 3,
    domain_tendency: Optional[Dict[str, int]] = None,
    bounds: Optional[Dict[str, Dict[str, float]]] = None,
) -> ActionPlan:
    """
    Generate an ordered list of actionable recommendations for a single beam.
    """
    domain_tendency = domain_tendency or DOMAIN_TENDENCY
    ref_stats = compute_reference_stats(df, adjustable_features)
    gap = float(scenario.get("gap_minutes", frt_pred - scenario.get("threshold", frt_pred)))
    scenario_summary = ScenarioSummary(
        exposure=float(scenario.get("exposure", np.nan)),
        margin=float(scenario.get("margin", np.nan)),
        threshold=float(scenario.get("threshold", np.nan)),
        frt_pred=float(frt_pred),
        gap_minutes=gap,
        pred_mode=pred_mode,
        prob_no_failure=float(base_proba[clf.classes.index("No Failure")]) if "No Failure" in clf.classes else float("nan"),
    )

    if gap >= 0 and pred_mode == "No Failure":
        return ActionPlan(
            scenario=scenario_summary,
            recommendations=[],
            references=ref_stats,
            notes=["Beam already satisfies the exposure scenario with positive margin. Maintain detailing and document assumptions."],
        )

    recommendations: List[Recommendation] = []
    base_row_df = row.to_frame().T
    _, base_proba_full = clf.predict_label(base_row_df)
    base_no_fail_prob = float(base_proba_full[0, clf.classes.index("No Failure")]) if "No Failure" in clf.classes else 0.0

    for feature in adjustable_features:
        if feature not in row.index:
            continue
        direction = domain_tendency.get(feature, 0)
        if direction == 0:
            continue
        try:
            current_val = float(row[feature])
        except (TypeError, ValueError):
            continue

        stats = ref_stats.get(feature, {})
        target_val = _choose_target_value(feature, current_val, stats, direction, bounds)
        if target_val is None:
            continue
        if abs(target_val - current_val) < 1e-3:
            continue

        mutant = row.copy()
        mutant[feature] = target_val
        mutant_df = mutant.to_frame().T
        pred_idx_new, proba_new = clf.predict_label(mutant_df)
        new_mode = clf.classes[int(pred_idx_new[0])]
        no_fail_prob_new = float(proba_new[0, clf.classes.index("No Failure")]) if "No Failure" in clf.classes else 0.0

        frt_new = float(frt.predict(mutant_df)[0])
        delta_frt = frt_new - frt_pred
        delta_prob = no_fail_prob_new - base_no_fail_prob

        driver_flags = (_feature_flag(feature, top_mode), _feature_flag(feature, top_frt))
        priority = _compute_priority(delta_frt, delta_prob, gap, driver_flags)
        if delta_frt <= 0 and delta_prob <= 0:
            # ignore detrimental moves
            continue

        title = _friendly_label(feature)
        action_text = _format_action(direction, feature, target_val - current_val, current_val, target_val)
        rationale_bits = []
        if driver_flags[0]:
            rationale_bits.append("key driver of current failure mode")
        if driver_flags[1]:
            rationale_bits.append("strong influence on predicted fire resistance time")
        if stats:
            if direction > 0 and current_val < stats.get("median", current_val):
                rationale_bits.append("currently below median for safe beams")
            if direction < 0 and current_val > stats.get("median", current_val):
                rationale_bits.append("above typical safe-beam median")

        rationale = "; ".join(rationale_bits) if rationale_bits else "Model expects this adjustment to improve safety margin."

        recommendations.append(
            Recommendation(
                feature=feature,
                title=title,
                action=action_text,
                current=current_val,
                target=target_val,
                delta_value=target_val - current_val,
                expected_frt=frt_new,
                expected_delta_frt=delta_frt,
                expected_prob_no_fail=no_fail_prob_new,
                expected_delta_prob=delta_prob,
                new_mode=new_mode,
                rationale=rationale,
                priority=priority,
                unit=_unit(feature),
            )
        )

    recommendations.sort(key=lambda r: r.priority, reverse=True)

    candidate_recs = recommendations[: max_actions * 2]
    top_recs = recommendations[:max_actions]

    combination = _best_combination(
        row=row,
        recommendations=candidate_recs,
        clf=clf,
        frt=frt,
        base_prob=base_no_fail_prob,
        base_frt=frt_pred,
        bounds=bounds,
        scenario_threshold=scenario_summary.threshold,
    )

    notes: List[str] = []
    if not recommendations:
        notes.append("No single-parameter adjustment within reference envelopes improved the scenario; consider combined modifications or extending exposure criteria.")
    if combination:
        if combination.predicted_mode == "No Failure":
            notes.append("Combined adjustments (see plan) are expected to satisfy the scenario and switch the classifier to No Failure.")
        elif combination.expected_delta_frt > 0 or combination.expected_delta_prob > 0:
            notes.append("Combined adjustments (see plan) provide the best uplift available within the reference envelopes, though residual risk remains.")

    return ActionPlan(
        scenario=scenario_summary,
        recommendations=top_recs,
        references=ref_stats,
        notes=notes,
        combination=combination,
    )
