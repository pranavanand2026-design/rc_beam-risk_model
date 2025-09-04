"""Case-level analysis and recommendation engine."""

from .insight import (
    ActionPlan,
    CombinationRecommendation,
    Recommendation,
    ScenarioSummary,
    build_case_action_plan,
    compute_reference_stats,
    DOMAIN_TENDENCY,
)

__all__ = [
    "ActionPlan",
    "Recommendation",
    "CombinationRecommendation",
    "ScenarioSummary",
    "build_case_action_plan",
    "compute_reference_stats",
    "DOMAIN_TENDENCY",
]
