from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class ScenarioResponse(BaseModel):
    exposure: float
    margin: float
    threshold: float
    frt_pred: float
    gap_minutes: float
    pred_mode: str
    prob_no_failure: float
    verdict: str


class RecommendationResponse(BaseModel):
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


class CombinationResponse(BaseModel):
    features: List[str]
    actions: List[str]
    expected_frt: float
    expected_delta_frt: float
    expected_prob_no_fail: float
    expected_delta_prob: float
    predicted_mode: str


class EurocodeCheckResponse(BaseModel):
    verdict: str
    criteria: List[Dict]
    adjustments: List[Dict]


class PredictionSummary(BaseModel):
    predicted_mode: str
    probabilities: Dict[str, float]
    frt_minutes: float
    threshold_minutes: float
    gap_minutes: float
    verdict: str


class AnalyzeResponse(BaseModel):
    scenario: ScenarioResponse
    recommendations: List[RecommendationResponse]
    combination: Optional[CombinationResponse] = None
    eurocode: Optional[EurocodeCheckResponse] = None
    notes: List[str]
    prediction: PredictionSummary
