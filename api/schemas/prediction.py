from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_mode: str
    probabilities: Dict[str, float]
    frt_minutes: float
    threshold_minutes: float
    gap_minutes: float
    verdict: str


class PlaygroundDeltas(BaseModel):
    frt_delta: float
    prob_no_failure_delta: float
    mode_changed: bool
    previous_mode: Optional[str] = None


class PlaygroundResponse(BaseModel):
    predicted_mode: str
    probabilities: Dict[str, float]
    frt_minutes: float
    threshold_minutes: float
    gap_minutes: float
    verdict: str
    deltas: Optional[PlaygroundDeltas] = None
