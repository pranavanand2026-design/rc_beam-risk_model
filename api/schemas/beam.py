from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class BeamInput(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="All model features as key-value pairs"
    )
    exposure_minutes: float = Field(default=90, ge=0)
    margin_minutes: float = Field(default=10, ge=0)


class PlaygroundInput(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="All model features as key-value pairs"
    )
    baseline_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Original feature values for computing deltas",
    )
    exposure_minutes: float = Field(default=90, ge=0)
    margin_minutes: float = Field(default=10, ge=0)
