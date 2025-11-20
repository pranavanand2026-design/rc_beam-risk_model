from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class FeatureBounds(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    scale: Optional[str] = None


class DatasetRange(BaseModel):
    min: float
    max: float
    mean: float
    step: float


class FeatureMeta(BaseModel):
    name: str
    label: str
    unit: str
    tendency: int
    adjustable: bool
    bounds: Optional[FeatureBounds] = None
    dataset_range: Optional[DatasetRange] = None


class MetaResponse(BaseModel):
    classes: List[str]
    classifier_features: List[str]
    regressor_features: List[str]
    adjustable_features: List[str]
    features: Dict[str, FeatureMeta]
    defaults: Dict[str, float]


class BeamSummary(BaseModel):
    id: str


class BeamListResponse(BaseModel):
    beams: List[BeamSummary]
    count: int


class GroundTruth(BaseModel):
    mode: Optional[str] = None
    frt: Optional[float] = None


class BeamDetailResponse(BaseModel):
    id: str
    features: Dict[str, float]
    ground_truth: Optional[GroundTruth] = None
