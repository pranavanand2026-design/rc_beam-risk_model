import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from rcbeam_fire.analysis.insight import (
    DOMAIN_TENDENCY,
    FEATURE_LABELS,
    FEATURE_UNITS,
)

from ..schemas.meta import (
    BeamDetailResponse,
    BeamListResponse,
    BeamSummary,
    DatasetRange,
    FeatureBounds,
    FeatureMeta,
    GroundTruth,
    MetaResponse,
)

router = APIRouter(prefix="/api/v1/meta", tags=["meta"])


@router.get("", response_model=MetaResponse)
async def get_meta(request: Request):
    state = request.app.state
    ranges = state.feature_ranges
    defaults = state.feature_defaults
    all_features = list(
        dict.fromkeys(state.classifier_features + state.regressor_features)
    )

    features_meta = {}
    for f in all_features:
        bounds_raw = state.bounds.get(f)
        bounds = None
        if bounds_raw:
            bounds = FeatureBounds(
                min=bounds_raw.get("min"),
                max=bounds_raw.get("max"),
                step=bounds_raw.get("step"),
                scale=bounds_raw.get("scale"),
            )
        ds_range = None
        if f in ranges:
            r = ranges[f]
            ds_range = DatasetRange(
                min=r["min"], max=r["max"], mean=r["mean"], step=r["step"]
            )
        features_meta[f] = FeatureMeta(
            name=f,
            label=FEATURE_LABELS.get(f, f),
            unit=FEATURE_UNITS.get(f, ""),
            tendency=DOMAIN_TENDENCY.get(f, 0),
            adjustable=f in state.adjustable_features,
            bounds=bounds,
            dataset_range=ds_range,
        )

    return MetaResponse(
        classes=state.classes,
        classifier_features=state.classifier_features,
        regressor_features=state.regressor_features,
        adjustable_features=state.adjustable_features,
        features=features_meta,
        defaults=defaults,
    )


@router.get("/beams", response_model=BeamListResponse)
async def list_beams(request: Request):
    ids = request.app.state.beam_ids
    return BeamListResponse(
        beams=[BeamSummary(id=bid) for bid in ids],
        count=len(ids),
    )


@router.get("/beams/{beam_id}", response_model=BeamDetailResponse)
async def get_beam(beam_id: str, request: Request):
    state = request.app.state
    ds = state.dataset_service
    all_features = list(
        dict.fromkeys(state.classifier_features + state.regressor_features)
    )

    # Single lookup instead of 3 separate calls
    row = ds.get_beam(beam_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Beam '{beam_id}' not found")

    features = {}
    for f in all_features:
        if f in row.index:
            val = row[f]
            try:
                features[f] = float(pd.to_numeric(val, errors="coerce"))
            except (TypeError, ValueError):
                features[f] = 0.0
        else:
            features[f] = 0.0

    mode = None
    for col in ("LimitState", "target"):
        if col in row.index and pd.notna(row[col]):
            mode = str(row[col])
            break
    frt = None
    if "FR" in row.index:
        try:
            val = float(pd.to_numeric(row["FR"], errors="coerce"))
            if np.isfinite(val):
                frt = val
        except (TypeError, ValueError):
            pass

    ground_truth = GroundTruth(mode=mode, frt=frt) if mode or frt else None

    return BeamDetailResponse(
        id=beam_id, features=features, ground_truth=ground_truth
    )
