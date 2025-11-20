from fastapi import APIRouter, Request

from ..schemas.beam import BeamInput
from ..schemas.prediction import PredictionResponse

router = APIRouter(prefix="/api/v1", tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(body: BeamInput, request: Request):
    model_svc = request.app.state.model_service
    predicted_mode, probabilities, frt_minutes = model_svc.predict(body.features)

    threshold = body.exposure_minutes + body.margin_minutes
    gap = frt_minutes - threshold
    verdict = (
        f"Meets scenario (+{gap:.1f} min margin)"
        if gap >= 0
        else f"At risk ({gap:.1f} min short)"
    )

    return PredictionResponse(
        predicted_mode=predicted_mode,
        probabilities=probabilities,
        frt_minutes=frt_minutes,
        threshold_minutes=threshold,
        gap_minutes=gap,
        verdict=verdict,
    )
