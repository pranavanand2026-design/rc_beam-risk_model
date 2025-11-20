from fastapi import APIRouter, Request

from ..schemas.beam import PlaygroundInput
from ..schemas.prediction import PlaygroundDeltas, PlaygroundResponse

router = APIRouter(prefix="/api/v1", tags=["playground"])


@router.post("/playground", response_model=PlaygroundResponse)
async def playground(body: PlaygroundInput, request: Request):
    model_svc = request.app.state.model_service
    predicted_mode, probabilities, frt_minutes = model_svc.predict(body.features)

    threshold = body.exposure_minutes + body.margin_minutes
    gap = frt_minutes - threshold
    verdict = (
        f"Meets scenario (+{gap:.1f} min margin)"
        if gap >= 0
        else f"At risk ({gap:.1f} min short)"
    )

    deltas = None
    if body.baseline_features:
        base_mode, base_probs, base_frt = model_svc.predict(body.baseline_features)
        deltas = PlaygroundDeltas(
            frt_delta=frt_minutes - base_frt,
            prob_no_failure_delta=(
                probabilities.get("No Failure", 0.0)
                - base_probs.get("No Failure", 0.0)
            ),
            mode_changed=predicted_mode != base_mode,
            previous_mode=base_mode if predicted_mode != base_mode else None,
        )

    return PlaygroundResponse(
        predicted_mode=predicted_mode,
        probabilities=probabilities,
        frt_minutes=frt_minutes,
        threshold_minutes=threshold,
        gap_minutes=gap,
        verdict=verdict,
        deltas=deltas,
    )
