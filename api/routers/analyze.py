from fastapi import APIRouter, Request

from ..schemas.beam import BeamInput
from ..schemas.analysis import AnalyzeResponse

router = APIRouter(prefix="/api/v1", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: BeamInput, request: Request):
    analysis_svc = request.app.state.analysis_service
    result = analysis_svc.full_analysis(
        features=body.features,
        exposure_minutes=body.exposure_minutes,
        margin_minutes=body.margin_minutes,
    )
    return AnalyzeResponse(**result)
