from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    has_models = (
        hasattr(request.app.state, "model_service")
        and request.app.state.model_service is not None
    )
    if not has_models:
        return {"status": "not_ready", "reason": "models not loaded"}
    return {"status": "ready"}
