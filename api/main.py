from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_cors_origins
from .core.lifespan import lifespan
from .routers import analyze, health, meta, playground, predict

app = FastAPI(
    title="RC Beam Fire Design API",
    version="1.0.0",
    description="ML-powered fire resistance prediction and design optimization for RC beams",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(meta.router)
app.include_router(predict.router)
app.include_router(analyze.router)
app.include_router(playground.router)
