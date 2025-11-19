from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI

from rcbeam_fire.config import load_config
from rcbeam_fire.utils.io import load_processed_dataset
from rcbeam_fire.utils.model_adapters import ClassifierAdapter, RegressorAdapter

from .config import CONFIG_PATH
from ..services.model_service import ModelService
from ..services.dataset_service import DatasetService
from ..services.analysis_service import AnalysisService


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(CONFIG_PATH)
    df = load_processed_dataset(cfg["paths"])

    models_dir = Path(cfg["paths"].get("models", "models")) / "checkpoints"
    clf_candidates = [
        "blend_xgb_ldam_nophys.joblib",
        "suite_lgbm_smote.joblib",
        "suite_xgb_smote.joblib",
    ]
    clf_path = next(
        (models_dir / fn for fn in clf_candidates if (models_dir / fn).exists()),
        None,
    )
    if clf_path is None:
        raise FileNotFoundError("No classifier checkpoint found in models/checkpoints/")

    frt_path = models_dir / "frt_regressor.joblib"
    if not frt_path.exists():
        raise FileNotFoundError("Missing frt_regressor.joblib in models/checkpoints/")

    clf = ClassifierAdapter(joblib.load(clf_path))
    frt = RegressorAdapter(joblib.load(frt_path))

    analysis_cfg = cfg.get("analysis", {})
    bounds_raw = analysis_cfg.get("bounds", {})
    bounds = _parse_bounds(bounds_raw)
    adjustable = analysis_cfg.get(
        "adjustable_features", ["Cc", "tins", "hi", "LR", "As", "Af"]
    )
    defaults = {
        "exposure_minutes": float(analysis_cfg.get("exposure_minutes", 90)),
        "margin_minutes": float(analysis_cfg.get("margin_minutes", 10)),
    }

    model_svc = ModelService(clf=clf, frt=frt)
    ds = DatasetService(df=df, id_col="BN")

    app.state.model_service = model_svc
    app.state.dataset_service = ds
    app.state.analysis_service = AnalysisService(
        model_service=model_svc,
        dataset_service=ds,
        bounds=bounds,
        adjustable_features=adjustable,
    )
    app.state.cfg = cfg
    app.state.bounds = bounds
    app.state.adjustable_features = adjustable
    app.state.defaults = defaults
    app.state.classes = clf.classes
    app.state.classifier_features = clf.features
    app.state.regressor_features = frt.features

    # Pre-compute static dataset stats so /meta doesn't recompute on every request
    all_features = list(dict.fromkeys(clf.features + frt.features))
    app.state.feature_ranges = ds.compute_feature_ranges(all_features)
    app.state.feature_defaults = ds.get_feature_defaults(all_features)
    app.state.beam_ids = ds.list_beam_ids()

    yield


def _parse_bounds(raw: dict) -> dict:
    parsed = {}
    for feat, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        norm = {}
        for key, value in entry.items():
            if key in {"min", "max", "step"}:
                try:
                    norm[key] = float(value)
                except (TypeError, ValueError):
                    continue
            elif key == "scale":
                norm[key] = str(value)
        if norm:
            parsed[str(feat)] = norm
    return parsed
