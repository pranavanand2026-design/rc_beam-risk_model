import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rcbeam_fire.config import load_config
from rcbeam_fire.data.preprocess import run_preprocess
from rcbeam_fire.utils.io import load_processed_dataset


@pytest.fixture(scope="session")
def config_path() -> Path:
    return ROOT / "config.yaml"


@pytest.fixture(scope="session")
def cfg(config_path):
    return load_config(config_path)


@pytest.fixture(scope="session")
def processed_df(cfg, config_path):
    try:
        return load_processed_dataset(cfg["paths"])
    except FileNotFoundError:
        df, _ = run_preprocess(config_path)
        return df


@pytest.fixture(scope="session")
def models_dir(cfg):
    return Path(cfg["paths"].get("models", "models")) / "checkpoints"


@pytest.fixture(scope="session")
def model_bundle(models_dir):
    import joblib
    from rcbeam_fire.utils.model_adapters import ClassifierAdapter, RegressorAdapter

    prefer = [
        "blend_xgb_ldam_nophys.joblib",
        "suite_lgbm_smote.joblib",
        "suite_xgb_smote.joblib",
    ]
    clf_path = next((models_dir / name for name in prefer if (models_dir / name).exists()), None)
    frt_path = models_dir / "frt_regressor.joblib"
    if clf_path is None or not frt_path.exists():
        pytest.skip("Trained model artifacts not found; run scripts/run_pipeline.py first.")
    clf = ClassifierAdapter(joblib.load(clf_path))
    frt = RegressorAdapter(joblib.load(frt_path))
    return clf, frt


@pytest.fixture(scope="session")
def case_module():
    import importlib.util

    script_path = ROOT / "scripts/12_case_analyzer.py"
    spec = importlib.util.spec_from_file_location("case_analyzer", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
