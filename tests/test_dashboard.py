from __future__ import annotations

from rcbeam_fire.dashboard import app as dashboard_app


def test_dashboard_resources_load(config_path):
    loader = getattr(dashboard_app.load_resources, "__wrapped__", dashboard_app.load_resources)
    cfg, df, clf, frt, adjustable = loader(config_path=config_path)
    assert df.shape[0] > 0
    assert callable(getattr(clf, "predict_label"))
    assert callable(getattr(frt, "predict"))
    assert isinstance(adjustable, list) and adjustable
