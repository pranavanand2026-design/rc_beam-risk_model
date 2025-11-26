from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rcbeam_fire.utils.model_adapters import ClassifierAdapter, RegressorAdapter


class ModelService:
    def __init__(self, clf: ClassifierAdapter, frt: RegressorAdapter):
        self.clf = clf
        self.frt = frt

    def _make_row(self, features: Dict[str, float]) -> pd.DataFrame:
        all_feats = set(self.clf.features) | set(self.frt.features)
        full = {f: features.get(f, 0.0) for f in all_feats}
        return pd.DataFrame([full])

    def predict(
        self, features: Dict[str, float]
    ) -> Tuple[str, Dict[str, float], float]:
        row = self._make_row(features)
        pred_idx, proba = self.clf.predict_label(row)
        predicted_mode = self.clf.classes[int(pred_idx[0])]
        probabilities = {
            cls: float(proba[0, i]) for i, cls in enumerate(self.clf.classes)
        }
        frt_minutes = float(self.frt.predict(row)[0])
        return predicted_mode, probabilities, frt_minutes

    def predict_proba_array(self, features: Dict[str, float]) -> np.ndarray:
        row = self._make_row(features)
        _, proba = self.clf.predict_label(row)
        return proba[0]

    @property
    def classes(self) -> List[str]:
        return self.clf.classes

    @property
    def classifier_features(self) -> List[str]:
        return self.clf.features

    @property
    def regressor_features(self) -> List[str]:
        return self.frt.features
