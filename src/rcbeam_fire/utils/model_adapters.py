from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ClassifierAdapter:
    """
    Supports either:
      - single-model pack: {"model", "classes", "features", "thresholds"}
      - blend pack: {"xgb_leg": {...}, "ldam_leg": {...}, "w_xgb", "classes", "features", "thresholds"}
    """

    pack: Dict

    def __post_init__(self):
        self.classes: List[str] = self.pack["classes"]
        self.features: List[str] = self.pack["features"]
        self.thresholds: Dict[str, float] = self.pack.get("thresholds", {})
        self.is_blend = ("xgb_leg" in self.pack) and ("ldam_leg" in self.pack)
        self.calibrators: Dict[str, object] = self.pack.get("calibrators", {})

        if self.is_blend:
            self.w_xgb = float(self.pack.get("w_xgb", 0.7))
            self.xgb_model = self.pack["xgb_leg"]["model"]
            self.ldam_scaler = self.pack["ldam_leg"]["scaler"]
            self.ldam_booster = self.pack["ldam_leg"]["booster"]
        else:
            self.model = self.pack["model"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        if self.is_blend:
            p_xgb = self.xgb_model.predict_proba(X)
            from xgboost import DMatrix  # lazy import

            Xs = self.ldam_scaler.transform(X.values)
            p_ldam = self.ldam_booster.predict(DMatrix(Xs))
            proba = self.w_xgb * p_xgb + (1.0 - self.w_xgb) * p_ldam
        else:
            proba = self.model.predict_proba(X)

        if self.calibrators:
            calibrated_cols = []
            for idx, cname in enumerate(self.classes):
                iso = self.calibrators.get(cname)
                if iso is not None:
                    calibrated_cols.append(iso.predict(proba[:, idx]))
                else:
                    calibrated_cols.append(proba[:, idx])
            proba = np.column_stack(calibrated_cols)
            row_sums = proba.sum(axis=1, keepdims=True)
            zero_mask = row_sums[:, 0] <= 0
            proba[~zero_mask] = proba[~zero_mask] / row_sums[~zero_mask]
            if np.any(zero_mask):
                proba[zero_mask] = 1.0 / len(self.classes)

        return proba

    def predict_label(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        proba = self.predict_proba(X)
        pred_idx = proba.argmax(axis=1)
        for ci, cname in enumerate(self.classes):
            thr = self.thresholds.get(cname)
            if thr is None:
                continue
            mask = proba[:, ci] >= thr
            pred_idx[mask] = ci
        return pred_idx, proba
