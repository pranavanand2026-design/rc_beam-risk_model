from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DatasetService:
    def __init__(self, df: pd.DataFrame, id_col: str = "BN"):
        self.df = df
        self.id_col = id_col

    def list_beam_ids(self) -> List[str]:
        return sorted(self.df[self.id_col].astype(str).unique().tolist())

    def get_beam(self, beam_id: str) -> Optional[pd.Series]:
        matches = self.df[self.df[self.id_col].astype(str) == str(beam_id)]
        if matches.empty:
            return None
        return matches.iloc[0]

    def get_beam_features(
        self, beam_id: str, feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        row = self.get_beam(beam_id)
        if row is None:
            return None
        features = {}
        for f in feature_names:
            if f in row.index:
                val = row[f]
                try:
                    features[f] = float(pd.to_numeric(val, errors="coerce"))
                except (TypeError, ValueError):
                    features[f] = 0.0
            else:
                features[f] = 0.0
        return features

    def get_ground_truth(
        self, beam_id: str
    ) -> Optional[Tuple[Optional[str], Optional[float]]]:
        row = self.get_beam(beam_id)
        if row is None:
            return None
        mode = None
        for col in ("LimitState", "target"):
            if col in row.index and pd.notna(row[col]):
                mode = str(row[col])
                break
        frt = None
        if "FR" in row.index:
            try:
                val = float(pd.to_numeric(row["FR"], errors="coerce"))
                if np.isfinite(val):
                    frt = val
            except (TypeError, ValueError):
                pass
        return mode, frt

    def compute_feature_ranges(
        self, features: List[str]
    ) -> Dict[str, Dict[str, float]]:
        ranges = {}
        for feat in features:
            if feat not in self.df.columns:
                continue
            col = pd.to_numeric(self.df[feat], errors="coerce").dropna()
            if col.empty:
                continue
            ranges[feat] = {
                "min": float(col.quantile(0.05)),
                "max": float(col.quantile(0.95)),
                "mean": float(col.mean()),
                "step": float(
                    max((col.quantile(0.75) - col.quantile(0.25)) / 20.0, 0.1)
                ),
            }
        return ranges

    def get_feature_defaults(self, features: List[str]) -> Dict[str, float]:
        defaults = {}
        for f in features:
            if f in self.df.columns:
                col = pd.to_numeric(self.df[f], errors="coerce").dropna()
                defaults[f] = float(col.median()) if not col.empty else 0.0
            else:
                defaults[f] = 0.0
        return defaults
