from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd

from rcbeam_fire.analysis.eurocode import evaluate_compliance, pick_requirement
from rcbeam_fire.analysis.insight import (
    DOMAIN_TENDENCY,
    ActionPlan,
    build_case_action_plan,
)

from .dataset_service import DatasetService
from .model_service import ModelService


class AnalysisService:
    def __init__(
        self,
        model_service: ModelService,
        dataset_service: DatasetService,
        bounds: Dict,
        adjustable_features: List[str],
    ):
        self.model = model_service
        self.dataset = dataset_service
        self.bounds = bounds
        self.adjustable_features = adjustable_features

    def full_analysis(
        self,
        features: Dict[str, float],
        exposure_minutes: float,
        margin_minutes: float,
    ) -> Dict:
        predicted_mode, probabilities, frt_minutes = self.model.predict(features)
        proba_array = self.model.predict_proba_array(features)

        threshold = exposure_minutes + margin_minutes
        gap = frt_minutes - threshold
        verdict = (
            f"Meets scenario (+{gap:.1f} min margin)"
            if gap >= 0
            else f"At risk ({gap:.1f} min short)"
        )

        row = pd.Series(features)
        scenario = {
            "exposure": exposure_minutes,
            "margin": margin_minutes,
            "threshold": threshold,
            "gap_minutes": gap,
        }

        action_plan: ActionPlan = build_case_action_plan(
            df=self.dataset.df,
            row=row,
            clf=self.model.clf,
            frt=self.model.frt,
            scenario=scenario,
            base_proba=proba_array,
            pred_mode=predicted_mode,
            frt_pred=frt_minutes,
            top_mode=[],
            top_frt=[],
            adjustable_features=self.adjustable_features,
            domain_tendency=DOMAIN_TENDENCY,
            bounds=self.bounds,
        )

        # Eurocode compliance
        requirement = pick_requirement(threshold)
        ec_verdict, ec_criteria, ec_adjustments = evaluate_compliance(
            row, requirement
        )

        result = {
            "scenario": {
                "exposure": exposure_minutes,
                "margin": margin_minutes,
                "threshold": threshold,
                "frt_pred": frt_minutes,
                "gap_minutes": gap,
                "pred_mode": predicted_mode,
                "prob_no_failure": probabilities.get("No Failure", 0.0),
                "verdict": action_plan.scenario.verdict,
            },
            "recommendations": [asdict(r) for r in action_plan.recommendations],
            "combination": (
                asdict(action_plan.combination)
                if action_plan.combination
                else None
            ),
            "eurocode": {
                "verdict": ec_verdict,
                "criteria": ec_criteria,
                "adjustments": [asdict(a) for a in ec_adjustments],
            },
            "notes": action_plan.notes,
            "prediction": {
                "predicted_mode": predicted_mode,
                "probabilities": probabilities,
                "frt_minutes": frt_minutes,
                "threshold_minutes": threshold,
                "gap_minutes": gap,
                "verdict": verdict,
            },
        }
        return result
