from __future__ import annotations

from typing import Dict, List

SAFE_FEATURES = [
    "L",
    "Ac",
    "Cc",
    "As",
    "Af",
    "tins",
    "hi",
    "fc",
    "fy",
    "Es",
    "fu",
    "Efrp",
    "Tg",
    "kins",
    "rinscins",
    "Ld",
    "LR",
]

MONOTONE_FEATURE_SIGNS = {
    "Cc": 1,
    "tins": 1,
    "hi": 1,
    "LR": -1,
}

RESAMPLERS = {"targeted": "targeted", "borderline": "borderline", "smoteenn": "smoteenn"}
OBJECTIVE_MODES = {"ldam_drw", "logit_adjusted", "focal"}


def build_monotone_constraints(feature_names: List[str]) -> List[int]:
    return [MONOTONE_FEATURE_SIGNS.get(name, 0) for name in feature_names]
