from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.model_adapters import ClassifierAdapter, RegressorAdapter

# ---------------------------------------------------------------------------
# Domain metadata
# ---------------------------------------------------------------------------

FEATURE_LABELS: Dict[str, str] = {
    "Cc": "Concrete cover",
    "tins": "Soffit insulation thickness",
    "hi": "Side insulation depth",
    "As": "Steel reinforcement area",
    "Af": "FRP area",
    "LR": "Load ratio",
    "Ld": "Applied load",
    "kins": "Insulation thermal conductivity",
}

FEATURE_UNITS: Dict[str, str] = {
    "Cc": "mm",
    "tins": "mm",
    "hi": "mm",
    "As": "mm\u00b2",
    "Af": "mm\u00b2",
    "LR": "%",
    "Ld": "kN",
    "kins": "W/mK",
}

# +1 -> increasing the feature is typically beneficial for fire performance,
# -1 -> decreasing is beneficial.
DOMAIN_TENDENCY: Dict[str, int] = {
    "Cc": +1,
    "tins": +1,
    "hi": +1,
    "As": +1,
    "Af": +1,
    "LR": -1,
    "Ld": -1,
    "kins": -1,
}

SMALL_EPS = 1e-6
