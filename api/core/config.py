from __future__ import annotations

import os
from typing import List


def get_cors_origins() -> List[str]:
    origins_env = os.getenv("CORS_ORIGINS", "")
    if origins_env:
        return [o.strip() for o in origins_env.split(",") if o.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


CONFIG_PATH = os.getenv("RCBEAM_CONFIG", "config.yaml")
