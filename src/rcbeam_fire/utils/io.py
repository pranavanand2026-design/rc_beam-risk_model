from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_processed_dataset(paths_cfg: Dict[str, str]) -> pd.DataFrame:
    processed_dir = Path(paths_cfg["processed"])
    parq = processed_dir / "dataset.parquet"
    csv_ = processed_dir / "dataset.csv"
    if parq.exists():
        return pd.read_parquet(parq)
    if csv_.exists():
        return pd.read_csv(csv_)
    raise FileNotFoundError(f"No processed dataset found in {processed_dir}")


__all__ = ["ensure_dirs", "load_processed_dataset"]

