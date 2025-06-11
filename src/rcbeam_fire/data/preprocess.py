from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ..config import load_config

NUMERIC_COLUMNS = [
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
    "F_to_EF",
    "FR",
    "df",
]


def _read_raw(cfg: Dict) -> pd.DataFrame:
    raw_dir = Path(cfg["paths"]["raw"])
    data_cfg = cfg["data"]
    path = raw_dir / data_cfg["file"]
    sheet = data_cfg.get("sheet", "Sheet1")
    df = pd.read_excel(path, sheet_name=sheet)
    if len(df) > 0:
        df = df.drop(index=0).reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {"F/EF": "F_to_EF"}
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    return df


def _coerce_numeric(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "BN" in df.columns:
        df["BN"] = df["BN"].astype(str)
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(str)
    return df


def _ensure_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    df = df.copy()
    df["target"] = df[target_col].astype(str)
    mask_missing = df["target"].isna() | (df["target"].str.lower().isin(["", "nan", "none"]))
    before = len(df)
    df = df[~mask_missing].copy()
    df.reset_index(drop=True, inplace=True)
    dropped = before - len(df)
    print(f"Dropped {dropped} unlabeled rows.")
    return df


def _save(df: pd.DataFrame, cfg: Dict, basename: str = "dataset") -> str:
    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = processed_dir / f"{basename}.parquet"
    try:
        df.to_parquet(out_parquet, index=False)
        print(f"Saved processed dataset → {out_parquet}  shape={df.shape}")
        return str(out_parquet)
    except Exception:
        out_csv = processed_dir / f"{basename}.csv"
        df.to_csv(out_csv, index=False)
        print(f"(Parquet unavailable) Saved CSV → {out_csv}  shape={df.shape}")
        return str(out_csv)


def run_preprocess(config_path: str | os.PathLike | None = None) -> Tuple[pd.DataFrame, str]:
    cfg = load_config(config_path)
    target_col = cfg["data"]["target_col"]
    df = _read_raw(cfg)
    df = _sanitize_columns(df)
    df = _coerce_numeric(df, target_col=target_col)
    df = _ensure_target(df, target_col=target_col)
    out_path = _save(df, cfg, basename="dataset")
    return df, out_path


__all__ = ["run_preprocess"]

