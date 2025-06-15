from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import load_config


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _load_raw(cfg: Dict[str, Any]) -> pd.DataFrame:
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    raw_path = Path(paths["raw"]) / data_cfg["file"]
    sheet = data_cfg.get("sheet", "Sheet1")
    df = pd.read_excel(raw_path, sheet_name=sheet)
    if len(df) > 0:
        df = df.drop(index=0).reset_index(drop=True)
    return df


def scan_dataset(config_path: str | os.PathLike | None = None) -> None:
    cfg = load_config(config_path)
    df = _load_raw(cfg)

    paths = cfg["paths"]
    out_tabs = Path(paths["outputs"]) / "tables"
    out_figs = Path(paths["outputs"]) / "figs"
    out_tabs.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print("Shape:", df.shape)
    with open(out_tabs / "info.txt", "w") as f:
        df.info(buf=f)
    df.head(10).to_csv(out_tabs / "preview.csv", index=False)

    mv = df.isna().sum().sort_values(ascending=False)
    mv.to_csv(out_tabs / "missing_values.csv")

    dtypes = df.dtypes.astype(str).value_counts()
    dtypes.to_csv(out_tabs / "column_types.csv")

    target_col = cfg["data"].get("target_col")
    if target_col and target_col in df.columns:
        vc = df[target_col].astype(str).value_counts(dropna=False)
        vc.to_csv(out_tabs / "label_counts.csv")
        plt.figure()
        sns.countplot(x=df[target_col].astype(str))
        plt.xticks(rotation=25)
        plt.title("Target class distribution")
        _savefig(out_figs / "target_balance.png")
    else:
        print(f"⚠ Target column '{target_col}' not found. Check config.yaml.")

    num_df = df.copy()
    for col in num_df.columns:
        if col == target_col:
            continue
        num_df[col] = pd.to_numeric(num_df[col], errors="coerce")
    num_only = num_df.select_dtypes(include=["number"])

    if num_only.empty:
        print("No numeric columns detected after coercion; please verify dataset.")
        return

    num_only.describe().T.to_csv(out_tabs / "numeric_summary.csv")
    corr = num_only.corr(numeric_only=True)
    corr.to_csv(out_tabs / "corr_matrix.csv")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (numeric)")
    _savefig(out_figs / "corr_heatmap.png")

    hist_cols = ["tins", "Ld", "L", "Af", "As", "LR"]
    for col in hist_cols:
        if col in num_only.columns:
            plt.figure()
            sns.histplot(num_only[col].dropna(), bins=40)
            plt.title(f"{col} distribution")
            _savefig(out_figs / f"hist_{col}.png")

    if target_col and target_col in df.columns:
        for col in ["tins", "LR", "Ld"]:
            if col in num_only.columns:
                plt.figure()
                sns.boxplot(x=df[target_col].astype(str), y=num_only[col])
                plt.title(f"{col} vs {target_col}")
                plt.xticks(rotation=25)
                _savefig(out_figs / f"{col}_vs_target.png")

    print(f"✓ Scan complete.\nTables → {out_tabs}\nFigs → {out_figs}")


__all__ = ["scan_dataset"]

