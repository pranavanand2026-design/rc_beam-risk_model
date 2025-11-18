#!/usr/bin/env python3
"""
run_pipeline.py
~~~~~~~~~~~~~~~
Convenience orchestrator that executes the staged workflow captured in Section
2.2.4 (Execution flow bullet list) of RC_Beam.pdf. Mirrors the “modular,
configuration-driven pipeline” narrative by chaining:

01_scan   → raw data QA evidence (FR-01/FR-03)
02_preprocess → deterministic cleaning (NFR-01)
05_train_hybrid → classification deliverable (FR-05/FR-06)
06_train_frt → regression deliverable (FR-03)
12_case_analyzer → interpretability artefacts (US-02/US-03)

This file keeps the published command sequence in sync with reality so the
report’s reproduction steps stay accurate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from rcbeam_fire.config import load_config


def _run(step: Sequence[str]) -> None:
    print(f"\nRunning: {' '.join(step)}")
    subprocess.run(step, check=True)


def _ensure_dataset(config_path: str) -> Path:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw"])
    dataset = raw_dir / cfg["data"]["file"]
    if not dataset.exists():
        raise FileNotFoundError(
            f"Expected dataset at '{dataset}'. Place the Excel file under data/raw/ and rerun."
        )
    return dataset


def build_steps(config_path: str, case_idx: int | None) -> Iterable[list[str]]:
    exe = [sys.executable]
    steps = [
        exe + ["scripts/01_scan.py", "--config", config_path],
        exe + ["scripts/02_preprocess.py", "--config", config_path],
        exe + ["scripts/05_train_hybrid.py", "--config", config_path],
        exe + ["scripts/06_train_frt.py", "--config", config_path],
    ]
    if case_idx is not None:
        steps.append(
            exe
            + [
                "scripts/12_case_analyzer.py",
                "--config",
                config_path,
                "--idx",
                str(case_idx),
            ]
        )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RC beam pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument(
        "--case-idx",
        type=int,
        default=0,
        help="Dataset row index for case analysis; -1 skips the analyzer.",
    )
    args = parser.parse_args()

    _ensure_dataset(args.config)
    case_idx = None if args.case_idx < 0 else args.case_idx
    for step in build_steps(args.config, case_idx):
        _run(step)

    print("\nPipeline complete. See outputs/ and models/ for artefacts.")


if __name__ == "__main__":
    main()
