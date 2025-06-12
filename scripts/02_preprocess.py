"""
02_preprocess.py
~~~~~~~~~~~~~~~~
Implements the data-conditioning step described in Section 2.2.4 (Component
`02_preprocess.py`) of RC_Beam.pdf. Responsibilities:

* cleanse/standardise the Bhatt dataset ahead of modelling (FR-02, technical
  constraint on schema consistency);
* persist a Parquet asset referenced by later scripts + documentation (evidence
  for reproducibility and traceability requirements NFR-01, NFR-08);
* log automation output so the report can cite where the cleaned dataset lives
  in the repo.
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rcbeam_fire.config import DEFAULT_CONFIG_PATH
from rcbeam_fire.data import run_preprocess


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Stage 02 – apply deterministic preprocessing to satisfy FR-02 and "
            "produce the data/processed artefact cited in the system specification."
        )
    )
    ap.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    args = ap.parse_args()
    _, out_path = run_preprocess(Path(args.config))
    print(f"Preprocessing complete. File saved at: {out_path}")


if __name__ == "__main__":
    main()
