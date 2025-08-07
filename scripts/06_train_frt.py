"""
06_train_frt.py
~~~~~~~~~~~~~~~
Produces the regression leg that estimates fire-resistance time (FRT), mapping
to Section 2.2.4 (`06_train_frt.py`) and FR-03/FR-04 requirements in the report.
Key expectations captured here:

* Generate MAE/RMSE/R² numbers cited in Section 5.6 of RC_Beam.pdf.
* Persist the trained regressor for downstream interpretability (US-02) and case
  analysis (Section 5.9 use cases).
* Enforce configuration-driven targets so experimentation remains reproducible
  per the “single open dataset” client request (Section 2.3.1).
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rcbeam_fire.config import DEFAULT_CONFIG_PATH
from rcbeam_fire.models import train_frt_regressor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fire-resistance time regressor.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--target", default="FR", help="Target column name for fire resistance time.")
    args = parser.parse_args()

    result = train_frt_regressor(config_path=Path(args.config), target_col=args.target)

    print("\n=== FRT REGRESSOR RESULTS ===")
    print(f" Model: {result.model_name}")
    print(f" Train metrics: {result.metrics_train}")
    print(f" Valid metrics: {result.metrics_valid}")
    print(f" Model pack → {result.model_path}")
    print(f" Metrics     → {result.metrics_path}")


if __name__ == "__main__":
    main()
