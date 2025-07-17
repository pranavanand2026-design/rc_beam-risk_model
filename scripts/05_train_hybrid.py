"""
05_train_hybrid.py
~~~~~~~~~~~~~~~~~~
Trains the blended classifier (XGBoost + LDAM-DRW).
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rcbeam_fire.config import DEFAULT_CONFIG_PATH
from rcbeam_fire.models import train_hybrid_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hybrid XGB + LDAM classifier.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--name",
        default="blend_xgb_ldam_nophys",
        help="Model name used for saved artifacts.",
    )
    parser.add_argument(
        "--objective_mode",
        choices=["ldam_drw", "logit_adjusted", "focal"],
        default=None,
        help="Objective variant for the LDAM leg (default: ldam_drw).",
    )
    parser.add_argument(
        "--resampler",
        choices=["targeted", "borderline", "smoteenn"],
        default=None,
        help="Resampling strategy for training folds (default: targeted).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter when using focal objective (default: 2.0).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap resamples for CI estimation (0 disables).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed used for bootstrap resampling.",
    )
    args = parser.parse_args()

    default_objective = args.objective_mode or "ldam_drw"
    default_resampler = args.resampler or "targeted"

    result = train_hybrid_classifier(
        config_path=Path(args.config),
        name=args.name,
        objective_mode=default_objective,
        resampler=default_resampler,
        focal_gamma=args.focal_gamma,
        bootstrap_samples=args.bootstrap,
        bootstrap_random_state=args.bootstrap_seed,
    )

    print("\n=== HYBRID CLASSIFIER RESULTS ===")
    print(f" Macro-F1: {result.macro_f1:.3f}")
    print(f" Balanced Accuracy: {result.bacc:.3f}")
    print(f" Accuracy: {result.acc:.3f}")
    print(f" Model pack → {result.model_path}")


if __name__ == "__main__":
    main()
