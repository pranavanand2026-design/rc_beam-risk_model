"""
05_train_hybrid.py
~~~~~~~~~~~~~~~~~~
Trains the blended classifier (XGBoost + LDAM-DRW) referenced throughout the
RC_Beam.pdf design chapter:

* Aligns with Section 2.2.4 component table (`05_train_hybrid.py`) and satisfies
  FR-05/FR-06 by producing tuned model artefacts plus CSV/PNG evidence.
* Implements the imbalance mitigation strategy (Borderline-SMOTE + LDAM) listed
  under Risks R-01/R-03 and Quality of Work Section 5 (classification metrics).
* Saves outputs under `models/checkpoints` and `outputs/` so the report’s links
  remain live inside the repo.
"""

import argparse
from pathlib import Path

from rcbeam_fire.config import DEFAULT_CONFIG_PATH
from rcbeam_fire.models import train_hybrid_classifier
from rcbeam_fire.models.blend import cv_train_and_eval, run_ablation_grid


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
        "--cv",
        action="store_true",
        help="Run grouped CV with nested threshold tuning and write a fold leaderboard.",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Run objective/resampler ablation grid (writes ablation_results.csv).",
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

    if args.ablate:
        run_ablation_grid(
            config_path=Path(args.config),
            objective_modes=[args.objective_mode] if args.objective_mode else None,
            resamplers=[args.resampler] if args.resampler else None,
            focal_gamma=args.focal_gamma,
            name=args.name,
        )
        return

    if args.cv:
        leaderboard, summary, summary_json = cv_train_and_eval(
            config_path=Path(args.config),
            name=args.name,
            objective_mode=default_objective,
            resampler=default_resampler,
            focal_gamma=args.focal_gamma,
        )
        print(f"\nCV leaderboard → {leaderboard}")
        print("Summary:", summary)
        print(f"Summary JSON → {summary_json}")
        return

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
    print(f" Report CSV → {result.report_csv}")
    print(f" Confusion → {result.cm_png}")
    print(f" ROC curve  → {result.roc_png}")
    print(f" PR curve   → {result.pr_png}")
    print(f" Top-K plot → {result.topk_png}")
    print(f" Metrics    → {result.metrics_json}")
    if result.data_stats_json:
        print(f" Data stats → {result.data_stats_json}")
    if result.summary_json:
        print(f" Summary    → {result.summary_json}")
    if result.bootstrap_json:
        print(f" Bootstrap  → {result.bootstrap_json}")
    print(f" Model pack → {result.model_path}")


if __name__ == "__main__":
    main()
