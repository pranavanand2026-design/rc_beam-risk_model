"""
01_scan.py
~~~~~~~~~~
Implements the first stage of the production pipeline described in *RC_BEAM – P26
system specification*. This script fulfils the data-inspection responsibilities
listed under Section 2.2.4 (Component `01_scan.py`) of the report by:

* validating that the Bhatt (2023) dataset can be opened (FR-01, FR-02);
* generating traceable summary tables/plots demanded by the client for reporting
  and visual evidence (FR-03, NFR-08);
* seeding later stages with consistent schema intelligence (risk R-01 mitigation).

It deliberately stays CLI-only so team members can reproduce the artefacts
documented in Figure A / Appendix outputs without touching notebooks.
"""

import argparse
from pathlib import Path

from rcbeam_fire.config import DEFAULT_CONFIG_PATH
from rcbeam_fire.data import scan_dataset


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Stage 01 – scan & QA the raw dataset, producing the CSV/PNG evidence "
            "referenced throughout RC_Beam.pdf (targets FR-03 visual artefacts)."
        )
    )
    ap.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    args = ap.parse_args()
    # Generate outputs/tables + outputs/figs so the documentation can cite
    # concrete artefacts (see Section 5 testing evidence).
    scan_dataset(Path(args.config))


if __name__ == "__main__":
    main()
