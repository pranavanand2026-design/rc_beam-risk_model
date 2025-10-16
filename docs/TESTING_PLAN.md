# Testing Expansion Plan

The specification in `RC_Beam.pdf` (Section 5 – Quality of Work) calls for richer
Normal/Boundary/Abnormal coverage than we currently exercise inside the repo. To
close the gap we will tackle the following test tracks:

## 1. Data & Schema Safety (Pre-Model)
- **Contract tests (`pytest tests/test_schema.py`)** to assert that every raw
  workbook column matches the expectations codified in `config.yaml`.
- **Boundary fixtures** for cover thickness (Cc), insulation (tins), and load
  ratio (LR) to honour Use Cases 3–5.
- **Abnormal fixtures** verifying we reject negative dimensions or missing
  material properties instead of silently coercing to NaN.

## 2. Model Training Regression Suite
- Archive the current metrics CSV/JSON as baselines; add a test that runs the
  trainers on a 200-row stratified sample and checks that macro-F1 / R² stay
  within ±5% of the recorded values.
- Parameterised tests for alternative seeds to prove the imbalance mitigation
  strategy remains stable (ties back to Risk R-01).

## 3. Case Analyzer Acceptance Tests
- Scripted CLI tests that replay the three representative scenarios from
  Section 5.9 (Normal, Boundary, Abnormal) and diff the generated
  `outputs/tables/case_*.json` against checked-in golden files.
- Add JSON schema validation for the recommendations payload so clients can rely
  on the format when embedding it in reports.

## 4. Dashboard / Streamlit Smoke Tests
- Lightweight `pytest` hook that boots the Streamlit app in headless mode to
  ensure layout + callbacks load (satisfies US-03 export readiness).

## 5. Automation & Reporting
- Integrate the above suites into `scripts/run_pipeline.py --mode test`, making
  it easy to collect evidence for the "Testing & Results" chapter during future
  sprints.

These work items should be broken into backlog issues so that each future commit
links code changes to corresponding evidence in the report.
