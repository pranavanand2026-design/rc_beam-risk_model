import os, json, joblib, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dataset.csv"
CLF = ROOT / "models" / "checkpoints" / "suite_lgbm_smote.joblib"
FRT = ROOT / "models" / "checkpoints" / "frt_regressor.joblib"
CASE_DIR = ROOT / "outputs" / "tables"

print("=== Paths ===")
print("DATA:", DATA.exists(), DATA)
print("CLF :", CLF.exists(), CLF)
print("FRT :", FRT.exists(), FRT)
print("CASE_DIR:", CASE_DIR.exists(), CASE_DIR)

# dataset
df = pd.read_csv(DATA)
print("\n=== Dataset head (5) ===")
print(df.head(5).to_string())
print("\nColumns:", list(df.columns))
print("\nDtypes:", df.dtypes.astype(str).to_dict())

# classifier pack
clf_pack = joblib.load(CLF)
print("\n=== Classifier pack keys ===", list(clf_pack.keys()))
clf_feats = clf_pack.get("features", [])
clf_classes = clf_pack.get("classes", [])
print("Classifier features (len):", len(clf_feats))
print("First 10:", clf_feats[:10])
print("Classes:", clf_classes)

# regressor pack
frt_pack = joblib.load(FRT)
print("\n=== Regressor pack keys ===", list(frt_pack.keys()))
frt_feats = frt_pack.get("features", [])
print("Regressor features (len):", len(frt_feats))
print("First 10:", frt_feats[:10])

# drift check
clf_only = [f for f in clf_feats if f not in frt_feats]
frt_only = [f for f in frt_feats if f not in clf_feats]
print("\n=== Feature drift ===")
print("In classifier only:", clf_only)
print("In regressor only :", frt_only)

# case json sample if exists
samples = [p for p in CASE_DIR.glob("case_*.json")]
if samples:
    with open(samples[0]) as f:
        js = json.load(f)
    print("\n=== Case JSON sample ===")
    print("Keys:", list(js.keys()))
    print("Scenario keys:", list(js.get("scenario", {}).keys()))
    print("Has recs:", "recommendations" in js, f"count={len(js.get('recommendations', []))}")
else:
    print("\nNo case_*.json files found in outputs/tables/")