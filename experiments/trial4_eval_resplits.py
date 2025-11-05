from __future__ import annotations
import os, argparse, warnings, json
import numpy as np, pandas as pd
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from imblearn.combine import SMOTETomek
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SAFE_FEATURES = ["L","Ac","Cc","As","Af","tins","hi","fc","fy","Es","fu","Efrp","Tg","kins","rinscins","Ld","LR"]

def load_cfg(p):
    import yaml
    with open(p,"r") as f: return yaml.safe_load(f)

def load_processed(paths):
    parq = os.path.join(paths["processed"], "dataset.parquet")
    csv_ = os.path.join(paths["processed"], "dataset.csv")
    return pd.read_parquet(parq) if os.path.exists(parq) else pd.read_csv(csv_)

def add_feats(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    eps=1e-9
    sdiv=lambda a,b: np.nan_to_num(a/np.where(np.abs(b)>eps,b,np.nan), nan=0.0)
    if {"As","Ac"}.issubset(X): X["rho_s_As_over_Ac"]=sdiv(X["As"],X["Ac"])
    if {"Af","Ac"}.issubset(X): X["rho_f_Af_over_Ac"]=sdiv(X["Af"],X["Ac"])
    if {"Af","As"}.issubset(X): X["Af_over_As"]=sdiv(X["Af"],X["As"])
    if {"Ld","L"}.issubset(X):  X["Ld_over_L"]=sdiv(X["Ld"],X["L"])
    if {"LR","fc"}.issubset(X): X["LR_times_fc"]=X["LR"]*X["fc"]
    if {"Efrp","Af","Es","As"}.issubset(X): X["EfrpAf_over_EsAs"]=sdiv(X["Efrp"]*X["Af"], X["Es"]*X["As"])
    if {"tins","kins","Tg"}.issubset(X): X["thermal_index"]=sdiv(X["tins"]*X["kins"], X["Tg"])
    return X

def prepare(df):
    X = df[[c for c in SAFE_FEATURES if c in df.columns]].copy()
    y_raw = df["target"].astype(str).str.strip().replace({"0":"No Failure","1":"Strength Failure","2":"Deflection Failure","3":"Other"})
    keep = y_raw.isin(["No Failure","Strength Failure","Deflection Failure"])
    X, y_raw = X[keep], y_raw[keep]
    X = add_feats(X)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    return X, y, classes

def tune_thresholds(P: np.ndarray, y_true: np.ndarray, classes):
    # only tune minorities
    grid = {c: np.linspace(0.22,0.55,8) for c in classes if c!="Deflection Failure"}
    keys=list(grid.keys()); best,bestf={},-1
    def rec(i,cur):
        nonlocal best,bestf
        if i==len(keys):
            th=cur.copy()
            pred = P.argmax(axis=1).copy()
            for ci,c in enumerate(classes):
                t=th.get(c,None)
                if t is not None:
                    pred[P[:,ci]>=t]=ci
            f = f1_score(y_true, pred, average="macro")
            if f>bestf: bestf,best=f,th.copy()
            return
        k=keys[i]
        for t in grid[k]:
            cur[k]=float(t); rec(i+1,cur)
    rec(0,{})
    return best

def run_cv(X, y, classes, n_splits=5, n_repeats=3, seed=42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    metrics=[]
    for fold,(tr,va) in enumerate(rskf.split(X,y),1):
        X_tr, y_tr = X.iloc[tr], y[tr]
        X_va, y_va = X.iloc[va], y[va]

        # targeted SMOTETomek
        _, counts = np.unique(y_tr, return_counts=True)
        maj = int(counts.max())
        # map class index names
        cname = dict(enumerate(classes))
        strat={}
        for i,c in cname.items():
            if c=="Strength Failure": strat[i]=maj
            if c=="No Failure": strat[i]=max(int(0.7*maj), counts[i])
        smt = SMOTETomek(random_state=seed, sampling_strategy=strat)
        Xb, yb = smt.fit_resample(X_tr, y_tr)

        # LGBM (same settings as winner-ish)
        model = LGBMClassifier(
            objective="multiclass", num_class=len(classes),
            n_estimators=800, learning_rate=0.06,
            min_child_samples=25, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.2, random_state=seed, n_jobs=-1
        )
        # inverse-frequency sample weights
        uniq,cnt=np.unique(yb,return_counts=True)
        inv={u:1.0/c for u,c in zip(uniq,cnt)}
        w=np.array([inv[i] for i in yb]); w=w/w.mean()
        model.fit(Xb,yb,sample_weight=w)

        P = model.predict_proba(X_va)
        th = tune_thresholds(P, y_va, classes)
        pred = P.argmax(axis=1)
        for ci,c in enumerate(classes):
            t = th.get(c,None)
            if t is not None:
                pred[P[:,ci]>=t]=ci

        metrics.append({
            "fold": fold,
            "macro_f1": f1_score(y_va,pred,average="macro"),
            "bacc": balanced_accuracy_score(y_va,pred),
            "acc": accuracy_score(y_va,pred)
        })
    return pd.DataFrame(metrics)

def main(cfg_path):
    cfg = load_cfg(cfg_path)
    df = load_processed(cfg["paths"])
    X,y,classes = prepare(df)

    out_dir = os.path.join(cfg["paths"]["outputs"], "cv")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    res = run_cv(X,y,classes, n_splits=5, n_repeats=3, seed=42)
    res.to_csv(os.path.join(out_dir,"cv_results.csv"), index=False)
    print(res.describe()[["macro_f1","bacc","acc"]])

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)