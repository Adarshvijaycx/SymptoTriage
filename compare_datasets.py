"""TEMP analysis: honest head-to-head of old vs new dataset using the
project's actual Stage-2 voting ensemble (DT+RF+LGBM, weights [1,3,5]).

Identical model config on both datasets => the only variable is the data.
Metrics: out-of-fold (3-fold stratified CV) macro-F1, accuracy, and ECE,
plus a single-split pre/post Platt calibration check using the project's
own compute_ece. The Stage-1 router is intentionally excluded: the new
dataset has no category map, so routing cannot run on it (a migration finding).
"""

import sys, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.stage2_ensemble import Stage2Ensemble
from src.models.calibrator import compute_ece
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

RANDOM_STATE = 42
N_SPLITS = 3

# Identical, moderate config for BOTH datasets (kept light so this runs quickly;
# fairness comes from it being the SAME for both, not from matching prod values).
DT_KW  = {"max_depth": 10, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 42}
RF_KW  = {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 2,
          "class_weight": "balanced", "n_jobs": -1, "random_state": 42}
LGB_KW = {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 31,
          "min_child_samples": 3, "reg_alpha": 0.1, "reg_lambda": 0.5,
          "class_weight": "balanced", "verbose": -1, "random_state": 42}
WEIGHTS = [1, 3, 5]


def build_old():
    """Old dataset -> binary one-hot matrix (present=1, else=0)."""
    df = pd.read_csv("dataset.csv")
    sym_cols = [c for c in df.columns if c != "Disease"]
    tokens = set()
    for c in sym_cols:
        for v in df[c].dropna():
            t = str(v).strip()
            if t:
                tokens.add(t)
    toks = sorted(tokens)
    idx = {t: i for i, t in enumerate(toks)}
    X = np.zeros((len(df), len(toks)), dtype=np.int8)
    for c in sym_cols:
        for ri, v in enumerate(df[c]):
            if pd.notna(v):
                t = str(v).strip()
                if t in idx:
                    X[ri, idx[t]] = 1
    y = pd.factorize(df["Disease"].str.strip())[0]
    return X, y, len(toks)


def build_new():
    """New dataset is already a binary matrix; first col is the label."""
    df = pd.read_csv("Diseases_and_Symptoms_dataset.csv")
    label = df.columns[0]
    X = df.drop(columns=[label]).to_numpy(dtype=np.int8)
    y = pd.factorize(df[label].str.strip())[0]
    return X, y, X.shape[1]


def fit_ensemble(Xtr, ytr):
    ens = Stage2Ensemble(dt_kwargs=DT_KW, rf_kwargs=RF_KW, lgb_kwargs=LGB_KW, weights=WEIGHTS)
    cats = np.zeros(len(ytr), dtype=int)   # single category => one flat voting ensemble
    ens.fit(Xtr, ytr, cats)
    return ens.ensembles_[0]               # the underlying VotingClassifier


def evaluate(name, X, y, n_feat):
    n_classes = len(np.unique(y))
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    print(f"rows={len(y):,}  features={n_feat}  classes={n_classes}", flush=True)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_conf = np.zeros(len(y), dtype=float)   # top-class probability (for ECE)
    t0 = time.time()
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        vc = fit_ensemble(X[tr], y[tr])
        proba = vc.predict_proba(X[te])
        classes = vc.classes_
        pred = classes[np.argmax(proba, axis=1)]
        oof_pred[te] = pred
        oof_conf[te] = proba.max(axis=1)
        print(f"  fold {k}/{N_SPLITS} done  (cum {time.time()-t0:.0f}s)  "
              f"fold_acc={accuracy_score(y[te], pred):.4f}", flush=True)

    macro_f1 = f1_score(y, oof_pred, average="macro")
    weighted_f1 = f1_score(y, oof_pred, average="weighted")
    acc = accuracy_score(y, oof_pred)

    # ECE on pooled OOF: build a 2-col [1-conf, conf] proxy vs correctness,
    # which is exactly what the project's compute_ece consumes for top-label ECE.
    correct = (oof_pred == y).astype(int)
    proba2 = np.column_stack([1 - oof_conf, oof_conf])
    ece = compute_ece(correct, proba2, n_bins=10)

    print(f"\n  OOF macro-F1     : {macro_f1:.4f}")
    print(f"  OOF weighted-F1  : {weighted_f1:.4f}")
    print(f"  OOF accuracy     : {acc:.4f}")
    print(f"  OOF top-label ECE: {ece:.4f}  (uncalibrated, lower=better)")
    print(f"  mean top conf    : {oof_conf.mean():.4f}", flush=True)

    # Single-split Platt calibration check (pre vs post)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30,
                                             random_state=RANDOM_STATE, stratify=y)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.50,
                                             random_state=RANDOM_STATE, stratify=ytmp)
    vc = fit_ensemble(Xtr, ytr)
    raw = vc.predict_proba(Xte)
    raw_pred = vc.classes_[np.argmax(raw, axis=1)]
    raw_ece = compute_ece((raw_pred == yte).astype(int),
                          np.column_stack([1 - raw.max(1), raw.max(1)]), 10)
    if FrozenEstimator is not None:
        cal = CalibratedClassifierCV(FrozenEstimator(vc), method="sigmoid")
    else:
        cal = CalibratedClassifierCV(vc, method="sigmoid", cv="prefit")
    cal.fit(Xcal, ycal)
    cp = cal.predict_proba(Xte)
    cp_pred = cal.classes_[np.argmax(cp, axis=1)]
    cal_ece = compute_ece((cp_pred == yte).astype(int),
                          np.column_stack([1 - cp.max(1), cp.max(1)]), 10)
    print(f"  Platt ECE pre/post: {raw_ece:.4f} -> {cal_ece:.4f}", flush=True)

    return {"name": name, "rows": len(y), "classes": n_classes,
            "macro_f1": macro_f1, "weighted_f1": weighted_f1, "acc": acc,
            "ece": ece, "mean_conf": float(oof_conf.mean()),
            "cal_ece_pre": raw_ece, "cal_ece_post": cal_ece}


if __name__ == "__main__":
    results = []
    Xo, yo, fo = build_old()
    results.append(evaluate("OLD  dataset.csv", Xo, yo, fo))
    Xn, yn, fn = build_new()
    results.append(evaluate("NEW  Diseases_and_Symptoms_dataset.csv", Xn, yn, fn))

    print(f"\n\n{'#'*64}\nSUMMARY\n{'#'*64}")
    hdr = f"{'dataset':<40}{'rows':>8}{'cls':>5}{'macroF1':>9}{'acc':>8}{'ECE':>8}{'calECE':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<40}{r['rows']:>8,}{r['classes']:>5}"
              f"{r['macro_f1']:>9.4f}{r['acc']:>8.4f}{r['ece']:>8.4f}{r['cal_ece_post']:>8.4f}")
    print("\nDONE", flush=True)
