"""
Random Forest Bioactivity Trainer
===================================
Trains and saves the best-performing model from the benchmark.
Run:
    python src/models/train_rf.py

Outputs:
    models/rf/rf_model.joblib
    models/rf/rf_scaler.joblib
    models/rf/rf_config.json
    models/rf/metrics.json
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)

DATA_PATH  = Path("src/data/data/himalayan_antimicrobial_compounds.csv")
OUTPUT_DIR = Path("models/rf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTOR_COLS = [
    'molecular_weight', 'logp', 'tpsa', 'hbd',
    'hba', 'rotatable_bonds', 'aromatic_rings', 'lipinski_violations',
]


def main():
    print("=" * 60)
    print("🌲 RANDOM FOREST BIOACTIVITY TRAINER")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    available = [c for c in DESCRIPTOR_COLS if c in df.columns]
    X = df[available].fillna(0).values.astype(np.float32)
    y = df['antimicrobial_active'].values.astype(int)

    print(f"\n📊 Dataset: {len(df)} compounds | {y.sum()} active | {(y==0).sum()} inactive")
    print(f"   Features: {available}")

    # ── Stratified train/test split ───────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # ── Scale ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"\n📊 Split — Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"   Train balance: {y_train.sum()} active / {(y_train==0).sum()} inactive")

    # ── 5-fold CV first to get honest numbers ─────────────────
    print("\n⚙️  Running 5-fold stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv = RandomForestClassifier(
        n_estimators=300, class_weight='balanced', random_state=42
    )
    X_s_full = scaler.fit_transform(X)   # refit on full for CV

    cv_f1  = cross_val_score(rf_cv, X_s_full, y, cv=skf, scoring='f1')
    cv_auc = cross_val_score(rf_cv, X_s_full, y, cv=skf, scoring='roc_auc')

    print(f"   CV F1:      {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(f"   CV ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # ── Train final model on full train set ───────────────────
    print("\n🌲 Training final Random Forest...")
    scaler = StandardScaler()   # refit scaler on train set only
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42,
        oob_score=True,
    )
    rf.fit(X_train_s, y_train)
    print(f"   OOB score: {rf.oob_score_:.3f}")

    # ── Evaluate on test set ──────────────────────────────────
    probs = rf.predict_proba(X_test_s)[:, 1]
    preds = rf.predict(X_test_s)

    metrics = {
        'accuracy':  float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall':    float(recall_score(y_test, preds, zero_division=0)),
        'f1':        float(f1_score(y_test, preds, zero_division=0)),
        'roc_auc':   float(roc_auc_score(y_test, probs)),
        'cv_f1_mean':       float(cv_f1.mean()),
        'cv_f1_std':        float(cv_f1.std()),
        'cv_roc_auc_mean':  float(cv_auc.mean()),
        'cv_roc_auc_std':   float(cv_auc.std()),
        'oob_score':        float(rf.oob_score_),
    }

    print(f"\n✅ Test Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\n{classification_report(y_test, preds, target_names=['Inactive','Active'])}")

    # ── Save everything ───────────────────────────────────────
    joblib.dump(rf,     OUTPUT_DIR / "rf_model.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "rf_scaler.joblib")

    config = {
        'model_type':     'RandomForestClassifier',
        'n_estimators':   300,
        'class_weight':   'balanced',
        'features':       available,
        'n_features':     len(available),
        'training_date':  datetime.now().isoformat(),
        'train_size':     len(y_train),
        'test_size':      len(y_test),
    }

    with open(OUTPUT_DIR / "rf_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    with open(OUTPUT_DIR / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Saved to models/rf/")
    print(f"   rf_model.joblib")
    print(f"   rf_scaler.joblib")
    print(f"   rf_config.json")
    print(f"   metrics.json")

    print("\n" + "=" * 60)
    print("✅ RF TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   CV F1:      {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(f"   CV ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    print(f"   Test F1:    {metrics['f1']:.3f}")
    print(f"   OOB:        {metrics['oob_score']:.3f}")
    print(f"\nNext: restart the API — RF will load automatically")


if __name__ == "__main__":
    main()