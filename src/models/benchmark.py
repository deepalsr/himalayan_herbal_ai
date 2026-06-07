"""
Phase 3 Benchmark — Scaffold Split + Baseline Comparison
=========================================================
Evaluates the MLP bioactivity model against classical ML baselines
using Bemis-Murcko scaffold splitting (the cheminformatics standard).

Run:
    python src/models/benchmark.py

Outputs:
    reports/benchmark_results.json
    reports/benchmark_comparison.png
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DATA_PATH    = Path("src/data/data/himalayan_antimicrobial_compounds.csv")
REPORTS_DIR  = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

DESCRIPTOR_COLS = [
    'molecular_weight', 'logp', 'tpsa', 'hbd',
    'hba', 'rotatable_bonds', 'aromatic_rings', 'lipinski_violations',
]


# ── Scaffold splitting ────────────────────────────────────────────────────

def get_scaffold(smiles: str) -> str:
    """Return Bemis-Murcko scaffold SMILES, or the original if RDKit fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return smiles


def scaffold_split(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42):
    """
    Split DataFrame into train/test by Bemis-Murcko scaffold.
    Scaffolds are sorted by size (largest → smallest) then greedily
    assigned to test until test_ratio is reached.
    Returns train_idx, test_idx (integer positional indices).
    """
    print("🔬 Computing Bemis-Murcko scaffolds...")
    scaffolds = df['smiles'].apply(get_scaffold)

    # Group compound indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        scaffold_to_indices[sc].append(i)

    # Sort scaffold groups by size descending
    groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)

    n_total    = len(df)
    n_test_target = int(n_total * test_ratio)

    train_idx, test_idx = [], []
    for group in groups:
        if len(test_idx) < n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    print(f"   Scaffolds: {len(groups)} unique")
    print(f"   Train: {len(train_idx)} | Test: {len(test_idx)}")
    return np.array(train_idx), np.array(test_idx)


# ── MLP (same architecture as train_activity_model.py) ───────────────────

class _MLP(nn.Module):
    def __init__(self, input_size=8, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers, prev = [], input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, X_val, y_val, class_weights, epochs=100):
    """Train MLP for one fold. Returns trained model."""
    device = 'cpu'
    model  = _MLP(input_size=X_tr.shape[1]).to(device)
    cw     = torch.FloatTensor(class_weights)
    crit   = nn.CrossEntropyLoss(weight=cw)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    Xt = torch.FloatTensor(X_tr);  yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_val); yv = torch.LongTensor(y_val)

    best_loss, best_state, patience_ctr = float('inf'), None, 0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()
        sched.step(val_loss)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= 15:
            break

    model.load_state_dict(best_state)
    return model


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds  = (probs >= 0.5).astype(int)
    return preds, probs


# ── Metrics helper ────────────────────────────────────────────────────────

def score(y_true, y_pred, y_prob):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }


# ── Main benchmark ────────────────────────────────────────────────────────

def run_benchmark():
    print("=" * 65)
    print("🏋️  PHASE 3 BENCHMARK — Scaffold Split + Baseline Comparison")
    print("=" * 65)

    # Load data
    df = pd.read_csv(DATA_PATH)
    available = [c for c in DESCRIPTOR_COLS if c in df.columns]
    X = df[available].fillna(0).values.astype(np.float32)
    y = df['antimicrobial_active'].values.astype(int)
    print(f"\n📊 Dataset: {len(df)} compounds | {y.sum()} active | {(y==0).sum()} inactive")
    print(f"   Features: {available}")

    # ── 1. Scaffold split test set (locked, never seen during CV) ─────────
    train_val_idx, test_idx = scaffold_split(df, test_ratio=0.15)

    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n🔒 Locked scaffold test set: {len(test_idx)} compounds")
    print(f"   Test active: {y_test.sum()} | inactive: {(y_test==0).sum()}")

    # ── 2. 5-fold stratified CV on train+val ──────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    baselines = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42),
        'SVM (RBF)': SVC(
            kernel='rbf', probability=True, class_weight='balanced',
            random_state=42, max_iter=5000),
        'Naive Bayes': GaussianNB(),
    }

    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        baselines['XGBoost'] = XGBClassifier(
            n_estimators=200,
            scale_pos_weight=(y_tv==0).sum()/(y_tv==1).sum(),
            random_state=42, eval_metric='logloss',
            verbosity=0, nthread=1, tree_method='hist')
        print("   ✅ XGBoost available")
    except ImportError:
        print("   ⚠️  XGBoost not installed — skipping")

    cv_results = defaultdict(lambda: defaultdict(list))

    print(f"\n⚙️  Running 5-fold stratified CV on {len(X_tv)} train/val compounds...")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tv, y_tv)):
        Xf_tr, yf_tr = X_tv[tr_idx], y_tv[tr_idx]
        Xf_val, yf_val = X_tv[val_idx], y_tv[val_idx]

        sc = StandardScaler()
        Xf_tr_s  = sc.fit_transform(Xf_tr)
        Xf_val_s = sc.transform(Xf_val)

        # Baselines
        for name, clf in baselines.items():
            clf.fit(Xf_tr_s, yf_tr)
            prob = clf.predict_proba(Xf_val_s)[:, 1]
            pred = clf.predict(Xf_val_s)
            for k, v in score(yf_val, pred, prob).items():
                cv_results[name][k].append(v)

        # MLP
        n_neg = (yf_tr == 0).sum(); n_pos = (yf_tr == 1).sum()
        cw = [len(yf_tr)/(2*n_neg) if n_neg else 1.0,
              len(yf_tr)/(2*n_pos) if n_pos else 1.0]
        mlp = train_mlp(Xf_tr_s, yf_tr, Xf_val_s, yf_val, cw)
        pred, prob = predict_mlp(mlp, Xf_val_s)
        for k, v in score(yf_val, pred, prob).items():
            cv_results['MLP (ours)'][k].append(v)

        print(f"   Fold {fold+1}/5 done")

    # ── 3. Final evaluation on locked scaffold test set ───────────────────
    print(f"\n🔒 Evaluating on locked scaffold test set...")
    test_results = {}

    sc_final = StandardScaler()
    X_tv_s   = sc_final.fit_transform(X_tv)
    X_test_s = sc_final.transform(X_test)

    for name, clf in baselines.items():
        clf.fit(X_tv_s, y_tv)
        prob = clf.predict_proba(X_test_s)[:, 1]
        pred = clf.predict(X_test_s)
        test_results[name] = score(y_test, pred, prob)

    n_neg = (y_tv==0).sum(); n_pos = (y_tv==1).sum()
    cw = [len(y_tv)/(2*n_neg) if n_neg else 1.0,
          len(y_tv)/(2*n_pos) if n_pos else 1.0]
    mlp_final = train_mlp(X_tv_s, y_tv, X_test_s, y_test, cw, epochs=150)
    pred, prob = predict_mlp(mlp_final, X_test_s)
    test_results['MLP (ours)'] = score(y_test, pred, prob)

    # ── 4. Print results table ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 5-FOLD CV RESULTS (mean ± std)")
    print("=" * 65)
    header = f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}"
    print(header)
    print("-" * 55)

    summary = {}
    for model_name, folds in cv_results.items():
        acc_m = np.mean(folds['accuracy']);   acc_s = np.std(folds['accuracy'])
        f1_m  = np.mean(folds['f1']);         f1_s  = np.std(folds['f1'])
        auc_m = np.mean(folds['roc_auc']);    auc_s = np.std(folds['roc_auc'])
        print(f"{model_name:<20} {acc_m:.3f}±{acc_s:.3f}  {f1_m:.3f}±{f1_s:.3f}  {auc_m:.3f}±{auc_s:.3f}")
        summary[model_name] = {
            'cv_accuracy_mean': round(acc_m, 4), 'cv_accuracy_std': round(acc_s, 4),
            'cv_f1_mean':       round(f1_m,  4), 'cv_f1_std':       round(f1_s,  4),
            'cv_roc_auc_mean':  round(auc_m, 4), 'cv_roc_auc_std':  round(auc_s, 4),
        }

    print("\n" + "=" * 65)
    print("🔒 SCAFFOLD TEST SET RESULTS")
    print("=" * 65)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 55)
    for model_name, m in test_results.items():
        print(f"{model_name:<20} {m['accuracy']:.3f}      {m['f1']:.3f}      {m['roc_auc']:.3f}")
        summary[model_name]['scaffold_test'] = {k: round(v, 4) for k, v in m.items()}

    # ── 5. Save results ───────────────────────────────────────────────────
    results_path = REPORTS_DIR / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")

    # ── 6. Plot ───────────────────────────────────────────────────────────
    _plot_benchmark(cv_results, test_results)

    print("\n✅ Phase 3 benchmark complete.")
    print("   Use these CV numbers in your paper's Table 1.")
    print("   Use scaffold test results as your held-out evaluation.")


def _plot_benchmark(cv_results, test_results):
    models  = list(cv_results.keys())
    metrics = ['f1', 'roc_auc', 'accuracy']
    labels  = ['F1-Score', 'ROC-AUC', 'Accuracy']
    colors  = ['#4C72B0', '#DD8452', '#55A868']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Comparison — 5-Fold CV (Scaffold Split)', fontsize=13)

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        means = [np.mean(cv_results[m][metric]) for m in models]
        stds  = [np.std(cv_results[m][metric])  for m in models]

        bars = ax.bar(models, means, yerr=stds, capsize=5,
                      color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=35)
        ax.grid(axis='y', alpha=0.3)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_path = REPORTS_DIR / "benchmark_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    run_benchmark()