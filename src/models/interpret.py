"""
SHAP Interpretability Analysis
================================
Explains which molecular descriptors drive bioactivity predictions.
Uses the Random Forest model (best CV performer) for SHAP analysis.

Run:
    python src/models/interpret.py

Outputs:
    reports/figures/shap_summary.png
    reports/figures/shap_beeswarm.png
    reports/figures/shap_dependence_<top_feature>.png
    reports/shap_values.csv
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

DATA_PATH   = Path("src/data/data/himalayan_antimicrobial_compounds.csv")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTOR_COLS = [
    'molecular_weight', 'logp', 'tpsa', 'hbd',
    'hba', 'rotatable_bonds', 'aromatic_rings', 'lipinski_violations',
]

FEATURE_LABELS = {
    'molecular_weight':    'Molecular Weight (Da)',
    'logp':                'LogP (lipophilicity)',
    'tpsa':                'TPSA (Å²)',
    'hbd':                 'H-Bond Donors',
    'hba':                 'H-Bond Acceptors',
    'rotatable_bonds':     'Rotatable Bonds',
    'aromatic_rings':      'Aromatic Rings',
    'lipinski_violations': 'Lipinski Violations',
}


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    available = [c for c in DESCRIPTOR_COLS if c in df.columns]
    X = df[available].fillna(0).values.astype(np.float32)
    y = df['antimicrobial_active'].values.astype(int)
    names = df['compound_name'].tolist() if 'compound_name' in df.columns else [str(i) for i in range(len(df))]
    labels = [FEATURE_LABELS.get(c, c) for c in available]
    return X, y, labels, names, df, available


def train_rf(X, y):
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    rf  = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    rf.fit(X_s, y)
    return rf, sc, X_s


def plot_feature_importance(rf, feature_labels):
    """Bar chart of mean decrease in impurity (MDI) importance."""
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_labels = [feature_labels[i] for i in indices]
    sorted_imp    = importances[indices]

    colors = ['#2E86AB' if i == 0 else '#A8DADC' for i in range(len(sorted_imp))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(sorted_imp)), sorted_imp[::-1],
                   color=colors[::-1], edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(sorted_imp)))
    ax.set_yticklabels(sorted_labels[::-1], fontsize=11)
    ax.set_xlabel('Mean Decrease in Impurity', fontsize=11)
    ax.set_title('Feature Importance — Bioactivity Prediction\n(Random Forest, n=309 compounds)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, sorted_imp[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Feature importance → {path}")
    return indices[0]   # index of top feature


def run_shap_analysis(rf, X_s, feature_labels, df, available_cols):
    try:
        import shap
    except ImportError:
        print("⚠️  SHAP not installed. Run: pip install shap")
        return None

    print("🔍 Computing SHAP values (this takes ~30 seconds)...")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_s)

    # shap_values is list of 2 arrays [class0, class1] — we want class1 (active)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # ── Beeswarm summary plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        sv, X_s,
        feature_names=feature_labels,
        plot_type="dot",
        show=False,
        plot_size=None,
    )
    plt.title('SHAP Summary — Impact on Antimicrobial Activity Prediction',
              fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    path = FIGURES_DIR / "shap_beeswarm.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 SHAP beeswarm → {path}")

    # ── Bar summary plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(
        sv, X_s,
        feature_names=feature_labels,
        plot_type="bar",
        show=False,
        plot_size=None,
    )
    plt.title('Mean |SHAP Value| per Feature', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    path = FIGURES_DIR / "shap_bar.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 SHAP bar → {path}")

    # ── Dependence plot for top 2 features ───────────────────────────────
    top2 = np.argsort(np.abs(sv).mean(axis=0)).flatten()[-2:][::-1].tolist()
    for feat_idx in top2:
        feat_name = feature_labels[feat_idx]
        fig, ax = plt.subplots(figsize=(8, 5))
        sv2 = sv if sv.ndim == 2 else sv[:, :, 1]
        feat_vals = X_s[:, feat_idx]
        shap_vals = sv2[:, feat_idx]
        sc = ax.scatter(feat_vals, shap_vals, c=shap_vals,
                        cmap='coolwarm', alpha=0.7, edgecolors='none', s=40)
        plt.colorbar(sc, ax=ax, label='SHAP value')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel(feat_name, fontsize=11)
        ax.set_ylabel('SHAP value (impact on prediction)', fontsize=11)
        ax.set_title(f'SHAP Dependence — {feat_name}',
                     fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        safe_name = feat_name.replace(' ', '_').replace('(','').replace(')','').replace('²','2')
        path = FIGURES_DIR / f"shap_dependence_{safe_name}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 SHAP dependence ({feat_name}) → {path}")
    # ── Save raw SHAP values as CSV ───────────────────────────────────────
    sv2 = sv if sv.ndim == 2 else sv[:, :, 1]
    shap_df = pd.DataFrame(sv2, columns=feature_labels)
    shap_df.insert(0, 'compound_name', df['compound_name'].values if 'compound_name' in df.columns else range(len(df)))
    shap_df.insert(1, 'true_label',    df['antimicrobial_active'].values)
    shap_df.to_csv("reports/shap_values.csv", index=False)
    print("💾 Raw SHAP values → reports/shap_values.csv")

    # ── Print top insights ────────────────────────────────────────────────
    sv2 = sv if sv.ndim == 2 else sv[:, :, 1]
    mean_abs = np.abs(sv2).mean(axis=0)
    ranked   = sorted(zip(feature_labels, mean_abs), key=lambda x: x[1], reverse=True)
    print("\n📊 Feature impact ranking (mean |SHAP|):")
    for feat, val in ranked:
        bar = '█' * int(val * 100)
        print(f"   {feat:<30} {val:.4f}  {bar}")

    return sv


def main():
    print("=" * 60)
    print("🔍 SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    X, y, feature_labels, names, df, available = load_and_prepare()
    print(f"\n📊 {len(X)} compounds | {y.sum()} active | {(y==0).sum()} inactive")

    print("\n🌲 Training Random Forest for SHAP analysis...")
    rf, sc, X_s = train_rf(X, y)
    print(f"   OOB score: {rf.oob_score_:.3f}" if hasattr(rf, 'oob_score_') else "")

    # MDI feature importance (fast, no SHAP needed)
    top_feat_idx = plot_feature_importance(rf, feature_labels)
    print(f"\n🏆 Top feature by MDI: {feature_labels[top_feat_idx]}")

    # SHAP (slower but more accurate)
    sv = run_shap_analysis(rf, X_s, feature_labels, df, available)

    print("\n" + "=" * 60)
    print("✅ Interpretability analysis complete")
    print("=" * 60)
    print("\nFigures for your paper:")
    print("  Fig 2 → reports/figures/feature_importance.png")
    print("  Fig 3 → reports/figures/shap_beeswarm.png")
    print("  Fig 4 → reports/figures/shap_dependence_*.png")
    print("\nKey narrative:")
    print("  'SHAP analysis reveals which physicochemical properties")
    print("   drive antimicrobial activity predictions, providing")
    print("   mechanistic interpretability beyond classification metrics.'")


if __name__ == "__main__":
    main()