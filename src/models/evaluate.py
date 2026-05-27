"""
Unified Model Evaluation Utilities
===================================

Comprehensive evaluation functions for all model types:
- Descriptor-based (bioactivity, toxicity)
- Graph Neural Networks
- Comparative analysis
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import json
import pickle

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Unified evaluator for all model types
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Args:
            output_dir: Directory to save evaluation reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def evaluate_predictions(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_proba: np.ndarray = None,
                            task_name: str = "model") -> Dict:
        """
        Comprehensive evaluation on predictions

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class
            task_name: Name of the task (for file naming)

        Returns:
            Dictionary with all metrics
        """

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        }

        # Add ROC-AUC if probabilities available
        if y_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        else:
            metrics['roc_auc'] = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'] = float(cm[0, 0])
        metrics['fp'] = float(cm[0, 1])
        metrics['fn'] = float(cm[1, 0])
        metrics['tp'] = float(cm[1, 1])

        # Specificity and sensitivity
        metrics['sensitivity'] = float(metrics['recall'])  # Same as recall
        if (metrics['tn'] + metrics['fp']) > 0:
            metrics['specificity'] = float(metrics['tn'] / (metrics['tn'] + metrics['fp']))
        else:
            metrics['specificity'] = None

        print("\n" + "="*70)
        print(f"📊 EVALUATION RESULTS - {task_name.upper()}")
        print("="*70)
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1']:.4f}")
        if metrics['roc_auc']:
            print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        if metrics['specificity']:
            print(f"Specificity: {metrics['specificity']:.4f}")

        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred,
                                   target_names=['Negative', 'Positive']))

        # Save metrics
        metrics_file = self.output_dir / f"{task_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metrics saved to {metrics_file}")

        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             task_name: str = "model", cmap: str = 'Blues'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                   cbar_kws={'label': 'Count'},
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - {task_name}')

        plot_file = self.output_dir / f"{task_name}_confusion_matrix.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix saved to {plot_file}")
        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      task_name: str = "model"):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {task_name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plot_file = self.output_dir / f"{task_name}_roc_curve.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"💾 ROC curve saved to {plot_file}")
        plt.close()

    def plot_pr_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                     task_name: str = "model"):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='green', lw=2,
               label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {task_name}')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plot_file = self.output_dir / f"{task_name}_pr_curve.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"💾 PR curve saved to {plot_file}")
        plt.close()

    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare results from multiple models

        Args:
            models_results: Dictionary with model names as keys and
                           metric dictionaries as values

        Returns:
            DataFrame with comparison
        """
        comparison_df = pd.DataFrame(models_results).T

        # Select relevant metrics
        metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        comparison_df = comparison_df[[col for col in metric_cols if col in comparison_df.columns]]

        print("\n" + "="*70)
        print("📊 MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string())

        # Save comparison
        comp_file = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comp_file)
        print(f"\n💾 Comparison saved to {comp_file}")

        return comparison_df

    def plot_metric_comparison(self, models_results: Dict[str, Dict],
                              metrics: list = None):
        """Plot metrics comparison across models"""
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        comparison_df = pd.DataFrame(models_results).T
        comparison_df = comparison_df[[m for m in metrics if m in comparison_df.columns]]

        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.plot(kind='bar', ax=ax)
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_file = self.output_dir / "model_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"💾 Comparison plot saved to {plot_file}")
        plt.close()


def main():
    """Demo evaluation pipeline"""
    print("="*70)
    print("📊 MODEL EVALUATION UTILITIES")
    print("="*70)

    # Create evaluator
    evaluator = ModelEvaluator(output_dir="reports")

    # Create demo predictions
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.1, 0.85, 0.95, 0.15, 0.6])

    # Evaluate
    metrics = evaluator.evaluate_predictions(y_true, y_pred, y_proba,
                                            task_name="demo_model")

    # Plot
    evaluator.plot_confusion_matrix(y_true, y_pred, "demo_model")
    evaluator.plot_roc_curve(y_true, y_proba, "demo_model")
    evaluator.plot_pr_curve(y_true, y_proba, "demo_model")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
