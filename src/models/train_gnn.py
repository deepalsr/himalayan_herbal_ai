"""
GNN Model Trainer for Antimicrobial Activity Prediction
========================================================

Trains Graph Neural Networks on molecular graph datasets.
Supports GAT and GCN architectures with full training pipeline,
validation, evaluation, and model checkpointing.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class GNNTrainer:
    """
    Trains and evaluates Graph Neural Networks for antimicrobial prediction
    """

    def __init__(self,
                 dataset,
                 model,
                 output_dir: str = "models/gnn",
                 device: str = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001):
        """
        Args:
            dataset: PyG Dataset object
            model: PyG GNN model
            output_dir: Directory to save models and results
            device: 'cuda' or 'cpu' (auto-detected if None)
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization
        """
        self.dataset = dataset
        self.model = model
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        os.makedirs(output_dir, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'train_acc': []
        }
        self.best_model_path = os.path.join(output_dir, 'best_model.pth')
        self.best_val_loss = float('inf')

        print(f"🔧 GNNTrainer initialized")
        print(f"   Model: {model.__class__.__name__}")
        print(f"   Device: {self.device}")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Output dir: {output_dir}")

    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """
        Split dataset into train/val/test
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (rest is test)
        """
        n = len(self.dataset)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        indices = torch.randperm(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        self.train_data = [self.dataset[i] for i in train_indices]
        self.val_data = [self.dataset[i] for i in val_indices]
        self.test_data = [self.dataset[i] for i in test_indices]

        print(f"\n📊 Data split:")
        print(f"   Train: {len(self.train_data)} ({n_train/n*100:.1f}%)")
        print(f"   Val:   {len(self.val_data)} ({n_val/n*100:.1f}%)")
        print(f"   Test:  {len(self.test_data)} ({n_test/n*100:.1f}%)")

        # Create dataloaders
        self.train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=32)
        self.test_loader = DataLoader(self.test_data, batch_size=32)

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        total_loss = 0.0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            out = self.model(batch)
            loss = criterion(out, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(self.train_data)

    def evaluate(self, loader) -> Tuple[float, float]:
        """
        Evaluate model on a loader
        
        Returns:
            loss, accuracy
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                out = self.model(batch)
                loss = criterion(out, batch.y)

                total_loss += loss.item() * batch.num_graphs
                correct += (out.argmax(dim=1) == batch.y).sum().item()

        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / len(loader.dataset)

        return avg_loss, accuracy

    def train(self, epochs: int = 200, patience: int = 30):
        """
        Train the model
        
        Args:
            epochs: Number of epochs
            patience: Early stopping patience
        """
        print(f"\n🚀 Starting GNN Training...")
        print(f"   Epochs: {epochs}")
        print(f"   Early stopping patience: {patience}")

        self.model = self.model.to(self.device)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            criterion = nn.CrossEntropyLoss()
            train_loss = 0.0

            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                out = self.model(batch)
                loss = criterion(out, batch.y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch.num_graphs

            train_loss /= len(self.train_data)

            # Validate
            val_loss, val_acc = self.evaluate(self.val_loader)
            train_acc = (self._get_predictions(self.train_loader)[0] == 
                        self._get_labels(self.train_loader)).mean()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['train_acc'].append(train_acc)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Train Acc: {train_acc:.4f}")

            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(torch.load(self.best_model_path))
        print(f"\n✅ Training completed. Best model saved to {self.best_model_path}")

    def _get_predictions(self, loader):
        """Get all predictions from a loader"""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                preds.append(out.argmax(dim=1).cpu())
        return torch.cat(preds, dim=0).numpy()

    def _get_labels(self, loader):
        """Get all labels from a loader"""
        labels = []
        for batch in loader:
            labels.append(batch.y.cpu())
        return torch.cat(labels, dim=0).numpy()

    def evaluate_test(self) -> Dict:
        """
        Comprehensive evaluation on test set
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n📈 Evaluating on test set...")

        test_loss, test_acc = self.evaluate(self.test_loader)

        # Get predictions and probabilities
        self.model.eval()
        y_true = []
        y_pred = []
        y_proba = []

        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                probs = torch.softmax(out, dim=1)

                y_true.append(batch.y.cpu())
                y_pred.append(out.argmax(dim=1).cpu())
                y_proba.append(probs[:, 1].cpu())

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        y_proba = torch.cat(y_proba).numpy()

        metrics = {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_proba)),
        }

        print(f"✅ Test Results:")
        print(f"   Loss: {test_loss:.4f}")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")

        print(f"\n📊 Classification Report:")
        print(classification_report(y_true, y_pred,
                                   target_names=['Inactive', 'Active']))

        # Visualizations
        self._plot_results(y_true, y_pred, y_proba)

        # Save metrics
        metrics_path = os.path.join(self.output_dir, 'gnn_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _plot_results(self, y_true, y_pred, y_proba):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Confusion matrix
        from sklearn.metrics import roc_curve
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        axes[0, 1].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Training history
        axes[1, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[1, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training History')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Validation accuracy
        axes[1, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[1, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy History')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'gnn_training_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plots saved to {plot_path}")
        plt.close()

    def save_model(self, name: str = 'gnn_model'):
        """Save the trained model"""
        model_path = os.path.join(self.output_dir, f'{name}.pth')
        config_path = os.path.join(self.output_dir, f'{name}_config.json')

        torch.save(self.model.state_dict(), model_path)

        config = {
            'model_type': self.model.__class__.__name__,
            'dataset_size': len(self.dataset),
            'training_date': datetime.now().isoformat(),
            'device': self.device,
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n💾 Model saved:")
        print(f"   Model: {model_path}")
        print(f"   Config: {config_path}")


def main():
    """Main GNN training pipeline"""
    print("="*70)
    print("🧬 GNN MODEL TRAINING PIPELINE")
    print("="*70)

    # Import dataset and model
    from src.gnn.graph_builder import build_dataset_from_csv
    from src.gnn.models.gnn_model import get_model

    csv_path = "src/data/data/himalayan_antimicrobial_compounds.csv"

    # Build dataset
    print("\n📊 Building graph dataset...")
    dataset = build_dataset_from_csv(csv_path, output_dir='data/processed')

    # Get model
    print("\n🧠 Creating GNN model...")
    model = get_model('gat', num_node_features=dataset.num_node_features)

    # Create trainer
    trainer = GNNTrainer(
        dataset=dataset,
        model=model,
        output_dir="models/gnn",
        learning_rate=0.001,
        weight_decay=0.0001
    )

    # Split data
    trainer.split_data(train_ratio=0.8, val_ratio=0.1)

    # Train
    trainer.train(epochs=200, patience=30)

    # Evaluate
    metrics = trainer.evaluate_test()

    # Save
    trainer.save_model('gnn_antimicrobial_model')

    print("\n" + "="*70)
    print("✅ GNN TRAINING COMPLETE")
    print("="*70)
    print(f"\n📊 Final Metrics:")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\n📁 Results saved to: models/gnn/")


if __name__ == "__main__":
    main()
