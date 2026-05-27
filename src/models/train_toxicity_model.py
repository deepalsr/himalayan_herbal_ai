"""
Toxicity Prediction Model Trainer
==================================

Trains a machine learning model to predict compound toxicity
from molecular descriptors. Uses similar architecture to bioactivity
but focuses on toxicity classification (toxic vs. non-toxic).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ToxicityPredictor(nn.Module):
    """
    Neural network for toxicity prediction from molecular descriptors
    """

    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int] = None,
                 dropout: float = 0.3,
                 num_classes: int = 2):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
            num_classes: Number of output classes (2 for binary)
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        self.input_size = input_size
        self.dropout_prob = dropout

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class ToxicityModelTrainer:
    """
    Trains and evaluates toxicity prediction models
    """

    def __init__(self,
                 data_path: str = "data/himalayan_antimicrobial_compounds.csv",
                 output_dir: str = "models/toxicity",
                 device: str = None):
        """
        Args:
            data_path: Path to the compound dataset
            output_dir: Directory to save models and results
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(output_dir, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_model_path = os.path.join(output_dir, 'best_model.pth')

        print(f"🔧 ToxicityModelTrainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Output: {output_dir}")

    def load_data(self) -> pd.DataFrame:
        """Load the compound dataset"""
        print(f"\n📂 Loading data from {self.data_path}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(df)} compounds")
        print(f"   Columns: {list(df.columns)}")

        return df

    def create_toxicity_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic toxicity labels based on compound properties
        In a real scenario, these would come from experimental data
        """
        print("\n⚠️  Creating synthetic toxicity labels (for demonstration)")

        # Heuristic: compounds with high MW or high violations more likely toxic
        df['toxicity'] = 0

        if 'molecular_weight' in df.columns:
            df.loc[df['molecular_weight'] > 450, 'toxicity'] = 1

        if 'lipinski_violations' in df.columns:
            df.loc[df['lipinski_violations'] > 1, 'toxicity'] = 1

        # Add some noise
        noise_indices = np.random.choice(len(df), size=int(0.1 * len(df)), replace=False)
        df.loc[noise_indices, 'toxicity'] = 1 - df.loc[noise_indices, 'toxicity']

        print(f"   Toxic compounds: {df['toxicity'].sum()}")
        print(f"   Non-toxic compounds: {len(df) - df['toxicity'].sum()}")

        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract molecular descriptor features for training"""
        print("\n🔬 Extracting features...")

        descriptor_cols = [
            'molecular_weight',
            'logp',
            'tpsa',
            'hbd',
            'hba',
            'rotatable_bonds',
            'aromatic_rings',
            'lipinski_violations'
        ]
        
        available_cols = [col for col in descriptor_cols if col in df.columns]

        if available_cols:
            X = df[available_cols].fillna(0).values
        else:
            print("⚠️  No molecular descriptors found. Creating synthetic features...")
            X = self._create_synthetic_features(df)

        y = df['toxicity'].values

        print(f"✅ Extracted {X.shape[1]} features from {X.shape[0]} samples")
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Label distribution: {np.bincount(y)}")

        return X, y

    def _create_synthetic_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create synthetic features"""
        features = []

        if 'molecular_weight' in df.columns:
            features.append(df['molecular_weight'].values)
        else:
            features.append(np.ones(len(df)) * 300)

        if 'smiles' in df.columns:
            logp = df['smiles'].str.len() / 100
            features.append(logp.values)
        else:
            features.append(np.ones(len(df)) * 2)

        if 'plant_source' in df.columns:
            tpsa = np.random.normal(60, 20, len(df))
            features.append(np.clip(tpsa, 0, 150))
        else:
            features.append(np.ones(len(df)) * 60)

        return np.column_stack(features)

    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                    test_size: float = 0.2, val_size: float = 0.1):
        """Split and normalize data"""
        print("\n📊 Preparing data...")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        print(f"✅ Data splits:")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Val:   {X_val.shape[0]} samples")
        print(f"   Test:  {X_test.shape[0]} samples")

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.LongTensor(y_test).to(self.device)

        return (
            (X_train_t, y_train_t),
            (X_val_t, y_val_t),
            (X_test_t, y_test_t)
        )

    def create_model(self, input_size: int, hidden_sizes: List[int] = None) -> ToxicityPredictor:
        """Create the model"""
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        model = ToxicityPredictor(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout=0.3,
            num_classes=2
        ).to(self.device)

        print(f"\n🧠 Model created:")
        print(f"   Input features: {input_size}")
        print(f"   Hidden layers: {hidden_sizes}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def train(self,
              train_data: Tuple[torch.Tensor, torch.Tensor],
              val_data: Tuple[torch.Tensor, torch.Tensor],
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              weight_decay: float = 0.0001):
        """Train the model"""
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")

        X_train, y_train = train_data
        X_val, y_val = val_data

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = (val_preds == y_val).float().mean().item()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")

            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(torch.load(self.best_model_path))
        print(f"\n✅ Training completed.")

    def evaluate(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """Evaluate model on test set"""
        print(f"\n📈 Evaluating model...")

        X_test, y_test = test_data

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            y_test_np = y_test.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()

        metrics = {
            'accuracy': float(accuracy_score(y_test_np, preds_np)),
            'precision': float(precision_score(y_test_np, preds_np, zero_division=0)),
            'recall': float(recall_score(y_test_np, preds_np, zero_division=0)),
            'f1': float(f1_score(y_test_np, preds_np, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test_np, probs_np)),
        }

        print(f"✅ Test Results:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

        print(f"\n📊 Classification Report:")
        print(classification_report(y_test_np, preds_np,
                                   target_names=['Non-toxic', 'Toxic']))

        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        self._plot_results(y_test_np, preds_np, probs_np)

        return metrics

    def _plot_results(self, y_test, y_pred, y_proba):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Oranges')
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        axes[0, 1].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[1, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training History')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.history['val_acc'], label='Val Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'evaluation_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plots saved to {plot_path}")
        plt.close()

    def save_model(self, name: str = 'toxicity_model'):
        """Save the trained model and scaler"""
        model_path = os.path.join(self.output_dir, f'{name}.pth')
        scaler_path = os.path.join(self.output_dir, f'{name}_scaler.pkl')
        config_path = os.path.join(self.output_dir, f'{name}_config.json')

        torch.save(self.model.state_dict(), model_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        config = {
            'input_size': self.model.input_size,
            'model_type': 'ToxicityPredictor',
            'device': self.device,
            'training_date': datetime.now().isoformat(),
            'history': self.history,
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n💾 Model saved:")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Config: {config_path}")


def main():
    """Main training pipeline"""
    print("="*70)
    print("☠️  TOXICITY PREDICTION MODEL TRAINER")
    print("="*70)

    trainer = ToxicityModelTrainer(
        data_path="src/data/data/himalayan_antimicrobial_compounds.csv",
        output_dir="models/toxicity"
    )

    df = trainer.load_data()
    df = trainer.create_toxicity_labels(df)
    X, y = trainer.extract_features(df)
    train_data, val_data, test_data = trainer.prepare_data(X, y)
    trainer.model = trainer.create_model(input_size=X.shape[1])

    trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=150,
        batch_size=16,
        learning_rate=0.001
    )

    metrics = trainer.evaluate(test_data)
    trainer.save_model('toxicity_model')

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"\n📊 Final Metrics:")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\n📁 Results saved to: models/toxicity/")


if __name__ == "__main__":
    main()
