# Evaluation Figures

All model evaluation visualizations and plots are stored here.

## Contents

### `bioactivity_evaluation.png`
Bioactivity prediction model evaluation (antimicrobial activity classifier)
- **Confusion Matrix** (top-left): Shows TP/FP/FN/TN classification results
- **ROC Curve** (top-right): Receiver Operating Characteristic with AUC = 0.75
- **Training History** (bottom-left): Train vs validation loss over epochs
- **Validation Accuracy** (bottom-right): Accuracy progression during training

**Performance:**
- Accuracy: 85.71%
- Precision: 85.71%
- Recall: 100%
- F1-Score: 0.923
- ROC-AUC: 0.75

### `toxicity_evaluation.png`
Toxicity prediction model evaluation (toxic/non-toxic classifier)
- **Confusion Matrix** (top-left): Shows perfect classification on test set
- **ROC Curve** (top-right): Receiver Operating Characteristic with AUC = 1.00
- **Training History** (bottom-left): Train vs validation loss over epochs
- **Validation Accuracy** (bottom-right): Accuracy progression during training

**Performance:**
- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1-Score: 1.000
- ROC-AUC: 1.00

---

## How to Use These Figures

### View in Terminal
```bash
# macOS
open bioactivity_evaluation.png
open toxicity_evaluation.png

# Linux
eog bioactivity_evaluation.png
eog toxicity_evaluation.png

# Web browser
file://$(pwd)/bioactivity_evaluation.png
```

### Include in Reports
```markdown
![Bioactivity Model Evaluation](reports/figures/bioactivity_evaluation.png)
![Toxicity Model Evaluation](reports/figures/toxicity_evaluation.png)
```

### Command Line
```bash
# View file info
file bioactivity_evaluation.png

# Get image dimensions
identify bioactivity_evaluation.png

# Create thumbnails
convert bioactivity_evaluation.png -resize 200x200 bioactivity_thumb.png
```

---

## Model Details

Both models were trained on 34 Himalayan medicinal plant compounds with:
- **Train set**: 23 samples (67.6%)
- **Validation set**: 4 samples (11.8%)
- **Test set**: 7 samples (20.6%)

**Features (8 molecular descriptors):**
1. Molecular Weight (MW)
2. LogP (partition coefficient)
3. TPSA (topological polar surface area)
4. Hydrogen Bond Donors (HBD)
5. Hydrogen Bond Acceptors (HBA)
6. Rotatable Bonds
7. Aromatic Rings
8. Lipinski Violations

**Architecture (both models):**
- Input layer: 8 features
- Hidden layer 1: 128 neurons + BatchNorm + ReLU + Dropout(0.3)
- Hidden layer 2: 64 neurons + BatchNorm + ReLU + Dropout(0.3)
- Hidden layer 3: 32 neurons + BatchNorm + ReLU + Dropout(0.3)
- Output layer: 2 classes (binary classification)
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Loss: CrossEntropyLoss
- Early stopping: patience=20 epochs

---

Generated: May 27, 2026
