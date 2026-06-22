# Himalayan Herbal AI 🌿

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20792814.svg)](https://doi.org/10.5281/zenodo.20792814)
[![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/deepalsr/himalayan-herbal-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)

**Machine learning pipeline for antimicrobial drug discovery from Himalayan medicinal plant compounds.**

A computational screening system that predicts antimicrobial bioactivity and toxicity of molecular compounds using a Random Forest + MLP ensemble, trained on 309 curated compounds from 12 Himalayan medicinal plant species supplemented with ChEMBL bioassay data.

> Final-year BE Computer Engineering project — Cosmos College of Management and Technology, Pokhara University.

---

## Live Demo

**[https://huggingface.co/spaces/deepalsr/himalayan-herbal-ai](https://huggingface.co/spaces/deepalsr/himalayan-herbal-ai)**

Enter any SMILES string and get:
- RF bioactivity score (primary model, F1=0.928)
- MLP bioactivity score (secondary model, F1=0.889)
- Toxicity prediction
- Composite candidate ranking (Promising / Moderate / Weak)
- Lipinski drug-likeness assessment

---

## Key Results

| Model | CV F1 | CV ROC-AUC | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| **Random Forest (ours)** | **0.891 ± 0.020** | **0.854 ± 0.037** | **0.928** | **0.889** |
| XGBoost | 0.858 ± 0.014 | 0.827 ± 0.048 | — | — |
| MLP (ours) | 0.811 ± 0.021 | 0.843 ± 0.046 | 0.889 | 0.900 |
| SVM (RBF) | 0.792 ± 0.051 | 0.818 ± 0.070 | — | — |
| Naive Bayes | 0.540 ± 0.124 | 0.675 ± 0.095 | — | — |

Evaluated with **Bemis-Murcko scaffold splitting** (5-fold CV) — no scaffold appears in both train and test.

**Key finding:** Molecular weight and H-bond donors are the strongest predictors of antimicrobial activity (SHAP analysis). Lipinski violations are the weakest predictor — consistent with known antibiotics that frequently violate the Rule of Five.

---

## Dataset

309 unique compounds built from two sources:

- **Curated Himalayan set** — 32 compounds from 12 medicinal plant species with literature-verified antimicrobial activity labels and DOI references
- **ChEMBL bioassay data** — 277 antibacterial activity records against *S. aureus* (IC50-based, pIC50 ≥ 5 = active, pIC50 ≤ 4 = inactive)

Deduplicated by canonical SMILES. Each compound includes 8 RDKit-computed molecular descriptors.

**Plant sources:** Azadirachta indica, Curcuma longa, Zingiber officinale, Berberis aristata, Tinospora cordifolia, Terminalia chebula, Phyllanthus emblica, Ocimum sanctum, Withania somnifera, Swertia chirayita, Thymus linearis, Juniperus communis

Dataset available at: [DOI: 10.5281/zenodo.20792814](https://doi.org/10.5281/zenodo.20792814)

---

## Project Structure

```
himalayan-herbal-ai/
├── src/
│   ├── data/
│   │   ├── data_collector.py         # ChEMBL + PubChem + built-in dataset builder
│   │   └── data/
│   │       └── himalayan_antimicrobial_compounds.csv
│   ├── features/
│   │   └── rdkit_descriptors.py      # RDKit descriptor extraction + scaler
│   ├── models/
│   │   ├── train_activity_model.py   # MLP bioactivity trainer
│   │   ├── train_toxicity_model.py   # MLP toxicity trainer
│   │   ├── train_rf.py               # Random Forest trainer
│   │   ├── benchmark.py              # Scaffold split + baseline comparison
│   │   └── interpret.py              # SHAP interpretability analysis
│   ├── ranking/
│   │   └── candidate_ranker.py       # Composite scoring and tier assignment
│   └── api/
│       └── main.py                   # FastAPI inference server
├── models/
│   ├── bioactivity/                  # MLP weights + scaler + config
│   ├── toxicity/                     # MLP toxicity weights + scaler
│   └── rf/                          # Random Forest joblib + scaler
├── reports/
│   ├── benchmark_results.json        # Full CV comparison table
│   ├── shap_values.csv              # Raw SHAP values per compound
│   └── figures/                     # Feature importance + SHAP plots
├── web/
│   └── frontend/
│       └── index.html               # Research-grade web UI
├── app.py                           # HuggingFace Spaces Gradio entry point
├── run_pipeline.py                  # Full pipeline runner
└── requirements.txt
```

---

## Pipeline

```
SMILES string
    │
    ▼
RDKit descriptor extraction (8 features)
    │
    ├──────────────────────┬─────────────────────┐
    ▼                      ▼                     ▼
Random Forest          MLP network           MLP toxicity
(primary)              (secondary)           model
    │                      │                     │
    ▼                      ▼                     ▼
RF bioactivity         MLP bioactivity       Toxicity
score                  score                 score
    │                      │                     │
    └──────────────────────┴─────────────────────┘
                           │
                           ▼
              Composite score formula:
         0.55 × bioactivity − 0.30 × toxicity
                − 0.15 × Lipinski penalty
                           │
                           ▼
              Promising / Moderate / Weak
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/deepalsr/himalayan-herbal-ai.git
cd himalayan-herbal-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

### Rebuild the dataset
```bash
python src/data/data_collector.py
```

### Retrain all models
```bash
python src/models/train_activity_model.py
python src/models/train_toxicity_model.py
python src/models/train_rf.py
```

### Run the benchmark
```bash
python src/models/benchmark.py
```

### Run SHAP interpretability
```bash
pip install shap
python src/models/interpret.py
```

### Start the API server
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000
```

### Start the Gradio demo locally
```bash
pip install gradio
python app.py
# Open http://localhost:7860
```

---

## API

The FastAPI backend exposes the following endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model load status |
| `/predict` | POST | Single compound prediction |
| `/batch-predict` | POST | Up to 1000 compounds |
| `/compounds` | GET | Browse dataset |
| `/dataset` | GET | Dataset statistics |
| `/docs` | GET | Interactive Swagger UI |

Example prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
    "compound_name": "Curcumin",
    "plant_source": "Curcuma longa"
  }'
```

Response:

```json
{
  "compound_name": "Curcumin",
  "rf_bioactivity": 0.9167,
  "rf_confidence": 0.9167,
  "mlp_bioactivity": 0.7392,
  "mlp_confidence": 0.7392,
  "toxicity_prediction": 0.017,
  "lipinski_violations": 0,
  "drug_like": true,
  "composite_score": 0.4991,
  "recommendation": "Moderate candidate — further wet-lab validation advised",
  "primary_model": "Random Forest"
}
```

---

## Methods

### Molecular descriptors
8 physicochemical descriptors computed by RDKit: molecular weight, LogP (lipophilicity), TPSA (topological polar surface area), hydrogen bond donors, hydrogen bond acceptors, rotatable bonds, aromatic rings, and Lipinski rule violations.

### Drug-likeness (Lipinski's Rule of Five)
A compound is considered drug-like if it has at most 1 violation of: MW ≤ 500 Da, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.

### Model training
Both MLP models use a 3-layer architecture (128→64→32) with BatchNorm, ReLU, and Dropout (p=0.3), trained with class-weighted CrossEntropyLoss to handle the 76/24 class imbalance. The Random Forest uses 300 trees with balanced class weights.

### Evaluation
All models evaluated with stratified Bemis-Murcko scaffold splitting — scaffolds are grouped and split so no scaffold appears in both train and test sets. 5-fold stratified CV on the train+validation portion, with a locked scaffold test set for final evaluation.

### Interpretability
SHAP (SHapley Additive exPlanations) TreeExplainer applied to the Random Forest to quantify per-feature contributions across all 309 compounds.

---

## Limitations

- Toxicity labels are a 5-rule heuristic proxy (MW, LogP, TPSA, Lipinski violations, aromatic rings ≥ 4), not experimental Ames or hERG assay data
- Scaffold generalisation is limited: all models show ROC-AUC drop from ~0.85 to ~0.45–0.60 on unseen scaffolds, motivating future graph-based approaches
- 309 compounds is sufficient for descriptor-based ML but insufficient for a Graph Attention Network — GNN is left as future work pending dataset expansion to 1000+ compounds
- Predictions are computational estimates only — not validated in a wet lab

---

## Citation

If you use this dataset, models, or code in your research, please cite:

```bibtex
@software{shrestha2026himalayan,
  author    = {Shrestha, Dipal Kumar},
  title     = {Himalayan Herbal AI: ML Pipeline for Antimicrobial
               Drug Discovery from Himalayan Medicinal Plant Compounds},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20792814},
  url       = {https://doi.org/10.5281/zenodo.20792814}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Dipal Kumar Shrestha**
BE Computer Engineering, Cosmos College of Management and Technology, Pokhara University
GitHub: [@deepalsr](https://github.com/deepalsr)
LinkedIn: [dipal-kumar-shrestha-78535725b](https://linkedin.com/in/dipal-kumar-shrestha-78535725b)

---

*For research use only. Predictions are computational estimates — not clinical recommendations.*