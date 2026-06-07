"""
Himalayan Herbal AI — Gradio Demo App
======================================
Run locally:
    pip install gradio
    python app_gradio.py

Deploy to HuggingFace Spaces:
    1. Create a new Space at huggingface.co/spaces (SDK: Gradio)
    2. git clone your space repo
    3. Copy this file + models/ + src/ + requirements_gradio.txt
    4. git push → auto-deploys with a public URL

The public URL becomes your paper's "live demo" link.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path

# ── Model architecture (mirrors train_activity_model.py) ─────────────────

class _Predictor(nn.Module):
    def __init__(self, input_size=8, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers, prev = [], input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.network = nn.Sequential(*layers)
        self.input_size = input_size

    def forward(self, x):
        return self.network(x)


# ── Load models ───────────────────────────────────────────────────────────

def _load():
    device = 'cpu'
    bio_model = _Predictor()
    bio_model.load_state_dict(torch.load("models/bioactivity/best_model.pth", map_location=device))
    bio_model.eval()

    tox_model = _Predictor()
    tox_model.load_state_dict(torch.load("models/toxicity/best_model.pth", map_location=device))
    tox_model.eval()

    with open("models/bioactivity/bioactivity_model_scaler.pkl", "rb") as f:
        bio_scaler = pickle.load(f)
    with open("models/toxicity/toxicity_model_scaler.pkl", "rb") as f:
        tox_scaler = pickle.load(f)

    return bio_model, tox_model, bio_scaler, tox_scaler


BIO_MODEL, TOX_MODEL, BIO_SCALER, TOX_SCALER = _load()


# ── Feature extraction ────────────────────────────────────────────────────

def _features(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    viol = int(mw>500) + int(logp>5) + int(hbd>5) + int(hba>10)
    return np.array([[mw, logp, tpsa, hbd, hba, rot, arom, viol]], dtype=np.float32)


def _predict_model(model, scaler, raw):
    x = torch.FloatTensor(scaler.transform(raw))
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return float(probs[1]), float(probs.max())


# ── Gradio prediction function ────────────────────────────────────────────

EXAMPLE_SMILES = [
    ["COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O", "Curcumin"],
    ["COC1=C(C2=C[N+]3=C(C=C2C=C1)C4=CC5=C(C=C4CC3)OCO5)OC", "Berberine"],
    ["CC1=CC=C(C=C1)C(C)O", "Thymol analog"],
    ["C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O", "Quercetin"],
]


def predict(smiles: str, compound_name: str = ""):
    if not smiles or not smiles.strip():
        return "❌ Please enter a SMILES string.", "", "", "", "", ""

    raw = _features(smiles.strip())
    if raw is None:
        return "❌ Invalid SMILES — could not parse molecule.", "", "", "", "", ""

    bio_prob, bio_conf = _predict_model(BIO_MODEL, BIO_SCALER, raw)
    tox_prob, tox_conf = _predict_model(TOX_MODEL, TOX_SCALER, raw)

    mw   = float(raw[0, 0])
    logp = float(raw[0, 1])
    viol = int(raw[0, 7])
    drug_like = viol <= 1

    lip_penalty = min(viol, 4) / 4.0
    composite   = max(0.0, min(1.0, 0.55*bio_prob - 0.30*tox_prob - 0.15*lip_penalty))

    if composite >= 0.55:
        tier = "🟢 Promising"
        rec  = "High bioactivity with acceptable toxicity profile. Recommended for wet-lab validation."
    elif composite >= 0.35:
        tier = "🟡 Moderate"
        rec  = "Moderate activity. Further structural optimisation or analog screening advised."
    else:
        tier = "🔴 Weak"
        rec  = "Low predicted activity or high toxicity concern. Consider structural modifications."

    name_str = f"**{compound_name}**\n\n" if compound_name.strip() else ""

    result = f"""{name_str}
### 🧬 Prediction Results

| Property | Value |
|---|---|
| Bioactivity Score | `{bio_prob:.3f}` ({bio_conf:.0%} confidence) |
| Toxicity Score | `{tox_prob:.3f}` |
| Composite Score | `{composite:.3f}` |
| Candidate Tier | {tier} |
| Drug-like (Lipinski) | {'✅ Yes' if drug_like else '❌ No'} |
| Lipinski Violations | {viol} / 4 |

### 📋 Recommendation
{rec}

### 📐 Molecular Properties
| Descriptor | Value |
|---|---|
| Molecular Weight | {mw:.1f} Da |
| LogP | {logp:.2f} |
| MW ≤ 500 | {'✅' if mw<=500 else '❌'} |
| LogP ≤ 5 | {'✅' if logp<=5 else '❌'} |

---
*Model: MLP trained on 309 Himalayan plant compounds (ChEMBL + curated). 
Bioactivity F1=0.89, ROC-AUC=0.90 (5-fold CV). 
For research use only — not a clinical recommendation.*
"""
    return result


# ── Gradio UI ─────────────────────────────────────────────────────────────

def build_app():
    import gradio as gr

    with gr.Blocks(
        title="Himalayan Herbal AI",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 860px !important; }"
    ) as demo:

        gr.Markdown("""
# 🌿 Himalayan Herbal AI
### Antimicrobial Bioactivity & Toxicity Predictor

Predicts the antimicrobial potential and toxicity of molecular compounds
using a neural network trained on **309 Himalayan medicinal plant compounds**.

> **Research tool** — predictions are computational estimates, not clinical results.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                smiles_input = gr.Textbox(
                    label="SMILES String",
                    placeholder="e.g. COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
                    lines=3,
                )
                name_input = gr.Textbox(
                    label="Compound Name (optional)",
                    placeholder="e.g. Curcumin"
                )
                predict_btn = gr.Button("🔬 Predict", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 📌 Example Compounds")
                for smi, name in EXAMPLE_SMILES:
                    gr.Button(name, size="sm").click(
                        fn=lambda s=smi, n=name: (s, n),
                        outputs=[smiles_input, name_input]
                    )

        output = gr.Markdown(label="Results")

        predict_btn.click(
            fn=predict,
            inputs=[smiles_input, name_input],
            outputs=output,
        )

        gr.Markdown("""
---
**Dataset:** 309 compounds from 12 Himalayan medicinal plants + ChEMBL antibacterial assays  
**Models:** Descriptor-based MLP | Bioactivity F1=0.89 | ROC-AUC=0.90  
**Paper:** *[Preprint link — add after submission]*  
**Code:** [github.com/deepalsr/himalayan-herbal-ai](https://github.com/deepalsr/himalayan-herbal-ai)
        """)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)