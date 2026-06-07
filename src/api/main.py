"""
FastAPI Server for Himalayan Herbal AI
======================================
Real model inference using trained BioactivityPredictor and
ToxicityPredictor weights. Scalers are fitted from the training
dataset at startup (since no .pkl was persisted after training).
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ── Model architecture ─────────────────────────────────────────────────────
# Copied verbatim from train_activity_model.py / train_toxicity_model.py
# so we can reconstruct the network and load saved weights.

class _Predictor(nn.Module):
    """Shared MLP architecture for both bioactivity and toxicity models."""

    def __init__(self, input_size: int,
                 hidden_sizes=(128, 64, 32),
                 dropout: float = 0.3,
                 num_classes: int = 2):
        super().__init__()
        self.input_size = input_size
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ── Constants ──────────────────────────────────────────────────────────────

DESCRIPTOR_COLS = [
    'molecular_weight', 'logp', 'tpsa', 'hbd',
    'hba', 'rotatable_bonds', 'aromatic_rings', 'lipinski_violations',
]

DATASET_PATH    = Path("src/data/data/himalayan_antimicrobial_compounds.csv")
BIO_MODEL_PATH  = Path("models/bioactivity/best_model.pth")
BIO_CONFIG_PATH = Path("models/bioactivity/bioactivity_model_config.json")
TOX_MODEL_PATH  = Path("models/toxicity/best_model.pth")
TOX_CONFIG_PATH = Path("models/toxicity/toxicity_model_config.json")


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Himalayan Herbal AI API",
    description="AI-driven discovery of herbal drug candidates",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Global inference state ─────────────────────────────────────────────────

_state: Dict = {
    "bio_model":  None,
    "tox_model":  None,
    "bio_scaler": None,
    "tox_scaler": None,
    "device":     "cpu",
}


# ── Startup: load models + fit scalers ────────────────────────────────────

def _fit_scaler_from_dataset():
    """
    Re-fit a StandardScaler on the training data.
    This is necessary because the training scripts did not persist
    the fitted scaler to disk — only best_model.pth was saved.
    Both the bioactivity and toxicity trainers used the same 8
    descriptor columns, so one scaler fits both.
    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    if not DATASET_PATH.exists():
        print(f"⚠️  Dataset not found at {DATASET_PATH}. Scaler will be identity.")
        return None

    df = pd.read_csv(DATASET_PATH)
    available = [c for c in DESCRIPTOR_COLS if c in df.columns]
    if not available:
        print("⚠️  No descriptor columns found in dataset. Scaler will be identity.")
        return None

    X = df[available].fillna(0).values
    scaler = StandardScaler()
    scaler.fit(X)
    print(f"✅ Scaler fitted on {len(X)} compounds ({len(available)} features)")
    return scaler


def _load_model(model_path: Path, config_path: Path, device: str) -> Optional[_Predictor]:
    """Load a _Predictor from saved weights and config JSON."""
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        return None

    input_size = 8  # default
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        input_size = cfg.get("input_size", 8)

    model = _Predictor(input_size=input_size)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✅ Loaded {model_path.name}  (input_size={input_size})")
    return model


@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Himalayan Herbal AI API...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _state["device"] = device

    _state["bio_model"] = _load_model(BIO_MODEL_PATH, BIO_CONFIG_PATH, device)
    _state["tox_model"] = _load_model(TOX_MODEL_PATH, TOX_CONFIG_PATH, device)

    scaler = _fit_scaler_from_dataset()
    _state["bio_scaler"] = scaler
    _state["tox_scaler"] = scaler   # same scaler — both models use identical features

    print("✅ API ready")


# ── Descriptor extraction ─────────────────────────────────────────────────

def _smiles_to_features(smiles: str) -> np.ndarray:
    """SMILES → scaled (1, 8) float32 array for model input."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    viol = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)

    raw = np.array([[mw, logp, tpsa, hbd, hba, rot, arom, viol]], dtype=np.float32)
    return raw


def _run_model(model: _Predictor, raw: np.ndarray, scaler) -> tuple[float, float]:
    """
    Run one forward pass. Returns (probability_of_active, confidence).
    probability_of_active = softmax score for class 1.
    confidence = max(softmax) — how decisive the model is.
    """
    x = scaler.transform(raw) if scaler else raw
    tensor = torch.tensor(x, dtype=torch.float32).to(_state["device"])

    with torch.no_grad():
        logits = model(tensor)                          # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]       # (2,)

    prob_active = float(probs[1].item())
    confidence  = float(probs.max().item())
    return prob_active, confidence


def _drug_likeness(smiles: str) -> tuple[bool, int]:
    """Returns (passes_lipinski, violation_count)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, 4
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    viol = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return viol <= 1, viol


# ── Request / Response models ─────────────────────────────────────────────

class CompoundRequest(BaseModel):
    smiles: str
    compound_name: Optional[str] = None
    plant_source: Optional[str] = None


class PredictionResponse(BaseModel):
    compound_name: str
    smiles: str
    bioactivity_prediction: float
    bioactivity_confidence: float
    toxicity_prediction: float
    toxicity_confidence: float
    lipinski_violations: int
    drug_like: bool
    composite_score: float
    recommendation: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded={
            "bioactivity": _state["bio_model"] is not None,
            "toxicity":    _state["tox_model"] is not None,
        },
    )


@app.get("/info")
async def info():
    return {
        "name": "Himalayan Herbal AI",
        "version": "1.1.0",
        "models": {
            "bioactivity": _state["bio_model"] is not None,
            "toxicity":    _state["tox_model"] is not None,
        },
        "descriptor_features": DESCRIPTOR_COLS,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: CompoundRequest):
    if not request.smiles:
        raise HTTPException(status_code=400, detail="SMILES string required")

    # Parse SMILES → feature array
    try:
        raw = _smiles_to_features(request.smiles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Bioactivity
    if _state["bio_model"] is not None:
        bio_prob, bio_conf = _run_model(_state["bio_model"], raw, _state["bio_scaler"])
    else:
        raise HTTPException(status_code=503, detail="Bioactivity model not loaded. Run training first.")

    # Toxicity
    if _state["tox_model"] is not None:
        tox_prob, tox_conf = _run_model(_state["tox_model"], raw, _state["tox_scaler"])
    else:
        raise HTTPException(status_code=503, detail="Toxicity model not loaded. Run training first.")

    # Drug-likeness
    drug_like, violations = _drug_likeness(request.smiles)

    # Composite score (mirrors candidate_ranker logic)
    lip_penalty = min(violations, 4) / 4.0
    composite = max(0.0, min(1.0,
        0.55 * bio_prob - 0.30 * tox_prob - 0.15 * lip_penalty
    ))

    # Recommendation
    if composite >= 0.55:
        rec = "Promising candidate — high bioactivity, acceptable toxicity"
    elif composite >= 0.35:
        rec = "Moderate candidate — further wet-lab validation advised"
    else:
        rec = "Weak candidate — low bioactivity or high toxicity concern"

    return PredictionResponse(
        compound_name=request.compound_name or f"compound_{request.smiles[:8]}",
        smiles=request.smiles,
        bioactivity_prediction=round(bio_prob, 4),
        bioactivity_confidence=round(bio_conf, 4),
        toxicity_prediction=round(tox_prob, 4),
        toxicity_confidence=round(tox_conf, 4),
        lipinski_violations=violations,
        drug_like=drug_like,
        composite_score=round(composite, 4),
        recommendation=rec,
    )


@app.get("/dataset")
async def get_dataset_info():
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    import pandas as pd
    df = pd.read_csv(DATASET_PATH)
    return {
        "total_compounds":  len(df),
        "active_compounds": int(df['antimicrobial_active'].sum()) if 'antimicrobial_active' in df.columns else 0,
        "plant_sources":    df['plant_source'].nunique() if 'plant_source' in df.columns else 0,
        "features":         list(df.columns),
    }


@app.get("/compounds")
async def list_compounds(limit: int = 50, active_only: bool = False):
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    import pandas as pd
    df = pd.read_csv(DATASET_PATH)
    if active_only and 'antimicrobial_active' in df.columns:
        df = df[df['antimicrobial_active'] == 1]
    df = df.head(limit)
    records = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            if hasattr(val, 'item'):
                record[col] = val.item()
            elif val != val:   # NaN check
                record[col] = None
            else:
                record[col] = val
        records.append(record)
    return {"count": len(records), "compounds": records}


@app.get("/models/status")
async def models_status():
    return {
        "models": {
            "bioactivity": {"loaded": _state["bio_model"] is not None, "path": str(BIO_MODEL_PATH)},
            "toxicity":    {"loaded": _state["tox_model"] is not None, "path": str(TOX_MODEL_PATH)},
        }
    }


@app.post("/batch-predict")
async def batch_predict(compounds: List[CompoundRequest]):
    if len(compounds) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 compounds per request")
    results = []
    for c in compounds:
        try:
            results.append(await predict(c))
        except HTTPException as e:
            results.append({"error": e.detail, "compound_name": c.compound_name})
    return {"predictions": results}


# ── Serve frontend (replaces Flask) ───────────────────────────────────────
# This mounts the HTML frontend directly on FastAPI so the Flask proxy
# in web/backend/app.py is no longer needed.
_frontend = Path("web/frontend")
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)