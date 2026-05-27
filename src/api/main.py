"""
FastAPI Server for Himalayan Herbal AI
======================================

REST API for:
- Compound prediction (bioactivity, toxicity)
- Model inference
- Dataset exploration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np
from pathlib import Path
import json

app = FastAPI(
    title="Himalayan Herbal AI API",
    description="AI-driven discovery of herbal drug candidates",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompoundRequest(BaseModel):
    """Request model for compound prediction"""
    smiles: str
    compound_name: Optional[str] = None
    plant_source: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    compound_name: str
    smiles: str
    bioactivity_prediction: float
    bioactivity_confidence: float
    toxicity_prediction: float
    toxicity_confidence: float
    drug_like: bool
    recommendation: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, bool]


# Global state
models_cache = {}
scalers_cache = {}


def load_model(model_path: str, scaler_path: str = None):
    """Load a saved model and optional scaler"""
    try:
        # For now, return mock predictions
        # In production, load actual PyTorch models
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("🚀 Starting Himalayan Herbal AI API...")

    # Attempt to load models
    bioactivity_path = Path("models/bioactivity/best_model.pth")
    toxicity_path = Path("models/toxicity/best_model.pth")

    models_cache['bioactivity_loaded'] = bioactivity_path.exists()
    models_cache['toxicity_loaded'] = toxicity_path.exists()

    if not models_cache['bioactivity_loaded']:
        print("⚠️  Bioactivity model not found. Train first: python src/models/train_activity_model.py")
    if not models_cache['toxicity_loaded']:
        print("⚠️  Toxicity model not found. Train first: python src/models/train_toxicity_model.py")

    print("✅ API Ready")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "bioactivity": models_cache.get('bioactivity_loaded', False),
            "toxicity": models_cache.get('toxicity_loaded', False)
        }
    )


@app.get("/info")
async def info():
    """Get API information"""
    return {
        "name": "Himalayan Herbal AI",
        "version": "1.0.0",
        "description": "AI-driven discovery of low-cost herbal drug candidates",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions for a compound",
            "/dataset": "Get dataset information",
            "/compounds": "List all compounds"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: CompoundRequest):
    """
    Predict bioactivity and toxicity for a compound

    Args:
        request: CompoundRequest with SMILES string

    Returns:
        PredictionResponse with predictions
    """

    if not request.smiles:
        raise HTTPException(status_code=400, detail="SMILES string required")

    # Mock predictions (replace with actual model inference)
    # In production, convert SMILES to features and use loaded models

    smiles_len = len(request.smiles)
    bioactivity_score = min(0.9, smiles_len / 200)  # Mock scoring
    toxicity_score = 1.0 - bioactivity_score

    recommendation = "✅ Promising candidate" if bioactivity_score > 0.6 and toxicity_score < 0.5 else "⚠️  Needs further study"

    return PredictionResponse(
        compound_name=request.compound_name or f"compound_{request.smiles[:10]}",
        smiles=request.smiles,
        bioactivity_prediction=float(bioactivity_score),
        bioactivity_confidence=0.85,
        toxicity_prediction=float(toxicity_score),
        toxicity_confidence=0.82,
        drug_like=smiles_len < 100,
        recommendation=recommendation
    )


@app.get("/dataset")
async def get_dataset_info():
    """Get dataset information"""
    dataset_path = Path("src/data/data/himalayan_antimicrobial_compounds.csv")

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Read dataset info
    import pandas as pd
    df = pd.read_csv(dataset_path)

    return {
        "total_compounds": len(df),
        "active_compounds": int(df['antimicrobial_active'].sum()) if 'antimicrobial_active' in df.columns else 0,
        "plant_sources": df['plant_source'].nunique() if 'plant_source' in df.columns else 0,
        "features": list(df.columns),
        "last_updated": dataset_path.stat().st_mtime
    }


@app.get("/compounds")
async def list_compounds(limit: int = 50, active_only: bool = False):
    """
    List compounds from dataset

    Args:
        limit: Maximum number of compounds to return
        active_only: Only return antimicrobial active compounds

    Returns:
        List of compounds
    """
    dataset_path = Path("src/data/data/himalayan_antimicrobial_compounds.csv")

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    import pandas as pd
    df = pd.read_csv(dataset_path)

    if active_only and 'antimicrobial_active' in df.columns:
        df = df[df['antimicrobial_active'] == 1]

    df = df.head(limit)

    # Convert to dict and replace NaN with None
    records = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            if pd.isna(val):
                record[col] = None
            elif isinstance(val, (np.integer, np.floating)):
                record[col] = float(val) if isinstance(val, np.floating) else int(val)
            else:
                record[col] = val
        records.append(record)

    return {
        "count": len(df),
        "compounds": records
    }


@app.get("/models/status")
async def models_status():
    """Get status of all trained models"""
    return {
        "models": {
            "bioactivity": {
                "loaded": models_cache.get('bioactivity_loaded', False),
                "path": "models/bioactivity/best_model.pth"
            },
            "toxicity": {
                "loaded": models_cache.get('toxicity_loaded', False),
                "path": "models/toxicity/best_model.pth"
            },
            "gnn": {
                "loaded": models_cache.get('gnn_loaded', False),
                "path": "models/gnn/best_model.pth"
            }
        }
    }


@app.post("/batch-predict")
async def batch_predict(compounds: List[CompoundRequest]):
    """
    Batch prediction for multiple compounds

    Args:
        compounds: List of CompoundRequest objects

    Returns:
        List of predictions
    """
    if len(compounds) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 compounds per request")

    results = []
    for compound in compounds:
        try:
            prediction = await predict(compound)
            results.append(prediction)
        except Exception as e:
            results.append({
                "error": str(e),
                "compound_name": compound.compound_name
            })

    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

