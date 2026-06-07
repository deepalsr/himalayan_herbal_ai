"""
RDKit Molecular Descriptor Extractor
=====================================
Computes the 8 molecular descriptors that the bioactivity and toxicity
models were trained on. Also handles StandardScaler fit/save/load so
inference uses the same scale as training.
"""

import numpy as np
import pickle
from pathlib import Path

# The exact 8 columns both training scripts use, in order.
# Order matters — the saved model weights expect this exact sequence.
DESCRIPTOR_COLS = [
    'molecular_weight',
    'logp',
    'tpsa',
    'hbd',
    'hba',
    'rotatable_bonds',
    'aromatic_rings',
    'lipinski_violations',
]


def smiles_to_descriptors(smiles: str) -> dict:
    """
    Convert a SMILES string to the 8 descriptor values.
    Returns a dict keyed by DESCRIPTOR_COLS names, or raises ValueError
    if the SMILES is invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    mw  = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Lipinski violations: count how many of the 4 rules are broken
    violations = sum([
        mw  > 500,
        logp > 5,
        hbd  > 5,
        hba  > 10,
    ])

    return {
        'molecular_weight': mw,
        'logp':             logp,
        'tpsa':             tpsa,
        'hbd':              hbd,
        'hba':              hba,
        'rotatable_bonds':  rot,
        'aromatic_rings':   arom,
        'lipinski_violations': violations,
    }


def descriptors_to_array(desc_dict: dict) -> np.ndarray:
    """
    Convert descriptor dict → (1, 8) float32 numpy array in the
    exact column order the model expects.
    """
    return np.array(
        [[desc_dict[col] for col in DESCRIPTOR_COLS]],
        dtype=np.float32
    )


def fit_and_save_scaler(df, scaler_path: str):
    """
    Fit a StandardScaler on a DataFrame that contains DESCRIPTOR_COLS
    and save it to disk. Call this once after training if no scaler
    was persisted — pass the same training DataFrame used by the trainer.
    """
    from sklearn.preprocessing import StandardScaler

    available = [c for c in DESCRIPTOR_COLS if c in df.columns]
    X = df[available].fillna(0).values

    scaler = StandardScaler()
    scaler.fit(X)

    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Scaler saved to {scaler_path}")
    return scaler


def load_scaler(scaler_path: str):
    """Load a previously saved StandardScaler."""
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)


def smiles_to_scaled_array(smiles: str, scaler) -> np.ndarray:
    """
    Full pipeline: SMILES → descriptors → scaled (1, 8) array.
    This is what the API calls at inference time.
    """
    desc = smiles_to_descriptors(smiles)
    raw  = descriptors_to_array(desc)        # shape (1, 8)
    scaled = scaler.transform(raw)           # same scale as training
    return scaled.astype(np.float32)