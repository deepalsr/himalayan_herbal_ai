"""
Candidate Ranker
================
Scores and ranks drug candidates using a weighted composite of
bioactivity, toxicity, and drug-likeness (Lipinski violations).

Score = w_bio * bioactivity  -  w_tox * toxicity  -  w_lip * (violations/4)

All inputs in [0, 1]. Higher score = better candidate.
"""

from typing import List, Dict


# Weights — must sum to 1.0 for interpretability
W_BIOACTIVITY = 0.55
W_TOXICITY    = 0.30
W_LIPINSKI    = 0.15


def compute_score(
    bioactivity: float,
    toxicity: float,
    lipinski_violations: int,
) -> float:
    """
    Composite drug-candidate score in [0, 1].
    lipinski_violations is normalised to [0, 1] by dividing by 4
    (max possible violations of the four Lipinski rules).
    """
    lip_penalty = min(lipinski_violations, 4) / 4.0
    score = (
        W_BIOACTIVITY * bioactivity
        - W_TOXICITY   * toxicity
        - W_LIPINSKI   * lip_penalty
    )
    # Clip to [0, 1] — negative scores mean outright rejection
    return float(max(0.0, min(1.0, score)))


def tier(score: float) -> str:
    """Human-readable tier label based on composite score."""
    if score >= 0.55:
        return "Promising"
    elif score >= 0.35:
        return "Moderate"
    else:
        return "Weak"


def rank_candidates(predictions: List[Dict]) -> List[Dict]:
    """
    Rank a list of prediction dicts by composite score.

    Each dict must have:
        compound_name        str
        bioactivity_prediction  float  (0–1, probability of active)
        toxicity_prediction     float  (0–1, probability of toxic)
        lipinski_violations     int    (0–4)   [optional, defaults to 0]

    Returns the list sorted descending by composite_score, with two
    new fields added: composite_score and tier.
    """
    ranked = []
    for p in predictions:
        bio = float(p.get('bioactivity_prediction', 0.0))
        tox = float(p.get('toxicity_prediction',    0.0))
        lip = int(p.get('lipinski_violations',      0))

        s = compute_score(bio, tox, lip)
        ranked.append({
            **p,
            'composite_score': round(s, 4),
            'tier': tier(s),
        })

    ranked.sort(key=lambda x: x['composite_score'], reverse=True)
    return ranked


def top_candidates(predictions: List[Dict], n: int = 10) -> List[Dict]:
    """Return the top-N ranked candidates."""
    return rank_candidates(predictions)[:n]