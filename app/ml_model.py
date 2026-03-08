"""
app/ml_model.py

Responsible for loading the trained joblib pipeline once at startup
and exposing a predict() function for the API layer.

Architecture
------------
  best_model.joblib  =  sklearn Pipeline(TfidfVectorizer, <best classifier>)
  Preprocessing      =  handled by preprocessing.preprocess() before vectorisation
"""

import os
import joblib
import numpy as np
from typing import Dict, Any

from app.preprocessing import preprocess

# ── Resolve model path relative to project root ──────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")

# ── Load once at module import time (not per-request) ────────────────────────
_pipeline = joblib.load(_MODEL_PATH)

# Label mapping
_LABEL_MAP = {1: "Positive", 0: "Negative"}


def predict(text: str) -> Dict[str, Any]:
    """
    Run end-to-end inference on raw input text.

    Parameters
    ----------
    text : str
        Raw review text from the API request.

    Returns
    -------
    dict
        {
            "sentiment":        "Positive" | "Negative",
            "confidence_score": float  (0.0 – 1.0),
            "probabilities":    {"Negative": float, "Positive": float}
        }
    """
    clean_text = preprocess(text)

    # predict_proba → shape (1, n_classes)
    proba  = _pipeline.predict_proba([clean_text])[0]
    pred   = int(np.argmax(proba))
    label  = _LABEL_MAP[pred]
    conf   = float(round(proba[pred], 4))

    return {
        "sentiment":        label,
        "confidence_score": conf,
        "probabilities": {
            "Negative": float(round(proba[0], 4)),
            "Positive": float(round(proba[1], 4)),
        },
    }
