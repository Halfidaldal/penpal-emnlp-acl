"""Turn valence by projection onto a sentiment concept vector.

Each turn is embedded on its own, L2-normalized and projected onto the unit
concept vector, giving a cosine score in [-1, 1]. Method and vector from
https://github.com/lauritswl/SemanticProjection.
"""

from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def _as_text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return None if text.lower() in {"", "nan", "none", "null"} else text


def load_concept_vector(path: str) -> np.ndarray:
    v = pd.read_csv(path).values.flatten().astype(float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Concept vector has zero norm.")
    return v / norm


class ValenceProjector:
    def __init__(self, model_name: str, vector_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.unit_vector = load_concept_vector(vector_path)

    def score(self, texts: List, batch_size: int = 32) -> np.ndarray:
        """Valence per text; NaN for missing or empty text."""
        texts = [_as_text(t) for t in texts]
        valid = [i for i, t in enumerate(texts) if t is not None]
        scores = np.full(len(texts), np.nan, dtype=float)
        if not valid:
            return scores

        emb = self.model.encode([texts[i] for i in valid], batch_size=batch_size,
                                show_progress_bar=True, convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Cannot normalize a zero-length embedding.")
        scores[valid] = np.clip((emb / norms) @ self.unit_vector, -1.0, 1.0)
        return scores
