"""Sentence embeddings for turns and full stories."""

import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _encode_batch(model: SentenceTransformer, batch: List[str]) -> np.ndarray:
    """Encode a batch, retrying with integer input_ids when the tokenizer emits floats.

    Some remote-code models (e.g. QZhou-Embedding) occasionally produce float
    token indices, which the embedding layer rejects.
    """
    try:
        return model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
    except RuntimeError as e:
        message = str(e)
        if not ("Expected tensor for argument #1 'indices'" in message and "FloatTensor" in message):
            raise

    features = model.tokenize(batch)
    features["input_ids"] = features["input_ids"].long()
    device = next(model.parameters()).device
    features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
    with torch.no_grad():
        emb = model.forward(features)["sentence_embedding"]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy()


def embed_story_columns(
    df: pd.DataFrame,
    text_columns: List[str],
    model_name: str,
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Embed each text column; adds `<col>_embedding` list columns and returns the arrays."""
    device = get_device()
    model = SentenceTransformer(model_name, trust_remote_code=True, device=str(device))
    for module in model:
        config = getattr(getattr(module, "auto_model", None), "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False

    df_out = df.copy()
    embeddings = {}
    for col in text_columns:
        texts = [_as_text(t) for t in df[col].tolist()]
        batches = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {col}"):
            batches.append(_encode_batch(model, texts[i:i + batch_size]))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        embeddings[col] = np.vstack(batches)
        df_out[f"{col}_embedding"] = embeddings[col].tolist()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return df_out, embeddings
