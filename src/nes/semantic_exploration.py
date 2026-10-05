"""Lag-based semantic distance along each story's turn sequence."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metadata import PAIRING_COLUMNS, chronological_slots, speaker_model_for_slot


def parse_embedding(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        arr = np.fromstring(x.strip("[]"), sep=",") if isinstance(x, str) else np.array(x)
    except Exception:
        return None
    return arr if arr.ndim == 1 else None


def compute_lag_distances(embeddings: np.ndarray, max_lag=None):
    """Mean cosine distance between all embedding pairs `k` steps apart, for k = 1..max_lag."""
    n = embeddings.shape[0]
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    unit = embeddings / norms

    results = []
    for k in range(1, max_lag + 1):
        dists = 1.0 - np.clip(np.sum(unit[:-k] * unit[k:], axis=1), -1.0, 1.0)
        results.append({
            "k": k,
            "distance": float(np.mean(dists)),
            "std_distance": float(np.std(dists)),
            "n_pairs": len(dists),
        })
    return results


def _story_metadata(grp: pd.DataFrame) -> dict:
    cols = ["condition", "starter", "starter_side", "starter_type", "llm_type",
            "author_1_type", "author_2_type", *PAIRING_COLUMNS]
    meta = {c: grp[c].iloc[0] if c in grp.columns else None for c in cols}
    if "analysis_turn" in grp.columns:
        meta["n_complete_exchanges"] = int(grp["analysis_turn"].notna().sum())
    elif "complete_exchange" in grp.columns:
        meta["n_complete_exchanges"] = int(grp["complete_exchange"].fillna(False).sum())
    else:
        meta["n_complete_exchanges"] = len(grp)
    return meta


def _record_metadata(conversation_id, agent: str, meta: dict) -> dict:
    record = {
        "conversation_id": conversation_id,
        "agent": agent,
        **{k: meta.get(k) for k in ["condition", "starter", "starter_side", "starter_type",
                                     "llm_type", "n_complete_exchanges"]},
    }
    # Pairing columns only for ai-ai-cross, so other conditions keep their schema.
    pairing = {c: meta[c] for c in PAIRING_COLUMNS if meta.get(c) is not None}
    record.update(pairing)

    if agent == "interleaved":
        record.update(speaker_slot=pd.NA, speaker_type=pd.NA, partner_type=pd.NA, speaker_is_starter=pd.NA)
        if pairing:
            record["speaker_model"] = pd.NA
        return record

    other = "author_2" if agent == "author_1" else "author_1"
    starter_side = meta.get("starter_side")
    record.update(
        speaker_slot=agent,
        speaker_type=meta.get(f"{agent}_type"),
        partner_type=meta.get(f"{other}_type"),
        speaker_is_starter=starter_side == agent if starter_side is not None else pd.NA,
    )
    if pairing:
        record["speaker_model"] = speaker_model_for_slot(meta, agent)
    return record


def compute_lag_exploration_metrics(
    df: pd.DataFrame,
    author_1_col: str = "author_1_embedding",
    author_2_col: str = "author_2_embedding",
    max_lag=None,
) -> pd.DataFrame:
    """Lag distances per story for the interleaved sequence and for each author slot."""
    df = df.copy()
    df["author_1_emb"] = df[author_1_col].apply(parse_embedding)
    df["author_2_emb"] = df[author_2_col].apply(parse_embedding)
    df = df[df["author_1_emb"].notnull() & df["author_2_emb"].notnull()].reset_index(drop=True)

    records = []
    for conversation_id, grp in tqdm(df.groupby(["conversation_id"]), desc="Lag exploration"):
        sort_cols = (["analysis_turn"] if grp["analysis_turn"].notna().any() else []) + ["turn"]
        grp = grp.sort_values(sort_cols)
        meta = _story_metadata(grp)
        try:
            slots = {"author_1": np.vstack(grp["author_1_emb"].tolist()),
                     "author_2": np.vstack(grp["author_2_emb"].tolist())}
        except Exception:
            continue

        if slots["author_1"].shape == slots["author_2"].shape:
            first, second = chronological_slots(meta["condition"], meta["starter_side"])
            n, d = slots[first].shape
            interleaved = np.empty((2 * n, d), dtype=float)
            interleaved[0::2] = slots[first]
            interleaved[1::2] = slots[second]
            for res in compute_lag_distances(interleaved, max_lag):
                records.append({**res, **_record_metadata(conversation_id, "interleaved", meta)})

        for agent, embs in slots.items():
            for res in compute_lag_distances(embs, max_lag):
                records.append({**res, **_record_metadata(conversation_id, agent, meta)})

    return pd.DataFrame.from_records(records)
