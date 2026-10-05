"""Story metadata shared across pipeline stages."""

from typing import Optional, Tuple

import pandas as pd


def chronological_slots(condition, starter_side) -> Tuple[str, str]:
    """Order in which the two slots of a row were written.

    Human-AI rows are always (human input, AI reply). In the other conditions
    author slots are randomly swapped per story, so `starter_side` decides.
    """
    if condition == "human-ai":
        return ("author_1", "author_2")
    if starter_side == "author_2":
        return ("author_2", "author_1")
    return ("author_1", "author_2")


# Cross-model pairing (`ai-ai-cross` only). Every stage whitelists the metadata
# it carries forward, so these must be listed wherever metadata is copied.
PAIRING_COLUMNS = [
    "model_starter",    # model that wrote turn 1
    "model_responder",  # model that replied
    "pair_id",          # ordered cell, "starter>responder"
    "dyad_id",          # order-insensitive pair
    "is_self_pair",     # both sides the same model
]


def derive_speaker_model(df: pd.DataFrame, is_starter: pd.Series) -> Optional[pd.Series]:
    """Model that wrote each slot, resolved via starter role (slots are counterbalanced)."""
    if not {"model_starter", "model_responder"}.issubset(df.columns):
        return None

    role = pd.Series(is_starter, index=df.index).astype("boolean")
    model = pd.Series(pd.NA, index=df.index, dtype="object")
    model = model.mask((role == True).fillna(False), df["model_starter"])
    model = model.mask((role == False).fillna(False), df["model_responder"])
    return model


def speaker_model_for_slot(metadata: dict, slot: str):
    """Scalar form of derive_speaker_model for per-conversation metadata."""
    starter_side = metadata.get("starter_side")
    if starter_side is None or metadata.get("model_starter") is None:
        return pd.NA
    return metadata["model_starter"] if starter_side == slot else metadata.get("model_responder")
