"""Surface text metrics (textdescriptives) per author slot, in long format."""

import pandas as pd
import spacy
import textdescriptives as td

from .metadata import PAIRING_COLUMNS, derive_speaker_model

SLOT_METADATA_COLUMNS = [
    "conversation_id", "condition", "turn", "analysis_turn", "interaction_count",
    "starter", "starter_side", "starter_type", "complete_exchange", "llm_type", "timestamp",
    "respondent_id", "respondent_id_u1", "respondent_id_u2", "model_id",
    *PAIRING_COLUMNS,
]
DERIVED_METADATA_COLUMNS = ["type", "speaker_slot", "speaker_type", "partner_type", "speaker_is_starter", "speaker_model"]


def _as_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def load_pipeline(spacy_mdl: str):
    nlp = spacy.load(spacy_mdl)
    try:
        nlp.add_pipe("textdescriptives/all")
    except Exception as e:
        # Some spaCy/pydantic combinations reject the bundled quality config;
        # add the components one by one with an explicit config instead.
        if "typing.Tuple" not in str(e) and "textdescriptives/quality" not in str(e):
            raise
        for comp in [
            "textdescriptives/descriptive_stats",
            "textdescriptives/readability",
            "textdescriptives/dependency_distance",
            "textdescriptives/pos_proportions",
            "textdescriptives/coherence",
            "textdescriptives/information_theory",
        ]:
            if not nlp.has_pipe(comp):
                nlp.add_pipe(comp)
        if not nlp.has_pipe("textdescriptives/quality"):
            nlp.add_pipe("textdescriptives/quality",
                         config={"top_ngram_range": (2, 4), "duplicate_n_gram_fraction_range": (5, 10)})
    return nlp


def _slot_metrics(nlp, df: pd.DataFrame, text_col: str, slot: str, batch_size: int, n_process: int) -> pd.DataFrame:
    text = df[text_col].apply(_as_text)
    metrics = td.extract_df(nlp.pipe(text, batch_size=batch_size, n_process=n_process), include_text=True)
    metrics.index = df.index

    other = "author_2" if slot == "author_1" else "author_1"
    metrics["type"] = slot
    metrics["speaker_slot"] = slot
    for col in SLOT_METADATA_COLUMNS:
        if col in df.columns:
            metrics[col] = df[col]
    if f"{slot}_type" in df.columns:
        metrics["speaker_type"] = df[f"{slot}_type"]
    if f"{other}_type" in df.columns:
        metrics["partner_type"] = df[f"{other}_type"]
    starter_col = "starter_side" if "starter_side" in df.columns else ("starter" if "starter" in df.columns else None)
    if starter_col is not None:
        metrics["speaker_is_starter"] = df[starter_col].eq(slot)
        speaker_model = derive_speaker_model(df, metrics["speaker_is_starter"])
        if speaker_model is not None:
            metrics["speaker_model"] = speaker_model

    # Metrics on empty or near-empty text are meaningless; set them to NA.
    missing = ~text.str.len().ge(2)
    if missing.any():
        metadata = set(SLOT_METADATA_COLUMNS) | set(DERIVED_METADATA_COLUMNS)
        metric_cols = [c for c in metrics.columns if c not in metadata]
        for col in metric_cols:
            dtype = metrics[col].dtype
            if pd.api.types.is_bool_dtype(dtype):
                metrics[col] = metrics[col].astype("boolean")
            elif pd.api.types.is_integer_dtype(dtype):
                metrics[col] = metrics[col].astype("Int64")
            elif pd.api.types.is_float_dtype(dtype):
                metrics[col] = metrics[col].astype("Float64")
            elif isinstance(dtype, pd.CategoricalDtype):
                metrics[col] = metrics[col].astype("object")
        metrics.loc[missing, metric_cols] = pd.NA
    return metrics


def compute_surface_metrics(
    df: pd.DataFrame,
    author_1_col: str,
    author_2_col: str,
    spacy_mdl: str = "en_core_web_md",
    batch_size: int = 10,
    n_process: int = 5,
) -> pd.DataFrame:
    """Metrics for both author slots stacked long (one row per text per slot)."""
    nlp = load_pipeline(spacy_mdl)
    return pd.concat([
        _slot_metrics(nlp, df, author_1_col, "author_1", batch_size, n_process),
        _slot_metrics(nlp, df, author_2_col, "author_2", batch_size, n_process),
    ], axis=0)
