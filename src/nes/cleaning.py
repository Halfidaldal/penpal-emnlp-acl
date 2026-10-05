"""Cleaning, quality control and exchange alignment of raw stories."""

import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from .metadata import PAIRING_COLUMNS, chronological_slots, derive_speaker_model

STORY_PREFIX = "This is the story of"


# ---------------------------------------------------------------------------
# Story boundaries and starters
# ---------------------------------------------------------------------------

def split_repeated_conversation_ids(
    df: pd.DataFrame,
    group_col: str = "conversation_id",
    expected_length: Optional[int] = None,
    source_col: str = "source_conversation_id",
) -> pd.DataFrame:
    """Give each story slice of a multi-story conversation its own ID (`<id>__story_NN`)."""
    if expected_length is None:
        return df.copy()

    df = df.copy()
    df[source_col] = df[group_col]
    for value, index in df[df[group_col].notna()].groupby(group_col, sort=False).groups.items():
        size = len(index)
        if size <= expected_length or size % expected_length != 0:
            continue
        for offset, idx in enumerate(index):
            df.at[idx, group_col] = f"{value}__story_{offset // expected_length + 1:02d}"
    return df


def filter_by_respondent_id(df: pd.DataFrame, threshold: int = 12, column: str = "respondent_id") -> pd.DataFrame:
    """Keep rows with a well-formed respondent ID (fixed length, not a test ID)."""
    ids = df[column].astype(str)
    out = df[ids.str.len() == threshold]
    out = out[~out[column].astype(str).str.startswith("test-")].copy()
    print(f"Respondent ID filter: {len(df)} -> {len(out)} rows")
    return out


def clean_user_ai_start(df: pd.DataFrame, max_turns: int = 10, experiment: str = "human-ai") -> pd.DataFrame:
    """Detect who started each human story and strip the story-prefix placeholder.

    If author_1's first turn holds only the placeholder, author_2 started.
    """
    prefix = re.compile(r"^\s*this\s+is\s+the\s+story\s+of\b[\s:,\-.]*", re.IGNORECASE)
    min_content_length = 10

    def strip_prefix(text: Any) -> Any:
        if pd.isna(text):
            return text
        return prefix.sub("", str(text), count=1).strip()

    def detect_starter(group: pd.DataFrame) -> str:
        first = group[group["turn"] == 1]
        if first.empty:
            return "author_1"
        text = first["author_1"].iloc[0]
        remainder = strip_prefix(str(text) if pd.notna(text) else "")
        return "author_2" if len(remainder) < min_content_length else "author_1"

    df = df.copy()
    df["turn"] = df.groupby("conversation_id").cumcount() + 1
    starters = df.groupby("conversation_id", group_keys=False).apply(detect_starter, include_groups=False)
    df["starter"] = df["conversation_id"].map(starters)
    print(f"Starters: {(starters == 'author_1').sum()} author_1, {(starters == 'author_2').sum()} author_2")

    turn_col = "interaction_count" if "interaction_count" in df.columns else "turn"
    df = df[df[turn_col] <= max_turns].copy()
    df["author_1"] = df["author_1"].map(strip_prefix)
    df["author_2"] = df["author_2"].map(strip_prefix)
    return df


def clean_ai_ai_data(df: pd.DataFrame, max_turns: int = 10) -> pd.DataFrame:
    """Strip the story prefix and replace model-revealing story IDs with opaque ones."""
    df = df.copy()
    df["author_1"] = df["author_1"].str.replace(f"{STORY_PREFIX}\n", "", regex=False)
    df["author_1"] = df["author_1"].str.replace(STORY_PREFIX, "", regex=False).str.strip()
    df["author_2"] = df["author_2"].str.replace(STORY_PREFIX, "", regex=False).str.strip()
    df = df[df["turn"] <= max_turns].copy()

    ids = {story_id: f"conv_{uuid4().hex}" for story_id in df["story_id"].dropna().unique()}
    df["conversation_id"] = df["story_id"].map(ids)
    df["starter"] = "author_1"
    df["interaction_count"] = df["turn"]
    print(f"AI-AI: {len(ids)} stories, {len(df)} rows")
    return df


def randomize_author_assignment(
    df: pd.DataFrame,
    group_col: str = "conversation_id",
    seed: int = 42,
    swap_probability: float = 0.5,
) -> pd.DataFrame:
    """Swap author_1/author_2 for a random half of stories.

    Where author_1 always starts (HH, AA), this matches the random starter
    assignment of the Human-AI condition.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    story_ids = df[group_col].unique()
    swapped = set(story_ids[rng.random(len(story_ids)) < swap_probability])
    print(f"Author randomization: swapped {len(swapped)}/{len(story_ids)} stories")

    rows = df[group_col].isin(swapped)
    df.loc[rows, ["author_1", "author_2"]] = df.loc[rows, ["author_2", "author_1"]].values
    if "starter" in df.columns:
        df.loc[rows, "starter"] = df.loc[rows, "starter"].map({"author_1": "author_2", "author_2": "author_1"})
    return df


def keep_complete_conversations(
    df: pd.DataFrame,
    group_col: str = "conversation_id",
    expected_length: Optional[int] = None,
) -> pd.DataFrame:
    """Drop stories whose row count differs from the expected (default: modal) length."""
    sizes = df.groupby(group_col).size()
    if sizes.empty:
        return df.copy()
    if expected_length is None:
        expected_length = int(sizes.mode().iloc[0])

    removed = sizes[sizes != expected_length]
    if len(removed):
        print(f"Dropping {len(removed)} incomplete stories (sizes {removed.value_counts().sort_index().to_dict()})")
    return df[df[group_col].isin(sizes[sizes == expected_length].index)].copy()


# ---------------------------------------------------------------------------
# Text-quality QC
# ---------------------------------------------------------------------------
# Deterministic flags for invalid, test or keyboard-smash input. Flagged text
# is never rewritten; flags only drive story-level exclusion.

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
_KEYBOARD_PATTERN_RE = re.compile(
    r"asdf|sdf|dfg|fgh|jkl|klj|qwer|wert|zxc|xcv|lkj|dflk|fkl|kldf|dfjk|jkas|ælk|ølk",
    re.IGNORECASE,
)
_VOWELS = set("aeiouyæøå")


def _text_quality_metrics(value: Any) -> Dict[str, Any]:
    text = "" if pd.isna(value) else str(value)
    stripped = text.strip()
    no_space = re.sub(r"\s+", "", stripped)
    letters = _LETTER_RE.findall(stripped)
    words = _WORD_RE.findall(stripped.lower())
    keyboard_chars = sum(len(m.group(0)) for m in _KEYBOARD_PATTERN_RE.finditer(stripped))

    nonspace = len(no_space)
    n_letters = len(letters)
    n_words = len(words)
    return {
        "char_count": len(stripped),
        "nonspace_chars": nonspace,
        "letter_chars": n_letters,
        "word_chars": sum(len(w) for w in words),
        "word_count": n_words,
        "alpha_ratio": n_letters / max(nonspace, 1),
        "symbol_ratio": sum(not c.isalnum() for c in no_space) / max(nonspace, 1),
        "keyboard_pattern_ratio": keyboard_chars / max(n_letters, 1),
        "word_vowel_ratio": sum(any(c in _VOWELS for c in w) for w in words) / max(n_words, 1),
        "long_no_vowel_tokens": sum(len(w) >= 8 and not any(c in _VOWELS for c in w) for w in words),
        "longest_token_chars": max((len(w) for w in words), default=0),
    }


def add_text_quality_qc(
    df: pd.DataFrame,
    text_columns: List[str],
    min_substantive_chars: int = 3,
    min_alpha_ratio: float = 0.45,
    max_symbol_ratio: float = 0.35,
    max_keyboard_pattern_ratio: float = 0.12,
    min_word_vowel_ratio: float = 0.35,
    long_token_chars: int = 24,
    min_word_count_for_vowel_check: int = 3,
) -> pd.DataFrame:
    """Add per-slot `<slot>_text_qc_*` metrics, flags and reasons, plus row-level `text_qc_flagged`."""
    df = df.copy()
    flag_cols, reason_cols = [], []

    for column in text_columns:
        metrics = df[column].map(_text_quality_metrics).apply(pd.Series)
        p = f"{column}_text_qc_"
        for name in metrics.columns:
            df[p + name] = metrics[name]

        # An AI-started HA story leaves the human slot of row 1 empty by design.
        protocol_empty = pd.Series(False, index=df.index)
        if column == "author_1" and {"starter", "turn"}.issubset(df.columns):
            protocol_empty = (
                df["starter"].eq("author_2") & df["turn"].eq(1) & df[p + "word_chars"].lt(min_substantive_chars)
            )

        has_text = df[p + "nonspace_chars"].ge(min_substantive_chars)
        checks = [
            ("non_substantive", df[p + "word_chars"].lt(min_substantive_chars) & ~protocol_empty),
            ("low_alpha_ratio", has_text & df[p + "alpha_ratio"].lt(min_alpha_ratio)),
            ("high_symbol_ratio", has_text & df[p + "symbol_ratio"].gt(max_symbol_ratio)),
            ("keyboard_smash_pattern",
             df[p + "keyboard_pattern_ratio"].gt(max_keyboard_pattern_ratio)
             | df[p + "long_no_vowel_tokens"].ge(2)
             | (df[p + "longest_token_chars"].ge(long_token_chars) & df[p + "keyboard_pattern_ratio"].gt(0))),
            ("low_word_vowel_ratio",
             df[p + "word_count"].ge(min_word_count_for_vowel_check)
             & df[p + "word_vowel_ratio"].lt(min_word_vowel_ratio)),
        ]

        flagged = pd.Series(False, index=df.index)
        reasons = pd.Series("", index=df.index, dtype="object")
        for reason, mask in checks:
            flagged |= mask
            reasons = reasons.mask(mask, reasons + reason + ";")
        reasons = reasons.str.rstrip(";")
        reasons = reasons.mask(protocol_empty & reasons.eq(""), "protocol_empty")

        df[p + "flagged"] = flagged
        df[p + "reasons"] = reasons
        flag_cols.append(p + "flagged")
        reason_cols.append(p + "reasons")

    df["text_qc_flagged"] = df[flag_cols].any(axis=1)

    def combine(row: pd.Series) -> str:
        parts = [
            f"{col}:{row[rc]}" for col, rc in zip(text_columns, reason_cols)
            if isinstance(row[rc], str) and row[rc] and row[rc] != "protocol_empty"
        ]
        return "|".join(parts)

    df["text_qc_reasons"] = df.apply(combine, axis=1)
    return df


def build_text_quality_story_summary(
    df: pd.DataFrame,
    group_col: str = "conversation_id",
    story_exclusion_min_flagged_rows: int = 2,
    exclude_if_first_human_flagged: bool = True,
    count_slots: bool = False,
) -> pd.DataFrame:
    """Story-level QC summary with an `exclude_for_text_qc` decision.

    Human conditions count flagged rows. Simulated conditions count flagged
    slots (`count_slots=True`), since a row where both model turns came back
    empty is two lost contributions.
    """
    agg = {
        "n_rows": (group_col, "size"),
        "n_text_qc_flagged": ("text_qc_flagged", "sum"),
        "starter": ("starter", "first"),
    }
    for col in ["respondent_id", "respondent_id_u1", "respondent_id_u2", "source_conversation_id", "condition"]:
        if col in df.columns:
            agg[col] = (col, "first")
    summary = df.groupby(group_col, dropna=False).agg(**agg).reset_index()

    first_human_flagged = pd.Series(False, index=summary.index)
    if exclude_if_first_human_flagged and {"starter", "turn", "author_1_text_qc_flagged"}.issubset(df.columns):
        opener = df["starter"].eq("author_1") & df["turn"].eq(1) & df["author_1_text_qc_flagged"]
        first_human_flagged = summary[group_col].isin(set(df.loc[opener, group_col]))
    summary["exclude_first_human_qc_flagged"] = first_human_flagged

    if count_slots:
        slot_cols = [c for c in df.columns if c.endswith("_text_qc_flagged") and c != "text_qc_flagged"]
        per_story = df.groupby(group_col, dropna=False)[slot_cols].sum().sum(axis=1)
        summary["n_text_qc_flagged_slots"] = summary[group_col].map(per_story).fillna(0).astype(int)
        n_flagged = summary["n_text_qc_flagged_slots"]
    else:
        n_flagged = summary["n_text_qc_flagged"]

    summary["exclude_for_text_qc"] = (
        n_flagged.ge(story_exclusion_min_flagged_rows) | summary["exclude_first_human_qc_flagged"]
    )
    return summary


def filter_by_text_quality_story_qc(
    df: pd.DataFrame,
    story_summary: pd.DataFrame,
    group_col: str = "conversation_id",
) -> pd.DataFrame:
    excluded = set(story_summary.loc[story_summary["exclude_for_text_qc"], group_col])
    out = df[~df[group_col].isin(excluded)].copy()
    print(f"Text-quality QC: excluded {len(excluded)} stories ({len(df)} -> {len(out)} rows)")
    return out


# ---------------------------------------------------------------------------
# Spell correction (human turns, gpt-4o-mini)
# ---------------------------------------------------------------------------

SPELLING_PROMPT = (
    "You are a helpful assistant that ONLY corrects spelling mistakes while keeping the "
    "original meaning and structure intact. Output only the corrected text without any explanations."
)


def apply_spell_correction(
    df: pd.DataFrame,
    text_columns: List[str],
    api_key: str,
    edit_distance_threshold: Optional[int] = 70,
    skip_if_qc_flagged: bool = True,
) -> pd.DataFrame:
    """Spell-correct text columns in place.

    The original text is kept in `<col>_raw` and the Levenshtein distance in
    `<col>_edit_distance`. QC-flagged slots are not sent to the API. Rows whose
    edit distance exceeds the threshold are marked in
    `spell_correction_excessive_edit` but not dropped.
    """
    import openai
    from Levenshtein import distance as levenshtein

    client = openai.OpenAI(api_key=api_key)

    def correct(text):
        if pd.isna(text) or not isinstance(text, str):
            return text
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SPELLING_PROMPT}, {"role": "user", "content": text}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    df = df.copy()
    excessive = pd.Series(False, index=df.index)

    for column in text_columns:
        raw_col, ed_col, flag_col = f"{column}_raw", f"{column}_edit_distance", f"{column}_text_qc_flagged"
        df[raw_col] = df[column]
        skip = (
            df[flag_col].fillna(False).astype(bool)
            if skip_if_qc_flagged and flag_col in df.columns
            else pd.Series(False, index=df.index)
        )
        df[column] = [value if skip.loc[idx] else correct(value) for idx, value in df[column].items()]
        df[ed_col] = [
            None if pd.isna(a) or pd.isna(b) else levenshtein(str(a), str(b))
            for a, b in zip(df[raw_col], df[column])
        ]
        if edit_distance_threshold is not None:
            excessive |= df[ed_col].fillna(0) > edit_distance_threshold
        print(f"Spell correction on {column}: {int((~skip).sum())} calls, mean edit distance {df[ed_col].mean():.2f}")

    df["spell_correction_excessive_edit"] = excessive
    return df


# ---------------------------------------------------------------------------
# Exchange alignment and exports
# ---------------------------------------------------------------------------

AUTHOR_TYPES = {
    "human-ai": ("human", "ai"),
    "human-human": ("human", "human"),
    "ai-ai": ("ai", "ai"),
    "ai-ai-cross": ("ai", "ai"),
}


def _is_substantive_text(series: pd.Series, min_substantive_chars: int = 2) -> pd.Series:
    """At least `min_substantive_chars` word characters once whitespace and punctuation are removed."""
    normalized = series.fillna("").astype(str).str.strip().str.replace(r"[\W_]+", "", regex=True)
    return normalized.str.len() >= min_substantive_chars


def add_exchange_aligned_metadata(
    df: pd.DataFrame,
    experiment: Optional[str] = None,
    group_col: str = "conversation_id",
    min_substantive_chars: int = 2,
) -> pd.DataFrame:
    """Add author/starter types, `complete_exchange` and `analysis_turn`.

    The story opener has no prior context, so the first complete row is
    marked pre-exchange (`analysis_turn` = NA) in every condition. In AI-started
    HA stories the primer sits in an incomplete row and is excluded anyway.
    """
    df = df.copy()
    if "condition" not in df.columns:
        df["condition"] = experiment
    if "turn" not in df.columns:
        df["turn"] = df.groupby(group_col).cumcount() + 1

    unknown = sorted(set(df["condition"].dropna()) - set(AUTHOR_TYPES))
    if unknown:
        raise ValueError(f"Unknown condition(s): {unknown}")
    invalid = sorted(set(df["starter"].dropna()) - {"author_1", "author_2"})
    if invalid:
        raise ValueError(f"Invalid starter values: {invalid}")

    df["author_1_type"] = df["condition"].map({k: v[0] for k, v in AUTHOR_TYPES.items()})
    df["author_2_type"] = df["condition"].map({k: v[1] for k, v in AUTHOR_TYPES.items()})
    df["starter_side"] = df["starter"]
    df["starter_type"] = np.where(df["starter_side"].eq("author_1"), df["author_1_type"], df["author_2_type"])
    df["complete_exchange"] = (
        _is_substantive_text(df["author_1"], min_substantive_chars)
        & _is_substantive_text(df["author_2"], min_substantive_chars)
    )

    ordered = df.sort_values([group_col, "turn"], kind="stable").copy()
    contextualized = ordered["complete_exchange"] & ~(ordered["turn"].eq(1) & ordered["complete_exchange"])
    ordered["_ctx"] = contextualized
    ordered["analysis_turn"] = (
        ordered.groupby(group_col)["_ctx"].cumsum().where(ordered["_ctx"], pd.NA).astype("Int64")
    )
    df["analysis_turn"] = ordered.sort_index()["analysis_turn"]

    print(f"Exchange alignment: {int(df['complete_exchange'].sum())} complete, "
          f"{int((~df['complete_exchange']).sum())} incomplete rows")
    return df


def build_long_format_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """One row per contribution (two per interaction row)."""
    shared = [
        "conversation_id", "condition", "turn", "analysis_turn",
        "complete_exchange", "starter_side", "starter_type",
    ] + [c for c in PAIRING_COLUMNS if c in df.columns]

    frames = []
    for slot, partner in [("author_1", "author_2"), ("author_2", "author_1")]:
        frame = df[shared].copy()
        frame["speaker_slot"] = slot
        frame["speaker_type"] = df[f"{slot}_type"]
        frame["speaker_is_starter"] = df["starter_side"].eq(slot)
        frame["partner_type"] = df[f"{partner}_type"]
        speaker_model = derive_speaker_model(df, frame["speaker_is_starter"])
        if speaker_model is not None:
            frame["speaker_model"] = speaker_model
        frame["text"] = df[slot]
        frames.append(frame)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["conversation_id", "turn", "speaker_slot"], kind="stable")
        .reset_index(drop=True)
    )


STORY_METADATA_COLUMNS = [
    "condition", "language", "client_id", "workshop_id", "timestamp",
    "respondent_id", "respondent_id_u1", "respondent_id_u2", "source_conversation_id",
    "interaction_count", "starter", "starter_side", "starter_type",
    "author_1_type", "author_2_type", "llm_type", "model_id", "turn",
    *PAIRING_COLUMNS,
]


def build_full_story_text(df: pd.DataFrame, experiment: str = "human-ai") -> pd.DataFrame:
    """One row per story with `full_story` (chronological) and per-author concatenations."""

    def norm(value: Any) -> str:
        return "" if pd.isna(value) else " ".join(str(value).split())

    def concat(series: pd.Series, suffix: str = "") -> str:
        return " ".join(f"{t}{suffix}" for t in series.map(norm) if t)

    def chronological(group: pd.DataFrame) -> str:
        parts = []
        for _, row in group.iterrows():
            condition = row.get("condition", experiment)
            condition = experiment if pd.isna(condition) else str(condition)
            starter = row.get("starter_side", row.get("starter", "author_1"))
            for slot in chronological_slots(condition, starter):
                text = norm(row[slot])
                if text:
                    parts.append(text)
        return " ".join(parts)

    sort_cols = ["conversation_id"]
    if "turn" in df.columns:
        sort_cols.append("turn")
    elif "interaction_count" in df.columns:
        sort_cols.append("interaction_count")
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")

    records = []
    for conversation_id, group in df.sort_values(sort_cols, kind="stable").groupby("conversation_id", sort=False):
        record = {
            "conversation_id": conversation_id,
            "full_author_1": concat(group["author_1"]),
            "full_author_2": concat(group["author_2"]),
        }
        for col in STORY_METADATA_COLUMNS:
            if col in group.columns:
                record[col] = group[col].iloc[0]
        record["full_author_1_dot"] = concat(group["author_1"], ".")
        record["full_author_2_dot"] = concat(group["author_2"], ".")
        record["full_story"] = chronological(group)
        records.append(record)

    return pd.DataFrame.from_records(records)
