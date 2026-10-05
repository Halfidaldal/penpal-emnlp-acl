"""Surprisal-based novelty and transience with a causal language model.

For turn T_t with prior story context C_t and the next turn F_{t+1}:

    novelty_t    = s(T_t | C_t) - s(T_t | BOS)
    transience_t = s(F_{t+1} | C_t, T_t) - s(F_{t+1} | C_t)

where s is mean surprisal in bits/token. Turns are scored in chronological
order (see `metadata.chronological_slots`), and context resets per story.
"""

import math

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metadata import chronological_slots

ANCHOR_TEXT = "Speaker:"
SLOTS = ("author_1", "author_2")


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


def load_language_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def calc_sentence_surprisal(context_ids, target_ids, model):
    """Mean (bits/token) and total (bits) surprisal of `target_ids` given `context_ids`."""
    max_context = getattr(model.config, "max_position_embeddings", 131072)
    context_ids = context_ids[-(max_context - len(target_ids) - 1):]
    combined = context_ids + target_ids
    if len(combined) < 2 or not target_ids:
        return 0.0, 0.0

    input_ids = torch.tensor([combined], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[:, :-1, :]

    total = 0.0
    for idx, token_id in enumerate(target_ids):
        pos = len(context_ids) + idx - 1
        if pos >= logits.shape[1]:
            break
        log_probs = torch.nn.functional.log_softmax(logits[0, pos], dim=-1)
        total += -(log_probs[token_id] / math.log(2)).item()
    return total / len(target_ids), total


def _prepare(df, tokenizer):
    """Sort by story/turn, tokenize both slots and build the BOS + anchor baseline context."""
    sort_cols = [c for c in ["conversation_id", "turn", "timestamp"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    has_text = {}
    for slot in SLOTS:
        text = df[slot].apply(_as_text)
        has_text[slot] = text.notna().to_numpy()
        if f"{slot}_ids" not in df.columns:
            df[f"{slot}_ids"] = text.apply(
                lambda t: [] if t is None else tokenizer(t, add_special_tokens=False)["input_ids"]
            )

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    base_context = ([bos] if bos is not None else []) + tokenizer(ANCHOR_TEXT, add_special_tokens=False)["input_ids"]
    return df, has_text, base_context


def _slot_order(row):
    return chronological_slots(row.get("condition"), row.get("starter_side"))


def compute_novelty_scores(df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """Adds `<slot>_surprise` (novelty), `<slot>_surprise_raw` (s(T|C)) and `<slot>_entropy` (total bits)."""
    df, has_text, base_context = _prepare(df.copy(), tokenizer)
    out = {f"{slot}_{k}": [] for slot in SLOTS for k in ("surprise", "surprise_raw", "entropy")}

    context = list(base_context)
    last_story = None
    for pos, row in tqdm(df.iterrows(), total=len(df), desc="Novelty"):
        story = row.get("conversation_id")
        if last_story is None or (story is not None and story != last_story):
            context = list(base_context)
            last_story = story

        results = {slot: (np.nan, np.nan, np.nan) for slot in SLOTS}
        for slot in _slot_order(row):
            ids = row[f"{slot}_ids"]
            if has_text[slot][pos] and ids:
                avg, total = calc_sentence_surprisal(context, ids, model)
                avg_base, _ = calc_sentence_surprisal(base_context, ids, model)
                results[slot] = (avg - avg_base, avg, total)
                context.extend(ids)

        for slot in SLOTS:
            novelty, raw, total = results[slot]
            out[f"{slot}_surprise"].append(novelty)
            out[f"{slot}_surprise_raw"].append(raw)
            out[f"{slot}_entropy"].append(total)

    for k in ("surprise", "surprise_raw", "entropy"):
        for slot in SLOTS:
            df[f"{slot}_{k}"] = out[f"{slot}_{k}"]
    return df


def compute_transience_scores(df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """Adds `<slot>_transience` plus its terms `_raw` = s(F|C,T) and `_baseline` = s(F|C).

    F is the next chronological turn in the same story; the final turn has none.
    """
    df, has_text, base_context = _prepare(df.copy(), tokenizer)
    out = {f"{slot}_{k}": [] for slot in SLOTS for k in ("transience", "transience_raw", "transience_baseline")}

    def next_turn_ids(pos, story, slot, order):
        i = order.index(slot)
        if i + 1 < len(order):
            return df.iloc[pos][f"{order[i + 1]}_ids"]
        if pos + 1 >= len(df):
            return []
        nxt = df.iloc[pos + 1]
        if nxt.get("conversation_id") != story:
            return []
        return nxt[f"{_slot_order(nxt)[0]}_ids"]

    context = list(base_context)
    last_story = None
    for pos in tqdm(range(len(df)), desc="Transience"):
        row = df.iloc[pos]
        story = row.get("conversation_id")
        if last_story is None or (story is not None and story != last_story):
            context = list(base_context)
            last_story = story

        order = _slot_order(row)
        results = {slot: (np.nan, np.nan, np.nan) for slot in SLOTS}
        for slot in order:
            ids = row[f"{slot}_ids"]
            future = next_turn_ids(pos, story, slot, order)
            if has_text[slot][pos] and ids and future:
                with_turn, _ = calc_sentence_surprisal(context + ids, future, model)
                without_turn, _ = calc_sentence_surprisal(list(context), future, model)
                results[slot] = (with_turn - without_turn, with_turn, without_turn)
            if has_text[slot][pos] and ids:
                context.extend(ids)

        for slot in SLOTS:
            transience, with_turn, without_turn = results[slot]
            out[f"{slot}_transience"].append(transience)
            out[f"{slot}_transience_raw"].append(with_turn)
            out[f"{slot}_transience_baseline"].append(without_turn)

    for k in ("transience", "transience_raw", "transience_baseline"):
        for slot in SLOTS:
            df[f"{slot}_{k}"] = out[f"{slot}_{k}"]
    return df
