# /// script
# dependencies = [
#     "torch",
#     "transformers>=4.48.0",
#     "accelerate",
#     "pandas",
#     "numpy",
#     "tqdm",
#     "huggingface-hub",
# ]
# ///
"""Shuffled-context control for Human-AI surprisal (paper, Robustness).

Each turn is scored against (a) no context, (b) its own story's context and
(c) the context of a different, randomly assigned story. The content gain is
G_content = s_shuffled - s_true = G_true - G_shuffled.

Runs standalone, locally or as a Hugging Face Job (see
launch_shuffled_context_control.py). Input is the Human-AI interim table, or
`input.csv` from TARGET_HF_REPO; output is shuffled_control_results.csv.
"""

import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
TARGET_HF_REPO = os.environ.get("TARGET_HF_REPO")
MODEL_ID = os.environ.get("MODEL_NAME", "google/gemma-4-12B")
OUTPUT = "shuffled_control_results.csv"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def as_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return None if text.lower() in {"", "nan", "none", "null"} else text


def surprisal(context_ids, target_ids, model):
    """Mean surprisal (bits/token) of target given context."""
    max_context = getattr(model.config, "max_position_embeddings", 131072)
    context_ids = context_ids[-(max_context - len(target_ids) - 1):]
    combined = context_ids + target_ids
    if len(combined) < 2 or not target_ids:
        return 0.0

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
    return total / len(target_ids)


def run_shuffled_control(df, model, tokenizer):
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    base_context = ([bos] if bos is not None else []) + tokenizer("Speaker:", add_special_tokens=False)["input_ids"]

    def ids(value):
        text = as_text(value)
        return tokenizer(text, add_special_tokens=False)["input_ids"] if text else []

    df = df.sort_values(["conversation_id", "turn", "timestamp"]).reset_index(drop=True)
    stories = {
        conv_id: [{"turn": row["turn"], "human": ids(row["author_1"]), "ai": ids(row["author_2"])}
                  for _, row in group.iterrows()]
        for conv_id, group in df.groupby("conversation_id")
    }

    # Derangement: no story is paired with itself.
    conv_ids = list(stories)
    shuffled = conv_ids.copy()
    while any(a == b for a, b in zip(conv_ids, shuffled)):
        random.shuffle(shuffled)

    results = []
    for conv_id, other_id in tqdm(zip(conv_ids, shuffled), total=len(conv_ids), desc="Stories"):
        true_story, other_story = stories[conv_id], stories[other_id]
        true_ctx, shuf_ctx = list(base_context), list(base_context)

        for i, turn in enumerate(true_story):
            other_turn = other_story[min(i, len(other_story) - 1)]
            for speaker in ("human", "ai"):
                target = turn[speaker]
                if target and i > 0:
                    s_base = surprisal(base_context, target, model)
                    s_true = surprisal(true_ctx, target, model)
                    s_shuf = surprisal(shuf_ctx, target, model)
                    results.append({
                        "conversation_id": conv_id,
                        "turn": turn["turn"],
                        "speaker": speaker,
                        "s_base": s_base,
                        "s_true": s_true,
                        "s_shuffled": s_shuf,
                        "gain_true": s_base - s_true,
                        "gain_shuffled": s_base - s_shuf,
                        "gain_content": s_shuf - s_true,
                    })
                true_ctx.extend(target)
                shuf_ctx.extend(other_turn[speaker])

    return pd.DataFrame(results)


def main():
    # Free disk on the HF Jobs runner before the model download.
    shutil.rmtree(os.path.expanduser("~/.cache/uv/archive"), ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.cache/uv/built-wheels"), ignore_errors=True)

    data_path = Path("data/human-ai/interim/interaction_level_stories_filtered.csv")
    if not data_path.exists():
        if not TARGET_HF_REPO:
            raise FileNotFoundError(f"{data_path} not found and TARGET_HF_REPO is not set")
        data_path = Path(hf_hub_download(TARGET_HF_REPO, "input.csv", repo_type="dataset", token=HF_TOKEN))
    df = pd.read_csv(data_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    model.eval()

    results = run_shuffled_control(df, model, tokenizer)
    results.to_csv(OUTPUT, index=False)

    summary = results.groupby("speaker")[["gain_true", "gain_shuffled", "gain_content"]].mean()
    print(summary)
    print("AI - human gap:")
    print((summary.loc["ai"] - summary.loc["human"]).to_string())

    if TARGET_HF_REPO and HF_TOKEN:
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=OUTPUT, path_in_repo=OUTPUT, repo_id=TARGET_HF_REPO, repo_type="dataset"
        )


if __name__ == "__main__":
    main()
