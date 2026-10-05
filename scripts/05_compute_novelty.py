#!/usr/bin/env python
"""Surprisal-based novelty and transience (google/gemma-4-31B).

interim/interaction_level_stories_filtered.csv -> processed/novelty_scores.csv

Gemma 4 needs transformers >= 5.5, which conflicts with the embedding stack;
run this script from an environment built from requirements-novelty.txt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.io import get_active_experiment, get_shared_config, load_csv, save_csv
from nes.novelty import compute_novelty_scores, compute_transience_scores, load_language_model


def main():
    model_name = get_shared_config()["novelty"]["model_name"]
    print(f"Novelty for {get_active_experiment()} with {model_name}")

    df = load_csv("interaction_level_stories_filtered.csv", stage="interim")
    tokenizer, model = load_language_model(model_name)
    df = compute_novelty_scores(df, tokenizer, model)
    df = compute_transience_scores(df, tokenizer, model)
    save_csv(df.drop(columns=["author_1_ids", "author_2_ids"]), "novelty_scores.csv")


if __name__ == "__main__":
    main()
