#!/usr/bin/env python
"""Embed turns and full stories (Kingsoft-LLM/QZhou-Embedding).

interim/stories_full_text_filtered.csv          -> processed/story_embeddings_full.{parquet,npy}
                                                   processed/story_author_{1,2}_embeddings_full.npy
interim/interaction_level_stories_filtered.csv  -> processed/story_embeddings_interaction_level.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.embeddings import embed_story_columns
from nes.io import get_active_experiment, get_shared_config, load_csv, save_npy, save_parquet


def main():
    cfg = get_shared_config()["embeddings"]
    print(f"Embedding {get_active_experiment()} with {cfg['model_name']}")

    stories = load_csv("stories_full_text_filtered.csv", stage="interim")
    stories, emb = embed_story_columns(
        stories, ["full_story", "full_author_1", "full_author_2"], cfg["model_name"], cfg["batch_size"]
    )
    save_parquet(stories, "story_embeddings_full.parquet")
    save_npy(emb["full_story"], "story_embeddings_full.npy")
    save_npy(emb["full_author_1"], "story_author_1_embeddings_full.npy")
    save_npy(emb["full_author_2"], "story_author_2_embeddings_full.npy")

    turns = load_csv("interaction_level_stories_filtered.csv", stage="interim")
    turns, _ = embed_story_columns(turns, ["author_1", "author_2"], cfg["model_name"], cfg["batch_size"])
    save_parquet(turns, "story_embeddings_interaction_level.parquet")


if __name__ == "__main__":
    main()
