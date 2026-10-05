#!/usr/bin/env python
"""Turn valence by projection onto the sentiment concept vector.

processed/story_embeddings_interaction_level.parquet -> processed/dyadic_sentiment_scores.parquet
(adds author_{1,2}_sentiment_projection)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.io import (
    backfill_interaction_metadata,
    get_active_experiment,
    get_project_root,
    get_shared_config,
    load_parquet,
    save_parquet,
)
from nes.sentiment import ValenceProjector


def main():
    cfg = get_shared_config()["sentiment"]
    print(f"Scoring valence for {get_active_experiment()} with {cfg['model_name']}")

    df = load_parquet("story_embeddings_interaction_level.parquet")
    df = backfill_interaction_metadata(df)

    projector = ValenceProjector(cfg["model_name"], str(get_project_root() / cfg["concept_vector_path"]))
    for slot in ("author_1", "author_2"):
        df[f"{slot}_sentiment_projection"] = projector.score(df[slot].tolist(), batch_size=cfg["batch_size"])

    save_parquet(df, "dyadic_sentiment_scores.parquet")


if __name__ == "__main__":
    main()
