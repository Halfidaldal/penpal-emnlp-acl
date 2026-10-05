#!/usr/bin/env python
"""Surface text metrics with textdescriptives.

interim/stories_full_text_filtered.csv         -> processed/full_story_surface_metrics.parquet
interim/interaction_level_stories_filtered.csv -> processed/interaction_level_surface_metrics.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.io import get_active_experiment, get_shared_config, load_csv, save_parquet
from nes.surface_metrics import compute_surface_metrics


def main():
    cfg = get_shared_config()["surface_metrics"]
    params = {"spacy_mdl": cfg["spacy_model"], "batch_size": cfg["batch_size"], "n_process": cfg["n_process"]}
    print(f"Surface metrics for {get_active_experiment()}")

    stories = load_csv("stories_full_text_filtered.csv", stage="interim")
    full = compute_surface_metrics(stories, "full_author_1_dot", "full_author_2_dot", **params)
    save_parquet(full, "full_story_surface_metrics.parquet")

    turns = load_csv("interaction_level_stories_filtered.csv", stage="interim")
    turn_level = compute_surface_metrics(turns, "author_1", "author_2", **params).reset_index(drop=True)
    save_parquet(turn_level, "interaction_level_surface_metrics.parquet")


if __name__ == "__main__":
    main()
