#!/usr/bin/env python
"""Lag-based semantic distances over complete exchanges (analysis_turn 1-9).

processed/story_embeddings_interaction_level.parquet -> processed/semantic_exploration_binned.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.io import backfill_interaction_metadata, get_active_experiment, get_shared_config, load_parquet, save_parquet
from nes.semantic_exploration import compute_lag_exploration_metrics


def main():
    max_lag = get_shared_config()["exploration"]["max_lag"]
    print(f"Semantic exploration for {get_active_experiment()}")

    df = backfill_interaction_metadata(load_parquet("story_embeddings_interaction_level.parquet"))
    mask = df["analysis_turn"].between(1, 9) & df["complete_exchange"].fillna(False)
    result = compute_lag_exploration_metrics(df[mask], max_lag=max_lag)
    save_parquet(result, "semantic_exploration_binned.parquet")

    print(result.groupby(["agent", "k"])["distance"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
