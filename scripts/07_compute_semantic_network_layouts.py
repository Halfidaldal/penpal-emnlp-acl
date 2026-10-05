#!/usr/bin/env python
"""2D layouts of turn embeddings across conditions for the semantic-network figure.

processed/story_embeddings_interaction_level.parquet (all conditions)
    -> analysis/comparison/semantic_network_layouts/{turn_nodes,turn_edges}.csv, metadata.json

By default the same number of stories and adjacent edges is sampled per
condition so panels are comparable.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.semantic_network_layouts import compute_semantic_network_layouts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", default="human-ai,human-human,ai-ai,ai-ai-cross")
    parser.add_argument("--method", choices=["pca", "umap", "auto"], default="pca")
    parser.add_argument("--turn-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis-turn-min", type=int, default=1)
    parser.add_argument("--analysis-turn-max", type=int, default=9)
    parser.add_argument("--no-balance-stories", action="store_true")
    parser.add_argument("--stories-per-condition", type=int, default=None, help="default: smallest condition")
    parser.add_argument("--no-balance-edges", action="store_true")
    parser.add_argument("--edges-per-condition", type=int, default=None, help="default: smallest condition")
    parser.add_argument("--output-dir", default="analysis/comparison/semantic_network_layouts")
    args = parser.parse_args()

    metadata = compute_semantic_network_layouts(
        conditions=[c.strip() for c in args.conditions.split(",") if c.strip()],
        method=args.method,
        turn_neighbors=args.turn_neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
        analysis_turn_min=args.analysis_turn_min,
        analysis_turn_max=args.analysis_turn_max,
        output_dir=args.output_dir,
        balance_stories=not args.no_balance_stories,
        stories_per_condition=args.stories_per_condition,
        balance_edges=not args.no_balance_edges,
        edges_per_condition=args.edges_per_condition,
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
