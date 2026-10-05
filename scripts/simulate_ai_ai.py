#!/usr/bin/env python
"""Generate the AI-AI condition (same-model pairs).

Writes data/ai-ai/raw/simulated_stories.csv. Models and sampling parameters
are under experiments.ai-ai.simulation in config.yaml; API keys are read from
the environment (or a .env file).
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.io import get_experiment_config, save_csv
from nes.simulation import simulate_ai_ai_dataset


def main():
    load_dotenv()
    cfg = get_experiment_config("ai-ai")["simulation"]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-stories", type=int, default=cfg["n_stories_per_model"], help="stories per model")
    parser.add_argument("--n-turns", type=int, default=cfg["n_turns_per_story"])
    parser.add_argument("--models", nargs="+", help="subset of model ids from config")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between stories")
    args = parser.parse_args()

    models = [m for m in cfg["models"] if not args.models or m["id"] in args.models]
    if not models:
        sys.exit(f"No configured models match {args.models}")

    df = simulate_ai_ai_dataset(
        model_configs=models,
        n_stories_per_model=args.n_stories,
        n_turns_per_story=args.n_turns,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        delay_between_stories=args.delay,
    )
    if df.empty:
        sys.exit("No stories generated; check API keys.")

    save_csv(df, "simulated_stories.csv", stage="raw", experiment="ai-ai")
    print(df.groupby("model_id")["story_id"].nunique().to_string())


if __name__ == "__main__":
    main()
