#!/usr/bin/env python
"""Clean raw stories into the exchange-aligned interim tables.

raw/<file>.csv -> interim/interaction_level_stories_filtered.csv
                  interim/interaction_level_stories_long_filtered.csv
                  interim/stories_full_text_filtered.csv
                  interim/text_quality_qc_*.csv

Human turns are spell-corrected with gpt-4o-mini when OPENAI_API_KEY is set.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.cleaning import (
    add_exchange_aligned_metadata,
    add_text_quality_qc,
    apply_spell_correction,
    build_full_story_text,
    build_long_format_analysis,
    build_text_quality_story_summary,
    clean_ai_ai_data,
    clean_user_ai_start,
    filter_by_respondent_id,
    filter_by_text_quality_story_qc,
    keep_complete_conversations,
    randomize_author_assignment,
    split_repeated_conversation_ids,
)
from nes.io import get_active_experiment, get_experiment_config, get_shared_config, load_csv, save_csv

QC_AUDIT_COLUMNS = [
    "conversation_id", "source_conversation_id", "turn", "interaction_count", "starter",
    "respondent_id", "respondent_id_u1", "respondent_id_u2", "llm_type", "timestamp", "text_qc_reasons",
    *[f"{slot}{suffix}" for slot in ("author_1", "author_2") for suffix in (
        "", "_text_qc_reasons", "_text_qc_char_count", "_text_qc_alpha_ratio", "_text_qc_symbol_ratio",
        "_text_qc_keyboard_pattern_ratio", "_text_qc_word_vowel_ratio", "_text_qc_long_no_vowel_tokens",
    )],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-text-qc", action="store_true")
    parser.add_argument("--skip-spell-correction", action="store_true")
    args = parser.parse_args()

    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared = get_shared_config()
    clean_cfg = exp_config["cleaning"]
    qc_cfg = shared["text_quality_qc"]
    seed = shared["random_seed"]
    simulated = experiment in ("ai-ai", "ai-ai-cross")
    print(f"Cleaning {experiment}")

    df = load_csv(exp_config["raw_file"], stage="raw")
    raw = exp_config["raw_columns"]
    df = df.rename(columns={raw["author_1"]: "author_1", raw["author_2"]: "author_2"})
    df["condition"] = experiment

    if simulated:
        df = clean_ai_ai_data(df, max_turns=clean_cfg["max_turns"])
        qc_slots = ["author_1", "author_2"]
        qc_params = {"min_substantive_chars": qc_cfg["min_substantive_chars"]}
    else:
        if experiment == "human-ai":
            df = split_repeated_conversation_ids(df, expected_length=clean_cfg["story_length"])
        df = clean_user_ai_start(df, max_turns=clean_cfg["max_turns"], experiment=experiment)
        if experiment == "human-ai":
            df = filter_by_respondent_id(df, threshold=12)
        qc_slots = ["author_1"] if experiment == "human-ai" else ["author_1", "author_2"]
        qc_params = {k: v for k, v in qc_cfg.items()
                     if k not in ("story_exclusion_min_flagged_rows", "exclude_if_first_human_flagged")}

    if not args.skip_text_qc:
        # Model output is only ever flagged as empty (non_substantive); a provider
        # returning nothing silently drops that contribution from the story.
        df = add_text_quality_qc(df, text_columns=qc_slots, **qc_params)
        audit = df[df["text_qc_flagged"]]
        if not simulated:
            audit = audit[[c for c in QC_AUDIT_COLUMNS if c in audit.columns]]
        save_csv(audit, "text_quality_qc_flagged_rows.csv", stage="interim")

        story_qc = build_text_quality_story_summary(
            df,
            story_exclusion_min_flagged_rows=qc_cfg["story_exclusion_min_flagged_rows"],
            exclude_if_first_human_flagged=qc_cfg["exclude_if_first_human_flagged"] and not simulated,
            count_slots=simulated,
        )
        save_csv(story_qc, "text_quality_qc_story_summary.csv", stage="interim")
        df = filter_by_text_quality_story_qc(df, story_qc)

    if not simulated and not args.skip_spell_correction:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            df = apply_spell_correction(df, text_columns=qc_slots, api_key=api_key)
        else:
            print("OPENAI_API_KEY not set; skipping spell correction")

    if experiment != "human-ai":
        df = randomize_author_assignment(df, seed=seed)

    df = keep_complete_conversations(df)
    df = add_exchange_aligned_metadata(df, experiment=experiment)

    save_csv(df, "interaction_level_stories_filtered.csv", stage="interim")
    save_csv(build_long_format_analysis(df), "interaction_level_stories_long_filtered.csv", stage="interim")
    stories = build_full_story_text(df, experiment=experiment)
    save_csv(stories, "stories_full_text_filtered.csv", stage="interim")
    print(f"{len(df)} interaction rows, {len(stories)} stories")


if __name__ == "__main__":
    main()
