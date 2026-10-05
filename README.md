# Directional Alignment and Narrative Agency in Human–LLM Co-Writing

Data and code for the paper comparing turn-based collaborative storytelling
across Human–Human (HH), Human–LLM (HA) and LLM–LLM (AA) dyads. The manuscript
source is in [`paper/`](paper/).

## Repository layout

```
data/<condition>/raw/        raw stories (where included)
data/<condition>/interim/    cleaned, exchange-aligned stories and QC audit
data/<condition>/processed/  embeddings, valence, novelty, surface metrics, semantic distances
scripts/                     processing pipeline
src/nes/                     pipeline modules
analysis/comparison/         cross-condition notebooks
analysis/human-ai/           within-HA notebook
analysis/figures/            figures written by the notebooks
config.yaml                  pipeline parameters
```

Conditions are `human-ai`, `human-human`, `ai-ai` and `ai-ai-cross` (two
different LLMs paired). Every table uses the same `author_1` / `author_2` slot
naming. In HH and AA the slots are randomly swapped per story, so the
chronological order of a row is given by `starter_side`.

## Reproducing the results

All processed data is included, so the analyses run without the GPU pipeline.
From the repository root, with R 4.4:

```bash
Rscript -e 'renv::restore()'
Rscript -e 'rmarkdown::render("analysis/comparison/valence_alignment_comparison.Rmd")'
Rscript -e 'rmarkdown::render("analysis/comparison/novelty_comparison.Rmd")'
Rscript -e 'rmarkdown::render("analysis/comparison/semantic_exploration_comparison.Rmd")'
Rscript -e 'rmarkdown::render("analysis/comparison/transience_uptake_and_scorer_checks.Rmd")'
Rscript -e 'rmarkdown::render("analysis/comparison/robustness_checks.Rmd")'
Rscript -e 'rmarkdown::render("analysis/human-ai/novelty.Rmd")'
```

The comparison notebooks include `ai-ai-cross`. The paper's planned contrasts
are always computed over HA, HH and AA; set `CONDITIONS <- PUBLISHED_CONDITIONS`
in a notebook to drop the cross-model condition entirely.

| Paper section | Notebook |
|---|---|
| Affective alignment: baseline asymmetry, directional alignment | `comparison/valence_alignment_comparison.Rmd` |
| Novelty, transience and resonance across conditions | `comparison/novelty_comparison.Rmd` |
| Transience 2x2 table, within-HA context gain and uptake, scorer check | `comparison/transience_uptake_and_scorer_checks.Rmd` |
| Semantic distance | `comparison/semantic_exploration_comparison.Rmd` |
| Within-HA agent contrasts | `human-ai/novelty.Rmd` |
| Robustness: shuffled context, lexical baselines, surface PERMANOVA | `comparison/robustness_checks.Rmd` |

| Figure | Notebook |
|---|---|
| `valence_baseline_asymmetry_per_story.png`, `valence_condition_baseline_density.png`, `valence_asymmetry_per_story.png`, `directional_valence_alignment_comparison_three_panel.png` | valence_alignment_comparison |
| `novelty_condition_distributions.png`, `novelty_metric_level_asymmetry.png`, `novelty_influence_per_condition.png` | novelty_comparison |
| `semantic_distance_interactionlevel.png`, `semantic_distance_storylevel.png`, `semantic_network.png` | semantic_exploration_comparison |
| `Novelty_Transience_Resonance_Distributions_Agent.png`, `novelty_resonance_vs_novelty.png` | human-ai/novelty |

The remaining figures in `analysis/figures/` are supplementary diagnostics.

As a check, the per-story baseline valence asymmetry
`|mean(slot 1 valence) − mean(slot 2 valence)|` should come out as:

| Condition | Stories | Mean |
|---|---|---|
| HA | 97 | 0.060 |
| HH | 36 | 0.041 |
| AA | 80 | 0.028 |

## Re-running the pipeline

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_md
python -m venv .venv-novelty && .venv-novelty/bin/pip install -r requirements-novelty.txt

PYTHON=.venv/bin/python NOVELTY_PYTHON=.venv-novelty/bin/python scripts/run_pipeline.sh
```

`run_pipeline.sh` recomputes `processed/` from `interim/` for every condition.
Run a single step for one condition with, for example,
`EXPERIMENT=human-ai python scripts/03_compute_sentiment.py`.

| Script | Step | Model |
|---|---|---|
| `01_clean_dataset.py` | QC, spell correction, exchange alignment (raw → interim; `--clean`) | `gpt-4o-mini` |
| `02_compute_embeddings.py` | turn and story embeddings | `Kingsoft-LLM/QZhou-Embedding` |
| `03_compute_sentiment.py` | valence by concept-vector projection | `paraphrase-multilingual-mpnet-base-v2` |
| `04_compute_surface_metrics.py` | textdescriptives surface metrics | `en_core_web_md` |
| `05_compute_novelty.py` | novelty and transience (excess surprisal) | `google/gemma-4-31B` |
| `06_compute_semantic_exploration.py` | lag-based semantic distances | |
| `07_compute_semantic_network_layouts.py` | 2D layouts for the network figure | |

Scripts 02 and 05 need a GPU in practice. Script 05 runs in its own
environment because Gemma 4 requires `transformers>=5.5`, which the embedding
stack does not support.

Two scripts are not part of the default pipeline:

- `simulate_ai_ai.py` generates the AA stories (API keys for the providers in
  `config.yaml`).
- `shuffled_context_control.py` computes the shuffled-context control reported
  in the paper (`google/gemma-4-12B`), locally or as a Hugging Face Job via
  `launch_shuffled_context_control.py`. Its output is included as
  `data/human-ai/processed/shuffled_control_results.csv`.

## Valence measure

The published valence is `author_*_sentiment_projection`: each turn embedded on
its own, L2-normalized, and projected onto the unit sentiment concept vector in
`src/nes/data/Sentiment.csv`.
