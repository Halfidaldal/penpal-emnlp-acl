#!/usr/bin/env bash
# Recompute data/<condition>/processed from interim for every condition, then
# the cross-condition network layouts.
#
#   --clean         regenerate interim from raw first (needs data/<condition>/raw)
#   --skip-novelty  skip the surprisal scores (needs a large GPU)
#   --render        knit the analysis notebooks afterwards
#
# PYTHON and NOVELTY_PYTHON select interpreters (see requirements-novelty.txt);
# CONDITIONS restricts the conditions processed.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
NOVELTY_PYTHON="${NOVELTY_PYTHON:-$PYTHON}"
CONDITIONS="${CONDITIONS:-human-ai human-human ai-ai ai-ai-cross}"

CLEAN=0
NOVELTY=1
RENDER=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --skip-novelty) NOVELTY=0 ;;
    --render) RENDER=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

for condition in $CONDITIONS; do
  export EXPERIMENT="$condition"
  if [[ $CLEAN -eq 1 ]]; then "$PYTHON" scripts/01_clean_dataset.py; fi
  "$PYTHON" scripts/02_compute_embeddings.py
  "$PYTHON" scripts/03_compute_sentiment.py
  "$PYTHON" scripts/04_compute_surface_metrics.py
  if [[ $NOVELTY -eq 1 ]]; then "$NOVELTY_PYTHON" scripts/05_compute_novelty.py; fi
  "$PYTHON" scripts/06_compute_semantic_exploration.py
done
unset EXPERIMENT

"$PYTHON" scripts/07_compute_semantic_network_layouts.py

if [[ $RENDER -eq 1 ]]; then
  for notebook in \
    analysis/comparison/valence_alignment_comparison.Rmd \
    analysis/comparison/novelty_comparison.Rmd \
    analysis/comparison/semantic_exploration_comparison.Rmd \
    analysis/comparison/transience_uptake_and_scorer_checks.Rmd \
    analysis/comparison/robustness_checks.Rmd \
    analysis/human-ai/novelty.Rmd; do
    Rscript -e "rmarkdown::render('$notebook')"
  done
fi
