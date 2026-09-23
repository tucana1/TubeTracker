#!/bin/zsh
# Opens the sparse-movie benchmark labelling tool in your browser.
# Answers are saved after every click to benchmark/labels/ld_v1.json.
# Close this window (or press Ctrl-C) when you are done.
cd "$(dirname "$0")" || exit 1
MOVIE="$HOME/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"
CACHE="runs/sparsetrack/ld"
LABELS="benchmark/labels/ld_v1.json"
if [ ! -f "$CACHE/meta.json" ]; then
  echo "Preparing the movie (about a minute, first time only)..."
  .venv/bin/python -m sparsetrack prepare "$MOVIE" --out "$CACHE" || { echo "Preparation failed."; read; exit 1; }
fi
.venv/bin/python -m sparsetrack bench "$CACHE" --labels "$LABELS" --annotator investigator
