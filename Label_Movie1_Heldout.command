#!/bin/zsh
# Label movie 1 (the third movie) blind, as movie 2 was: the honest test of a frozen SparseTrack version.
# Grain census, onset brackets and traces for a random sample of 30 isolated grains; answers are saved after
# every click to benchmark/labels/m1_v1.json. Do not open the model's analysis of this movie before labelling.
cd "$(dirname "$0")" || exit 1
MOVIE="$HOME/Downloads/Pollen tube movie 1 7-14-26.mp4"
CACHE="runs/sparsetrack/m1"
LABELS="benchmark/labels/m1_v1.json"
if [ ! -f "$CACHE/meta.json" ]; then
  echo "Preparing the movie (a few minutes, first time only)..."
  .venv/bin/python -m sparsetrack prepare "$MOVIE" --out "$CACHE" --flatfield || { echo "Preparation failed."; read; exit 1; }
fi
.venv/bin/python -m sparsetrack bench "$CACHE" --labels "$LABELS" --annotator investigator --port 8767 --sample 30
