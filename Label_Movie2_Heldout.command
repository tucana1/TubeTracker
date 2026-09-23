#!/bin/zsh
# Held-out benchmark labelling for "Pollen tube movie 2 7-14-26" (never used for tuning;
# scored once per frozen SparseTrack version). Answers save after every click to
# benchmark/labels/m2_v1.json. Close this window (or press Ctrl-C) when you are done.
cd "$(dirname "$0")" || exit 1
MOVIE="$HOME/Downloads/Pollen tube movie 2 7-14-26.mp4"
CACHE="runs/sparsetrack/m2"
LABELS="benchmark/labels/m2_v1.json"
if [ ! -f "$CACHE/meta.json" ]; then
  echo "Preparing the movie (about two minutes, first time only)..."
  .venv/bin/python -m sparsetrack prepare "$MOVIE" --out "$CACHE" --flatfield || { echo "Preparation failed."; read; exit 1; }
fi
.venv/bin/python -m sparsetrack bench "$CACHE" --labels "$LABELS" --annotator investigator --port 8766
