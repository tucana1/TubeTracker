#!/bin/bash
# Start TubeTracker Annotator (P0A v1). Double-click on the Mac.
cd "$(dirname "$0")"
MOVIE="${1:-/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4}"
PROJ="${2:-$HOME/Documents/TubeTracker-annotator}"
exec ./.venv-annotator/bin/python scripts/run_annotation_app.py \
  --movie "$MOVIE" --project-dir "$PROJ"
