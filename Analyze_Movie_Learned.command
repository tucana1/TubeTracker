#!/bin/zsh
# Prototype: analyse a movie with learned evidence. Pick the movie, wait, and the review gallery opens.
# The movie's cache is SparseTrack's (runs/sparsetrack/<movie name>/cache, built the first time only);
# results go to runs/learned_evidence/<movie name>/: per_grain.csv, and in perbin/ the review gallery,
# the population curve and growth curves. Uses the model trained on your dev movie when it exists
# (runs/learned_evidence/ld/unet.pt), else the model shipped in prototypes/learned_evidence/models/.
cd "$(dirname "$0")" || exit 1
MOVIE=$(osascript -e 'POSIX path of (choose file with prompt "Choose a pollen movie to analyse")' 2>/dev/null)
[ -z "$MOVIE" ] && { echo "No movie chosen."; exit 0; }
NAME=$(.venv/bin/python -c 'import re, sys; from pathlib import Path; print(re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(sys.argv[1]).stem).strip("_."))' "$MOVIE")
CACHE="runs/sparsetrack/$NAME/cache"
if [ ! -f "$CACHE/meta.json" ] || [ ! -f "$CACHE/grains.json" ]; then
  echo "Preparing $MOVIE (first time only, a few minutes)..."
  .venv/bin/python -c 'import sys
from pathlib import Path
from sparsetrack import stack
from sparsetrack.cli import auto_frames_per_bin, write_census
movie, cache = Path(sys.argv[1]), Path(sys.argv[2])
stack.prepare(movie, cache, frames_per_bin=auto_frames_per_bin(movie), ref_bins=3, ref_start="auto")
write_census(cache, 3, False)' "$MOVIE" "$CACHE" || { echo "Preparation failed."; read; exit 1; }
fi
MODEL="runs/learned_evidence/ld/unet.pt"
[ -f "$MODEL" ] || MODEL="prototypes/learned_evidence/models/unet_v2_sample_field.pt"
echo "Analysing with $MODEL (about 10-20 minutes on a laptop)..."
.venv/bin/python -m prototypes.learned_evidence.pipeline --field "$CACHE" --work "runs/learned_evidence/$NAME" \
    --model "$MODEL" || { echo "Analysis failed."; read; exit 1; }
open "runs/learned_evidence/$NAME/perbin/index.html"
echo "Done. Results: runs/learned_evidence/$NAME/ (per_grain.csv, perbin/). You can close this window."
