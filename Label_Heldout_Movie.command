#!/bin/zsh
# Label a NEW movie blind (as movies 1 and 2 were): the honest test of the frozen SparseTrack candidates.
# Pick the movie; it is prepared once (a few minutes), then the labelling tool opens on a fixed random sample of 30
# isolated grains: grain census, onset brackets and traces, saved after every click to
# benchmark/labels/<movie>_v1.json. Do not open any analysis of this movie before the labelling is finished; then
# run scripts/score_heldout.sh <movie> --final (it scores the versions fixed before the labels existed).
cd "$(dirname "$0")" || exit 1
MOVIE=$(osascript -e 'POSIX path of (choose file with prompt "Choose the movie to label blind" of type {"public.movie"})' 2>/dev/null)
[ -n "$MOVIE" ] || { echo "No movie chosen."; exit 1; }
KEY=$(basename "$MOVIE" | sed -E 's/\.[A-Za-z0-9]+$//; s/[^A-Za-z0-9_.-]+/_/g; s/^[_.]+//; s/[_.]+$//')
CACHE="runs/sparsetrack/heldout_$KEY"
LABELS="benchmark/labels/heldout_${KEY}_v1.json"
echo "Movie: $MOVIE"
echo "Labels: $LABELS"
if [ ! -f "$CACHE/meta.json" ]; then
  echo "Preparing the movie (a few minutes, first time only)..."
  FLAT=""
  [ -n "$TT_FLATFIELD" ] && FLAT="--flatfield"   # TT_FLATFIELD=1 for uneven illumination (as movie 1)
  .venv/bin/python -m sparsetrack prepare "$MOVIE" --out "$CACHE" $FLAT || { echo "Preparation failed."; read; exit 1; }
fi
.venv/bin/python -m sparsetrack bench "$CACHE" --labels "$LABELS" --annotator investigator --port 8768 --sample 30
