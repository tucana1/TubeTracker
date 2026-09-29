#!/bin/zsh
# Analyse a movie with SparseTrack: pick the movie, wait, and the review gallery opens.
# Results go to runs/sparsetrack/<movie name>/ (the cache is built the first time only).
cd "$(dirname "$0")" || exit 1
MOVIE=$(osascript -e 'POSIX path of (choose file with prompt "Choose a pollen movie to analyse")' 2>/dev/null)
[ -z "$MOVIE" ] && { echo "No movie chosen."; exit 0; }
echo "Analysing $MOVIE (5-10 minutes)..."
.venv/bin/python -m sparsetrack run "$MOVIE" || { echo "Analysis failed."; read; exit 1; }
echo "Done. To check and correct the answers, double-click Review_Movie_SparseTrack.command. You can close this window."
