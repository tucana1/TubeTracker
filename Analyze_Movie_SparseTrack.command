#!/bin/zsh
# Analyse a movie with SparseTrack: pick the movie, wait, and the review gallery opens.
# Results go to runs/sparsetrack/<movie name>/ (the cache is built the first time only).
cd "$(dirname "$0")" || exit 1
MOVIE=$(osascript -e 'POSIX path of (choose file with prompt "Choose a pollen movie to analyse")' 2>/dev/null)
[ -z "$MOVIE" ] && { echo "No movie chosen."; exit 0; }
if [ ! -f calibration.json ]; then
  echo "Once: your microscope's pixel size and frame interval give lengths in um and times in minutes."
  read "UM?Pixel size in um (Enter to skip): "
  read "SPF?Seconds between frames (Enter to skip): "
  if [[ -n "$UM" && -n "$SPF" ]]; then
    printf '{"um_per_px": %s, "s_per_frame": %s}\n' "$UM" "$SPF" > calibration.json && echo "Saved to calibration.json (edit or delete it to change)."
  fi
fi
echo "Analysing $MOVIE (5-10 minutes)..."
.venv/bin/python -m sparsetrack run "$MOVIE" || { echo "Analysis failed."; read; exit 1; }
echo "Done. To check and correct the answers, double-click Review_Movie_SparseTrack.command. You can close this window."
