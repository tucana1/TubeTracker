#!/bin/zsh
# Analyse movies with SparseTrack: pick one movie or several (Cmd-click), wait, and each review gallery opens.
# Results go to runs/sparsetrack/<movie name>/ (a movie's cache is built the first time only).
cd "$(dirname "$0")" || exit 1
PICKED=$(osascript -e 'set fs to choose file with prompt "Choose pollen movies to analyse (Cmd-click for several)" with multiple selections allowed' \
                   -e 'set out to ""' -e 'repeat with f in fs' -e 'set out to out & POSIX path of f & linefeed' -e 'end repeat' \
                   -e 'return out' 2>/dev/null)
MOVIES=("${(@f)PICKED}")
MOVIES=(${MOVIES:#})  # drop empty lines
[ ${#MOVIES} -eq 0 ] && { echo "No movie chosen."; exit 0; }
if [ ! -f calibration.json ]; then
  echo "Once: your microscope's pixel size and frame interval give lengths in um and times in minutes."
  read "UM?Pixel size in um (Enter to skip): "
  read "SPF?Seconds between frames (Enter to skip): "
  if [[ -n "$UM" && -n "$SPF" ]]; then
    printf '{"um_per_px": %s, "s_per_frame": %s}\n' "$UM" "$SPF" > calibration.json && echo "Saved to calibration.json (edit or delete it to change)."
  fi
fi
FAILED=()
for MOVIE in "${MOVIES[@]}"; do
  echo "Analysing $MOVIE (5-10 minutes)..."
  .venv/bin/python -m sparsetrack run "$MOVIE" || { echo "Analysis of $MOVIE failed."; FAILED+=("$MOVIE"); }
done
[ ${#FAILED} -gt 0 ] && { echo "Failed: ${FAILED[*]}"; read; exit 1; }
echo "Done. To check and correct the answers, double-click Review_Movie_SparseTrack.command. You can close this window."
