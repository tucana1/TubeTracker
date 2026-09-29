#!/bin/zsh
# Check and correct a movie that Analyze_Movie_SparseTrack.command has analysed. The labelling tool opens with
# the model's answers filled in (an onset bracket and traced tubes for every grain): Enter confirms an answer,
# or fix it as you would label it. Answers are saved as you go. Press Ctrl-C in this window when you stop:
# the reviewed results are then written to runs/sparsetrack/<movie name>/review/ (reviewed_grains.csv,
# reviewed_traces.csv, population.png). Run it again to carry on; answers not yet checked stay the model's.
cd "$(dirname "$0")" || exit 1
MOVIE=$(osascript -e 'POSIX path of (choose file with prompt "Choose the movie to review")' 2>/dev/null)
[ -z "$MOVIE" ] && { echo "No movie chosen."; exit 0; }
NAME=$(.venv/bin/python -c 'import sys; from sparsetrack.cli import run_folder; print(run_folder(sys.argv[1]))' "$MOVIE")
[ -f "$NAME/analysis/predictions.json" ] || { echo "Analyse $MOVIE first (Analyze_Movie_SparseTrack.command)."; read; exit 1; }
EXTRA=()
if [ -f "$NAME/review/review_labels.model.json" ] && [ "$NAME/analysis/predictions.json" -nt "$NAME/review/review_labels.model.json" ]; then
  echo "This movie was analysed again after its review was started."
  read "ans?Start a new review of the new analysis? The earlier review is kept, whole, in a folder beside it. [y/N] "
  [[ "$ans" == [yY]* ]] && EXTRA=(--new)
fi
trap 'echo' INT
.venv/bin/python -m sparsetrack review "$MOVIE" "${EXTRA[@]}" || { echo "Review failed."; read; exit 1; }
trap - INT
open "$NAME/review/reviewed_grains.csv"
echo "Done. Results: $NAME/review/ (reviewed_grains.csv, reviewed_traces.csv, population.png). You can close this window."
