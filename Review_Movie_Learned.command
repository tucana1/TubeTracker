#!/bin/zsh
# Review a movie that Analyze_Movie_Learned.command has analysed. The model's answers open pre-filled in the
# labelling tool (an onset bracket and traced tubes for every grain): confirm or fix each grain, and your
# answers are saved as you go. Press Ctrl-C in this window when you stop: the reviewed results are then
# written to runs/learned_evidence/<movie name>/ (reviewed_grains.csv, reviewed_traces.csv, population.png).
# Run it again to carry on where you left off; answers you have not checked stay the model's.
cd "$(dirname "$0")" || exit 1
MOVIE=$(osascript -e 'POSIX path of (choose file with prompt "Choose the movie to review")' 2>/dev/null)
[ -z "$MOVIE" ] && { echo "No movie chosen."; exit 0; }
NAME=$(.venv/bin/python -c 'import re, sys; from pathlib import Path; print(re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(sys.argv[1]).stem).strip("_."))' "$MOVIE")
CACHE="runs/sparsetrack/$NAME/cache"
WORK="runs/learned_evidence/$NAME"
[ -f "$WORK/perbin/predictions.json" ] || { echo "Analyse $MOVIE first (Analyze_Movie_Learned.command)."; read; exit 1; }
if [ ! -f "$WORK/review_labels.json" ]; then
  echo "Pre-filling the review with the model's answers (a minute or two)..."
  .venv/bin/python -m prototypes.learned_evidence.prefill --field "$CACHE" --work "$WORK" \
      || { echo "Pre-filling failed."; read; exit 1; }
fi
echo "Opening the labelling tool on the model's answers. Press Ctrl-C here when you stop."
trap 'echo' INT
.venv/bin/python -m sparsetrack bench "$CACHE" --labels "$WORK/review_labels.json" --annotator reviewer
trap - INT
.venv/bin/python -m prototypes.learned_evidence.export_review --labels "$WORK/review_labels.json" \
    || { echo "Export failed."; read; exit 1; }
open "$WORK/reviewed_grains.csv"
echo "Done. Results: $WORK/ (reviewed_grains.csv, reviewed_traces.csv, population.png). You can close this window."
