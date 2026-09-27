#!/bin/zsh
# Adapts the learned pipeline to your dev movie: the dev test scored on your labels (benchmark/labels/ld_v1.json),
# then decoder calibration and fine-tuning, each kept only if its check says so. About 2.5 hours the first time
# (1.5 of them training on your movie's field). It can be stopped and started again: finished steps are skipped.
# If your labels have changed since, or a step read the movie otherwise, it offers to run every step again (the
# trained model is kept). If your labels find the way of reading movies adopted on 27 Sep 2026 worse than the one
# before, it offers to switch back; the choice is kept for later runs and Analyze_Movie_Learned.command.
# At the end runs/learned_evidence/SUMMARY.md opens: what each step found, what Analyze_Movie_Learned.command now
# uses, and the one command that scores movie 2, once.
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || { echo "No .venv here. In a separate worktree, link it: ln -s ../TubeTracker/.venv .venv"; read; exit 1; }
.venv/bin/python -c "import torch" 2>/dev/null || { echo "torch is needed once: .venv/bin/pip install torch==2.13.0"; read; exit 1; }
CACHE="runs/sparsetrack/ld"
[ -f "$CACHE/meta.json" ] || { echo "No prepared dev movie at $CACHE: open Label_Sparse_Benchmark.command once first."; read; exit 1; }
# your current labels: those of the checkout that holds runs/ (the main one when runs/ is a link to it)
LABELS="$(cd runs && pwd -P)/../benchmark/labels/ld_v1.json"
[ -f "$LABELS" ] || LABELS="benchmark/labels/ld_v1.json"
echo "Adapting to $CACHE with $LABELS ..."
.venv/bin/python -m prototypes.learned_evidence.adapt --field "$CACHE" --labels "$LABELS" \
    || { echo "Adaptation failed."; read; exit 1; }
if grep -q "find the fused reading worse" runs/learned_evidence/SUMMARY.md; then
    read "ans?Your labels find the new way of reading the movie (since 27 Sep) worse than the one before. Switch back to it and run every step again (about an hour; the model trained on your field is kept)? [y/N] "
    if [[ "$ans" == [yY]* ]]; then
        .venv/bin/python -m prototypes.learned_evidence.adapt --field "$CACHE" --labels "$LABELS" --reading plain \
            --redo || { echo "Adaptation failed."; read; exit 1; }
    fi
elif grep -q -e "The labels have changed" -e "read the movie otherwise" runs/learned_evidence/SUMMARY.md; then
    read "ans?Your labels, or how the movie is read, have changed since these results were made. Run every step again now (about an hour; the model trained on your field is kept)? [y/N] "
    if [[ "$ans" == [yY]* ]]; then
        .venv/bin/python -m prototypes.learned_evidence.adapt --field "$CACHE" --labels "$LABELS" --redo \
            || { echo "Adaptation failed."; read; exit 1; }
    fi
fi
open runs/learned_evidence/SUMMARY.md
echo "Done. Summary: runs/learned_evidence/SUMMARY.md. You can close this window."
