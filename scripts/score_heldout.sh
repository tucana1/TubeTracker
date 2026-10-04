#!/bin/zsh
# Score the versions fixed BEFORE a new movie's blind labels existed, once each, and compare them (paired over grains).
#   scripts/score_heldout.sh <key> --final
# <key> as Label_Heldout_Movie.command named the movie (labels benchmark/labels/heldout_<key>_v1.json, cache
# runs/sparsetrack/heldout_<key>). Run it once, after the labelling is finished; tune nothing on that movie afterwards.
#
# Candidates, fixed 4 Oct 2026 (no fourth movie existed yet):
#   0.8.8          tag sparsetrack-0.8.8: the shipped tracker.
#   0.9.0-tiptraj  tag sparsetrack-0.9.0-candidate: 0.8.8 plus the global tip-trajectory reader on the grains the hybrid
#                  floods (tiptraj="flood"), tip detector sparsetrack/models/tips_v3_all.pt (trained on ld, m2 and
#                  m1), weights prototypes/tip_trajectory/weights_ld_v3.json (tuned on ld), lengths along the tube's
#                  middle (tiptraj_mid). Its leave-one-movie-out estimates: ld 77/104, m2 37/54, m1 27/50 traced
#                  lengths in tolerance (0.8.8: 75, 27, 15).
# Each version is analysed from its own worktree (../tt-<version>), so it runs its own code and files.
set -e
cd "$(dirname "$0")/.."
REPO=$PWD
KEY=$1
[ -n "$KEY" ] || { echo "usage: scripts/score_heldout.sh <movie key> --final"; exit 1; }
LABELS=benchmark/labels/heldout_${KEY}_v1.json
CACHE=runs/sparsetrack/heldout_${KEY}
[ -f $LABELS ] || { echo "No $LABELS yet: label the movie first (Label_Heldout_Movie.command)."; exit 1; }
[ -f $CACHE/meta.json ] || { echo "No cache at $CACHE."; exit 1; }
# the labelling tool creates the labels file when labelling starts: score only when told the labelling is finished,
# and only if every sampled grain has its onset and every planned trace answered
[ "$2" = "--final" ] || { echo "Run as scripts/score_heldout.sh $KEY --final once the labelling is finished."; exit 1; }
.venv/bin/python - "$LABELS" <<'PY' || exit 1
import json, sys
from sparsetrack.bench.server import trace_bins
d = json.load(open(sys.argv[1]))
nb, todo = int(d["n_bins"]), []
for gid, g in sorted(d["grains"].items()):
    if g.get("excluded"):
        continue
    lab = d["labels"].get(gid) or {}
    on = lab.get("onset") or {}
    if not on.get("verdict"):
        todo.append(f"{gid}: no onset answer")
    elif on["verdict"] in ("emerged_within", "emerged_at_start"):
        missing = [b for b in trace_bins(on.get("first_visible_bin") or 0, nb) if str(b) not in (lab.get("traces") or {})]
        if missing:
            todo.append(f"{gid}: traces not answered at bins {missing}")
if todo:
    print(f"The labelling is not finished ({len(todo)} grains):", *todo[:12], sep="\n  ")
    sys.exit(1)
PY
OUT0=runs/research/heldout_${KEY}_0.8.8
[ -d ../tt-0.8.8 ] || git worktree add ../tt-0.8.8 sparsetrack-0.8.8
if [ ! -f $OUT0/predictions.json ]; then
  echo "== SparseTrack 0.8.8 on $KEY"
  (cd ../tt-0.8.8 && $REPO/.venv/bin/python -m sparsetrack analyze $REPO/$CACHE --out $REPO/$OUT0 --grains $REPO/$LABELS)
fi
OUT1=runs/research/heldout_${KEY}_0.9.0_tiptraj
[ -d ../tt-0.9.0-candidate ] || git worktree add ../tt-0.9.0-candidate sparsetrack-0.9.0-candidate
DET=$REPO/../tt-0.9.0-candidate/sparsetrack/models/tips_v3_all.pt
W=$REPO/../tt-0.9.0-candidate/prototypes/tip_trajectory/weights_ld_v3.json
[ "$(shasum $DET | cut -c1-40)" = "87fcc28f2267e5f8cda51d77fdd489b5b2f82c71" ] || { echo "The candidate's tip detector changed since it was fixed."; exit 1; }
[ "$(shasum $W | cut -c1-40)" = "613795a1abf19718264e9f57fd30d476ef33058b" ] || { echo "The candidate's weights changed since they were fixed."; exit 1; }
if [ ! -f $OUT1/predictions.json ]; then
  echo "== SparseTrack 0.9.0 candidate (tip-trajectory reader) on $KEY"
  (cd ../tt-0.9.0-candidate && $REPO/.venv/bin/python -c "
from sparsetrack.analyze import Params, analyze
analyze('$REPO/$CACHE', '$REPO/$OUT1', grains_path='$REPO/$LABELS',
        params=Params(tiptraj='flood', tiptraj_model='$DET', tiptraj_weights='$W', tiptraj_mid=True))")
fi
.venv/bin/python scripts/compare_predictions.py --labels $LABELS --baseline 0.8.8 \
    --pred 0.8.8=$OUT0/predictions.json \
    --pred 0.9.0-tiptraj=$OUT1/predictions.json \
    --out benchmark/reports/heldout_${KEY}_frozen.md
echo "Report: benchmark/reports/heldout_${KEY}_frozen.md"
