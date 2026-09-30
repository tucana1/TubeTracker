#!/bin/zsh
# Score the versions fixed before movie 1's blind labels existed, once each, and compare them (paired over grains).
# Run once, after benchmark/labels/m1_v1.json is finished (scripts/score_movie1.sh --final); do not tune
# anything on movie 1 afterwards.
#   0.6.0          tag sparsetrack-0.6.0
#   0.7.0          tag sparsetrack-0.7.0 (grains followed, tips continued)
#   0.7.0+real     0.7.0 with the tube network fine-tuned on ld and m2 traces (prototypes/learned_flood/models/)
#   0.8.0          = tag sparsetrack-0.7.0-bn (0.7.0 + the BatchNorm / local-background checkpoint options) with the
#                  network retrained on those fixes and both movies' traces (prototypes/tube_net/README.md): 0.8.0's
#                  default network, same readings
#   0.8.0+radial   tag sparsetrack-0.7.0-bn-radial with the same network and flood_tip="radial"
# Each version is analysed from its own worktree (../tt-<version>), so it runs its own code.
set -e
cd "$(dirname "$0")/.."
REPO=$PWD
LABELS=benchmark/labels/m1_v1.json
CACHE=runs/sparsetrack/m1
[ -f $LABELS ] || { echo "No $LABELS yet: label movie 1 first (Label_Movie1_Heldout.command)."; exit 1; }
# the labelling tool creates the labels file when labelling starts: score only when told the labelling is finished,
# and only if every sampled grain has its onset and every planned trace answered
[ "$1" = "--final" ] || { echo "Run as scripts/score_movie1.sh --final once movie 1's labelling is finished."; exit 1; }
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
    print(f"Movie 1's labelling is not finished ({len(todo)} grains):", *todo[:12], sep="\n  ")
    sys.exit(1)
PY
for v in 0.6.0 0.7.0; do
  [ -d ../tt-$v ] || git worktree add ../tt-$v sparsetrack-$v
  OUT=runs/research/m1_frozen_$v
  if [ ! -f $OUT/predictions.json ]; then
    echo "== SparseTrack $v on movie 1"
    (cd ../tt-$v && $REPO/.venv/bin/python -m sparsetrack analyze $REPO/$CACHE --out $REPO/$OUT --grains $REPO/$LABELS)
  fi
done
MODEL=$REPO/prototypes/learned_flood/models/tubes_real_ld_m2.pt
[ "$(shasum $MODEL | cut -c1-40)" = "e8c14145b0663d2d29e4f73836bf951e21078753" ] || { echo "The candidate network changed since it was fixed."; exit 1; }
OUT=runs/research/m1_frozen_0.7.0_real
if [ ! -f $OUT/predictions.json ]; then
  echo "== SparseTrack 0.7.0 with the real-trace network on movie 1"
  (cd ../tt-0.7.0 && $REPO/.venv/bin/python -c "
from sparsetrack.analyze import Params, analyze
analyze('$REPO/$CACHE', '$REPO/$OUT', grains_path='$REPO/$LABELS', params=Params(model='$MODEL'))")
fi
[ -d ../tt-0.7.0-bn ] || git worktree add ../tt-0.7.0-bn sparsetrack-0.7.0-bn
MODEL=$REPO/prototypes/learned_flood/models/tubes_bn_real_ld_m2.pt
[ "$(shasum $MODEL | cut -c1-40)" = "91cb95715eae5122d3f70eb89aa7246ce5b34aaa" ] || { echo "The BatchNorm candidate network changed since it was fixed."; exit 1; }
OUT=runs/research/m1_frozen_0.7.0_bn
if [ ! -f $OUT/predictions.json ]; then
  echo "== SparseTrack 0.7.0-bn with the BatchNorm real-trace network on movie 1"
  (cd ../tt-0.7.0-bn && $REPO/.venv/bin/python -c "
from sparsetrack.analyze import Params, analyze
analyze('$REPO/$CACHE', '$REPO/$OUT', grains_path='$REPO/$LABELS', params=Params(model='$MODEL'))")
fi
[ -d ../tt-0.7.0-bn-radial ] || git worktree add ../tt-0.7.0-bn-radial sparsetrack-0.7.0-bn-radial
OUT=runs/research/m1_frozen_0.7.0_bn_radial
if [ ! -f $OUT/predictions.json ]; then
  echo "== SparseTrack 0.7.0-bn-radial (the BatchNorm network, radial flood tip) on movie 1"
  (cd ../tt-0.7.0-bn-radial && $REPO/.venv/bin/python -c "
from sparsetrack.analyze import Params, analyze
analyze('$REPO/$CACHE', '$REPO/$OUT', grains_path='$REPO/$LABELS', params=Params(model='$MODEL', flood_tip='radial'))")
fi
.venv/bin/python scripts/compare_predictions.py --labels $LABELS --baseline 0.6.0 \
    --pred 0.6.0=runs/research/m1_frozen_0.6.0/predictions.json \
    --pred 0.7.0=runs/research/m1_frozen_0.7.0/predictions.json \
    --pred 0.7.0+real=runs/research/m1_frozen_0.7.0_real/predictions.json \
    --pred 0.8.0=runs/research/m1_frozen_0.7.0_bn/predictions.json \
    --pred 0.8.0+radial=runs/research/m1_frozen_0.7.0_bn_radial/predictions.json \
    --out benchmark/reports/m1_v1_frozen.md
echo "Report: benchmark/reports/m1_v1_frozen.md"
