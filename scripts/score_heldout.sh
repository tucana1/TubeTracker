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
#   0.9.0-tiptraj-selftrained  tag sparsetrack-0.9.0-selftrained-candidate: the same reader, detector and weights on
#                  tube maps adapted to the movie first, label-free: `sparsetrack selftrain` on the movie's cache
#                  (sparsetrack/selftrain.py with its settings at that tag: the shipped network read on every grain
#                  of the cache's census, confident pseudo-traces, traced + propagated crops, one fine-tune of 1500
#                  steps, seed 0, one run, replay of the shipped network's own training shards below), then the
#                  analysis with the adapted network as Params.model (flood, centring and reader all read its maps).
#                  Estimates (prototypes/self_train/README.md, "The step"): m1 28/50 lengths, onsets 15-17/28 (seeds
#                  0, 1; the reader on the shipped maps 27/50, 12/28); m2, from a network that never saw m2, 24, 31,
#                  31/54 for seeds 0-2 (on that network's own maps 28/54): the training seed moves m2 by up to 7.
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
OUT2=runs/research/heldout_${KEY}_0.9.0_tiptraj_selftrained
[ -d ../tt-0.9.0-selftrained ] || git worktree add ../tt-0.9.0-selftrained sparsetrack-0.9.0-selftrained-candidate
WT2=$REPO/../tt-0.9.0-selftrained
DET2=$WT2/sparsetrack/models/tips_v3_all.pt
W2=$WT2/prototypes/tip_trajectory/weights_ld_v3.json
NET0=$WT2/sparsetrack/models/tubes_bn_real_ld_m2.pt
[ -f $WT2/sparsetrack/selftrain.py ] || { echo "The self-trained candidate's worktree has no sparsetrack/selftrain.py."; exit 1; }
[ "$(shasum $DET2 | cut -c1-40)" = "87fcc28f2267e5f8cda51d77fdd489b5b2f82c71" ] || { echo "The self-trained candidate's tip detector changed since it was fixed."; exit 1; }
[ "$(shasum $W2 | cut -c1-40)" = "613795a1abf19718264e9f57fd30d476ef33058b" ] || { echo "The self-trained candidate's weights changed since they were fixed."; exit 1; }
[ "$(shasum $NET0 | cut -c1-40)" = "91cb95715eae5122d3f70eb89aa7246ce5b34aaa" ] || { echo "The starting tube network changed since it was fixed."; exit 1; }
# replay: the shipped network's own training data (its ld and movie 2 trace crops, one v5 and one v6 synthetic shard)
REPLAY=()
for f_sha in runs/tube_net/shards/real3_ld.npz:5696dec8587f04afa4919dbb0fb56ef482746592 \
             runs/tube_net/shards/real3_m2.npz:4c5ae2d5076a1479bc36acb6d0711c44302c47f4 \
             runs/learned_flood/shards/train_v5m2s10.npz:2fffb9a8067c1d272d00e297a8e67ffe90df8ddf \
             runs/synth_v6/shards/train_v6m1_s50.npz:2991b701c6f43e6a7e815ddedf493a0e5e734562; do
  f=${f_sha%%:*}
  [ "$(shasum $f | cut -c1-40)" = "${f_sha##*:}" ] || { echo "Replay shard $f changed since it was fixed."; exit 1; }
  REPLAY+=($REPO/$f)
done
if [ ! -f $OUT2/selftrained.pt ]; then
  echo "== self-training the tube network on $KEY (label-free: the cache's census, no labels)"
  (cd $WT2 && $REPO/.venv/bin/python -m sparsetrack selftrain $REPO/$CACHE --out $REPO/$OUT2/selftrained.pt \
      --steps 1500 --seed 0 --runs 1 --replay $REPLAY)
fi
if [ ! -f $OUT2/predictions.json ]; then
  echo "== SparseTrack 0.9.0 candidate (tip-trajectory reader) on the self-trained maps on $KEY"
  (cd $WT2 && $REPO/.venv/bin/python -c "
from sparsetrack.analyze import Params, analyze
analyze('$REPO/$CACHE', '$REPO/$OUT2', grains_path='$REPO/$LABELS',
        params=Params(model='$REPO/$OUT2/selftrained.pt', tiptraj='flood', tiptraj_model='$DET2',
                      tiptraj_weights='$W2', tiptraj_mid=True))")
fi
.venv/bin/python scripts/compare_predictions.py --labels $LABELS --baseline 0.8.8 \
    --pred 0.8.8=$OUT0/predictions.json \
    --pred 0.9.0-tiptraj=$OUT1/predictions.json \
    --pred 0.9.0-tiptraj-selftrained=$OUT2/predictions.json \
    --out benchmark/reports/heldout_${KEY}_frozen.md
echo "Report: benchmark/reports/heldout_${KEY}_frozen.md"
