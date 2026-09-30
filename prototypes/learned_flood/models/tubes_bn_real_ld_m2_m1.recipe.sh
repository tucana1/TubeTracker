#!/bin/zsh
# How tubes_bn_real_ld_m2_m1.pt was made (30 Sep 2026): 0.8.0's recipe (tubes_bn_real_ld_m2.recipe.sh) with the traces
# of all three labelled movies - the dev movie ld, movie 2 (m2) and movie 1 (m1, labelled blind 30 Sep and scored once,
# frozen, before any of this: benchmark/reports/m1_v1_frozen.md).
#
# Judged leave one movie out over the three movies (prototypes/tube_net/README.md, "Three movies"): each movie read
# with the same recipe trained on the other two movies' traces. There is no fourth movie: the leave-one-out numbers
# are its estimate, and all three movies are development sets for it.
#
# Differences from tubes_bn_real_ld_m2.pt: movie 1's trace crops join the trace half of each batch
# (prototypes/tube_net/realdata.py m1 --flat-cap --over-grain 0.6 --follow: the 6 traces that start inside their
# grain - tubes from a pore facing the camera - leave the grain's inside unscored; movie 1's grains move, so a
# neighbouring-bin crop is skipped where the grain moved > 1.5 px and the negatives before onset are placed where the
# grain is). ld's and m2's crops are exactly 0.8.0's (real3_ld / real3_m2, rebuilt identically by the same script).
# The three movies' crops are pooled: each movie counts by its number of crops (ld 1508, m2 866, m1 1143).
# sha1 of the file this script made: e2bcf871827ac996bcbf15d4d3110f860206054d (log runs/tube_net/logs/
# train_tubes_bn_real_ld_m2_m1.log; synthetic validation at the end: v5 loss 0.237, recall 76.1%, false marks 0.13%;
# v6 0.168, 85.2%, 0.15%). Checkpoint fields: norm "batch", bg_px 96. Not to be changed.
# MPS training is not bit-reproducible: this script rebuilds an equivalent network, not the same file.
set -e
cd "$(dirname "$0")/../../.."  # the repository root
PY=${PY:-.venv/bin/python}
V5="runs/learned_flood/shards/train_v5s[0-2].npz,runs/learned_flood/shards/train_v5m2s1[0-2].npz"  # tubes_synth_v1's shards
V6="runs/synth_v6/shards/train_v6*.npz"                                                             # prototypes/synth_v6
# 1. the base: tubes_synth_v1's recipe with BatchNorm (as for 0.8.0)
[ -f runs/tube_net/tn_bn_syn.pt ] || $PY -u -m prototypes.tube_net.train --out runs/tube_net/tn_bn_syn.pt \
    --norm batch --steps 8000 --lr 2e-3 --data "$V5=1"
# 2. trace crops of the three labelled movies, flat caps (movie 1 with its two options)
for mv in ld m2; do
  [ -f runs/tube_net/shards/real3_$mv.npz ] || $PY -m prototypes.tube_net.realdata $mv runs/tube_net/shards/real3_$mv.npz --flat-cap
done
[ -f runs/tube_net/shards/real3_m1.npz ] || $PY -m prototypes.tube_net.realdata m1 runs/tube_net/shards/real3_m1.npz \
    --flat-cap --over-grain 0.6 --follow
# 3. fine-tune: 25% v5, 25% v6, 50% trace crops of the three movies; 3000 steps, lr 1e-3; bg_px 96 in the checkpoint
$PY -u -m prototypes.tube_net.train --out prototypes/learned_flood/models/tubes_bn_real_ld_m2_m1.pt \
    --init runs/tube_net/tn_bn_syn.pt --norm batch --steps 3000 --lr 1e-3 --bg-px 96 --seed 0 \
    --data "$V5=0.25" "$V6=0.25" \
           "runs/tube_net/shards/real3_ld.npz,runs/tube_net/shards/real3_m2.npz,runs/tube_net/shards/real3_m1.npz=0.5"
