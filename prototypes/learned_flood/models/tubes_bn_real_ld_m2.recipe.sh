#!/bin/zsh
# How tubes_bn_real_ld_m2.pt was made (30 Sep 2026; fixed before movie 1's tube labels existed - do not retrain it).
#
# The recipe that did best leave-one-movie-out in prototypes/tube_net/README.md ("tn_bn_r3v6"), trained once on BOTH
# labelled movies' traces. Differences from the earlier candidate tubes_real_ld_m2.pt (the unet_d recipe):
#   - BatchNorm instead of GroupNorm (a pixel's output depends only on its surroundings, not on the whole frame);
#   - inputs relative to the "before" image's local median over 96 px (checkpoint field bg_px = 96), not the frame's
#     median: movie 2's illumination varies by -77..+38 grey levels across the frame;
#   - trace crops v3 (prototypes/tube_net/realdata.py --flat-cap): the body ends at the clicked apex, grain masks at
#     the grain's place at that bin, no background band on touching traces, exit-centred crops, neighbouring bins,
#     negatives at several bins before onset;
#   - v6 synthetic shards (bulbs, rounded tips, the grain's own change) in half of the synthetic share.
# sha1 of the file this script made: 91cb95715eae5122d3f70eb89aa7246ce5b34aaa.
# Loading it needs sparsetrack/learned.py with the ``norm`` and ``bg_px`` checkpoint options (commit ba89c96, "learned:
# checkpoint options for BatchNorm networks and inputs relative to the local background"); SparseTrack 0.7.0 as tagged
# refuses the file (BatchNorm keys). Scoring on movie 1: prototypes/tube_net/README.md, "Candidate record".
# MPS training is not bit-reproducible: this script rebuilds an equivalent network, not the same file.
set -e
cd "$(dirname "$0")/../../.."  # the repository root
PY=${PY:-.venv/bin/python}
V5="runs/learned_flood/shards/train_v5s[0-2].npz,runs/learned_flood/shards/train_v5m2s1[0-2].npz"  # tubes_synth_v1's shards
V6="runs/synth_v6/shards/train_v6*.npz"                                                             # prototypes/synth_v6
# 1. the base: tubes_synth_v1's recipe (prototypes/learned_flood/recipe.py; 8000 steps, lr 2e-3) with BatchNorm
[ -f runs/tube_net/tn_bn_syn.pt ] || $PY -u -m prototypes.tube_net.train --out runs/tube_net/tn_bn_syn.pt \
    --norm batch --steps 8000 --lr 2e-3 --data "$V5=1"
# 2. trace crops of both labelled movies, flat caps
for mv in ld m2; do
  [ -f runs/tube_net/shards/real3_$mv.npz ] || $PY -m prototypes.tube_net.realdata $mv runs/tube_net/shards/real3_$mv.npz --flat-cap
done
# 3. fine-tune: 25% v5, 25% v6, 50% trace crops of both movies (8 + 8 + 16 of each batch of 32); 3000 steps, lr 1e-3
$PY -u -m prototypes.tube_net.train --out prototypes/learned_flood/models/tubes_bn_real_ld_m2.pt \
    --init runs/tube_net/tn_bn_syn.pt --norm batch --steps 3000 --lr 1e-3 --bg-px 96 \
    --data "$V5=0.25" "$V6=0.25" "runs/tube_net/shards/real3_ld.npz,runs/tube_net/shards/real3_m2.npz=0.5"
