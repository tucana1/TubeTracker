#!/bin/zsh
# Stage A of the four-movie study: the wild type's training data, and the leave-one-movie-out networks.
#   prototypes/four_movies/train.sh
# Tip detector "version 4" = version 3's recipe (prototypes/tip_detector/README.md) with the wild type's traces;
# tube network "tn4" = 0.8.0's recipe (prototypes/learned_flood/models/tubes_bn_real_ld_m2.recipe.sh) with three
# movies' traces per fold. A fold's name lists the movies it was trained on; the movie it never saw is read with it.
# Existing outputs are kept (delete a file to rebuild it). MPS training is not bit-reproducible.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
TD=runs/research/tip_detector
[ -f $TD/offsets_wt.npz ] || $PY -m prototypes.tip_detector.offsets wt
[ -f $TD/samples3_wt.npy ] || $PY -m prototypes.tip_detector.build3 wt
for fold in "m2 m1 wt" "ld m1 wt" "ld m2 wt" "ld m2 m1 wt"; do
  name=tip4_${fold// /}; [ "$fold" = "ld m2 m1 wt" ] && name=tip4_all
  [ -f $TD/$name.pt ] || $PY -m prototypes.tip_detector.train3 --train ${=fold} --name $name --shift --long-w 2 --protect
done
V5="runs/learned_flood/shards/train_v5s[0-2].npz,runs/learned_flood/shards/train_v5m2s1[0-2].npz"
V6="runs/synth_v6/shards/train_v6*.npz"
# the wild type's trace crops, with movie 1's options (tubes from a pore facing the camera; grains that move)
[ -f runs/tube_net/shards/real3_wt.npz ] || $PY -m prototypes.tube_net.realdata wt runs/tube_net/shards/real3_wt.npz \
    --flat-cap --over-grain 0.6 --follow
for fold in "ld m2 wt" "ld m1 wt" "m2 m1 wt" "ld m2 m1 wt"; do
  name=tn4_${fold// /}; [ "$fold" = "ld m2 m1 wt" ] && name=tn4_all
  shards=""; for m in ${=fold}; do shards="$shards,runs/tube_net/shards/real3_$m.npz"; done
  [ -f runs/tube_net/$name.pt ] || $PY -u -m prototypes.tube_net.train --out runs/tube_net/$name.pt \
      --init runs/tube_net/tn_bn_syn.pt --norm batch --steps 3000 --lr 1e-3 --bg-px 96 --seed 0 \
      --data "$V5=0.25" "$V6=0.25" "${shards#,}=0.5"
done
echo "stage A done"
