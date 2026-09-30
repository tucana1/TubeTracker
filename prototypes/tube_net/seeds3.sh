#!/bin/zsh
# Training noise (30 Sep 2026): MPS training is not bit-reproducible, and each leave-one-out network is one draw. The
# movie-2 fold trained again with another seed - ld traces only (tn_bn_r3v6_ld's recipe) and ld + m1 traces (tn3_ldm1's)
# - and pixel-checked on movie 2, to see how far two trainings of one recipe differ.
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
$PY -m prototypes.tube_net.loo3 tn3s1 --train ld --seed 1 --no-check
$PY -m prototypes.tube_net.loo3 tn3s1 --train ld m1 --seed 1 --no-check
for n in tn3s1_ld tn3s1_ldm1; do
  [ -f runs/tube_net/pix3_${n}_on_m2.json ] || \
    $PY -u -m prototypes.tube_net.pixels runs/tube_net/$n.pt m2 --json runs/tube_net/pix3_${n}_on_m2.json --grains \
        2>&1 | grep -v "^  m2: [0-9]* bins"
done
