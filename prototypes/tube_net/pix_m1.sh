#!/bin/zsh
# Pixel checks on movie 1 (in memory; nothing is written to its cache) for the networks that did not see its traces:
# 0.8.0's network (ld + m2 traces: the three-movie fold for movie 1), the two single-movie folds of the two-movie
# leave-one-out (tn_bn_r3v6_ld / _m2) and 0.7.0's synthetic-only network. Output: runs/tube_net/pix3_*_on_m1.json.
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
run() {  # MODEL BG
  $PY -u -m prototypes.tube_net.pixels "$1" m1 --bg "$2" --json "runs/tube_net/pix3_$(basename "$1" .pt)_on_m1.json" --grains \
      2>&1 | grep -v "^  m1:"
}
run sparsetrack/models/tubes_bn_real_ld_m2.pt 96
run runs/tube_net/tn_bn_r3v6_ld.pt 96
run runs/tube_net/tn_bn_r3v6_m2.pt 96
run sparsetrack/models/tubes_synth_v1.pt 0
