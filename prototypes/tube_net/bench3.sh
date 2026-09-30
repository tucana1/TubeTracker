#!/bin/zsh
# End-to-end benches of the three-movie leave-one-out (30 Sep 2026), one at a time, maps in memory (bench.py): each
# movie read by SparseTrack 0.8.0's defaults with only the tube network changed, into runs/tube_net/e2e3_*.json.
#   m2: tn3_ldm1 (ld + m1 traces) against tn_bn_r3v6_ld (ld traces only: the two-movie fold)
#   ld: tn3_m2m1 (m2 + m1 traces) against tn_bn_r3v6_m2 (m2 traces only)
#   m1: 0.8.0's network (ld + m2 traces: the three-movie fold for movie 1) against the single-movie folds; primary
#       labels (as traced) and secondary (tubes over the grain measured from the grain's edge: edge_labels.py,
#       the same predictions rescored: rescore.py)
#   the flood's radial tip on the three-movie fold maps (ld, m2; movie 1's is in its frozen report)
# Movie 1 only once its frozen report exists (benchmark/reports/m1_v1_frozen.md): until then nothing may be run on it.
# Usage: bench3.sh (runs every bench whose output is missing; rerun it once the frozen report exists)
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
REPORT=${REPORT:-benchmark/reports/m1_v1_frozen.md}
EDGE=runs/tube_net/m1_v1_edge.json
bench() {  # NAME MODEL MOVIE [extra bench.py args]
  local out=runs/tube_net/e2e3_$1.json
  [ -f $out ] && return 0
  if [ "$3" = m1 ] && [ ! -f $REPORT ]; then echo "skip $1: movie 1's frozen report does not exist yet"; return 0; fi
  if [ ! -f $2 ]; then echo "skip $1: no $2 yet"; return 0; fi
  echo "== $1"
  $PY -u -m prototypes.tube_net.bench $2 $3 --out $out "${@[4,-1]}" 2>&1 | grep -v "^  m[12]: \|^  ld: " \
      | tee runs/tube_net/logs/bench3_$1.log
}
[ -f $EDGE ] || $PY -m prototypes.tube_net.edge_labels --out $EDGE
bench tn3_ldm1_on_m2    runs/tube_net/tn3_ldm1.pt      m2
bench tn3_m2m1_on_ld    runs/tube_net/tn3_m2m1.pt      ld
bench r3v6_m2_on_ld     runs/tube_net/tn_bn_r3v6_m2.pt ld --bg 96   # the two-movie fold again at HEAD (its record:
                                                                     # runs/tube_net/e2e_tn_bn_r3v6_m2_bg96_on_ld.json)
bench r3v6_ld_on_m1     runs/tube_net/tn_bn_r3v6_ld.pt m1 --bg 96
bench r3v6_m2_on_m1     runs/tube_net/tn_bn_r3v6_m2.pt m1 --bg 96
# the flood's radial tip (0.8.0+radial, pre-registered for movie 1: its frozen report) on the three-movie fold maps
bench tn3_ldm1_on_m2_radial runs/tube_net/tn3_ldm1.pt  m2 --set flood_tip=radial
bench tn3_m2m1_on_ld_radial runs/tube_net/tn3_m2m1.pt  ld --set flood_tip=radial
# last: the two-movie fold on m2 and 0.8.0 on m1 again at HEAD (their records: e2e_tn_bn_r3v6_ld_bg96_on_m2.json and
# the frozen 0.8.0 predictions, runs/sparsetrack/m1_frozen_0.7.0_bn)
bench r3v6_ld_on_m2     runs/tube_net/tn_bn_r3v6_ld.pt m2 --bg 96
bench v080_on_m1        sparsetrack/models/tubes_bn_real_ld_m2.pt m1
for n in v080_on_m1 r3v6_ld_on_m1 r3v6_m2_on_m1; do
  [ -f runs/tube_net/e2e3_$n.json ] && [ ! -f runs/tube_net/e2e3_${n}edge.json ] && \
    $PY -m prototypes.tube_net.rescore runs/tube_net/e2e3_$n.json m1=$EDGE --out runs/tube_net/e2e3_${n}edge.json
done
exit 0
