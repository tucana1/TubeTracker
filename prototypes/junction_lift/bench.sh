#!/bin/zsh
# usage: bench.sh TAG key=value ...   (runs m2, ld, m1 one after another, each once the 1-minute load is below 12)
WT=/Users/joshjiang/Documents/TubeTracker/.claude/worktrees/agent-acd9ac5f4741507c2
PY=/Users/joshjiang/Documents/TubeTracker/.venv/bin/python
SP=/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/scratchpad
BASE=$SP/bt/base088.json
tag=$1; shift
mkdir -p $SP/jt/bench/$tag
cd $WT
for mv in m2 ld m1; do
  until [ "$(sysctl -n vm.loadavg | awk '{print int($2)}')" -lt 12 ]; do sleep 20; done
  echo "== $mv start $(date +%H:%M:%S) load $(sysctl -n vm.loadavg)"
  $PY -u scripts/synth_bench.py --real $mv --no-synth --no-legacy --set "$@" \
      --dump-real $SP/jt/bench/$tag/$mv.json --baseline $BASE --work $SP/jt/bench/$tag/work 2>&1
  rc=$?; echo "== $mv done $(date +%H:%M:%S) exit $rc"
done
