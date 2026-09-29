#!/bin/bash
# workers.sh start N | stop | restart N : my TTA workers (one thread each), PIDs in logs/worker<i>.pid.
# stop releases the claims of unfinished jobs (their chunks stay, so a pass resumes where it stopped).
ME=/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/tta
PY=/home/user/TubeTracker/.venv/bin/python
stop() {
  for f in "$ME"/logs/worker*.pid; do
    [ -f "$f" ] || continue
    pid=$(cat "$f"); kill "$pid" 2>/dev/null && echo "stopped $pid"; rm -f "$f"
  done
  sleep 3
  # release claims whose output does not exist (pass: sparse/<model>_x<t>_<movie>.npz; decode: preds/<movie>/<tag>.json)
  for c in "$ME"/claims/*; do
    [ -d "$c" ] || continue
    [ -f "$c/FAILED" ] && continue
    IFS=_ read -r kind a b rest <<< "$(basename "$c")"
    if [ "$kind" = pass ]; then
      out="$ME/sparse/${a}_x${b}_${rest}.npz"
      [ "$b" = e ] && out="$ME/sparse/${a}_xe_${rest}.npz"
    elif [ "$kind" = decode ]; then
      name=$(basename "$c"); name=${name#decode_}; movie=${name##*_}; tag=${name%_*}
      out="$ME/preds/$movie/$tag.json"
    else
      out="$ME/logs/timing_pipeline_${a}.json"
    fi
    [ -f "$out" ] || { rmdir "$c" && echo "released $(basename "$c")"; }
  done
}
start() {
  n=${1:-2}
  for i in $(seq 1 "$n"); do
    nohup $PY "$ME/tta.py" worker "$ME/queue.txt" --threads 1 >> "$ME/logs/worker$i.log" 2>&1 &
    echo $! > "$ME/logs/worker$i.pid"; renice -n 0 -p $! > /dev/null; echo "started worker$i $!"
    sleep 2
  done
}
case "$1" in
  stop) stop ;;
  start) start "$2" ;;
  restart) stop; start "$2" ;;
esac
