#!/bin/bash
# Network pass + fused decode of one model on development movies in two lanes (1 thread each).
#   run_eval.sh TAG MODEL_PT LANE1_MOVIES LANE2_MOVIES      (comma-separated movie lists)
ME=/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata
PY=/home/user/TubeTracker/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TAG=$1; M=$2
( $PY $ME/evalnet.py $TAG $M $3 || echo "FAILED lane1"; echo LANE1_DONE ) >> $ME/logs/eval_${TAG}_1.log 2>&1 &
[ -n "$4" ] && ( $PY $ME/evalnet.py $TAG $M $4 || echo "FAILED lane2"; echo LANE2_DONE ) >> $ME/logs/eval_${TAG}_2.log 2>&1 &
wait
echo EVAL_ALL_DONE >> $ME/logs/eval_${TAG}_1.log
