#!/bin/bash
# Two lanes (1 thread each): 4 faint (amplitude 0.35-0.7) and 2 intermediate (0.5-1.0) training movies, seeds 50-55.
ME=/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata
PY=/home/user/TubeTracker/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
( $PY $ME/gen.py v5faint 50 52 && $PY $ME/gen.py v5mid 54; echo LANE_A_DONE ) >> $ME/logs/gen_a.log 2>&1 &
( $PY $ME/gen.py v5faint 51 53 && $PY $ME/gen.py v5mid 55; echo LANE_B_DONE ) >> $ME/logs/gen_b.log 2>&1 &
wait
echo GEN_ALL_DONE >> $ME/logs/gen_a.log
