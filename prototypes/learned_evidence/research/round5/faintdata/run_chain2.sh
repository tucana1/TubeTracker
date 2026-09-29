#!/bin/bash
# Under the shared machine's load a 2-thread training ran at 1.5-3 s/step (OpenMP barriers on an oversubscribed
# machine), so my two threads are two 1-thread processes:
#   M1 "fds"  : from scratch, the repository's recipe (6000 steps, lr 2e-3), resumed from its step-250 checkpoint;
#   M2 "fdft" : continued from v2 on the same 16 shards, 2000 steps, OneCycle peak lr 5e-4.
# When M2 is saved, M1 is paused (SIGSTOP) while M2 is evaluated on every development movie and the real movie in two
# lanes, then resumed (SIGCONT).
ME=/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata
SCR=/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad
PY=/home/user/TubeTracker/.venv/bin/python
V2=/home/user/TubeTracker/prototypes/learned_evidence/models/unet_v2_sample_field.pt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
M1=$ME/models/unet_fd_scratch.pt
M2=$ME/models/unet_fd_ft.pt
if [ ! -f $M1 ]; then
  ( $PY $ME/train5.py --shards "$SCR/le/shards/train_*.npz" "$ME/shards/train_*.npz" --out $M1 --steps 6000 \
      --threads 1 >> $ME/logs/train_fds.log 2>&1; echo TRAIN_EXIT $? >> $ME/logs/train_fds.log ) &
  sleep 5
  pgrep -f "train5.py --shards .* --out $M1" | head -1 > $ME/logs/m1.pid
fi
if [ ! -f $M2 ]; then
  $PY $ME/train5.py --shards "$SCR/le/shards/train_*.npz" "$ME/shards/train_*.npz" --out $M2 --steps 2000 --lr 5e-4 \
    --threads 1 --init $V2 >> $ME/logs/train_fdft.log 2>&1
  echo TRAIN_EXIT $? >> $ME/logs/train_fdft.log
fi
[ -f $M2 ] || exit 1
P1=$(cat $ME/logs/m1.pid 2>/dev/null)
[ -n "$P1" ] && kill -STOP $P1 2>/dev/null && echo "M1 paused (pid $P1) $(date)" >> $ME/logs/train_fds.log
$ME/run_eval.sh fdft $M2 v5faints30,v5s3,v5s5 v5faints31,v5s4,v5s6
$ME/run_eval.sh fdft $M2 real,v5s7,v5s13,v5thicks30,v5ws26 v5s8,v5s14,v5s15,v5thicks31,v5ws27
[ -n "$P1" ] && kill -CONT $P1 2>/dev/null && echo "M1 resumed (pid $P1) $(date)" >> $ME/logs/train_fds.log
echo CHAIN2_DONE >> $ME/logs/train_fdft.log
wait
