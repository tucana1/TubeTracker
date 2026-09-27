#!/bin/bash
# M1 "fds": the repository's recipe (train.py: 6000 steps, batch 32, lr 2e-3 OneCycle, widths 16-128, seed 0, crop 64)
# from scratch on v2's ten shards + my six (4 faint, 2 intermediate), 2 threads; then its first look (faint s30, s31,
# thin s3-s6) in two lanes. Resumes from the checkpoint after a restart.
ME=/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata
SCR=/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad
PY=/home/user/TubeTracker/.venv/bin/python
until grep -q GEN_ALL_DONE $ME/logs/gen_a.log 2>/dev/null; do sleep 30; done
n=$(ls $ME/shards/train_*.npz | grep -vc "\.tmp\.npz")
[ "$n" -eq 6 ] || { echo "expected 6 shards, found $n" >> $ME/logs/train_fds.log; exit 1; }
M=$ME/models/unet_fd_scratch.pt
if [ ! -f $M ]; then
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY $ME/train5.py --shards "$SCR/le/shards/train_*.npz" "$ME/shards/train_*.npz" \
    --out $M --steps 6000 --threads 2 >> $ME/logs/train_fds.log 2>&1
  echo TRAIN_EXIT $? >> $ME/logs/train_fds.log
fi
[ -f $M ] && $ME/run_eval.sh fds $M v5faints30,v5s3,v5s5 v5faints31,v5s4,v5s6
