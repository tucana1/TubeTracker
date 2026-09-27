#!/bin/bash
# One lane (1 thread): runs the lines of queue_<NAME>.txt in order as they appear (a line "STOP" ends it).
#   worker.sh NAME [START_MARKER_FILE START_MARKER_TEXT]   (waits for the marker before its first job)
ME=/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
N=$1; Q=$ME/queue_$N.txt; D=$ME/logs/worker_$N.done; L=$ME/logs/worker_$N.log
touch $Q $D
if [ -n "$2" ]; then until grep -q "$3" "$2" 2>/dev/null; do sleep 15; done; fi
while true; do
  i=$(( $(wc -l < $D) + 1 ))
  line=$(sed -n "${i}p" $Q)
  if [ -z "$line" ]; then sleep 20; continue; fi
  [ "$line" = "STOP" ] && { echo "WORKER_STOP $(date)" >> $L; exit 0; }
  echo "JOB_START $i $(date +%T): $line" >> $L
  bash -c "$line" >> $L 2>&1
  echo "JOB_END $i $(date +%T) exit $?" >> $L
  echo "$line" >> $D
done
