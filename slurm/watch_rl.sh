#!/bin/bash
# Print one line per state change of RL jobs: job state, step 1 / step 5 reached, first error, exit.
#   bash slurm/watch_rl.sh <jobid>:<run_id> [<jobid>:<run_id> ...]
# Exits when every run has reached step 5 or its job has left the queue.
D="${EDGE_SCRATCH:-/scratch/ebasaran22/chasing-the-edge}/outputs/runs"
SEEN=" "
emit() { case "$SEEN" in *" $1 "*) ;; *) SEEN="$SEEN$1 "; echo "$2";; esac; }
while true; do
  left=0
  for p in "$@"; do
    j="${p%%:*}"; r="${p#*:}"; L="$D/$r/train.log"
    st=$(squeue -h -j "$j" -o "%T %N" 2>/dev/null)
    if [ -z "$st" ]; then st="ENDED $(sacct -X -n -j "$j" -o State | head -1 | tr -d ' ')"; fi
    emit "$j:${st// /_}" "$r job $j: $st"
    if [ -f "$L" ]; then
      for s in 1 5; do
        if grep -q "step:$s - " "$L"; then emit "$r:s$s" "$r reached step $s at $(date +%T)"; fi
      done
      e=$(grep -m1 -E "Traceback|Error|WATCHDOG|OutOfMemory|Killed" "$L")
      if [ -n "$e" ]; then emit "$r:err" "$r ERROR: ${e:0:200}"; fi
    fi
    case "$SEEN" in *" $r:s5 "*|*" $j:ENDED"*) ;; *) left=$((left + 1));; esac
  done
  [ "$left" -eq 0 ] && exit 0
  sleep 30
done
