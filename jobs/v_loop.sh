#!/bin/bash
# The review experiments of 2026-11-13 (second round) (jobs/v_items.py, queue_v, logs/v) in four-node interactive allocations, one at a
# time: an allocation that ends (time limit, node failure) is replaced by a new one until every run is done.
#   nohup setsid bash jobs/v_loop.sh > logs/v/loop.log 2>&1 &
# Stop: touch queue_v/stop (checked between allocations; the runs of the current allocation finish first).
# In each allocation: the one-node runs side by side (one runner per node), then the four-node runs.
# One interactive allocation only: the account may hold two and the other one is left to the other users.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
PY=$PP_ROOT/env/bin/python3
mkdir -p logs/v
echo $$ > queue_v/loop.pid

open_runs() { $PY jobs/v_run.py --count; }

allocate() {   # prints the job id of a running four-node interactive allocation
  local out job
  for try in $(seq 1 400); do
    [ -e queue_v/stop ] && return 1
    while [ "$(squeue -h -u $USER -q interactive -o %i 2>/dev/null | wc -l)" -ge 2 ]; do sleep 180; done
    out=$(salloc -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 04:00:00 --gpus-per-node=4 --no-shell -J pp_v 2>&1)
    job=$(echo "$out" | grep -o 'Granted job allocation [0-9]*' | awk '{print $4}')
    [ -n "$job" ] && break
    echo "$(date +%m-%dT%H:%M:%S) no allocation yet: $(echo "$out" | tail -1)" >&2
    sleep 180
  done
  [ -z "$job" ] && return 1
  for w in $(seq 1 60); do [ -n "$(squeue -h -j $job -t RUNNING -o %i 2>/dev/null)" ] && break; sleep 10; done
  echo $job
}

stuck=0
for round in $(seq 1 30); do
  before=$(open_runs)
  echo "$(date +%m-%dT%H:%M:%S) round $round: $before runs open"
  [ "$before" -le 0 ] && break
  [ -e queue_v/stop ] && { echo "stop file found"; break; }
  JOB=$(allocate) || { echo "no allocation"; break; }
  HOSTS=$(scontrol show hostnames $(squeue -h -j $JOB -o %N) | paste -sd,)
  echo "$(date +%m-%dT%H:%M:%S) allocation $JOB on $HOSTS"
  for h in ${HOSTS//,/ }; do $PY jobs/v_run.py --job $JOB --hosts $h > logs/v/runner1.$JOB.$h.log 2>&1 & done
  wait
  $PY jobs/v_run.py --job $JOB --hosts $HOSTS > logs/v/runner4.$JOB.log 2>&1
  [ -n "$(squeue -h -j $JOB -o %i 2>/dev/null)" ] && scancel $JOB
  after=$(open_runs)
  echo "$(date +%m-%dT%H:%M:%S) allocation $JOB done: $after runs open"
  if [ "$after" -ge "$before" ]; then stuck=$((stuck + 1)); else stuck=0; fi
  [ $stuck -ge 2 ] && { echo "no progress in two allocations; stopping"; break; }
done
echo "$(date +%m-%dT%H:%M:%S) loop finished: $(open_runs) runs open"
rm -f queue_v/loop.pid
