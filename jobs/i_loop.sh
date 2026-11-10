#!/bin/bash
# The runs on four nodes or fewer in interactive allocations (user, 2026-11-10: "16 GPU는 interactive 쓰면 됨"), in
# place of the regular jobs pp_w_l32, pp_p_g16, pp_p_g8 and pp_p_g1. One four-node, four-hour allocation at a time:
#   1) queue_g runs on four nodes (jobs/gw_run.py: the 1,024-qubit run m.eng.L32.1e-4.g16.t2hw),
#   2) paulistrings-rs on 16 GPUs (mg16), then on 8 GPUs (mg8),
#   3) paulistrings-rs on one node, the modes g1, g2 and g4 side by side on nodes 0, 1 and 2.
# When these are done, the n2count campaign continues with its own interactive allocations (jobs/n2_count.sh, pinned
# to the source and binaries of its campaign) until it is complete.
# One loop only: the account may hold two interactive allocations and the other one is left to the other users.
#   nohup setsid bash jobs/i_loop.sh > logs/i_loop.log 2>&1 &
# Stop: touch queue_g/stop_i (checked between allocations).
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
PY=$PP_ROOT/env/bin/python3
SMALL=mg16,mg8,g1,g2,g4
CAMPAIGN=full-20261109T060832Z
echo $$ > queue_g/i_loop.pid

open_p() {
  $PY - <<EOF
import sys; sys.path.insert(0, "jobs")
import p_drive as p
print(sum(1 for rid, m, c, x in p.runs() if m in "$SMALL".split(",") and p.record(rid) is None))
EOF
}

allocate() {   # prints the job id of a running four-node interactive allocation
  local out job
  for try in $(seq 1 400); do
    out=$(salloc -A m5320_g -C "gpu&hbm40g" -q interactive -N 4 -t 04:00:00 --gpus-per-node=4 --no-shell -J pp_i_small 2>&1)
    job=$(echo "$out" | grep -o 'Granted job allocation [0-9]*' | awk '{print $4}')
    [ -n "$job" ] && break
    echo "$out" | grep -q -E 'QOSMaxSubmitJobPerUserLimit|QOSMaxJobsPerUserLimit' || { echo "$out" >&2; sleep 600; continue; }
    [ $((try % 10)) -eq 1 ] && echo "$(date +%m-%dT%H:%M:%S) both interactive slots of the account are taken; waiting" >&2
    sleep 180
  done
  [ -z "$job" ] && return 1
  for w in $(seq 1 60); do [ -n "$(squeue -h -j $job -t RUNNING -o %i 2>/dev/null)" ] && break; sleep 10; done
  echo $job
}

for round in $(seq 1 8); do
  og=$($PY jobs/gw_run.py --count); op=$(open_p)
  echo "$(date +%m-%dT%H:%M:%S) round $round: queue_g $og open, paulistrings-rs ($SMALL) $op open"
  [ "$og" -le 0 ] && [ "$op" -le 0 ] && break
  [ -e queue_g/stop_i ] && { echo "stop file found"; exit 0; }
  JOB=$(allocate) || { echo "no allocation"; exit 1; }
  HOSTS=$(scontrol show hostnames $(squeue -h -j $JOB -o %N) | paste -sd,)
  echo "$(date +%m-%dT%H:%M:%S) allocation $JOB on $HOSTS"
  [ "$og" -gt 0 ] && $PY jobs/gw_run.py --job $JOB --hosts $HOSTS
  export MPICH_GPU_SUPPORT_ENABLED=0
  $PY jobs/p_drive.py --job $JOB --modes mg16
  $PY jobs/p_drive.py --job $JOB --modes mg8
  $PY jobs/p_drive.py --job $JOB --modes g1 --node 0 > logs/p/i_g1.$JOB.log 2>&1 &
  $PY jobs/p_drive.py --job $JOB --modes g2 --node 1 > logs/p/i_g2.$JOB.log 2>&1 &
  $PY jobs/p_drive.py --job $JOB --modes g4 --node 2 > logs/p/i_g4.$JOB.log 2>&1 &
  wait
  cat logs/p/i_g1.$JOB.log logs/p/i_g2.$JOB.log logs/p/i_g4.$JOB.log
  unset MPICH_GPU_SUPPORT_ENABLED
  [ -n "$(squeue -h -j $JOB -o %i 2>/dev/null)" ] && scancel $JOB
  og2=$($PY jobs/gw_run.py --count); op2=$(open_p)
  if [ "$og2" -ge "$og" ] && [ "$op2" -ge "$op" ]; then echo "no progress in a fresh allocation; stopping"; exit 1; fi
done

# n2count: the pinned campaign, one interactive allocation per call (the driver requests it itself)
for round in $(seq 1 6); do
  [ -e queue_g/stop_i ] && { echo "stop file found"; exit 0; }
  status=$(python3 -c "import json; print(json.load(open('logs/n2count/campaign.$CAMPAIGN.json'))['status'])")
  echo "$(date +%m-%dT%H:%M:%S) n2count campaign status $status"
  case $status in complete|halted|allocation_limit) break;; esac
  # the driver's request uses --immediate=30: wait for a free interactive slot first
  while [ "$(squeue -h -u $USER -q interactive -o %i 2>/dev/null | wc -l)" -ge 2 ]; do sleep 180; done
  while pgrep -u $USER -f "n2_count.py" > /dev/null; do sleep 180; done   # a regular continuation holds the campaign lock
  bash jobs/n2_count.sh --source-commit cdbca69ce5353d6532e3778810acbd0d0b368ad0 \
    --binary-sha256 6bbe1e3f203b6a570de7ac3d5dc238f16eff153c4eb458ea5b5ae8d08edacab5 \
    --reference-sha256 d426695d7053e6bf629120bb61c1ea0d808e8e771fbdbd80627a2da9f4fe20be \
    --campaign $CAMPAIGN >> logs/n2count/i_loop.log 2>&1 || sleep 300
done
echo "$(date +%m-%dT%H:%M:%S) loop finished: queue_g $($PY jobs/gw_run.py --count), paulistrings-rs $(open_p) open"
