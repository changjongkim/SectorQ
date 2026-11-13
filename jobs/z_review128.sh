#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -N 32
#SBATCH -t 04:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_z_g128
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Review runs on 128 GPUs (2026-11-12), approved by the user:
#   1. SectorQ on noisy kicked Ising 7e-7, 5e-7, 3e-7 (queue_z, jobs/z_items.py; paulistrings-rs completed 7e-7 and
#      5e-7 there and failed at 3e-7).
#   2. XXZ 8x8 under the matched rule: SectorQ with a truncation after every operation (as jobs/p_batch.sh,
#      logs/p/sq.e1.xxz.g128.<cutoff>.out) at the cutoffs where paulistrings-rs ran on 128 GPUs.
#   3. SectorQ on TFIM 7e-7, 5e-7 (queue_z; paulistrings-rs failed at 7e-7).
#   env/bin/python3 jobs/z_items.py && sbatch jobs/z_review128.sh
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
export MPICH_GPU_SUPPORT_ENABLED=0
JOB=$SLURM_JOB_ID
N=$SLURM_NNODES
G=$((N * 4))
HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)
echo "$(date +%m-%dT%H:%M:%S) job $JOB: $N nodes, $G GPUs; queue_z open $(env/bin/python3 jobs/z_run.py --count)"
matched() {   # cutoff: SectorQ on XXZ with a truncation after every operation (jobs/p_batch.sh run())
  local x=$1 log=logs/p/sq.e1.xxz.g$G.$1.out
  [ -s $log ] && grep -q "^exit=" $log && return
  [ -e logs/p/sq.stop.xxz.g$G ] && return
  echo "===== SectorQ, truncation after every operation, xxz, $G GPUs, cutoff $x" > $log
  timeout 2400 srun --jobid=$JOB --nodelist=$HOSTS --overlap -N $N -n $N --ntasks-per-node=1 -c 128 --gpus-per-node=4 --cpu-bind=none \
    stdbuf -oL ./bin/pp_engine_i --gpus 4 --circuit xxz --lx 8 --ly 8 --jxy 0.1 --jz 0.15 --steps 6 --target 4000000 \
    --coef $x --trace 0 --vram-frac 0.9 --trunc-every 1 >> $log 2>&1
  local code=$?
  echo "exit=$code" >> $log
  [ $code -ne 0 ] && touch logs/p/sq.stop.xxz.g$G
  echo "$(date +%m-%dT%H:%M:%S) SectorQ matched xxz $G GPUs $x: $(grep -m1 '^expectation=' $log) $(grep -o '^wall=[0-9.]* s' $log | head -1) exit=$code"
}
env/bin/python3 jobs/z_run.py --job $JOB --hosts $HOSTS
for x in 1e-4 3e-5 1e-5 5e-6 3e-6 2e-6 1e-6 7e-7; do matched $x; done
env/bin/python3 jobs/z_items.py --tfim
env/bin/python3 jobs/z_run.py --job $JOB --hosts $HOSTS
echo "$(date +%m-%dT%H:%M:%S) done after ${SECONDS}s; queue_z open $(env/bin/python3 jobs/z_run.py --count)"
