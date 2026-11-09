#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 08:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_w_fill16
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# Campaign g on four nodes (16 GPUs) in a batch allocation: the open runs of queue_g that need four nodes
# (list: jobs/u_items.py -> queue_g/todo_w.jsonl), with the stall watch (jobs/gw_run.py).
#   sbatch jobs/w_fill16.sh
# A run that does not fit into the time that is left stays open; the same script submitted again continues.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)
open=$(timeout 300 env/bin/python3 jobs/gw_run.py --count) || { echo "the runner did not answer within 5 minutes on $(hostname); the job gives its nodes back"; exit 1; }
echo "fill16 job $SLURM_JOB_ID: nodes=$SLURM_NNODES open=$open"
env/bin/python3 jobs/gw_run.py --job $SLURM_JOB_ID --hosts $HOSTS
echo "done after ${SECONDS}s; open=$(env/bin/python3 jobs/gw_run.py --count)"
