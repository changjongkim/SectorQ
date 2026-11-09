#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_p_fill
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/%x-%j.out
# paulistrings-rs with its own truncation rule at the cutoffs and GPU counts of the main figures (jobs/p_drive.py,
# 2026-11-09): one job per placement, the modes given as the argument.
#   sbatch -N 1  -J pp_p_g1   -t 02:00:00 jobs/p_fill.sh g1,g2,g4
#   sbatch -N 2  -J pp_p_g8   -t 01:00:00 jobs/p_fill.sh mg8
#   sbatch -N 4  -J pp_p_g16  -t 04:00:00 jobs/p_fill.sh mg16
#   sbatch -N 8  -J pp_p_g32f -t 01:30:00 jobs/p_fill.sh mg32
#   sbatch -N 16 -J pp_p_g64f -t 02:00:00 jobs/p_fill.sh mg64
#   sbatch -N 32 -J pp_p_g128 -t 05:00:00 jobs/p_fill.sh mg128
#   sbatch -N 64 -J pp_p_g256 -t 02:30:00 jobs/p_fill.sh mg256
# A run that does not fit into the time that is left stays open; the same command submitted again continues.
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd $PP_ROOT
export MPICH_GPU_SUPPORT_ENABLED=0
MODES=${1:?modes, e.g. mg128}
echo "$(date +%m-%dT%H:%M:%S) job $SLURM_JOB_ID: $SLURM_NNODES nodes, modes $MODES; $(env/bin/python3 jobs/p_drive.py --count) runs open in all"
env/bin/python3 jobs/p_drive.py --job $SLURM_JOB_ID --modes $MODES
echo "$(date +%m-%dT%H:%M:%S) done after ${SECONDS}s; $(env/bin/python3 jobs/p_drive.py --count) runs open in all"
