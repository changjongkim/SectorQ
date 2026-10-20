#!/bin/bash
#SBATCH -A m5320_g
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 04:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -J pp_n2count
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH -o /pscratch/sd/s/sgkim/kcj/pauliprop/logs/n2count/batch-%j.out
# The batch shell becomes the single experiment loop. Source and binaries are
# pinned when submitting, so a later edit cannot silently change queued work.
set -euo pipefail
cd /pscratch/sd/s/sgkim/kcj/pauliprop
exec bash jobs/n2_count.sh --batch \
    --source-commit "${PP_N2_SOURCE_COMMIT:?submission commit required}" \
    --binary-sha256 "${PP_N2_BINARY_SHA256:?experiment hash required}" \
    --reference-sha256 "${PP_N2_REFERENCE_SHA256:?reference hash required}" \
    --campaign "${PP_N2_CAMPAIGN:?stable campaign identifier required}"
