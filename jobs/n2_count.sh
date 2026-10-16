#!/bin/bash
# One finite pilot campaign, one loop and one srun per fresh four-node allocation.
set -euo pipefail
source /pscratch/sd/s/sgkim/kcj/pauliprop/jobs/env.sh
cd "$PP_ROOT"
export MPICH_GPU_SUPPORT_ENABLED=0
export PYTHONDONTWRITEBYTECODE=1
exec python3 jobs/n2_count.py "$@"
