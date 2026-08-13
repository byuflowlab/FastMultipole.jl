#!/bin/bash
# Sync the preregistered task-035 analysis harnesses, then run decomposition
# followed by the four-case Nsight Compute array.
set -euo pipefail
REMOTE=orc
FMDIR=FastMultipole-034
rsync -az --delete --exclude '*.mem' MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/"
analysis_id=$(ssh "$REMOTE" "bash -lc 'cd $FMDIR && sbatch --parsable MATRIX_OPERATOR_REFACTOR/scripts/cuda_035_analysis_run.sh'")
analysis_id="${analysis_id%%;*}"
ncu_id=$(ssh "$REMOTE" "bash -lc 'cd $FMDIR && sbatch --parsable --dependency=afterok:$analysis_id MATRIX_OPERATOR_REFACTOR/scripts/cuda_035_ncu_run.sh'")
ncu_id="${ncu_id%%;*}"
echo "analysis job: $analysis_id"
echo "Nsight array: $ncu_id (afterok:$analysis_id)"
