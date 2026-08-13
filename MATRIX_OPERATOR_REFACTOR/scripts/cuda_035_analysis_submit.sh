#!/bin/bash
# Sync the preregistered task-035 analysis harnesses, then run decomposition
# followed by the four-case Nsight Compute array.
set -euo pipefail
REMOTE=orc
FMDIR=FastMultipole-034
rsync -az --delete --exclude '*.mem' MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/"
analysis_raw=$(ssh "$REMOTE" "bash -lc 'cd $FMDIR && sbatch --parsable MATRIX_OPERATOR_REFACTOR/scripts/cuda_035_analysis_run.sh'")
analysis_id=$(printf '%s\n' "$analysis_raw" | sed -nE 's/^([0-9]+)(;.*)?$/\1/p' | tail -1)
[ -n "$analysis_id" ] || { echo "could not parse analysis job id" >&2; exit 1; }
ncu_raw=$(ssh "$REMOTE" "bash -lc 'cd $FMDIR && sbatch --parsable --dependency=afterok:$analysis_id MATRIX_OPERATOR_REFACTOR/scripts/cuda_035_ncu_run.sh'")
ncu_id=$(printf '%s\n' "$ncu_raw" | sed -nE 's/^([0-9]+)(;.*)?$/\1/p' | tail -1)
[ -n "$ncu_id" ] || { echo "could not parse Nsight job id" >&2; exit 1; }
echo "analysis job: $analysis_id"
echo "Nsight array: $ncu_id (afterok:$analysis_id)"
