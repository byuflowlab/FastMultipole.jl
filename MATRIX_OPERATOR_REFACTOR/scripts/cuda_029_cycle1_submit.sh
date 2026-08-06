#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to
# the BYU cluster and submit the 029 cycle-1 verification job.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_cycle1_submit.sh
# Requires a live ssh master session to `orc`.
#
# CAUTION: this refreshes src/test in the shared FastMultipole-023 remote tree
# and touches the shared fm023env manifest — do not run while another fm* job
# from that tree is queued or running.
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023

ssh "$REMOTE" "mkdir -p $RDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling \
  $RDIR/MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms"

rsync -az --delete --exclude .git \
    src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_029/"

rsync -az MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references \
    "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/"

ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_029/src staging_029/test . \
  && cp staging_029/Project.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR \
  && cp -r staging_029/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia/1.11.7-6bmogfl \
      && julia --project=\$HOME/fm023env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && julia --project=test -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && sbatch MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_cycle1_run.sh'"

echo "Submitted. Poll with:  ssh orc 'squeue -u \$USER'"
echo "Fetch results with:    bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_cycle1_fetch.sh <jobid>"
