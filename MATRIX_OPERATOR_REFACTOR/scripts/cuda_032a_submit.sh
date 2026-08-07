#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to
# the BYU cluster and submit the 032a Stage C H200 job (binned-nearfield test
# preflight + the §6.3 mechanism-selection benchmark).
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_032a_submit.sh
# HARD RULE: ~/FastMultipole-023 has one owner at a time — verify no other job
# is running from it (squeue) before submitting.
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023

ssh "$REMOTE" "mkdir -p $RDIR/MATRIX_OPERATOR_REFACTOR/data/split_nearfield \
  $RDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling"

rsync -az --delete --exclude .git \
    src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_032a/"

# the stage-D scalar no-regression cases hard-gate on the checksummed 024b
# direct references (030 pattern)
rsync -az MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references \
    "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/"

ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_032a/src staging_032a/test . \
  && cp staging_032a/Project.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR \
  && cp -r staging_032a/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia/1.11.7-6bmogfl \
      && julia --project=\$HOME/fm023env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && sbatch --export=ALL,FM032A_SENTINELS=${FM032A_SENTINELS:-0},FM032A_RUN_STAGEC=${FM032A_RUN_STAGEC:-0} \
          MATRIX_OPERATOR_REFACTOR/scripts/cuda_032a_run.sh'"

echo "Submitted. Poll with:  ssh orc 'bash -lc \"squeue -u \\\$USER\"'"
echo "Fetch results with:    bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_032a_fetch.sh <jobid>"
