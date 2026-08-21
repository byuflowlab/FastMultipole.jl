#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to the BYU
# cluster and submit the 019 tuning job.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_019_submit.sh
# Requires a live ssh master session to `orc` (if auth expired: ssh -fN orc and
# complete the password/Duo prompts once; ControlPersist keeps it alive).
set -euo pipefail
REMOTE=orc
RDIR=projects/FastMultipole-022

rsync -az --delete --exclude .git \
    src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_019/"

ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_019/src staging_019/test . \
  && cp staging_019/Project.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR \
  && cp -r staging_019/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'sbatch MATRIX_OPERATOR_REFACTOR/scripts/cuda_019_run.sh'"

echo "Submitted. Poll with:  ssh orc 'squeue -u rander39'"
echo "Fetch results with:    bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_019_fetch.sh"
