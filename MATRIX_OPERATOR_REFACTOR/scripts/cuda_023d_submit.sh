#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to the BYU
# cluster and submit the 023d precomputed-y-CUDA validation + benchmark job.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_023d_submit.sh
# Requires a live ssh master session to `orc` (if auth expired: ssh -fN orc and
# complete the password/Duo prompts once; ControlPersist keeps it alive).
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023

rsync -az --delete --exclude .git \
    src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_023d/"

# refresh the tree, run the login-node instantiate (compute nodes have no
# internet), then submit
ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_023d/src staging_023d/test . \
  && cp staging_023d/Project.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR \
  && cp -r staging_023d/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia \
      && julia --project=\$HOME/fm023env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && julia --project=test -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && sbatch MATRIX_OPERATOR_REFACTOR/scripts/cuda_023d_run.sh'"

echo "Submitted. Poll with:  ssh orc 'squeue -u \$USER'"
echo "Fetch results with:    bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_023d_fetch.sh <jobid>"
