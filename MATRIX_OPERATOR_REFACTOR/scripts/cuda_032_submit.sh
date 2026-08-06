#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to the
# BYU cluster and submit the 032 stage-2 H200 job (device correctness preflight
# plus the functor-abstraction and erf-free-vs-custom_erf nearfield benchmarks).
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_032_submit.sh
# Requires a live ssh master session to `orc` (if auth expired: ssh -fN orc and
# complete the password/Duo prompts once; ControlPersist keeps it alive).
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023

ssh "$REMOTE" "mkdir -p $RDIR/MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms"

rsync -az --delete --exclude .git \
    src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_032/"

# refresh the tree, run the login-node instantiate (compute nodes have no
# internet), then submit
ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_032/src staging_032/test . \
  && cp staging_032/Project.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR \
  && cp -r staging_032/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia/1.11.7-6bmogfl \
      && julia --project=\$HOME/fm023env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && julia --project=test -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && sbatch MATRIX_OPERATOR_REFACTOR/scripts/cuda_032_run.sh'"

echo "Submitted. Poll with:  ssh orc 'squeue -u \$USER'"
echo "Fetch results with:    bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_032_fetch.sh <jobid>"
