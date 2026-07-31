#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to the
# BYU cluster and submit the 028 feasibility job.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_submit.sh [pilot|sweep]
# Default mode: pilot. Requires a live ssh master session to `orc` (if auth
# expired: ssh -fN orc and complete the password/Duo prompts once;
# ControlPersist keeps it alive).
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
MODE="${1:-pilot}"

case "$MODE" in
  pilot) SBATCH_EXTRA="--time=03:00:00" ;;
  sweep) SBATCH_EXTRA="--time=06:00:00" ;;   # 25 tiered cases at ~2.5 min each
  *) echo "usage: cuda_028_submit.sh [pilot|sweep]" >&2; exit 1 ;;
esac

# the remote tree can vanish (2026-07-31: FastMultipole-023 disappeared during
# the cluster archive migration) — always recreate it
ssh "$REMOTE" "mkdir -p $RDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling"

rsync -az --delete --exclude .git \
    src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_028/"

# the 028 accuracy gate reads the checksummed 024b direct references
rsync -az MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references \
    "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/"

# refresh the tree, run the login-node instantiate (compute nodes have no
# internet), then submit
ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_028/src staging_028/test . \
  && cp staging_028/Project.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR \
  && cp -r staging_028/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia \
      && julia --project=\$HOME/fm023env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && julia --project=test -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && sbatch $SBATCH_EXTRA --export=ALL,FM028_MODE=$MODE MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_run.sh'"

echo "Submitted ($MODE). Poll with:  ssh orc 'squeue -u \$USER'"
echo "Fetch results with:           bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_fetch.sh <jobid>"
