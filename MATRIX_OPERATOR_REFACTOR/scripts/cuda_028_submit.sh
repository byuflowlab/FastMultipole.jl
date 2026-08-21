#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to the
# BYU cluster and submit the 028 feasibility job.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_submit.sh [pilot|sweep|derisk|verify|attribute|stage5|stage6|stage6sort|stage56gate|stage7|stage8|stage8repro]
# Default mode: pilot. Requires a live ssh master session to `orc` (if auth
# expired: ssh -fN orc and complete the password/Duo prompts once;
# ControlPersist keeps it alive).
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
MODE="${1:-pilot}"
SBATCH_EXPORT="ALL,FM028_MODE=$MODE"

case "$MODE" in
  pilot) SBATCH_EXTRA="--time=03:00:00" ;;
  sweep) SBATCH_EXTRA="--time=06:00:00" ;;   # 25 tiered cases at ~2.5 min each
  derisk) SBATCH_EXTRA="--time=02:00:00" ;;  # 6 cache builds + profiling
  verify) SBATCH_EXTRA="--time=01:00:00" ;;  # 2 cases at the verdict config
  attribute) SBATCH_EXTRA="--time=03:00:00" ;; # residual A/Bs + q endpoints
  stage5) SBATCH_EXTRA="--time=04:00:00" ;; # nine full-verdict radius cases
  stage6) SBATCH_EXTRA="--time=04:00:00" ;; # depth bracket + launch A/B
  stage6sort) SBATCH_EXTRA="--time=03:00:00" ;; # counting-sort full verdict A/B
  stage56gate) SBATCH_EXTRA="--time=03:00:00" ;; # all-q parity + final verdict
  stage7) SBATCH_EXTRA="--time=05:00:00" ;; # schedule parity + five verdicts
  stage8) SBATCH_EXTRA="--time=08:00:00" ;; # gates + low-rank + kernel bake-off
  stage8repro) SBATCH_EXTRA="--time=00:15:00 --mem=32G" ;; # independent tensor winner gate
  *) echo "usage: cuda_028_submit.sh [pilot|sweep|derisk|verify|attribute|stage5|stage6|stage6sort|stage56gate|stage7|stage8|stage8repro]" >&2; exit 1 ;;
esac

if [[ "$MODE" == "stage6" || "$MODE" == "stage6sort" ]]; then
  : "${FM028_SELECTED_Q:?$MODE requires FM028_SELECTED_Q}"
  SBATCH_EXPORT+=",FM028_SELECTED_Q=$FM028_SELECTED_Q"
fi
if [[ "$MODE" == "stage8" || "$MODE" == "stage8repro" ]]; then
  if [[ -n "${FM028_SELECTED_POLICY:-}" || -n "${FM028_SELECTED_SCHEDULE:-}" ]]; then
    : "${FM028_SELECTED_POLICY:?set both selected policy and schedule, or neither}"
    : "${FM028_SELECTED_SCHEDULE:?set both selected policy and schedule, or neither}"
    SBATCH_EXPORT+=",FM028_SELECTED_POLICY=$FM028_SELECTED_POLICY,FM028_SELECTED_SCHEDULE=$FM028_SELECTED_SCHEDULE"
  fi
  if [[ -n "${FM028_STAGE7_JOB:-}" ]]; then
    SBATCH_EXTRA+=" --dependency=afterok:$FM028_STAGE7_JOB"
  fi
fi

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
      && sbatch $SBATCH_EXTRA --export=$SBATCH_EXPORT MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_run.sh'"

echo "Submitted ($MODE). Poll with:  ssh orc 'squeue -u \$USER'"
echo "Fetch results with:           bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_fetch.sh <jobid>"
