#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to the
# BYU cluster and submit the 030 cost-vs-n job.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_030_submit.sh [sweep|spotcheck]
# Default mode: sweep. Requires a live ssh master session to `orc` (if auth
# expired: ssh -fN orc and complete the password/Duo prompts once;
# ControlPersist keeps it alive).
#
# spotcheck mode requires FM030_SPOTCHECKS, a space-separated list of
#   <n>:<baseline_geometry>:<candidate_geometry>:<tf>:<fmt>
# where a geometry is a bare ell (the shipped (6,5,...,5) schedule at that
# depth) or an explicit sched<q2>-...-<qell> string, e.g.
#   FM030_SPOTCHECKS="10000:5:4:Float32:fp16 1000000:5:sched6-5-5-5:Float32:fp16"
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
MODE="${1:-sweep}"
SBATCH_EXPORT="ALL,FM030_MODE=$MODE"

case "$MODE" in
  sweep) SBATCH_EXTRA="--time=06:00:00" ;;      # 42 cases at ~2.5 min each + preflight
  spotcheck) SBATCH_EXTRA="--time=02:00:00" ;;  # baseline/candidate pairs at 2-3 n
  *) echo "usage: cuda_030_submit.sh [sweep|spotcheck]" >&2; exit 1 ;;
esac

if [[ "$MODE" == "spotcheck" ]]; then
  : "${FM030_SPOTCHECKS:?spotcheck requires FM030_SPOTCHECKS}"
  SBATCH_EXPORT+=",FM030_SPOTCHECKS=$FM030_SPOTCHECKS"
fi
# optional narrowing overrides for a resume or a partial rerun
for v in FM030_NS FM030_ELLS FM030_REPS FM030_BOUND; do
  if [[ -n "${!v:-}" ]]; then SBATCH_EXPORT+=",$v=${!v}"; fi
done

# the remote tree can vanish (2026-07-31: FastMultipole-023 disappeared during
# the cluster archive migration) — always recreate it
ssh "$REMOTE" "mkdir -p $RDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling \
  $RDIR/MATRIX_OPERATOR_REFACTOR/data/cost_vs_n"

rsync -az --delete --exclude .git \
    src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_030/"

# every 030 sweep point is referenced against the checksummed 024b directs, and
# the runner hard-gates on their presence and checksums
rsync -az MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references \
    "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/"

# refresh the tree, run the login-node instantiate (compute nodes have no
# internet), then submit
ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_030/src staging_030/test . \
  && cp staging_030/Project.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR \
  && cp -r staging_030/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia \
      && julia --project=\$HOME/fm023env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && julia --project=test -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && sbatch $SBATCH_EXTRA --export=$SBATCH_EXPORT MATRIX_OPERATOR_REFACTOR/scripts/cuda_030_run.sh'"

echo "Submitted ($MODE). Poll with:  ssh orc 'squeue -u \$USER'"
echo "Fetch results with:           bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_030_fetch.sh <jobid>"
