#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to the BYU
# cluster and submit the 019b exploratory job.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_019b_submit.sh
# Requires a live ssh master session to `orc` (if auth expired: ssh -fN orc and
# complete the password/Duo prompts once; ControlPersist keeps it alive).
#
# After syncing, this resolves/instantiates $HOME/fm022env ON THE LOGIN NODE
# (which has internet; compute nodes do not) so a Project.toml change cannot
# leave the env with a stale Manifest that fails at `using CUDA`/`using
# FastMultipole` inside the jobs. If that resolve step fails, run
#   ssh orc; module load julia; julia --project=$HOME/fm022env -e 'import Pkg; Pkg.update()'
# on the login node once, then re-run this script (per past experience,
# Pkg.update() is occasionally needed before CUDA loads; the local-toolkit
# preferences keep CUDA JLL artifacts off the network).
set -euo pipefail
REMOTE=orc
RDIR=projects/FastMultipole-022
# Optional argument selects which job(s) to submit: gpu | cpu | all (default all).
WHICH="${1:-all}"
case "$WHICH" in
  gpu) SUBMIT="sbatch MATRIX_OPERATOR_REFACTOR/scripts/cuda_019b_run.sh" ;;
  cpu) SUBMIT="sbatch MATRIX_OPERATOR_REFACTOR/scripts/cpu_019b_run.sh" ;;
  all) SUBMIT="sbatch MATRIX_OPERATOR_REFACTOR/scripts/cuda_019b_run.sh; sbatch MATRIX_OPERATOR_REFACTOR/scripts/cpu_019b_run.sh" ;;
  *) echo "usage: $0 [gpu|cpu|all]"; exit 1 ;;
esac

rsync -az --delete --exclude .git \
    src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_019b/"

ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_019b/src staging_019b/test . \
  && cp staging_019b/Project.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR \
  && cp -r staging_019b/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia \
       && julia --project=\$HOME/fm022env -e \"import Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.status()\"' \
  && bash -lc '$SUBMIT'"

echo "Submitted ($WHICH)."
echo "Poll with:           ssh orc 'squeue -u rander39'"
echo "Fetch results with:  bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_019b_fetch.sh"
