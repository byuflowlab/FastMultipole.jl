#!/bin/bash
# Local driver (run from the repo/worktree root): sync the working tree to the
# BYU cluster and submit the 029 step-2 floor-attribution profiling job.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_profile_submit.sh
# Requires a live ssh master session to `orc`.
#
# CAUTION: refreshes src/test in the shared FastMultipole-023 remote tree —
# do not run while another fm* job from that tree is queued or running.
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023

ssh "$REMOTE" "mkdir -p $RDIR/MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms"

rsync -az --delete --exclude .git \
    src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_029p/"

ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_029p/src staging_029p/test . \
  && cp staging_029p/Project.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR \
  && cp -r staging_029p/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia/1.11.7-6bmogfl \
      && julia --project=\$HOME/fm023env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && sbatch MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_profile_run.sh'"

echo "Submitted. Poll with:  ssh orc 'squeue -u \$USER'"
