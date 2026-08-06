#!/bin/bash
# Local driver (run on the Mac, from the repo root): sync the working tree to
# the BYU cluster and submit the 029 P2 two-GPU feasibility job.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_p2_submit.sh
# Requires a live ssh master session to `orc`.
#
# Task-owned-tree pattern (034 precedent): this task owns ~/FastMultipole-029p2
# and ~/fm029p2env exclusively, so P2 jobs can run concurrently with jobs from
# other task trees (e.g. 032a in ~/FastMultipole-023) without a mid-run rsync
# corrupting their source. The env is cloned once from ~/fm023env (same depot,
# same CUDA local-runtime preferences) and re-pointed at this tree.
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-029p2
RENV=fm029p2env

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
  && if [ ! -s \$HOME/$RENV/Project.toml ]; then \
       mkdir -p \$HOME/$RENV \
       && cp \$HOME/fm023env/Project.toml \$HOME/$RENV/ \
       && cp \$HOME/fm023env/Manifest.toml \$HOME/$RENV/ \
       && { cp \$HOME/fm023env/LocalPreferences.toml \$HOME/$RENV/ 2>/dev/null || true; }; \
     fi \
  && bash -lc 'module load julia/1.11.7-6bmogfl \
      && julia --project=\$HOME/'$RENV' -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && julia --project=test -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && sbatch --export=ALL,FM029_DIR=\$HOME/'$RDIR',FM029_ENV=\$HOME/'$RENV' \
           MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_p2_run.sh'"

echo "Submitted from ~/$RDIR (env ~/$RENV). Poll with:  ssh orc 'squeue -u \$USER'"
echo "Output: ssh orc 'ls -t $RDIR/fm029p2-*.out'"
