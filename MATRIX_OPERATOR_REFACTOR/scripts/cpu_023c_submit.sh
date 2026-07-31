#!/bin/bash
# Sync the dirty task-023c worktree to ORC, instantiate on the login node, and
# submit the host benchmark. Run from the repository root.
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
COMMIT=$(git rev-parse HEAD)
if git diff --quiet && git diff --cached --quiet; then WORKTREE=clean; else WORKTREE=dirty; fi

rsync -az --delete --exclude .git \
    src test Project.toml Manifest.toml MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$RDIR/staging_023c/"

ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_023c/src staging_023c/test . \
  && cp staging_023c/Project.toml staging_023c/Manifest.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR/data/precomputed_y_resident_m2l_host \
  && cp -r staging_023c/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia \
      && julia --project=\$HOME/fm023env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && julia --project=test -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && FM023C_GIT_COMMIT=$COMMIT FM023C_GIT_WORKTREE=$WORKTREE \
         sbatch --export=ALL,FM023C_GIT_COMMIT,FM023C_GIT_WORKTREE \
         MATRIX_OPERATOR_REFACTOR/scripts/cpu_023c_run.sh'"

echo "Submitted. Poll with: ssh orc 'squeue -u \$USER'"
echo "Fetch with: bash MATRIX_OPERATOR_REFACTOR/scripts/cpu_023c_fetch.sh <jobid>"
