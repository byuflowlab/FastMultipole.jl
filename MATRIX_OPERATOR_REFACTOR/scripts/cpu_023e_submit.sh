#!/bin/bash
# Sync the task-023e worktree to ORC, instantiate, and submit the authorized CPU job.
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
COMMIT=$(git rev-parse HEAD)
TREE=$(git rev-parse 'HEAD^{tree}')
if git diff --quiet && git diff --cached --quiet; then WORKTREE=clean; else WORKTREE=dirty; fi

rsync -az --delete --exclude .git src test Project.toml Manifest.toml \
    MATRIX_OPERATOR_REFACTOR/scripts "$REMOTE:$RDIR/staging_023e/"

ssh "$REMOTE" "cd $RDIR \
  && rm -rf src test MATRIX_OPERATOR_REFACTOR/scripts \
  && cp -r staging_023e/src staging_023e/test . \
  && cp staging_023e/Project.toml staging_023e/Manifest.toml . \
  && mkdir -p MATRIX_OPERATOR_REFACTOR/data/dense_translation_m2l_host \
  && cp -r staging_023e/scripts MATRIX_OPERATOR_REFACTOR/ \
  && bash -lc 'module load julia \
      && julia --project=\$HOME/fm023env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && julia --project=test -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
      && export FM023E_TFS=\"${FM023E_TFS:-Float32 Float64}\" \
         FM023E_LHS=\"${FM023E_LHS:-false true}\" \
         FM023E_PS=\"${FM023E_PS:-4 8 12}\" \
         FM023E_NS=\"${FM023E_NS:-150 2000 20000}\" \
         FM023E_VARIANTS=\"${FM023E_VARIANTS:-dense materialized_concat factored precomputed_y}\" \
         FM023E_BLAS_THREAD_LIST=\"${FM023E_BLAS_THREAD_LIST:-1 64}\" \
      && FM023E_GIT_COMMIT=$COMMIT FM023E_GIT_TREE=$TREE FM023E_GIT_WORKTREE=$WORKTREE \
         FM023E_LABEL=${FM023E_LABEL:-functional_baseline} \
         sbatch --export=ALL,FM023E_GIT_COMMIT,FM023E_GIT_TREE,FM023E_GIT_WORKTREE,FM023E_LABEL \
         MATRIX_OPERATOR_REFACTOR/scripts/cpu_023e_run.sh'"
