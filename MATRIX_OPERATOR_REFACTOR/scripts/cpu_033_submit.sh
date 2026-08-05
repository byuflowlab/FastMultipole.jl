#!/bin/bash
# Task 033: sync the FLOWVPM baseline worktree (e2bd487) + harness to the
# cluster, build the pinned env on the login node, and submit the CPU job.
# Usage: bash MATRIX_OPERATOR_REFACTOR/scripts/cpu_033_submit.sh
# Run from the FastMultipole repo root. Requires a live ssh ControlMaster
# session to `orc` (ssh -fN orc).
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-033
STAGE=staging_033
FLOWVPM_BASELINE="${FM033_FLOWVPM_BASELINE:-../FLOWVPM-baseline-e2bd487}"

test -f "$FLOWVPM_BASELINE/Project.toml" || {
  echo "missing FLOWVPM baseline worktree at $FLOWVPM_BASELINE" >&2; exit 1; }
grep -q '^version = "4.0.4"' "$FLOWVPM_BASELINE/Project.toml" || {
  echo "baseline worktree is not at v4.0.4 (e2bd487)" >&2; exit 1; }

ssh "$REMOTE" "mkdir -p $RDIR/$STAGE/flowvpm"
rsync -az --delete --exclude .git \
  "$FLOWVPM_BASELINE/src" "$FLOWVPM_BASELINE/examples" \
  "$FLOWVPM_BASELINE/Project.toml" "$REMOTE:$RDIR/$STAGE/flowvpm/"
rsync -az --delete --exclude .git \
  MATRIX_OPERATOR_REFACTOR/scripts "$REMOTE:$RDIR/$STAGE/"

ssh "$REMOTE" "cd $RDIR \
  && mkdir -p MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/references FLOWVPM_baseline \
  && rsync -a --delete $STAGE/flowvpm/ FLOWVPM_baseline/ \
  && rsync -a --delete $STAGE/scripts/ MATRIX_OPERATOR_REFACTOR/scripts/ \
  && bash -lc 'module load julia \
    && julia --project=\$HOME/fm033env -e \"using Pkg; \
         Pkg.develop(path=\\\"./FLOWVPM_baseline\\\"); \
         Pkg.add([\\\"FastMultipole\\\"]); \
         Pkg.compat(\\\"FastMultipole\\\", \\\"2.0.0 - 2.0.4\\\"); \
         Pkg.resolve(); Pkg.instantiate(); Pkg.status()\" \
    && julia --project=\$HOME/fm033env -e \"using FastMultipole; \
         @assert pkgversion(FastMultipole) < v\\\"2.1\\\"; \
         println(\\\"pinned FastMultipole \\\", pkgversion(FastMultipole))\" \
    && sbatch MATRIX_OPERATOR_REFACTOR/scripts/cpu_033_run.sh'"
