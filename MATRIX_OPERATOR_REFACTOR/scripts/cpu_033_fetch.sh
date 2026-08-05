#!/bin/bash
# Task 033: fetch the job log and benchmark data back from the cluster.
# Usage: bash MATRIX_OPERATOR_REFACTOR/scripts/cpu_033_fetch.sh <jobid>
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-033
JOBID="${1:?usage: cpu_033_fetch.sh <jobid>}"

mkdir -p MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline
scp "$REMOTE:$RDIR/fm033cpu-$JOBID.out" MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/ || true
rsync -az "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/" \
  MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/
ls -la MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/
