#!/bin/bash
# Fetch 030 job output + campaign CSVs from the cluster (run on the Mac, from
# the repo root): bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_030_fetch.sh <jobid>
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
JOBID="${1:?usage: cuda_030_fetch.sh <jobid>}"

mkdir -p MATRIX_OPERATOR_REFACTOR/data/cost_vs_n
# `|| true`: tolerate a still-running job whose log is not yet complete
scp "$REMOTE:$RDIR/fm030-$JOBID.out" MATRIX_OPERATOR_REFACTOR/data/cost_vs_n/ || true
rsync -az "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/cost_vs_n/" \
    MATRIX_OPERATOR_REFACTOR/data/cost_vs_n/
echo "Fetched into MATRIX_OPERATOR_REFACTOR/data/cost_vs_n/"
