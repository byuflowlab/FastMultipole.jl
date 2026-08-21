#!/bin/bash
# Fetch 028 job output + benchmark CSVs from the cluster (run on the Mac, from
# the repo root): bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_028_fetch.sh <jobid>
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
JOBID="${1:?usage: cuda_028_fetch.sh <jobid>}"

mkdir -p MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms
scp "$REMOTE:$RDIR/fm028-$JOBID.out" MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms/ || true
rsync -az "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms/" \
    MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms/
echo "Fetched into MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms/"
