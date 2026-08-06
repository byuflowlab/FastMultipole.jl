#!/bin/bash
# Fetch 029 job outputs from the cluster into the local data directory.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_fetch.sh <jobid>
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
JOBID="${1:?usage: cuda_029_fetch.sh <jobid>}"
LOCAL="MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms"
mkdir -p "$LOCAL"
rsync -az "$REMOTE:$RDIR/fm029-$JOBID.out" "$LOCAL/"
rsync -az "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms/" "$LOCAL/"
echo "Fetched to $LOCAL"
