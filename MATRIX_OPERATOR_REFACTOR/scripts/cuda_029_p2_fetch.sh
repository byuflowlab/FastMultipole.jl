#!/bin/bash
# Fetch 029 P2 job artifacts from the cluster into the local data directory.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_p2_fetch.sh <jobid>
set -euo pipefail
JOB="${1:?usage: cuda_029_p2_fetch.sh <jobid>}"
REMOTE=orc
RDIR=FastMultipole-023
LOCAL="MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms"

rsync -av "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms/cuda029p2_*_${JOB}.csv*" "$LOCAL/" || true
rsync -av "$REMOTE:$RDIR/fm029p2-${JOB}.out" "$LOCAL/" || true
ls -la "$LOCAL" | grep "$JOB" || echo "no artifacts for job $JOB yet"
