#!/bin/bash
# Fetch the 032 stage-2 job log and nearfield benchmark CSVs from the cluster.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_032_fetch.sh <jobid>
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
JOBID="${1:?usage: cuda_032_fetch.sh <jobid>}"
DEST="MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms"

mkdir -p "$DEST"
rsync -az "$REMOTE:$RDIR/fm032-$JOBID.out" "$DEST/"
rsync -az "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms/nearfield032_*.csv" \
    "$DEST/" || echo "(no nearfield032 CSVs yet)"
echo "fetched into $DEST"
tail -40 "$DEST/fm032-$JOBID.out"
