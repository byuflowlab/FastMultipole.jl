#!/bin/bash
# Fetch the 032a Stage C job log and benchmark CSVs from the cluster.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_032a_fetch.sh <jobid>
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
JOBID="${1:?usage: cuda_032a_fetch.sh <jobid>}"
DEST="MATRIX_OPERATOR_REFACTOR/data/split_nearfield"

mkdir -p "$DEST"
rsync -az "$REMOTE:$RDIR/fm032a-$JOBID.out" "$DEST/"
rsync -az "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/split_nearfield/cuda032a_*.csv" \
    "$DEST/" || echo "(no cuda032a CSVs yet)"
echo "fetched into $DEST"
tail -40 "$DEST/fm032a-$JOBID.out"
