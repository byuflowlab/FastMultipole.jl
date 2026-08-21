#!/bin/bash
# Fetch the task-023c Slurm log and all generated host CSVs from ORC.
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
JOBID="${1:?usage: cpu_023c_fetch.sh <jobid>}"
OUTDIR=MATRIX_OPERATOR_REFACTOR/data/precomputed_y_resident_m2l_host

mkdir -p "$OUTDIR"
scp "$REMOTE:$RDIR/fm023ccpu-$JOBID.out" "$OUTDIR/"
rsync -az "$REMOTE:$RDIR/$OUTDIR/" "$OUTDIR/"
echo "Fetched job $JOBID into $OUTDIR/"
