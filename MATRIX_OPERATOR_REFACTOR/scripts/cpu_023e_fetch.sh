#!/bin/bash
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
JOBID="${1:?usage: cpu_023e_fetch.sh <jobid>}"
OUTDIR=MATRIX_OPERATOR_REFACTOR/data/dense_translation_m2l_host
mkdir -p "$OUTDIR"
scp "$REMOTE:$RDIR/fm023ecpu-$JOBID.out" "$OUTDIR/"
rsync -az "$REMOTE:$RDIR/$OUTDIR/" "$OUTDIR/"
echo "Fetched job $JOBID into $OUTDIR/"
