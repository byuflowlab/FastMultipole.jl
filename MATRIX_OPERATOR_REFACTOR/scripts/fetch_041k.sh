#!/bin/bash
# Usage: fetch_041k.sh <jobid>
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-041e
JOB="${1:?usage: fetch_041k.sh <jobid>}"
OUTDIR=MATRIX_OPERATOR_REFACTOR/data/direct_bruteforce_ceiling
mkdir -p "$OUTDIR"

scp "$REMOTE:fm041k_${JOB}.out" "$OUTDIR/"
rsync -az "$REMOTE:$RDIR/$OUTDIR/" "$OUTDIR/"
test -s "$OUTDIR/sweep.csv"
test -s "$OUTDIR/accuracy.csv"
echo "Fetched task 041k artifacts (job $JOB)"
