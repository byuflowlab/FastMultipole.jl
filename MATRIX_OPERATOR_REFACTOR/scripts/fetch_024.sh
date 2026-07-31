#!/bin/bash
# Fetch one or both task-024 jobs, then validate the combined campaign.
# Usage: fetch_024.sh <cpu-jobid|-> <h200-jobid|->
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-024
CPU_JOB="${1:?usage: fetch_024.sh <cpu-jobid|-> <h200-jobid|->}"
GPU_JOB="${2:?usage: fetch_024.sh <cpu-jobid|-> <h200-jobid|->}"
OUTDIR=MATRIX_OPERATOR_REFACTOR/data/operator_ab_benchmark
mkdir -p "$OUTDIR/raw" "$OUTDIR/summary"

if [[ "$CPU_JOB" != "-" ]]; then
  scp "$REMOTE:$RDIR/fm024cpu-$CPU_JOB.out" "$OUTDIR/" 
fi
if [[ "$GPU_JOB" != "-" ]]; then
  scp "$REMOTE:$RDIR/fm024h200-$GPU_JOB.out" "$OUTDIR/"
fi
rsync -az "$REMOTE:$RDIR/$OUTDIR/raw/" "$OUTDIR/raw/"
rsync -az "$REMOTE:$RDIR/$OUTDIR/source_manifest.sha256" "$OUTDIR/" || true

FM024_REQUIRE_COMPLETE=1 julia \
  MATRIX_OPERATOR_REFACTOR/scripts/summarize_024.jl \
  "$OUTDIR/raw" "$OUTDIR/summary"
echo "Fetched and validated task 024 into $OUTDIR/"
