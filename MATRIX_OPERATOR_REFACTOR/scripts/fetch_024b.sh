#!/bin/bash
# Usage: fetch_024b.sh <reference-jobid|-> <cpu-jobid|-> <h200-jobid|->
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-024b
REF_JOB="${1:?usage: fetch_024b.sh <reference-jobid|-> <cpu-jobid|-> <h200-jobid|->}"
CPU_JOB="${2:?usage: fetch_024b.sh <reference-jobid|-> <cpu-jobid|-> <h200-jobid|->}"
GPU_JOB="${3:?usage: fetch_024b.sh <reference-jobid|-> <cpu-jobid|-> <h200-jobid|->}"
OUTDIR=MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling
REFDIR=MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references
mkdir -p "$OUTDIR" "$REFDIR"

if [[ "$REF_JOB" != "-" ]]; then
  scp "$REMOTE:$RDIR/fm024bref-$REF_JOB.out" "$REFDIR/"
  rsync -az "$REMOTE:$RDIR/$REFDIR/" "$REFDIR/"
  for n in 1000 3162 10000 31623 100000 316228 1000000; do
    test -s "$REFDIR/direct_reference_n${n}.csv"
  done
  (cd "$REFDIR" && shasum -a 256 -c direct_reference_checksums.sha256)
fi
if [[ "$CPU_JOB" != "-" ]]; then
  scp "$REMOTE:$RDIR/fm024bcpu-$CPU_JOB.out" "$OUTDIR/"
fi
if [[ "$GPU_JOB" != "-" ]]; then
  scp "$REMOTE:$RDIR/fm024bh200-$GPU_JOB.out" "$OUTDIR/"
fi
if [[ "$CPU_JOB" != "-" || "$GPU_JOB" != "-" ]]; then
  rsync -az "$REMOTE:$RDIR/$OUTDIR/" "$OUTDIR/"
fi
echo "Fetched task 024b artifacts"
