#!/bin/bash
# Task 035: fetch campaign results from the cluster.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_035_fetch.sh [jobid ...]
# Copies the sweep CSV(s) and job logs into
# MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign/.
set -euo pipefail
REMOTE=orc
FMDIR=FastMultipole-034
DEST=MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign
mkdir -p "$DEST"
rsync -az "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign/" "$DEST/"
for j in "$@"; do
    rsync -az "$REMOTE:$FMDIR/fm035-$j.out" "$DEST/" || true
done
ls -la "$DEST"
