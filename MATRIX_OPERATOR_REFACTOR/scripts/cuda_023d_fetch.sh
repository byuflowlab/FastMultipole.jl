#!/bin/bash
# Fetch 023d job output + benchmark CSVs from the cluster (run on the Mac, from
# the repo root): bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_023d_fetch.sh <jobid>
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-023
JOBID="${1:?usage: cuda_023d_fetch.sh <jobid>}"

mkdir -p MATRIX_OPERATOR_REFACTOR/data/precomputed_y_resident_m2l_cuda
scp "$REMOTE:$RDIR/fm023d-$JOBID.out" MATRIX_OPERATOR_REFACTOR/data/precomputed_y_resident_m2l_cuda/
rsync -az "$REMOTE:$RDIR/MATRIX_OPERATOR_REFACTOR/data/precomputed_y_resident_m2l_cuda/" \
    MATRIX_OPERATOR_REFACTOR/data/precomputed_y_resident_m2l_cuda/
echo "Fetched into MATRIX_OPERATOR_REFACTOR/data/precomputed_y_resident_m2l_cuda/"
