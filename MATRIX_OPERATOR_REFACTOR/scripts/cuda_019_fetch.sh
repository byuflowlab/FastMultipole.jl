#!/bin/bash
# Local driver (run on the Mac, from the repo root): show queue state and fetch fm019
# sbatch outputs into the 019 data dir.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_019_fetch.sh
set -euo pipefail
DEST=MATRIX_OPERATOR_REFACTOR/data/operator_performance_tuning/raw
mkdir -p "$DEST"
ssh orc 'squeue -u rander39' || true
scp 'orc:projects/FastMultipole-022/fm019-*.out' "$DEST/" 2>/dev/null \
    || echo "no fm019-*.out files on the cluster yet"
ls -l "$DEST"
