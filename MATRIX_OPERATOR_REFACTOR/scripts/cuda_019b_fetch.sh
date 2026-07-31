#!/bin/bash
# Local driver (run on the Mac, from the repo root): show queue state and fetch fm019b
# sbatch outputs into the 019b data dir.
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_019b_fetch.sh
set -euo pipefail
DEST=MATRIX_OPERATOR_REFACTOR/data/smallp_fallback_layout/raw
mkdir -p "$DEST"
ssh orc 'squeue -u rander39' || true
scp 'orc:projects/FastMultipole-022/fm019b-*.out' "$DEST/" 2>/dev/null \
    || echo "no fm019b-*.out files on the cluster yet"
scp 'orc:projects/FastMultipole-022/fm019bcpu-*.out' "$DEST/" 2>/dev/null \
    || echo "no fm019bcpu-*.out files on the cluster yet"
# machine-tagged CPU CSV dirs written by impl_019b_smallp_layout.jl on the node
rsync -az 'orc:projects/FastMultipole-022/MATRIX_OPERATOR_REFACTOR/data/smallp_fallback_layout/' \
    MATRIX_OPERATOR_REFACTOR/data/smallp_fallback_layout/ 2>/dev/null \
    || echo "no remote smallp_fallback_layout data dirs yet"
ls -lR MATRIX_OPERATOR_REFACTOR/data/smallp_fallback_layout/ | head -40
