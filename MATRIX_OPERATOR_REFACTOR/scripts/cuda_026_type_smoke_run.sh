#!/bin/bash
#SBATCH --job-name=fm026types
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -euo pipefail
# julia pinned: module default moved to 1.12.6 on 2026-08-05 which segfaults host LLVM JIT (job 13058191); 1.11.7 is the toolchain of record
module load cuda julia/1.11.7-6bmogfl

WORKDIR="${FM026_DIR:-$HOME/FastMultipole-026}"
ENVDIR="${FM026_ENV:-$HOME/fm026env}"
cd "$WORKDIR"

nvidia-smi -L
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()'
julia --project="$ENVDIR" \
    MATRIX_OPERATOR_REFACTOR/scripts/cuda_026_type_smoke.jl
