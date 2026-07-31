#!/bin/bash
#SBATCH --job-name=fm026types
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -euo pipefail
module load cuda julia

WORKDIR="${FM026_DIR:-$HOME/FastMultipole-026}"
ENVDIR="${FM026_ENV:-$HOME/fm026env}"
cd "$WORKDIR"

nvidia-smi -L
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()'
julia --project="$ENVDIR" \
    MATRIX_OPERATOR_REFACTOR/scripts/cuda_026_type_smoke.jl
