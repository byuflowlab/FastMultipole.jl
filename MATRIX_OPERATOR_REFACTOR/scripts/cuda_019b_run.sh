#!/bin/bash
#SBATCH --job-name=fm019b
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Item 019b: GPU small-P / tiny-batch exploratory benchmark. Precompiles CUDA.jl
# against the local toolkit, then runs the 019b sweep.
source /etc/profile
set -o pipefail
# julia pinned: module default moved to 1.12.6 on 2026-08-05 which segfaults host LLVM JIT (job 13058191); 1.11.7 is the toolchain of record
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

cd "$HOME/projects/FastMultipole-022"
ENVDIR="$HOME/fm022env"

echo "=== CUDA.jl precompile/versioninfo"
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()' || { echo "CUDA_JL_LOAD_FAIL"; exit 1; }

echo "=== cuda_019b_tuning.jl"
julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/cuda_019b_tuning.jl
val_status=$?
echo "VALIDATION_EXIT=$val_status"

exit $val_status
