#!/bin/bash
#SBATCH --job-name=fm019
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Item 019: GPU-node tuning benchmark. Precompiles CUDA.jl against the local
# toolkit on this node's CPU target, then runs the 019 tuning script and the CUDA
# radix lifecycle test.
source /etc/profile
set -o pipefail
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

cd "$HOME/projects/FastMultipole-022"
ENVDIR="$HOME/fm022env"

echo "=== CUDA.jl precompile/versioninfo"
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()' || { echo "CUDA_JL_LOAD_FAIL"; exit 1; }

echo "=== cuda_019_tuning.jl"
julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/cuda_019_tuning.jl
val_status=$?
echo "VALIDATION_EXIT=$val_status"

echo "=== test/cuda_radix_lifecycle_test.jl"
julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
test_status=$?
echo "LIFECYCLE_TEST_EXIT=$test_status"

exit $(( val_status || test_status ))
