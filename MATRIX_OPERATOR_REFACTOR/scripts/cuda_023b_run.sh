#!/bin/bash
#SBATCH --job-name=fm023b
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Task 023b: H200 validation + benchmark of the device-resident factored M2L.
# Runs the CUDA lifecycle and integration tests (factored sections included),
# then the 023b concat-vs-factored benchmark sweep. Pattern: cuda_022_run.sh.
# Prereq (login node, has internet): instantiate the env and test deps first —
#   julia --project=$HOME/fm023env -e 'using Pkg; Pkg.instantiate()'
#   cd $WORKDIR && julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
source /etc/profile
set -o pipefail
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${FM023B_DIR:-$HOME/FastMultipole-023}"
ENVDIR="$HOME/fm023env"
cd "$WORKDIR"

echo "=== CUDA.jl precompile/versioninfo"
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()' || { echo "CUDA_JL_LOAD_FAIL"; exit 1; }

echo "=== test/cuda_radix_lifecycle_test.jl"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
lifecycle_status=$?
echo "LIFECYCLE_TEST_EXIT=$lifecycle_status"

echo "=== test/cuda_radix_integration_test.jl (023b factored sections)"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_integration_test.jl
integration_status=$?
echo "INTEGRATION_TEST_EXIT=$integration_status"

echo "=== benchmark_023b_factored_cuda.jl"
julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023b_factored_cuda.jl
bench_status=$?
echo "BENCH_EXIT=$bench_status"

exit $(( lifecycle_status || integration_status || bench_status ))
