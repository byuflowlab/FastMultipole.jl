#!/bin/bash
#SBATCH --job-name=fm027
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Task 027: H200 validation + benchmark of the device-resident hierarchical rigid
# M2L. Runs the CUDA lifecycle, integration, and new hierarchical test files, then
# the flat-vs-hierarchical benchmark sweep. Pattern: cuda_023f_run.sh.
# Prereq (login node, has internet): instantiate the env and test deps first —
#   julia --project=$HOME/fm023env -e 'using Pkg; Pkg.instantiate()'
#   cd $WORKDIR && julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
source /etc/profile
set -o pipefail
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${FM027_DIR:-$HOME/FastMultipole-023}"
ENVDIR="$HOME/fm023env"
cd "$WORKDIR"

echo "=== source manifest"
julia --project="$ENVDIR" -e '
    using SHA
    files = sort(filter(f -> endswith(f, ".jl"), readdir("src")))
    ctx = SHA.SHA256_CTX()
    for f in files
        SHA.update!(ctx, codeunits(f)); SHA.update!(ctx, read(joinpath("src", f)))
    end
    println("SOURCE_MANIFEST=", bytes2hex(SHA.digest!(ctx))[1:16])'

echo "=== CUDA.jl precompile/versioninfo"
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()' || { echo "CUDA_JL_LOAD_FAIL"; exit 1; }

echo "=== test/cuda_radix_lifecycle_test.jl"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
lifecycle_status=$?
echo "LIFECYCLE_TEST_EXIT=$lifecycle_status"

echo "=== test/cuda_radix_integration_test.jl"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_integration_test.jl
integration_status=$?
echo "INTEGRATION_TEST_EXIT=$integration_status"

echo "=== test/cuda_radix_hierarchical_test.jl (027)"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_hierarchical_test.jl
hier_status=$?
echo "HIERARCHICAL_TEST_EXIT=$hier_status"

echo "=== benchmark_027_hierarchical_cuda.jl"
julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_027_hierarchical_cuda.jl
bench_status=$?
echo "BENCH_EXIT=$bench_status"

exit $(( lifecycle_status || integration_status || hier_status || bench_status ))
