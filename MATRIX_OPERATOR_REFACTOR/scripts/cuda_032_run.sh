#!/bin/bash
#SBATCH --job-name=fm032
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Task 032 stage 2 H200 job:
#   1. device correctness preflight — the stage 1+2 CUDA interface tests
#      (generalized packing, vortex B2M + LH, 13-row hessian, direct-kernel
#      functors, RegularizedVortex parity, adequacy gate) plus the shipped
#      lifecycle regression test;
#   2. the two mandated stage-2 nearfield benchmarks (functor-abstraction cost,
#      erf-free vs custom_erf) via benchmark_032_nearfield.jl.
# Environment pattern from cuda_030_run.sh: the fm023env project carries CUDA
# (local-toolkit preferences; compute nodes have no internet).
# no -u: /etc/profile.d scripts reference unset vars on the cluster
set -eo pipefail
source /etc/profile
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${FM032_DIR:-$HOME/FastMultipole-023}"
ENVDIR="${FM032_ENV:-$HOME/fm023env}"
OUTDIR="${FM032_OUTDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms}"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
export JULIA_NUM_THREADS=8

echo "=== preflight: CUDA interface tests (stage 1 + 2) ==="
julia --project="$ENVDIR" test/cuda_radix_interface_test.jl
echo "=== preflight: shipped lifecycle regression test ==="
julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl

echo "=== stage 2 nearfield benchmarks ==="
FM032_OUTDIR="$OUTDIR" julia --project="$ENVDIR" \
    MATRIX_OPERATOR_REFACTOR/scripts/benchmark_032_nearfield.jl

echo "fm032 job complete"
