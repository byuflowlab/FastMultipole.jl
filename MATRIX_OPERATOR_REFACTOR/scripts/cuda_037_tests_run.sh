#!/bin/bash
#SBATCH --job-name=fm037t
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Task 037 device test job: run the CUDA suites touched by stages 1-2
# (rectangular radix geometry) with mandatory CUDA. 034/035 tree layout.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L

FMDIR="${FM037_FMDIR:-$HOME/FastMultipole-034}"
ENVDIR="${FM037_ENV:-$HOME/fm034env}"
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
export JULIA_NUM_THREADS=8

cd "$FMDIR"
echo "=== 037 stage 1-2 device suites ==="
julia --project="$ENVDIR" test/cuda_radix_interface_test.jl
julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
julia --project="$ENVDIR" test/cuda_radix_graph_test.jl
julia --project="$ENVDIR" test/radix_fmm_integration_test.jl
julia --project="$ENVDIR" test/radix_grid_clustering_test.jl
echo "fm037 test job complete"
