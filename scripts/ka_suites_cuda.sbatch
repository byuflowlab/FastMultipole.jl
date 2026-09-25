#!/bin/bash
#SBATCH --job-name=fm_ka_cuda
#SBATCH --qos=eng
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# CUDA arm of the ka_*_correctness.jl suites for the ka-migration PR
# (ported buffer-cache and box-tolerance fixes), plus the CPU-side cache test.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"; nvidia-smi -L
export JULIA_NUM_THREADS=8
ENV="$HOME/fm_ka_cuda_env"
FM="$HOME/algpu/FastMultipole"
# environment built on the login node (compute nodes have no network)
echo "=== buffer cache test (CPU backend) $(date +%H:%M:%S)"
cd "$FM"
julia --startup-file=no --project="$ENV" -e 'using Test, FastMultipole; include("test/ka_target_buffer_cache_test.jl")' 2>&1 | grep -E "Test Summary" -A1
echo "=== device-resident example on CUDA $(date +%H:%M:%S)"
julia --startup-file=no --project="$ENV" examples/device_resident_system.jl 2>&1 | tail -5
echo "=== KA correctness suites on CUDA $(date +%H:%M:%S)"
cd "$FM/test/metal_env"
FASTMULTIPOLE_GPU_TEST_PROJECT="$ENV" bash run_suites.sh
echo "=== suites exit $? (number of failing suites) $(date +%H:%M:%S)"
