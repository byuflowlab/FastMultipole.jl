#!/bin/bash
# fm041e_submit_gpu.sh — task 041e H200 job: fused-nearfield parity tests +
# Stage B pre-registered shape microbenchmarks (fm041e_target_owned_bench.jl).
# Copy OUTSIDE the repo snapshot before sbatch:
#   cp to ~/fm041e_gpu.sh, then `sbatch ~/fm041e_gpu.sh`.
# Requires: ~/FastMultipole-041e (repo snapshot) and ~/fm041env.
#SBATCH --time=04:00:00
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus=h200:1
#SBATCH --mem=192G
#SBATCH -J fm041eG
#SBATCH -o /home/rander39/fm041eG_%j.out

source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L

REPO=/home/rander39/FastMultipole-041e
ENVDIR=/home/rander39/fm041eenv
export JULIA_NUM_THREADS=1
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1

cd "$REPO"

echo "=== preflight: CUDA loads + package sanity ==="
julia --project="$ENVDIR" -e 'using CUDA; @info "CUDA" CUDA.functional(); using FastMultipole; FastMultipole.load_cuda_radix_lifecycle!() || error("CUDA radix lifecycle failed to load: " * FastMultipole.cuda_radix_status())' || exit 1

echo "=== fused nearfield parity + CSR permutation tests ==="
julia --project="$ENVDIR" -t 1 test/cuda_radix_fused_nearfield_test.jl || exit 1

echo "=== Stage B microbenchmarks: fm041e_target_owned_bench.jl ==="
julia --project="$ENVDIR" -t 1 MATRIX_OPERATOR_REFACTOR/scripts/fm041e_target_owned_bench.jl

echo "=== done ==="
