#!/bin/bash
# fm041a_submit_gpu_contrast.sh — task 041a H200 job: contrast sweep + stage
# breakdown + sigma-heterogeneous variant (fm041a_gpu_contrast.jl).
# Copy OUTSIDE the repo snapshot before sbatch:
#   cp to ~/fm041a_gpu_contrast.sh, then `sbatch ~/fm041a_gpu_contrast.sh`.
# Requires: ~/FastMultipole-041 (repo snapshot) and ~/fm041env.
#SBATCH --time=08:00:00
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus=h200:1
#SBATCH --mem=192G
#SBATCH -J fm041aC
#SBATCH -o /home/rander39/fm041aC_%j.out

source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L

REPO=/home/rander39/FastMultipole-041
ENVDIR=/home/rander39/fm041env
export JULIA_NUM_THREADS=1
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1

cd "$REPO"

echo "=== preflight: CUDA loads + package sanity ==="
julia --project="$ENVDIR" -e 'using CUDA; @info "CUDA" CUDA.functional(); using FastMultipole; FastMultipole.load_cuda_radix_lifecycle!() || error("CUDA radix lifecycle failed to load: " * FastMultipole.cuda_radix_status())' || exit 1

echo "=== measurement of record: fm041a_gpu_contrast.jl ==="
julia --project="$ENVDIR" -t 1 MATRIX_OPERATOR_REFACTOR/scripts/fm041a_gpu_contrast.jl

echo "=== done ==="
