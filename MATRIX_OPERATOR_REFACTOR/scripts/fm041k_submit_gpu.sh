#!/bin/bash
# fm041k_submit_gpu.sh — task 041k H200 job: brute-force direct UJ(+SFS)
# ceiling sweep (fm041k_direct_bruteforce.jl).
# Copy OUTSIDE the repo snapshot before sbatch:
#   cp to ~/fm041k_submit.sh, then `sbatch ~/fm041k_submit.sh`.
# Requires: ~/FastMultipole-041e (repo snapshot) and ~/fm041eenv.
#SBATCH --time=02:00:00
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH -J fm041k
#SBATCH -o /home/rander39/fm041k_%j.out

source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L

REPO=/home/rander39/FastMultipole-041e
ENVDIR=/home/rander39/fm041eenv
export JULIA_NUM_THREADS=8

cd "$REPO"

echo "=== preflight: CUDA functional ==="
julia --project="$ENVDIR" -e 'using CUDA; CUDA.functional() || error("CUDA not functional"); @info "CUDA" CUDA.name(CUDA.device())' || exit 1

echo "=== 041k brute-force ceiling sweep (amendment: tiled vs opt A/B) ==="
julia --project="$ENVDIR" -t 8 MATRIX_OPERATOR_REFACTOR/scripts/fm041k_direct_bruteforce.jl \
    MATRIX_OPERATOR_REFACTOR/data/direct_bruteforce_ceiling tiled,opt

echo "=== done ==="
