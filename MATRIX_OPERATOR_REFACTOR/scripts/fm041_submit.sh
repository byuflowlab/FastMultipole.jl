#!/bin/bash
# fm041_submit.sh — task 041 H200 job: CUDA adaptive test gate + pre-registered
# measurement (fm041_cuda_cost.jl). Copy OUTSIDE the repo snapshot before
# sbatch (rsync would delete it): cp to ~/fm041_submit.sh, then `sbatch ~/fm041_submit.sh`.
# Requires: ~/FastMultipole-041 (repo snapshot) and ~/fm041env (copy of
# ~/fm034env with the FastMultipole dev path repointed):
#   cp -r ~/fm034env ~/fm041env
#   sed -i 's#/home/rander39/FastMultipole-034#/home/rander39/FastMultipole-041#' ~/fm041env/Manifest.toml
#SBATCH --time=08:00:00
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=192G
#SBATCH -J fm041
#SBATCH -o /home/rander39/fm041_%j.out

module load julia cuda
echo "=== node: $(hostname)"
nvidia-smi -L

REPO=/home/rander39/FastMultipole-041
ENVDIR=/home/rander39/fm041env
export JULIA_NUM_THREADS=1
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1

cd "$REPO"

echo "=== preflight: CUDA loads + package sanity ==="
julia --project="$ENVDIR" -e 'using CUDA; @info "CUDA" CUDA.functional(); using FastMultipole; @info "FastMultipole loaded"; FastMultipole.load_cuda_radix_lifecycle!() || error("CUDA radix lifecycle failed to load: " * FastMultipole.cuda_radix_status())' || exit 1

echo "=== gate 1: task 041 CUDA adaptive tests ==="
julia --project="$ENVDIR" test/cuda_radix_adaptive_test.jl || exit 1

echo "=== gate 2: uniform-device non-regression (existing CUDA suites) ==="
julia --project="$ENVDIR" test/cuda/runtests.jl || exit 1

echo "=== measurement of record: fm041_cuda_cost.jl ==="
julia --project="$ENVDIR" -t 1 MATRIX_OPERATOR_REFACTOR/scripts/fm041_cuda_cost.jl

echo "=== done ==="
