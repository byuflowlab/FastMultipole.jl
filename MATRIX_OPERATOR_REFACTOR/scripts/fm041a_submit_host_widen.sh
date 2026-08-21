#!/bin/bash
# fm041a_submit_host_widen.sh — task 041a CPU-node job: pre-registered widened
# host depth/K sweep with same-job anchors (fm041a_host_widen.jl).
# Copy OUTSIDE the repo snapshot before sbatch:
#   cp to ~/fm041a_host_widen.sh, then `sbatch ~/fm041a_host_widen.sh`.
# Requires: ~/FastMultipole-041 (repo snapshot; host path needs no CUDA env —
# uses the repo project like the fm040 job).
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -J fm041aH
#SBATCH -o /home/rander39/fm041aH_%j.out

source /etc/profile
module load julia
echo "=== node: $(hostname)"

REPO=/home/rander39/FastMultipole-041
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$REPO"
julia --project=. -e 'using Pkg; Pkg.instantiate(); using FastMultipole; @info "FastMultipole loaded"' || exit 1

echo "=== measurement of record: fm041a_host_widen.jl ==="
julia --project=. -t 1 MATRIX_OPERATOR_REFACTOR/scripts/fm041a_host_widen.jl

echo "=== done ==="
