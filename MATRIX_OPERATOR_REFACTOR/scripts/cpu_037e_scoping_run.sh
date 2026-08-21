#!/bin/bash
#SBATCH --job-name=fm037e_scope
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Task 037e E0: full-n host scoping runs (CPU only, no GPU request).
# Pattern: cuda_035_run.sh env conventions minus CUDA. Uses the 034-layout
# trees (~/FastMultipole-034 synced from the matrix-ops working tree by
# cuda_035_submit.sh; ~/fm034env has FLOWVPM + FastMultipole dev'ed, so
# fm033_build serves all three cases including the rotor).
#
#   sbatch MATRIX_OPERATOR_REFACTOR/scripts/cpu_037e_scoping_run.sh
#
# The whole computation is route generation + bucket/AABB counting on host
# arrays — single pass per config, no benchmark timing, so thread count is
# uncritical (HPC: the local 4-thread cap does not apply).
# no -u: /etc/profile.d scripts reference unset vars on the cluster
set -eo pipefail
source /etc/profile
module load julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"

FMDIR="${FM037E_FMDIR:-$HOME/FastMultipole-034}"
ENVDIR="${FM037E_ENV:-$HOME/fm034env}"
DATADIR="$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign"
mkdir -p "$DATADIR"

cd "$FMDIR"
export FM037E_OUT="$DATADIR/fm037e_scoping_${SLURM_JOB_ID:-manual}.csv"
export JULIA_NUM_THREADS=8
julia --project="$ENVDIR" \
    MATRIX_OPERATOR_REFACTOR/scripts/fm037e_scoping.jl

echo "fm037e scoping complete: $FM037E_OUT"
