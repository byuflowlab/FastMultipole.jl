#!/bin/bash
#SBATCH --job-name=fm029p
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Task 029 step 2 (evidence-only): launch/sync floor attribution of the robust
# baseline config (sched6-5-4-4, FP16-WMMA, F32) via CUDA.@profile trace, at
# n=1e6 plus an n=1e3 pure-floor control. No production src/ changes; the tree
# is the step-1 baseline manifest (expected 5a5d41312dc113d2).
set -o pipefail
source /etc/profile
# julia pinned: 1.12.6 segfaults host LLVM JIT on the device step (job 13058336)
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${FM029_DIR:-$HOME/FastMultipole-023}"
ENVDIR="${FM029_ENV:-$HOME/fm023env}"
OUTDIR="${FM029_OUTDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms}"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

FM029P_N=1000000,1000 FM029P_POLICY=sched6-5-4-4 FM029P_REPS=5 \
FM029P_OUT="$OUTDIR/cuda029_profile_$(hostname)_${SLURM_JOB_ID:-manual}" \
julia --project="$ENVDIR" \
    MATRIX_OPERATOR_REFACTOR/scripts/profile_029_floor.jl
status=$?
echo "PROFILE_JOB_EXIT=$status"
exit $status
