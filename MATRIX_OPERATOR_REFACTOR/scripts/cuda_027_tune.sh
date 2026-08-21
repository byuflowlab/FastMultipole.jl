#!/bin/bash
#SBATCH --job-name=fm027t
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Task 027 optimize phase: window-width (K) sweep and depth (ell) feasibility probe.
#
# Job 12977812 established that hierarchical route generation costs a fixed
# 46-61 microseconds per (level, offset window) -- the blocking device-to-host
# window-prefix copy -- and that this is 92-96% of hierarchical dense M2L at the
# host default K = 4. K amortizes that fixed cost directly:
#     windows = (ell - 1) * ceil(noffsets / K)
# bounded by the window flag/route buffers, which scale as K * max_level_nodes.
# Phase 1 finds the knee. Phase 2 answers the outstanding ell = 6/7 question that
# 024b could not construct.
source /etc/profile
set -o pipefail
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L

WORKDIR="${FM027_DIR:-$HOME/FastMultipole-023}"
ENVDIR="$HOME/fm023env"
cd "$WORKDIR"

echo "=== source manifest"
julia --project="$ENVDIR" -e '
    using SHA
    files = sort(filter(f -> endswith(f, ".jl"), readdir("src")))
    ctx = SHA.SHA256_CTX()
    for f in files
        SHA.update!(ctx, codeunits(f)); SHA.update!(ctx, read(joinpath("src", f)))
    end
    println("SOURCE_MANIFEST=", bytes2hex(SHA.digest!(ctx))[1:16])'

DATA=MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_cuda

# --- phase 1: K sweep -------------------------------------------------------
# 7 K values x 2 policies x 2 strategies x 2 n x 2 ell = 112 cases.
# K is clamped to noffsets internally, so K=1740 is "one whole level per window"
# for both radii (q=3 clamps to 316).
echo "=== phase 1: K sweep"
FM027_K="1,4,8,32,64,256,1740" \
FM027_POLICY="hier3,hier12" \
FM027_STRAT="precomputed_y,dense" \
FM027_N="20000,200000" \
FM027_ELL="4,5" \
FM027_P="4" FM027_TF="Float64" FM027_LH="0" FM027_REPS="5" \
FM027_OUT="$DATA/ksweep_$(hostname)_${SLURM_JOB_ID}.csv" \
julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_027_hierarchical_cuda.jl
k_status=$?
echo "KSWEEP_EXIT=$k_status"

# --- phase 2: depth feasibility probe ---------------------------------------
# ell 5/6/7 at two moderate K values, which keep the window buffers bounded
# (K * max_level_nodes) while already amortizing most of the per-window cost.
# Flat is included at ell=5 only as the reference point it can still reach.
echo "=== phase 2: ell = 5/6/7 feasibility"
FM027_K="64,256" \
FM027_POLICY="hier3,hier12" \
FM027_STRAT="precomputed_y,dense" \
FM027_N="200000" \
FM027_ELL="5,6,7" \
FM027_P="4" FM027_TF="Float64" FM027_LH="0" FM027_REPS="3" \
FM027_DIRECT_MAX="0" \
FM027_OUT="$DATA/depth_$(hostname)_${SLURM_JOB_ID}.csv" \
julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_027_hierarchical_cuda.jl
d_status=$?
echo "DEPTH_EXIT=$d_status"

exit $(( k_status || d_status ))
