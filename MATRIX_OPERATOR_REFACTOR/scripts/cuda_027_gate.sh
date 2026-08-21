#!/bin/bash
#SBATCH --job-name=fm027g
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Task 027 Checkpoint D + new-default verification, in one job on one node.
#
# Phase 1  re-run every CUDA test file against the NEW production default
#          (HierarchicalRigidStencil, near_radius2=12, device window_classes=256).
#          Jobs 12977812/12992039 exercised hierarchical only via an explicitly
#          passed policy; the default itself has never executed on device.
# Phase 2  the mandatory 026 old-versus-new regression gate. `src/` is swapped in
#          place between the pre-026 snapshot and the current tree and the same
#          case matrix is run alternately (new, old, new, old), so both sides see
#          the same node, GPU, Julia/CUDA, BLAS, seeds, and case order. Restricted
#          to the FLAT path across all four strategies: the pre-026 source has no
#          hierarchical policy at all (026 introduced it), so flat is the only
#          common surface. It does cover the two paths 026 actually targeted,
#          flat precomputed-y and flat dense.
source /etc/profile
set -o pipefail
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L

WORKDIR="${FM027_DIR:-$HOME/FastMultipole-023}"
ENVDIR="$HOME/fm023env"
# pre-026 snapshot: containers.jl declares `CUDARadixLifecycleOptions{TF}` with
# `operator::Any`. NEVER write into the snapshot directories.
OLDSRC="${FM026_OLD_SRC:-$HOME/FastMultipole-026/staging_024b_reference/src}"
cd "$WORKDIR"

GATE=MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_cuda/gate_026_${SLURM_JOB_ID}
mkdir -p "$GATE"

# `src/` is swapped below; always put the current tree back, even on failure.
NEWSRC_BACKUP="$(mktemp -d)/src_new"
cp -r src "$NEWSRC_BACKUP"
restore_new() { rm -rf src && cp -r "$NEWSRC_BACKUP" src; }
trap restore_new EXIT

manifest() {
    julia --project="$ENVDIR" -e '
        using SHA
        files = sort(filter(f -> endswith(f, ".jl"), readdir("src")))
        ctx = SHA.SHA256_CTX()
        for f in files
            SHA.update!(ctx, codeunits(f)); SHA.update!(ctx, read(joinpath("src", f)))
        end
        println("MANIFEST=", bytes2hex(SHA.digest!(ctx))[1:16])'
}

echo "=== new source manifest"; manifest
grep -n "struct CUDARadixLifecycleOptions" "$OLDSRC/containers.jl" | head -1
echo "=== CUDA.jl"
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()' || { echo "CUDA_JL_LOAD_FAIL"; exit 1; }

# ---- phase 1: CUDA suites against the NEW DEFAULT --------------------------
echo "=== phase 1: CUDA test suites (new default policy)"
for t in cuda_radix_lifecycle_test cuda_radix_integration_test cuda_radix_hierarchical_test; do
    echo "--- test/$t.jl"
    FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" "test/$t.jl"
    echo "${t}_EXIT=$?"
done | tee "$GATE/phase1_tests.log"
tests_status=$(grep -c "_EXIT=[^0]" "$GATE/phase1_tests.log")
echo "PHASE1_NONZERO_EXITS=$tests_status"

# ---- phase 2: 026 old-versus-new gate, interleaved -------------------------
run_side() {  # $1 = label, $2 = outdir
    FM026G_OUT="$2" FM026G_LABEL="$1" FM026G_REPS=7 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/gate_026_cases.jl
}

for round in 1 2; do
    echo "=== phase 2 round $round: NEW source"
    restore_new; manifest
    run_side "new_r$round" "$GATE/new_r$round"

    echo "=== phase 2 round $round: OLD (pre-026) source"
    rm -rf src && cp -r "$OLDSRC" src && manifest
    run_side "old_r$round" "$GATE/old_r$round"
done
restore_new
gate_status=$?
echo "GATE_RUNS_EXIT=$gate_status"

# ---- phase 3: verdict ------------------------------------------------------
# Gates m2l_ms_median AND step_ms_median independently: >5% flagged, >10% blocks,
# cross-cell geomean must be <= 1.00, no allocation or counter regression.
echo "=== phase 3: verdict (new vs old, round 1)"
julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/compare_026_regression.jl \
    "$GATE/new_r1" "$GATE/old_r1"
verdict1=$?
echo "VERDICT_R1_EXIT=$verdict1"
echo "=== phase 3: verdict (new vs old, round 2)"
julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/compare_026_regression.jl \
    "$GATE/new_r2" "$GATE/old_r2"
verdict2=$?
echo "VERDICT_R2_EXIT=$verdict2"

echo "GATE_DIR=$GATE"
exit 0
