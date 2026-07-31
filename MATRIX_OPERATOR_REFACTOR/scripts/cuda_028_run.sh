#!/bin/bash
#SBATCH --job-name=fm028
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Task 028: H200 feasibility measurement — can n=1e6 run in 10 ms per step?
# Preflight: CUDA lifecycle test + the new 028 convection test, then the
# feasibility benchmark. FM028_MODE=pilot|sweep selects the case matrix
# (individual FM028_* overrides exported at submit time win over the presets).
# Pattern: cuda_027_run.sh.
source /etc/profile
set -o pipefail
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${FM028_DIR:-$HOME/FastMultipole-023}"
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

echo "=== CUDA.jl precompile/versioninfo"
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()' || { echo "CUDA_JL_LOAD_FAIL"; exit 1; }

echo "=== test/cuda_radix_lifecycle_test.jl"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
lifecycle_status=$?
echo "LIFECYCLE_TEST_EXIT=$lifecycle_status"

echo "=== test/cuda_radix_convection_test.jl (028)"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_convection_test.jl
convection_status=$?
echo "CONVECTION_TEST_EXIT=$convection_status"

# ---- case matrix presets (any FM028_* already exported wins) ----------------
MODE="${FM028_MODE:-pilot}"
echo "=== FM028_MODE=$MODE"
if [ "$MODE" = "pilot" ]; then
    # 03-phase-pilot.md: validate n=1e6 hierarchical construction, convection,
    # counters, and error plumbing; flat dense ell=4 F64 is the 024b baseline
    # cross-check (host_step_ms should land near 0.4246 s).
    : "${FM028_N:=1000000}"
    : "${FM028_P:=3}"
    : "${FM028_TF:=Float64}"
    : "${FM028_LH:=0}"
    : "${FM028_REPS:=5}"
    : "${FM028_STEPS:=3}"
    : "${FM028_DT:=1e-5}"
    export FM028_N FM028_P FM028_TF FM028_LH FM028_REPS FM028_STEPS FM028_DT

    echo "=== pilot case 1: hier12 dense ell=5 K=256"
    FM028_POLICY=hier12 FM028_STRAT=dense FM028_ELL=5 FM028_K=256 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    bench1_status=$?
    echo "PILOT1_EXIT=$bench1_status"

    echo "=== pilot case 2: flat dense ell=4 (024b cross-check)"
    FM028_POLICY=flat FM028_STRAT=dense FM028_ELL=4 FM028_K=256 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    bench2_status=$?
    echo "PILOT2_EXIT=$bench2_status"
    bench_status=$(( bench1_status || bench2_status ))
elif [ "$MODE" = "verify" ]; then
    # Lever 3 before/after: re-run the exact Phase A verdict config so
    # verdict_step_ms, verdict_step_host_alloc_bytes, the counter contract and
    # the accuracy gate are all directly comparable to job 12997508.
    # Phase A baseline: F64 104.0-106.3 ms / F32 91.4 ms, 57.7 MB/step.
    : "${FM028_N:=1000000}"
    : "${FM028_P:=3}"
    : "${FM028_ELL:=5}"
    : "${FM028_K:=1740}"
    : "${FM028_POLICY:=hier12}"
    : "${FM028_STRAT:=dense}"
    : "${FM028_TF:=Float64,Float32}"
    : "${FM028_LH:=0}"
    : "${FM028_REPS:=5}"
    : "${FM028_STEPS:=5}"
    : "${FM028_STALE:=0}"
    export FM028_N FM028_P FM028_ELL FM028_K FM028_POLICY FM028_STRAT \
        FM028_TF FM028_LH FM028_REPS FM028_STEPS FM028_STALE
    echo "=== verify (lever 3 before/after)"
    julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    bench_status=$?
    echo "VERIFY_EXIT=$bench_status"
elif [ "$MODE" = "derisk" ]; then
    # Phase B step 2: resolve the three Phase A inferences (exact L2B/nearfield
    # split, host-allocation profile, per-kernel trace). No src/ changes.
    : "${FM028_N:=1000000}"
    : "${FM028_P:=3}"
    # the {4,5,6} L2B/nearfield bracket was measured in job 12998146 (part A);
    # subsequent runs only need the verdict ell
    : "${FM028_ELL:=5}"
    : "${FM028_K:=1740}"
    : "${FM028_REPS:=5}"
    export FM028_N FM028_P FM028_ELL FM028_K FM028_REPS
    echo "=== derisk"
    julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_derisk.jl
    bench_status=$?
    echo "DERISK_EXIT=$bench_status"
else
    # 04-phase-sweep.md core matrix, TIERED using pilot 12996475 evidence.
    # The original full cross product (2 n x 3 ell x 2 K x 2 policy x 2 strat x
    # 2 TF = 96 hier cases, ~4h at the measured ~2.5 min/case) is pruned: the
    # pilot showed ell=5 is the integer optimum at n=1e6 (ell=4/6 extrapolate to
    # ~430/~220 ms vs 127 ms), so ell is bracketed with one confirming case per
    # side instead of being swept across every other axis.
    : "${FM028_P:=3}"
    : "${FM028_REPS:=5}"
    : "${FM028_STEPS:=5}"
    : "${FM028_DT:=1e-5}"
    export FM028_P FM028_REPS FM028_STEPS FM028_DT
    sweep_status=0
    _run() {   # _run <label> <env assignments...>
        local label="$1"; shift
        echo "=== $label"
        env "$@" julia --project="$ENVDIR" \
            MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
        local s=$?
        echo "EXIT[$label]=$s"
        sweep_status=$(( sweep_status || s ))
    }

    # Tier A - ell bracket confirmation at n=1e6 (2 cases). Confirms by
    # measurement what the pilot only extrapolated.
    _run "tierA ell bracket" FM028_N=1000000 FM028_ELL=4,6 FM028_K=256 \
        FM028_POLICY=hier12 FM028_STRAT=dense FM028_TF=Float64 FM028_LH=0 \
        FM028_STALE=0

    # Tier B - main matrix at the optimum ell=5, n=1e6 (16 cases):
    # policy x strategy x K x precision. First hierarchical Float32 data
    # anywhere; K is untested at n=1e6 and drives the route_gen sync count.
    _run "tierB main ell=5" FM028_N=1000000 FM028_ELL=5 FM028_K=256,1740 \
        FM028_POLICY=hier12,hier3 FM028_STRAT=dense,precomputed_y \
        FM028_TF=Float64,Float32 FM028_LH=0 FM028_STALE=0

    # Tier C - scaling tie-in to the 027 record (3 cases): exposes the exponent
    # between n=2e5 and n=1e6 on the hierarchical path.
    _run "tierC scaling n" FM028_N=200000,316228 FM028_ELL=5 FM028_K=1740 \
        FM028_POLICY=hier12 FM028_STRAT=dense FM028_TF=Float64 FM028_LH=0 \
        FM028_STALE=0
    _run "tierC 027 tie-in" FM028_N=200000 FM028_ELL=4 FM028_K=1740 \
        FM028_POLICY=hier12 FM028_STRAT=dense FM028_TF=Float64 FM028_LH=0 \
        FM028_STALE=0

    # Tier D - Lamb-Helmholtz on (1 case): no hierarchical LH data exists.
    _run "tierD LH on" FM028_N=1000000 FM028_ELL=5 FM028_K=1740 \
        FM028_POLICY=hier12 FM028_STRAT=dense FM028_TF=Float64 FM028_LH=1 \
        FM028_STALE=0

    # Tier E - stale-tree refresh policy accuracy (1 case): timing already
    # known weak (saves 13.4 ms), this measures what the accuracy costs.
    _run "tierE stale policy" FM028_N=1000000 FM028_ELL=5 FM028_K=1740 \
        FM028_POLICY=hier12 FM028_STRAT=dense FM028_TF=Float64 FM028_LH=0 \
        FM028_STALE=5

    # flat baseline rows (024b cross-check), F64 + F32
    _run "flat baseline" FM028_N=1000000 FM028_ELL=4 FM028_K=256 \
        FM028_POLICY=flat FM028_STRAT=dense FM028_TF=Float64,Float32 \
        FM028_LH=0 FM028_STALE=0
    bench_status=$sweep_status
fi

exit $(( lifecycle_status || convection_status || bench_status ))
