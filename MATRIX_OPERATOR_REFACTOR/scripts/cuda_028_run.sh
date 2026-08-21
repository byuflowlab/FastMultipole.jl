#!/bin/bash
#SBATCH --job-name=fm028
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Task 028: H200 feasibility measurement — can n=1e6 run in 10 ms per step?
# Preflight: CUDA lifecycle test + the new 028 convection test, then the
# feasibility benchmark. FM028_MODE selects the case matrix
# (individual FM028_* overrides exported at submit time win over the presets).
# Pattern: cuda_027_run.sh.
source /etc/profile
set -o pipefail
# julia pinned: module default moved to 1.12.6 on 2026-08-05 which segfaults host LLVM JIT (job 13058191); 1.11.7 is the toolchain of record
module load cuda julia/1.11.7-6bmogfl
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

# Small and fast, and the counting sort is on by default for bounded depths, so
# it belongs in the preflight rather than only in the stage6sort A/B mode.
echo "=== test/cuda_radix_counting_sort_test.jl (028 stage 6)"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_counting_sort_test.jl
counting_sort_status=$?
echo "COUNTING_SORT_TEST_EXIT=$counting_sort_status"

if (( lifecycle_status || convection_status || counting_sort_status )); then
    echo "PREFLIGHT_EXIT=1"
    exit 1
fi

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
elif [ "$MODE" = "attribute" ]; then
    # Cycle 4 evidence-only pass: fixed-work A/Bs and launch-shape sweeps for
    # the two residual kernels, followed by fresh optimized q=3/q=12 endpoint
    # timings + sampled-direct errors. No src changes belong to this pass.
    : "${FM028_N:=1000000}"
    : "${FM028_P:=3}"
    : "${FM028_ELL:=5}"
    : "${FM028_K:=1740}"
    : "${FM028_REPS:=7}"
    export FM028_N FM028_P FM028_ELL FM028_K FM028_REPS

    echo "=== cycle 4 residual kernel attribution"
    julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_attribution.jl
    attr_status=$?
    echo "ATTRIBUTION_EXIT=$attr_status"

    echo "=== cycle 4 optimized radius endpoints (q=3/q=12)"
    FM028_POLICY=hier3,hier12 FM028_STRAT=dense FM028_TF=Float32 FM028_LH=0 \
        FM028_STEPS=0 FM028_STALE=0 FM028_BOUND=ab \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    endpoint_status=$?
    echo "RADIUS_ENDPOINT_EXIT=$endpoint_status"
    bench_status=$(( attr_status || endpoint_status ))
elif [ "$MODE" = "stage5" ]; then
    # Complete intermediate-radius accuracy/performance frontier.  Each shell
    # uses one full union-class window, eliminating window-count as a confounder.
    : "${FM028_N:=1000000}"
    : "${FM028_P:=3}"
    : "${FM028_ELL:=5}"
    : "${FM028_K:=full}"
    : "${FM028_POLICY:=hier3,hier4,hier5,hier6,hier8,hier9,hier10,hier11,hier12}"
    : "${FM028_STRAT:=dense}"
    : "${FM028_TF:=Float32}"
    : "${FM028_LH:=0}"
    : "${FM028_REPS:=7}"
    : "${FM028_STEPS:=0}"
    : "${FM028_STALE:=0}"
    : "${FM028_BOUND:=ab}"
    export FM028_N FM028_P FM028_ELL FM028_K FM028_POLICY FM028_STRAT \
        FM028_TF FM028_LH FM028_REPS FM028_STEPS FM028_STALE FM028_BOUND

    echo "=== stage 5: complete rigid-radius frontier"
    julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    bench_status=$?
    echo "STAGE5_EXIT=$bench_status"
elif [ "$MODE" = "stage6" ]; then
    # Selected-shell depth bracket plus complete-verdict M2L launch A/B.
    # FM028_SELECTED_Q is set after Stage 5; no public API is introduced.
    : "${FM028_SELECTED_Q:?stage6 requires FM028_SELECTED_Q}"
    : "${FM028_N:=1000000}"
    : "${FM028_P:=3}"
    : "${FM028_REPS:=9}"
    export FM028_N FM028_P FM028_REPS
    common=(FM028_N="$FM028_N" FM028_P="$FM028_P" FM028_K=full
        FM028_POLICY="hier$FM028_SELECTED_Q" FM028_STRAT=dense FM028_LH=0
        FM028_STEPS=0 FM028_STALE=0 FM028_BOUND=ab FM028_REPS="$FM028_REPS")

    echo "=== stage 6a: q=$FM028_SELECTED_Q ell=4,5,6 bracket"
    env "${common[@]}" FM028_ELL=4,5,6 FM028_TF=Float32 \
        FM028_M2L_THREADS=128 FM028_M2L_BLOCK_CAP=16384 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    bracket_status=$?

    echo "=== stage 6b: current tiled-M2L launch, independent F32/F64 verdict"
    env "${common[@]}" FM028_ELL=5 FM028_TF=Float32,Float64 \
        FM028_M2L_THREADS=128 FM028_M2L_BLOCK_CAP=16384 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    launch_a_status=$?

    echo "=== stage 6b: candidate tiled-M2L launch, independent F32/F64 verdict"
    env "${common[@]}" FM028_ELL=5 FM028_TF=Float32,Float64 \
        FM028_M2L_THREADS=64 FM028_M2L_BLOCK_CAP=65536 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    launch_b_status=$?
    bench_status=$(( bracket_status || launch_a_status || launch_b_status ))
    echo "STAGE6_EXIT=$bench_status"
elif [ "$MODE" = "stage6sort" ]; then
    : "${FM028_SELECTED_Q:?stage6sort requires FM028_SELECTED_Q}"
    : "${FM028_N:=1000000}"
    : "${FM028_P:=3}"
    : "${FM028_REPS:=9}"
    common=(FM028_N="$FM028_N" FM028_P="$FM028_P" FM028_ELL=5 FM028_K=full
        FM028_POLICY="hier$FM028_SELECTED_Q" FM028_STRAT=dense
        FM028_TF=Float32,Float64 FM028_LH=0 FM028_STEPS=0 FM028_STALE=0
        FM028_BOUND=ab FM028_REPS="$FM028_REPS")

    echo "=== stage 6c counting-sort parity/lifecycle gate"
    FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" \
        test/cuda_radix_counting_sort_test.jl
    counting_test_status=$?

    echo "=== stage 6c: existing device sortperm!"
    env "${common[@]}" FM028_COUNTING_SORT=0 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    sort_a_status=$?
    echo "=== stage 6c: bounded Morton counting sort"
    env "${common[@]}" FM028_COUNTING_SORT=1 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    sort_b_status=$?
    bench_status=$(( counting_test_status || sort_a_status || sort_b_status ))
    echo "STAGE6SORT_EXIT=$bench_status"
elif [ "$MODE" = "stage56gate" ]; then
    # Final combined production-default gate: every supported shell's compact
    # host/device geometry parity plus q=6 F32/F64 full verdicts with both
    # retained Stage 6 optimizations enabled by default.
    echo "=== stages 5-6 all-radius CUDA parity gate"
    FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" \
        test/cuda_radix_hierarchical_test.jl
    radius_gate_status=$?
    echo "=== stages 5-6 final selected production defaults"
    FM028_N=1000000 FM028_P=3 FM028_ELL=5 FM028_K=full \
        FM028_POLICY=hier6 FM028_STRAT=dense FM028_TF=Float32,Float64 \
        FM028_LH=0 FM028_REPS=9 FM028_STEPS=0 FM028_STALE=0 FM028_BOUND=ab \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    final_status=$?
    bench_status=$(( radius_gate_status || final_status ))
    echo "STAGE56GATE_EXIT=$bench_status"
elif [ "$MODE" = "stage7" ]; then
    # Stage 7 adjacent q=5/q=6 shell frontier. The four schedule entries are
    # levels 2:5; every entry selects a complete rigid cubic-orbit table and
    # the leaf entry controls the direct list.
    echo "=== stage 7 schedule host/device route and output gate"
    FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" \
        test/cuda_radix_hierarchical_test.jl
    schedule_gate_status=$?
    if (( schedule_gate_status )); then
        echo "STAGE7_GATE_EXIT=$schedule_gate_status"
        exit 1
    fi
    echo "=== stage 7 q5/q6 sampled-field replay attribution"
    FM028_N=1000000 FM028_P=3 FM028_ELL=5 FM028_TF=Float32 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_stage7_replay.jl
    replay_status=$?
    if (( replay_status )); then
        echo "STAGE7_REPLAY_EXIT=$replay_status"
        exit 1
    fi
    echo "=== stage 7 complete adjacent-shell schedule frontier"
    FM028_N=1000000 FM028_P=3 FM028_ELL=5 FM028_K=full \
        FM028_POLICY=hier5,sched6-5-5-5,sched6-6-5-5,sched6-6-6-5,hier6 \
        FM028_STRAT=dense FM028_TF=Float32 FM028_LH=0 FM028_REPS=9 \
        FM028_STEPS=0 FM028_STALE=0 FM028_BOUND=ab \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    frontier_status=$?
    bench_status=$(( schedule_gate_status || replay_status || frontier_status ))
    echo "STAGE7_EXIT=$bench_status"
elif [ "$MODE" = "stage8" ]; then
    if [[ -z "${FM028_SELECTED_POLICY:-}" || -z "${FM028_SELECTED_SCHEDULE:-}" ]]; then
        read -r FM028_SELECTED_POLICY FM028_SELECTED_SCHEDULE < <(
            julia --project="$ENVDIR" \
                MATRIX_OPERATOR_REFACTOR/scripts/select_028_stage7_winner.jl)
    fi
    export FM028_SELECTED_POLICY FM028_SELECTED_SCHEDULE
    echo "=== stage 8 selected geometry: $FM028_SELECTED_POLICY ($FM028_SELECTED_SCHEDULE)"
    echo "=== stage 8 symmetric/tensor CUDA parity and fallback gate"
    FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" \
        test/cuda_radix_hierarchical_test.jl
    bakeoff_gate_status=$?
    if (( bakeoff_gate_status )); then
        echo "STAGE8_GATE_EXIT=$bakeoff_gate_status"
        exit 1
    fi

    echo "=== stage 8 Float64 singular-spectrum audit"
    FM028_P=3 FM028_ELL=5 FM028_SCHEDULE="$FM028_SELECTED_SCHEDULE" \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/audit_028_low_rank.jl
    lowrank_status=$?

    common=(FM028_N=1000000 FM028_P=3 FM028_ELL=5 FM028_K=full
        FM028_POLICY="$FM028_SELECTED_POLICY" FM028_STRAT=dense FM028_LH=0
        FM028_REPS=9 FM028_STEPS=0 FM028_STALE=0 FM028_BOUND=ab)
    echo "=== stage 8 baseline F32/F64 budgets"
    env "${common[@]}" FM028_TF=Float32,Float64 FM028_SYMMETRIC=0 \
        FM028_TENSOR_FORMAT=off julia --project="$ENVDIR" \
        MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    baseline_status=$?
    echo "=== stage 8 unordered symmetric nearfield F32/F64"
    env "${common[@]}" FM028_TF=Float32,Float64 FM028_SYMMETRIC=1 \
        FM028_TENSOR_FORMAT=off julia --project="$ENVDIR" \
        MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    symmetric_status=$?

    tensor_status=0
    for format in tf32 fp16 bf16; do
        echo "=== stage 8 tensor M2L format=$format"
        env "${common[@]}" FM028_TF=Float32 FM028_SYMMETRIC=0 \
            FM028_TENSOR_FORMAT="$format" julia --project="$ENVDIR" \
            MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
        s=$?
        echo "TENSOR_${format}_EXIT=$s"
        tensor_status=$(( tensor_status || s ))
    done
    bench_status=$(( bakeoff_gate_status || lowrank_status || baseline_status || symmetric_status || tensor_status ))
    echo "STAGE8_EXIT=$bench_status"
elif [ "$MODE" = "stage8repro" ]; then
    : "${FM028_SELECTED_POLICY:?stage8repro requires FM028_SELECTED_POLICY}"
    : "${FM028_SELECTED_SCHEDULE:?stage8repro requires FM028_SELECTED_SCHEDULE}"
    echo "=== stage 8 independent tensor reproduction: $FM028_SELECTED_POLICY ($FM028_SELECTED_SCHEDULE)"
    FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" \
        test/cuda_radix_hierarchical_test.jl
    repro_gate_status=$?
    if (( repro_gate_status )); then
        echo "STAGE8_REPRO_GATE_EXIT=$repro_gate_status"
        exit 1
    fi
    common=(FM028_N=1000000 FM028_P=3 FM028_ELL=5 FM028_K=full
        FM028_POLICY="$FM028_SELECTED_POLICY" FM028_STRAT=dense FM028_TF=Float32
        FM028_LH=0 FM028_REPS=9 FM028_STEPS=0 FM028_STALE=0 FM028_BOUND=ab
        FM028_SYMMETRIC=0)
    repro_status=0
    for format in fp16 bf16; do
        echo "=== stage 8 independent tensor reproduction format=$format"
        env "${common[@]}" FM028_TENSOR_FORMAT="$format" \
            julia --project="$ENVDIR" \
            MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
        s=$?
        echo "STAGE8_REPRO_${format}_EXIT=$s"
        repro_status=$(( repro_status || s ))
    done
    echo "STAGE8_REPRO_EXIT=$repro_status"
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
