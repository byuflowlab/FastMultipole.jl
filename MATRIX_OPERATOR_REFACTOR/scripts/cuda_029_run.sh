#!/bin/bash
#SBATCH --job-name=fm029
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Task 029 step 1: fresh single-H200 baseline profile of the frozen 1M-body,
# literature-P=4 workload (028 seeds/bounds/gate), in separately initialized
# processes, with the complete leaderboard environment record (power limit,
# clock/persistence mode, GPU topology) that the 029 record format requires.
#
# Baseline geometries (per the 029 Reading Gate Record):
#   sched6-5-5-5  fp16/F32  the 028 shipped default (record 9.591/9.653 ms)
#   sched6-5-4-4  fp16/F32  the 030 robust winner  (record 7.556 ms, 0.87x gate)
#   sched6-4-4-3  fp16/F32  the 030 knife-edge     (record 7.092 ms, 1.00x gate)
#   sched6-5-5-5  off/F64   dual-precision context row
#
# benchmark_028_feasibility.jl is reused UNCHANGED (one case per process; the
# 030 skip/ledger idiom). Accuracy gate stays 1.19e-3; FM028_BOUND=ab records
# the eval-only and verdict boundaries.
DRYRUN="${FM029_DRYRUN:-0}"

set -o pipefail
if [ "$DRYRUN" != "1" ]; then
    source /etc/profile
    # julia pinned: module default 1.12.6 segfaults host LLVM JIT on the device
    # step (clean-env repro job 13058336); 1.11.7 is the toolchain of record
    module load cuda julia/1.11.7-6bmogfl
    echo "=== node: $(hostname)"
    nvidia-smi -L
    echo "CUDA_HOME=${CUDA_HOME:-unset}"
    # 029 leaderboard environment record: driver/power/clock/persistence state
    # and the intra-node GPU topology (for the later multi-H200 track)
    echo "=== nvidia-smi power/clock record"
    nvidia-smi -q -d POWER,CLOCK,PERFORMANCE | grep -E \
      "Power Limit|Persistence|Performance State|Clocks Event|SM *:|Memory *:|Applications" | head -40
    echo "=== nvidia-smi topology"
    nvidia-smi topo -m
fi

WORKDIR="${FM029_DIR:-$HOME/FastMultipole-023}"
ENVDIR="${FM029_ENV:-$HOME/fm023env}"
OUTDIR="${FM029_OUTDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms}"
REFDIR="${FM029_REFDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references}"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

if [ "$DRYRUN" = "1" ]; then
    echo "=== FM029_DRYRUN=1: skipping preflight, reference gate, and all julia runs"
else

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

echo "=== test/cuda_radix_convection_test.jl"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_convection_test.jl
convection_status=$?
echo "CONVECTION_TEST_EXIT=$convection_status"

echo "=== test/cuda_radix_counting_sort_test.jl"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_counting_sort_test.jl
counting_sort_status=$?
echo "COUNTING_SORT_TEST_EXIT=$counting_sort_status"

if (( lifecycle_status || convection_status || counting_sort_status )); then
    echo "PREFLIGHT_EXIT=1"
    exit 1
fi

echo "=== 024b reference integrity (n=1e6)"
if [ ! -s "$REFDIR/direct_reference_n1000000.csv" ]; then
    echo "MISSING_REFERENCE n=1000000 in $REFDIR" >&2
    echo "REFERENCE_GATE_EXIT=1"
    exit 1
fi
(cd "$REFDIR" && shasum -a 256 -c direct_reference_checksums.sha256) || {
    echo "REFERENCE_GATE_EXIT=1"; exit 1; }
echo "REFERENCE_GATE_EXIT=0"

fi   # end non-dry-run preflight

COMMON=(FM028_P=3 FM028_STRAT=dense FM028_LH=0 FM028_K=full
        FM028_REPS="${FM029_REPS:-15}" FM028_STEPS=0 FM028_STALE=0
        FM028_BOUND="${FM029_BOUND:-ab}" FM028_SYMMETRIC=0
        FM028_M2L_THREADS=64 FM028_M2L_BLOCK_CAP=65536 FM028_COUNTING_SORT=1
        FM028_REFDIR="$REFDIR")

geometry_tag () { echo "${1//-/_}"; }

failed_cases=0
run_case () {   # run_case <policy> <ell> <tf> <fmt>
  local policy=$1 ell=$2 tf=$3 fmt=$4
  local gtag out
  gtag="$(geometry_tag "$policy")"
  out="$OUTDIR/cuda029_base_${gtag}_${fmt}_n1000000_$(hostname)_${SLURM_JOB_ID:-manual}.csv"
  echo "=== 029 baseline policy=$policy ell=$ell tf=$tf tensor=$fmt"
  if [ "$DRYRUN" = "1" ]; then
    echo "    DRYRUN env ${COMMON[*]} FM028_N=1000000 FM028_ELL=$ell FM028_POLICY=$policy FM028_TF=$tf FM028_TENSOR_FORMAT=$fmt FM028_OUT=$(basename "$out")"
    return 0
  fi
  if ! env "${COMMON[@]}" \
        FM028_N=1000000 FM028_ELL="$ell" FM028_POLICY="$policy" \
        FM028_TF="$tf" FM028_TENSOR_FORMAT="$fmt" FM028_OUT="$out" \
        julia --project="$ENVDIR" \
          MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl; then
    echo "CASE_FAILED policy=$policy fmt=$fmt" >&2
    failed_cases=$((failed_cases + 1))
  fi
}

run_case sched6-5-5-5 5 Float32 fp16
run_case sched6-5-4-4 5 Float32 fp16
run_case sched6-4-4-3 5 Float32 fp16
run_case sched6-5-5-5 5 Float64 off

echo "=== 029 baseline complete; failed_cases=$failed_cases"
if (( failed_cases > 0 )); then
  echo "BASELINE_EXIT=1"
  exit 1
fi
echo "BASELINE_EXIT=0"
