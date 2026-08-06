#!/bin/bash
#SBATCH --job-name=fm029c1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Task 029 cycle 1 (user-approved 2026-08-06): graph-captured/sync-free
# far-field chain + occupancy-epoch route-window fold, verified against the
# step-1 baselines (job 13059710) on the same frozen workload.
#
# Sections:
#   1. full CUDA preflight suites (incl. the new cuda_radix_graph_test.jl)
#   2. verdict A/B at the three baseline geometries, FP16/F32, REPS>=15:
#        on      = cached windows + graph capture (new production defaults)
#        cachedo = cached windows only (graph off) — per-item attribution
#        off     = both flags off (pre-029 path, same manifest)
#      plus the F64/off context row with defaults on
#   3. re-profile (profile_029_floor.jl, defaults on) — launch/sync counts
#      must actually fall vs job 13059955
#   4. P3 rider: ISOLATED fixed-work nearfield ILP A/B (no production change)
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

preflight_fail=0
for t in cuda_radix_lifecycle_test cuda_radix_convection_test \
         cuda_radix_counting_sort_test cuda_radix_hierarchical_test \
         cuda_radix_graph_test cuda_radix_interface_test; do
    echo "=== test/$t.jl"
    FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" "test/$t.jl"
    s=$?
    echo "${t^^}_EXIT=$s"
    (( s )) && preflight_fail=1
done
if (( preflight_fail )); then
    echo "PREFLIGHT_EXIT=1"
    exit 1
fi
echo "PREFLIGHT_EXIT=0"

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
        FM028_REPS="${FM029_REPS:-15}" FM028_STEPS=5 FM028_STALE=0
        FM028_BOUND="${FM029_BOUND:-ab}" FM028_SYMMETRIC=0
        FM028_M2L_THREADS=64 FM028_M2L_BLOCK_CAP=65536 FM028_COUNTING_SORT=1
        FM028_REFDIR="$REFDIR")

geometry_tag () { echo "${1//-/_}"; }

failed_cases=0
run_case () {   # run_case <mode> <policy> <ell> <tf> <fmt>
  local mode=$1 policy=$2 ell=$3 tf=$4 fmt=$5
  local cached=1 graph=1
  case "$mode" in
    on)      cached=1; graph=1 ;;
    cachedo) cached=1; graph=0 ;;
    off)     cached=0; graph=0 ;;
    *) echo "bad mode $mode"; exit 2 ;;
  esac
  local gtag out
  gtag="$(geometry_tag "$policy")"
  out="$OUTDIR/cuda029c1_${mode}_${gtag}_${fmt}_n1000000_$(hostname)_${SLURM_JOB_ID:-manual}.csv"
  echo "=== 029c1 mode=$mode policy=$policy ell=$ell tf=$tf tensor=$fmt"
  if [ "$DRYRUN" = "1" ]; then
    echo "    DRYRUN FM029_CACHED=$cached FM029_GRAPH=$graph ${COMMON[*]} FM028_POLICY=$policy FM028_TF=$tf FM028_TENSOR_FORMAT=$fmt OUT=$(basename "$out")"
    return 0
  fi
  if ! env "${COMMON[@]}" \
        FM029_CACHED="$cached" FM029_GRAPH="$graph" \
        FM028_N=1000000 FM028_ELL="$ell" FM028_POLICY="$policy" \
        FM028_TF="$tf" FM028_TENSOR_FORMAT="$fmt" FM028_OUT="$out" \
        julia --project="$ENVDIR" \
          MATRIX_OPERATOR_REFACTOR/scripts/benchmark_029_cycle1.jl; then
    echo "CASE_FAILED mode=$mode policy=$policy fmt=$fmt" >&2
    failed_cases=$((failed_cases + 1))
  fi
}

# after (defaults): the cycle-1 candidate rows
run_case on sched6-5-5-5 5 Float32 fp16
run_case on sched6-5-4-4 5 Float32 fp16
run_case on sched6-4-4-3 5 Float32 fp16
run_case on sched6-5-5-5 5 Float64 off
# per-item attribution: cached windows without graph capture
run_case cachedo sched6-5-4-4 5 Float32 fp16
run_case cachedo sched6-4-4-3 5 Float32 fp16
# before: pre-029 path on this manifest (baseline continuity vs job 13059710)
run_case off sched6-5-5-5 5 Float32 fp16
run_case off sched6-5-4-4 5 Float32 fp16
run_case off sched6-4-4-3 5 Float32 fp16

echo "=== 029c1 verdict A/B complete; failed_cases=$failed_cases"

if [ "$DRYRUN" != "1" ]; then
echo "=== 029c1 re-profile (defaults on; compare against job 13059955)"
FM029P_OUT="$OUTDIR/cuda029c1_profile_$(hostname)_${SLURM_JOB_ID:-manual}" \
FM029P_N=1000000,1000 FM029P_REPS=5 \
  julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/profile_029_floor.jl \
  || { echo "PROFILE_FAILED" >&2; failed_cases=$((failed_cases + 1)); }

echo "=== 029c1 P3 rider: isolated nearfield ILP A/B (no production change)"
for pol in sched6-5-4-4 sched6-4-4-3; do
  FM029N_POLICY="$pol" FM029N_N=1000000 FM029N_REPS=25 \
  FM029N_OUT="$OUTDIR/cuda029c1_nearfield_ilp_${pol//-/_}_$(hostname)_${SLURM_JOB_ID:-manual}.csv" \
    julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/prototype_029_nearfield_ilp.jl \
    || { echo "NEARFIELD_ILP_FAILED policy=$pol" >&2; failed_cases=$((failed_cases + 1)); }
done
fi

echo "=== 029c1 complete; failed_cases=$failed_cases"
if (( failed_cases > 0 )); then
  echo "CYCLE1_EXIT=1"
  exit 1
fi
echo "CYCLE1_EXIT=0"
