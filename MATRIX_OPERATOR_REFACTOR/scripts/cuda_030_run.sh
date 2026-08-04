#!/bin/bash
#SBATCH --job-name=fm030
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Task 030: per-time-step cost vs n at the 028 shipped defaults, as three
# fixed-depth series (ell = 3/4/5) in FP16-WMMA/Float32 and Float64.
#
# Depth bracket (user direction 2026-08-03, revised from 4/5/6): ell=5 is the
# 028 optimum at n=1e6, and the optimal depth tracks n *downward* — 028 §4.8
# measured ell=4 as optimal at n=2e5, and 024b selected ell=2/3 below n=1e5. A
# sweep spanning n=1e3..1e6 therefore needs the coarse side of ell=5 bracketed,
# not the fine side; ell=6 was only competitive above the target n.
#
# Preflight is the 028 set (lifecycle + convection + counting sort), then the
# case loop. FM030_MODE selects the case matrix:
#   sweep      42 cases: n in {1e3..1e6} x ell in {4,5,6} x {fp16-F32, off-F64}
#   spotcheck  baseline-vs-candidate pairs at selected n (phase 5 validation)
#
# Structure note (why one julia process per case rather than one comma-list
# invocation per series): benchmark_028_feasibility.jl writes its CSV only after
# the whole in-process sweep finishes, so a walltime kill or a hard abort loses
# every row in that process. A scheduled `sched*` policy also pins exactly one
# ell (its schedule must have ell-1 entries) and FM028_TENSOR_FORMAT is read
# once at process load, so the series axes cannot be swept in-process anyway.
# One case per process makes the completed-case skip a file test and makes a
# resubmission resume at case granularity.
#
# benchmark_028_feasibility.jl is reused UNCHANGED; every 028 mode stays
# reproducible. Pattern: cuda_028_run.sh + the cuda_024b_run.sh skip/ledger.
#
# FM030_DRYRUN=1 skips the GPU preflight and prints each case's julia command
# instead of running it, so the 42-case matrix, the schedule mapping, the
# completed-case skip, and the ledger can be exercised on a laptop. The 024b
# campaign burned several cluster jobs on driver-script defects; this makes that
# class of defect cheap to catch. It has no effect on a real run.
DRYRUN="${FM030_DRYRUN:-0}"

set -o pipefail
if [ "$DRYRUN" != "1" ]; then
    source /etc/profile
    module load cuda julia
    echo "=== node: $(hostname)"
    nvidia-smi -L
    echo "CUDA_HOME=${CUDA_HOME:-unset}"
fi

WORKDIR="${FM030_DIR:-$HOME/FastMultipole-023}"
ENVDIR="${FM030_ENV:-$HOME/fm023env}"
OUTDIR="${FM030_OUTDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cost_vs_n}"
REFDIR="${FM030_REFDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references}"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

if [ "$DRYRUN" = "1" ]; then
    echo "=== FM030_DRYRUN=1: skipping preflight, reference gate, and all julia runs"
    lifecycle_status=0; convection_status=0; counting_sort_status=0
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

# ---- reference integrity gate ----------------------------------------------
# A missing reference is NOT an error inside the harness: it silently falls back
# to reference_source=device_direct, which would quietly detach a sweep point
# from the checksummed 024b baseline. Gate here instead.
echo "=== 024b reference integrity"
for n in 1000 3162 10000 31623 100000 316228 1000000; do
  if [ ! -s "$REFDIR/direct_reference_n${n}.csv" ]; then
    echo "MISSING_REFERENCE n=$n in $REFDIR" >&2
    echo "REFERENCE_GATE_EXIT=1"
    exit 1
  fi
done
(cd "$REFDIR" && shasum -a 256 -c direct_reference_checksums.sha256) || {
    echo "REFERENCE_GATE_EXIT=1"; exit 1; }
echo "REFERENCE_GATE_EXIT=0"

fi   # end of the non-dry-run preflight block

# ---- frozen workload --------------------------------------------------------
# 030 §"Fixed Comparable Workload": seed/box/P/lh/strategy/K/threads are pinned
# by the harness or here; only n, ell, policy, precision, tensor format vary.
COMMON=(FM028_P=3 FM028_STRAT=dense FM028_LH=0 FM028_K=full
        FM028_REPS="${FM030_REPS:-9}" FM028_STEPS=0 FM028_STALE=0
        FM028_BOUND="${FM030_BOUND:-ab}" FM028_SYMMETRIC=0
        FM028_M2L_THREADS=64 FM028_M2L_BLOCK_CAP=65536 FM028_COUNTING_SORT=1
        FM028_REFDIR="$REFDIR")

# A case's geometry is given either as a bare ell (which selects the shipped
# (6,5,...,5) schedule for that depth) or as an explicit sched<q2>-...-<qell>
# string, so a spot-check can vary the radius schedule and not only the depth.
# Either way the schedule must have exactly ell-1 non-increasing entries, and
# the harness re-validates that at `_cache_kwargs`.
schedule_for_ell () {
  case "$1" in
    2) echo "sched5" ;;
    3) echo "sched6-5" ;;
    4) echo "sched6-5-5" ;;
    5) echo "sched6-5-5-5" ;;
    6) echo "sched6-5-5-5-5" ;;
    *) echo "unsupported ell=$1" >&2; return 1 ;;
  esac
}

# resolve_geometry <spec> -> "<ell> <policy>"; spec is an integer ell or a
# sched... string whose ell is (number of entries) + 1.
resolve_geometry () {
  local spec=$1 ell policy entries
  if [[ "$spec" =~ ^[0-9]+$ ]]; then
    policy="$(schedule_for_ell "$spec")" || return 1
    ell="$spec"
  elif [[ "$spec" == sched* ]]; then
    policy="$spec"
    entries="${spec#sched}"
    ell=$(( $(tr -cd '-' <<< "$entries" | wc -c) + 2 ))
  else
    echo "unsupported geometry spec '$spec'" >&2
    return 1
  fi
  echo "$ell $policy"
}

# A filesystem-safe tag for a geometry, used in the per-case CSV name so two
# different schedules at the same depth cannot collide.
geometry_tag () { echo "${1//-/_}"; }

failures="$OUTDIR/cost_vs_n_failures_$(hostname)_${SLURM_JOB_ID}.csv"
failed_cases=0

# A device-capacity failure is a hardware result and belongs in the published
# ledger; anything else is unexplained and must not be recorded as capacity.
classify_failure () {
  local log=$1
  if grep -qiE 'out of (gpu )?memory|OutOfMemoryError|exceeds the free-memory budget|max_persistent_bytes|estimated device peak' "$log"; then
    echo "failed_capacity"
  else
    echo "failed_other"
  fi
}

ledger () {   # ledger <tag> <geometry> <tf> <fmt> <n> <status>
  if [[ ! -s "$failures" ]]; then
    echo "tag,geometry,tf,tensor_format,n,status,job_id" > "$failures"
  fi
  echo "$1,$2,$3,$4,$5,$6,$SLURM_JOB_ID" >> "$failures"
  failed_cases=$((failed_cases + 1))
}

# One case per process, one CSV per case. `fit=true` is field 21 of the 86-column
# 028 schema, so completion is checked by field position rather than by a
# `^`-anchored prefix grep (the 028 CSV starts with manifest,job,host,...).
# The `! -name '*.classes.csv'` guard avoids the companion-file trap.
case_csv () {   # case_csv <tag> <geometry_tag> <fmt> <n>
  echo "$OUTDIR/cuda030_$1_$2_$3_n$4_$(hostname)_${SLURM_JOB_ID}.csv"
}

case_done () {  # case_done <tag> <geometry_tag> <fmt> <n>
  local f
  while read -r f; do
    [[ -n "$f" ]] || continue
    awk -F, 'NR>1 && $21=="true" {found=1} END {exit !found}' "$f" && return 0
  done < <(find "$OUTDIR" -maxdepth 1 -name "cuda030_$1_$2_$3_n$4_*.csv" \
             ! -name '*.classes.csv')
  return 1
}

# run_case <tag> <geometry: ell or sched-string> <tf> <fmt> <n> [extra env...]
run_case () {
  local tag=$1 geom=$2 tf=$3 fmt=$4 n=$5; shift 5
  local ell policy gtag out caselog status resolved
  resolved="$(resolve_geometry "$geom")" || {
      ledger "$tag" "$geom" "$tf" "$fmt" "$n" "bad_geometry"; return 0; }
  read -r ell policy <<< "$resolved"
  gtag="$(geometry_tag "$policy")"
  if case_done "$tag" "$gtag" "$fmt" "$n"; then
    echo "=== skip completed $tag $policy $fmt n=$n"
    return 0
  fi
  out="$(case_csv "$tag" "$gtag" "$fmt" "$n")"
  echo "=== $tag ell=$ell policy=$policy tf=$tf tensor=$fmt n=$n"
  if [ "$DRYRUN" = "1" ]; then
    echo "    DRYRUN env ${COMMON[*]} $* FM028_N=$n FM028_ELL=$ell" \
         "FM028_POLICY=$policy FM028_TF=$tf FM028_TENSOR_FORMAT=$fmt" \
         "FM028_OUT=$(basename "$out")"
    return 0
  fi
  caselog="$(mktemp)"
  if ! env "${COMMON[@]}" "$@" \
        FM028_N="$n" FM028_ELL="$ell" FM028_POLICY="$policy" \
        FM028_TF="$tf" FM028_TENSOR_FORMAT="$fmt" FM028_OUT="$out" \
        julia --project="$ENVDIR" \
          MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl \
          2>&1 | tee "$caselog"; then
    status="$(classify_failure "$caselog")"
    ledger "$tag" "$policy" "$tf" "$fmt" "$n" "$status"
  elif ! case_done "$tag" "$gtag" "$fmt" "$n"; then
    # The harness catches per-case failures inside `measure` and writes a
    # fit=false row with a zero exit code, so a clean exit is not proof of a
    # measured point.
    status="$(classify_failure "$caselog")"
    ledger "$tag" "$policy" "$tf" "$fmt" "$n" "unfit_${status}"
  fi
  rm -f "$caselog"
}

MODE="${FM030_MODE:-sweep}"
echo "=== FM030_MODE=$MODE"

if [ "$MODE" = "sweep" ]; then
    # 7 n x 3 ell x 2 precisions = 42 cases. Ordered n-outer so that a job that
    # runs out of walltime still leaves complete small-n series for every ell.
    for n in ${FM030_NS:-1000 3162 10000 31623 100000 316228 1000000}; do
      for ell in ${FM030_ELLS:-3 4 5}; do
        run_case sweep "$ell" Float32 fp16 "$n"
        run_case sweep "$ell" Float64 off  "$n"
      done
    done

elif [ "$MODE" = "spotcheck" ]; then
    # Phase 5: pre-registered baseline-vs-candidate pairs, run back-to-back in
    # one job so the difference is measured under matched clock/thermal state.
    # FM030_SPOTCHECKS is a space-separated list of
    #   <n>:<baseline_geometry>:<candidate_geometry>:<tf>:<fmt>
    # where a geometry is a bare ell (shipped (6,5,...,5) schedule at that
    # depth) or an explicit sched<q2>-...-<qell> string, so the candidate may
    # differ from the baseline in depth, in radius schedule, or in both.
    : "${FM030_SPOTCHECKS:?spotcheck mode requires FM030_SPOTCHECKS}"
    for spec in $FM030_SPOTCHECKS; do
      IFS=: read -r n base_geom cand_geom tf fmt <<< "$spec"
      run_case spotbase "$base_geom" "$tf" "$fmt" "$n"
      run_case spotcand "$cand_geom" "$tf" "$fmt" "$n"
    done

else
    echo "unknown FM030_MODE=$MODE (expected sweep|spotcheck)" >&2
    exit 1
fi

echo "=== 030 case loop complete; failed_cases=$failed_cases"
if (( failed_cases > 0 )); then
  echo "FM030_FAILED_CASES=$failed_cases (see $failures)" >&2
  echo "SWEEP_EXIT=1"
  exit 1
fi
echo "SWEEP_EXIT=0"
