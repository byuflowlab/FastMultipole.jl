#!/bin/bash
#SBATCH --job-name=fm026cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -euo pipefail
module load julia

# The m12 partition mixes CPU targets and concurrent jobs share ~/.julia, so a
# job precompiling on one node can invalidate the pkgimage another running job
# depends on. A per-job first-layer depot isolates each job's compiled cache
# (reads still fall back to the shared depot for artifacts/registries).
export JULIA_DEPOT_PATH="$HOME/.julia_fm026/${SLURM_JOB_ID:-manual}:$HOME/.julia"

WORKDIR="${FM026_DIR:-$HOME/FastMultipole-026}"
ENVDIR="${FM026_ENV:-$HOME/fm026env}"
JOBTAG="${SLURM_JOB_ID:-manual}"
RAW="${FM026_RAW:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_host/raw/$JOBTAG}"
SUMMARY="${FM026_SUMMARY:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_host/summary/$JOBTAG}"
PHASE="${FM026_PHASE:-all}"
SOURCE_HASH="${FM026_SOURCE_HASH:?set FM026_SOURCE_HASH to the synchronized source manifest hash}"
DRIVER="MATRIX_OPERATOR_REFACTOR/scripts/benchmark_026_hierarchical_host.jl"
SUMMARIZER="MATRIX_OPERATOR_REFACTOR/scripts/summarize_026_hierarchical_host.jl"

cd "$WORKDIR"
mkdir -p "$RAW" "$SUMMARY"
echo "task=026 node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-unknown} job=$JOBTAG phase=$PHASE"
echo "source_hash=$SOURCE_HASH"
uname -a
lscpu
julia --version

echo "=== focused validation"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project="$ENVDIR" -e \
  'using FastMultipole, FastMultipole.StaticArrays, Test;
   include("test/gravitational.jl");
   include("test/hierarchical_m2l_host_test.jl")'

run_case () {
  local bt=$1 policy=$2 strategy=$3 n=$4 ell=$5 window=$6
  local tf=$7 lh=$8 distribution=$9 accuracy=${10}
  local performance=${11:-true} stage_detail=${12:-false}
  local step_timing=${13:-true} reps=${14:-5}
  local key="bt${bt}_${policy}_${strategy}_n${n}_ell${ell}_w${window}_${tf}_lh${lh}_${distribution}_acc${accuracy}"
  local out="$RAW/${key}.csv"
  local levels="$RAW/${key}_levels.csv"
  if [[ -s "$out" && -s "$levels" ]]; then
    echo "skip existing $key"
    return
  fi
  echo "=== $key"
  OPENBLAS_NUM_THREADS="$bt" OMP_NUM_THREADS="$bt" \
    FM026_BLAS_THREADS="$bt" FM026_POLICY="$policy" \
    FM026_STRATEGY="$strategy" FM026_N="$n" FM026_ELL="$ell" \
    FM026_WINDOW_CLASSES="$window" FM026_TF="$tf" FM026_LH="$lh" \
    FM026_DISTRIBUTION="$distribution" FM026_ACCURACY="$accuracy" \
    FM026_PERFORMANCE="$performance" FM026_STAGE_DETAIL="$stage_detail" \
    FM026_STEP_TIMING="$step_timing" FM026_P=4 \
    FM026_WARMUPS="${FM026_WARMUPS:-1}" FM026_REPS="$reps" \
    FM026_ALLOC_REPS="${FM026_ALLOC_REPS:-1}" FM026_SOURCE_HASH="$SOURCE_HASH" \
    FM026_OUT="$out" FM026_LEVEL_OUT="$levels" \
    julia --project="$ENVDIR" "$DRIVER"
}

# Comma-separated FM026_STRATEGIES restricts the strategy matrix (e.g. a
# targeted re-measurement after a strategy-path fix). Partial matrices must not
# apply the full-campaign count validation.
IFS=',' read -r -a strategies <<< "${FM026_STRATEGIES:-concat,factored,precomputed_y,dense}"
blas=(1 64)

if [[ "$PHASE" == "all" || "$PHASE" == "tuning" ]]; then
  echo "=== window tuning"
  common_windows=(1 2 4 8 16 32 64 128)
  for bt in "${blas[@]}"; do
    for policy in hierarchical_12 hierarchical_3; do
      for strategy in "${strategies[@]}"; do
        for window in "${common_windows[@]}"; do
          run_case "$bt" "$policy" "$strategy" 20000 4 "$window" \
            Float64 false uniform false true false false 3
        done
      done
    done
  done
  tuning_expected=tuning
  [[ -n "${FM026_STRATEGIES:-}" ]] && tuning_expected=none
  FM026_EXPECTED_PHASE="$tuning_expected" \
    julia --project="$ENVDIR" "$SUMMARIZER" "$RAW" "$SUMMARY"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "scaling" ]]; then
  selected="${FM026_SELECTED_WINDOW:-}"
  if [[ -z "$selected" ]]; then
    selected=$(tr -d '[:space:]' < "$SUMMARY/selected_window.txt")
  fi
  echo "=== scaling selected_window=$selected"
  for bt in "${blas[@]}"; do
    for policy in flat hierarchical_12 hierarchical_3; do
      for strategy in "${strategies[@]}"; do
        detail=false
        [[ "$bt" == 1 && "$policy" == "hierarchical_12" &&
           "$strategy" == "concat" ]] && detail=true
        for n in 64 128 256 512 1024 2048; do
          run_case "$bt" "$policy" "$strategy" "$n" 3 "$selected" \
            Float64 false uniform false true "$detail" true 5
        done
        for n in 128 256 512 1024 2048 4096 8192; do
          run_case "$bt" "$policy" "$strategy" "$n" 4 "$selected" \
            Float64 false uniform false true "$detail" true 5
        done
        for n in 256 512 1024 2048 4096; do
          run_case "$bt" "$policy" "$strategy" "$n" 5 "$selected" \
            Float64 false uniform false true "$detail" true 5
        done
        run_case "$bt" "$policy" "$strategy" 20000 4 "$selected" \
          Float64 false clustered false true "$detail" true 5
      done
    done
  done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "accuracy" ]]; then
  selected="${FM026_SELECTED_WINDOW:-}"
  if [[ -z "$selected" ]]; then
    selected=$(tr -d '[:space:]' < "$SUMMARY/selected_window.txt")
  fi
  echo "=== accuracy selected_window=$selected"
  for tf in Float32 Float64; do
    for lh in false true; do
      for ell in 3 4 5; do
        for policy in flat_12 flat_3 hierarchical_12 hierarchical_3; do
          for strategy in "${strategies[@]}"; do
            run_case 1 "$policy" "$strategy" 5000 "$ell" "$selected" \
              "$tf" "$lh" uniform true false false false 1
          done
        done
      done
    done
  done
fi

expected_phase=none
[[ "$PHASE" == "all" && -z "${FM026_STRATEGIES:-}" ]] && expected_phase=all
FM026_EXPECTED_PHASE="$expected_phase" \
FM026_SELECTED_WINDOW="${selected:-}" \
  julia --project="$ENVDIR" "$SUMMARIZER" "$RAW" "$SUMMARY"
echo "task 026 campaign complete: $SUMMARY"
