#!/bin/bash
#SBATCH --job-name=fm033cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -euo pipefail
# julia pinned: module default moved to 1.12.6 on 2026-08-05 which segfaults host LLVM JIT (job 13058191); 1.11.7 is the toolchain of record
module load julia/1.11.7-6bmogfl

WORKDIR="${FM033_DIR:-$HOME/FastMultipole-033}"
ENVDIR="${FM033_ENV:-$HOME/fm033env}"
OUTDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline"
REFDIR="$OUTDIR/references"
SCRIPTS="$WORKDIR/MATRIX_OPERATOR_REFACTOR/scripts"
cd "$WORKDIR"
mkdir -p "$OUTDIR" "$REFDIR"
out="$OUTDIR/cpu_$(hostname)_${SLURM_JOB_ID}.csv"
export JULIA_DEPOT_PATH="$HOME/.julia_fm033/$SLURM_JOB_ID:$HOME/.julia"

echo "=== task 033 CPU environment"
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK job=$SLURM_JOB_ID"
lscpu
julia --version
julia --project="$ENVDIR" -e 'using Pkg; Pkg.status()'
julia --project="$ENVDIR" -e 'using FastMultipole;
  v = pkgversion(FastMultipole);
  println("FastMultipole version: ", v);
  @assert v < v"2.1" "baseline requires FastMultipole 2.0.x (shrink_recenter kwarg)"'

# Phase 1: sampled-direct references (skips per-file if already present).
# The manifest must cover the current case set (wake replaced ring on
# 2026-08-05): a stale pre-amendment manifest that verifies but lacks wake
# entries must not short-circuit reference generation.
if ! (cd "$REFDIR" && grep -q "direct_reference_wake_" direct_reference_checksums.sha256 2>/dev/null \
      && shasum -a 256 -c direct_reference_checksums.sha256 2>/dev/null); then
  echo "=== generating 033 direct references"
  # -t 1: FastMultipole 2.0.4 direct_multithread! on the (target, source)
  # path is broken (UndefVarError: n_source_bodies, direct.jl:111); the
  # single-thread direct is correct and cheap at <=512 samples per case.
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FM033_REFERENCE_DIR="$REFDIR" \
    julia -t 1 --project="$ENVDIR" "$SCRIPTS/prepare_033_references.jl"
fi
(cd "$REFDIR" && shasum -a 256 -c direct_reference_checksums.sha256)

# Phase 2: benchmark cases, resumable at case granularity.
run_case () {
  local case=$1 threads=$2 n=$3 profile=${4:-0}
  local mode="cpu${threads}"
  if grep -q "^${case},${mode},${n}," "$OUTDIR"/cpu_*.csv 2>/dev/null; then
    echo "=== skip completed $case $mode n=$n"
    return
  fi
  echo "=== $case threads=$threads n=$n profile=$profile"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    FM033_CASE="$case" FM033_N="$n" FM033_OUT="$out" \
    FM033_REFERENCE_DIR="$REFDIR" FM033_PROFILE="$profile" \
    julia -t "$threads" --project="$ENVDIR" "$SCRIPTS/benchmark_033_cpu.jl"
}

# Keep the longest cases (n=1e6, then single-thread 1e6) last so earlier
# results survive a wall-time kill. Profile capture at the representative
# n=1e5 for both cases and both thread counts.
for n in 1000 3162 10000 31623; do
  for case in cube wake; do
    run_case "$case" 1 "$n"
    run_case "$case" 64 "$n"
  done
done
for case in cube wake; do
  run_case "$case" 1 100000 1
  run_case "$case" 64 100000 1
done
for case in cube wake; do
  run_case "$case" 1 316228
  run_case "$case" 64 316228
done
for case in cube wake; do
  run_case "$case" 64 1000000
done
for case in cube wake; do
  run_case "$case" 1 1000000
done
echo "=== task 033 complete"
