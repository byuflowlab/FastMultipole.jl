#!/bin/bash
#SBATCH --job-name=fm024cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -euo pipefail
# julia pinned: module default moved to 1.12.6 on 2026-08-05 which segfaults host LLVM JIT (job 13058191); 1.11.7 is the toolchain of record
module load julia/1.11.7-6bmogfl

WORKDIR="${FM024_DIR:-$HOME/FastMultipole-024}"
ENVDIR="${FM024_ENV:-$HOME/fm024env}"
OUTDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/operator_ab_benchmark/raw"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

echo "=== task 024 CPU environment"
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK job=$SLURM_JOB_ID"
uname -a
lscpu
julia --version
echo "git_commit=${FM024_GIT_COMMIT:-unknown}"
echo "git_tree=${FM024_GIT_TREE:-unknown}"
echo "git_worktree=${FM024_GIT_WORKTREE:-unknown}"
echo "source_manifest=${FM024_SOURCE_MANIFEST:-unknown}"

echo "=== focused host validation"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project="$ENVDIR" test/radix_fmm_integration_test.jl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project="$ENVDIR" test/radix_fmm_timestepping_test.jl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project="$ENVDIR" -e \
  'using Test, FastMultipole, FastMultipole.StaticArrays, Random;
   include("test/gravitational.jl");
   include("test/precomputed_y_resident_m2l_test.jl")'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project="$ENVDIR" test/dense_translation_m2l_test.jl

stamp=$(date +%Y%m%d-%H%M%S)
run_case () {
  local bt=$1 tf=$2 lh=$3 p=$4 n=$5 ell=$6 distribution=$7
  local out="$OUTDIR/cpu_$(hostname)_blas${bt}_${distribution}_${tf}_lh${lh}_p${p}_n${n}_ell${ell}_${stamp}.csv"
  echo "=== $(basename "$out")"
  OPENBLAS_NUM_THREADS="$bt" OMP_NUM_THREADS="$bt" \
    FM024_BLAS_THREADS="$bt" FM024_TF="$tf" FM024_LH="$lh" \
    FM024_P="$p" FM024_N="$n" FM024_ELL="$ell" \
    FM024_DISTRIBUTION="$distribution" FM024_OUT="$out" \
    julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_024_host.jl
}

# BLAS settings are separate Julia processes and fixed before startup.
for bt in 1 64; do
  for tf in Float32 Float64; do
    for lh in false true; do
      for p in 4 8 12; do
        for n in 150 2000 20000; do
          run_case "$bt" "$tf" "$lh" "$p" "$n" 3 uniform
        done
      done
    done
  done
  # Occupancy/skew matrix: Float64, N=20k, ell=4, every P and LH mode.
  for lh in false true; do
    for p in 4 8 12; do
      run_case "$bt" Float64 "$lh" "$p" 20000 4 clustered
    done
  done
done

julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/summarize_024.jl \
  "$OUTDIR" "$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/operator_ab_benchmark/summary"
