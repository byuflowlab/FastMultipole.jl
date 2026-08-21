#!/bin/bash
#SBATCH --job-name=fm024h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -euo pipefail
# julia pinned: module default moved to 1.12.6 on 2026-08-05 which segfaults host LLVM JIT (job 13058191); 1.11.7 is the toolchain of record
module load cuda julia/1.11.7-6bmogfl

WORKDIR="${FM024_DIR:-$HOME/FastMultipole-024}"
ENVDIR="${FM024_ENV:-$HOME/fm024env}"
OUTDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/operator_ab_benchmark/raw"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

echo "=== task 024 H200 environment"
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK job=$SLURM_JOB_ID"
uname -a
lscpu
nvidia-smi
julia --version
echo "git_commit=${FM024_GIT_COMMIT:-unknown}"
echo "git_tree=${FM024_GIT_TREE:-unknown}"
echo "git_worktree=${FM024_GIT_WORKTREE:-unknown}"
echo "source_manifest=${FM024_SOURCE_MANIFEST:-unknown}"
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()'

echo "=== required CUDA lifecycle/integration gate"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_integration_test.jl

stamp=$(date +%Y%m%d-%H%M%S)
run_case () {
  local tf=$1 lh=$2 p=$3 n=$4 ell=$5 distribution=$6
  local out="$OUTDIR/cuda_$(hostname)_${distribution}_${tf}_lh${lh}_p${p}_n${n}_ell${ell}_${stamp}.csv"
  echo "=== $(basename "$out")"
  FM024_TF="$tf" FM024_LH="$lh" FM024_P="$p" FM024_N="$n" \
    FM024_ELL="$ell" FM024_DISTRIBUTION="$distribution" FM024_OUT="$out" \
    julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_024_cuda.jl
}

for tf in Float32 Float64; do
  for lh in false true; do
    for p in 4 8 12; do
      for n in 150 2000 20000; do
        run_case "$tf" "$lh" "$p" "$n" 3 uniform
      done
    done
  done
done
for lh in false true; do
  for p in 4 8 12; do
    run_case Float64 "$lh" "$p" 20000 4 clustered
  done
done

julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/summarize_024.jl \
  "$OUTDIR" "$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/operator_ab_benchmark/summary"
