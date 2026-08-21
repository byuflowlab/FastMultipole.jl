#!/bin/bash
#SBATCH --job-name=fm023ccpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -uo pipefail
# julia pinned: module default moved to 1.12.6 on 2026-08-05 which segfaults host LLVM JIT (job 13058191); 1.11.7 is the toolchain of record
module load julia/1.11.7-6bmogfl

cd "$HOME/FastMultipole-023"
ENVDIR="$HOME/fm023env"
OUTDIR="MATRIX_OPERATOR_REFACTOR/data/precomputed_y_resident_m2l_host"
STAMP=$(date +%Y%m%d-%H%M%S)

echo "=== task 023c environment"
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK job=$SLURM_JOB_ID"
uname -a
lscpu | grep -E "Model name|Socket|Core|Thread|CPU\(s\)" || true
julia --version
echo "git_commit=${FM023C_GIT_COMMIT:-unknown}"
echo "worktree=${FM023C_GIT_WORKTREE:-unknown}"

echo "=== focused 023c tests"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=test -e '
    using FastMultipole, Test, Random, LinearAlgebra
    using FastMultipole.StaticArrays
    include("test/gravitational.jl")
    include("test/precomputed_y_resident_m2l_test.jl")
' || { echo "FOCUSED_023C_TESTS_FAIL"; exit 1; }
echo "FOCUSED_023C_TESTS_PASS"

echo "=== single-thread OpenBLAS sweep"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FM023C_BLAS_THREADS=1 \
  FM023C_SWEEP_COLS="${FM023C_SWEEP_COLS:-4,8,12,16,24}" \
  FM023C_OUT="$OUTDIR/host_$(hostname)_blas1_$STAMP.csv" \
  julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023c_precomputed_y_host.jl
s1=$?
echo "BLAS1_EXIT=$s1"

echo "=== ${SLURM_CPUS_PER_TASK}-thread OpenBLAS sweep"
OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  FM023C_BLAS_THREADS=$SLURM_CPUS_PER_TASK \
  FM023C_SWEEP_COLS="${FM023C_SWEEP_COLS:-4,8,12,16,24}" \
  FM023C_OUT="$OUTDIR/host_$(hostname)_blas${SLURM_CPUS_PER_TASK}_$STAMP.csv" \
  julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023c_precomputed_y_host.jl
s2=$?
echo "BLAS${SLURM_CPUS_PER_TASK}_EXIT=$s2"
ls -lh "$OUTDIR" || true
exit $(( s1 || s2 ))
