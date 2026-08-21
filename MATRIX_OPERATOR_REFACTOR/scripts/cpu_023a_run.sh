#!/bin/bash
#SBATCH --job-name=fm023acpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Item 023a: factored resident M2L host benchmark (functional baseline, scalar,
# GEMM, and crossover variants vs concat) on a non-macOS HPC CPU node. Runs the
# sweep TWICE: genuinely single-thread BLAS, then multithread BLAS at the
# allocated core count (thread control must be the process-start env var;
# runtime set_num_threads is unreliable -- see 008c).
source /etc/profile
set -o pipefail
# julia pinned: module default moved to 1.12.6 on 2026-08-05 which segfaults host LLVM JIT (job 13058191); 1.11.7 is the toolchain of record
module load julia/1.11.7-6bmogfl
echo "=== node: $(hostname)  cpus=$SLURM_CPUS_PER_TASK"
lscpu | grep -E "Model name|Socket|Core|Thread" || true

cd "$HOME/FastMultipole-023"
ENVDIR="$HOME/fm023env"
OUTDIR="MATRIX_OPERATOR_REFACTOR/data/factored_resident_m2l_host"
STAMP=$(date +%Y%m%d-%H%M%S)

echo "=== FastMultipole load preflight"
julia --project="$ENVDIR" -e 'using FastMultipole; println("FASTMULTIPOLE_LOAD_OK")' \
    || { echo "FASTMULTIPOLE_LOAD_FAIL (instantiate fm023env on the login node first; compute nodes have no internet)"; exit 1; }

echo "=== single-thread BLAS pass"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  FM023A_BLAS_THREADS=1 \
  FM023A_OUT="$OUTDIR/host_$(hostname)_blas1_$STAMP.csv" \
  julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023a_factored_host.jl
s1=$?
echo "BLAS1_EXIT=$s1"

echo "=== multithread BLAS pass (threads=$SLURM_CPUS_PER_TASK)"
OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  FM023A_BLAS_THREADS=$SLURM_CPUS_PER_TASK \
  FM023A_OUT="$OUTDIR/host_$(hostname)_blas${SLURM_CPUS_PER_TASK}_$STAMP.csv" \
  julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023a_factored_host.jl
s2=$?
echo "BLASN_EXIT=$s2"

echo "=== data files"
ls -l "$OUTDIR" || true

exit $(( s1 || s2 ))
