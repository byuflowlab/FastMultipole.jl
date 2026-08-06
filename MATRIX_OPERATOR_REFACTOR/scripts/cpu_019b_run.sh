#!/bin/bash
#SBATCH --job-name=fm019bcpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Item 019b: CPU exploratory benchmark (small-P fallback crossover + chi layout)
# on an HPC CPU node. Runs the sweep TWICE: genuinely single-thread BLAS, then
# multithread BLAS at the allocated core count (thread control must be the
# process-start env var; runtime set_num_threads is unreliable -- see 008c).
source /etc/profile
set -o pipefail
# julia pinned: module default moved to 1.12.6 on 2026-08-05 which segfaults host LLVM JIT (job 13058191); 1.11.7 is the toolchain of record
module load julia/1.11.7-6bmogfl
echo "=== node: $(hostname)  cpus=$SLURM_CPUS_PER_TASK"
lscpu | grep -E "Model name|Socket|Core|Thread" || true

cd "$HOME/projects/FastMultipole-022"
ENVDIR="$HOME/fm022env"

echo "=== FastMultipole load preflight"
julia --project="$ENVDIR" -e 'using FastMultipole; println("FASTMULTIPOLE_LOAD_OK")' \
    || { echo "FASTMULTIPOLE_LOAD_FAIL (stale fm022env Manifest? resolve on the login node — see cuda_019b_submit.sh)"; exit 1; }

echo "=== single-thread BLAS pass"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/impl_019b_smallp_layout.jl
s1=$?
echo "BLAS1_EXIT=$s1"

# preserve the single-thread env record before the second pass overwrites env.md
OUT="MATRIX_OPERATOR_REFACTOR/data/smallp_fallback_layout/$(hostname)"
[ -f "$OUT/env.md" ] && cp "$OUT/env.md" "$OUT/env_blas1.md"

echo "=== multithread BLAS pass (threads=$SLURM_CPUS_PER_TASK)"
OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/impl_019b_smallp_layout.jl
s2=$?
echo "BLASN_EXIT=$s2"

echo "=== data files"
ls -l "$OUT" || true

exit $(( s1 || s2 ))
