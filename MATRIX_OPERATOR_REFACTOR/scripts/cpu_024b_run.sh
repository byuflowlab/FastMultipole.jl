#!/bin/bash
#SBATCH --job-name=fm024bcpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -euo pipefail
module load julia

WORKDIR="${FM024B_DIR:-$HOME/FastMultipole-024b}"
ENVDIR="${FM024B_ENV:-$HOME/fm024env}"
OUTDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling"
REFDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references"
cd "$WORKDIR"
mkdir -p "$OUTDIR"
out="$OUTDIR/cpu_$(hostname)_${SLURM_JOB_ID}.csv"

echo "=== task 024b CPU environment"
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK job=$SLURM_JOB_ID"
lscpu
julia --version

for n in 1000 3162 10000 31623 100000 316228 1000000; do
  test -s "$REFDIR/direct_reference_n${n}.csv"
done
(cd "$REFDIR" && shasum -a 256 -c direct_reference_checksums.sha256)

run_case () {
  local threads=$1 n=$2
  local mode="cpu${threads}"
  if grep -q "^${mode},${n}," "$OUTDIR"/cpu_*.csv 2>/dev/null; then
    echo "=== skip completed CPU threads=$threads n=$n"
    return
  fi
  echo "=== CPU threads=$threads n=$n"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FM024B_N="$n" FM024B_OUT="$out" \
    FM024B_REFERENCE_DIR="$REFDIR" \
    julia -t "$threads" --project="$ENVDIR" \
      MATRIX_OPERATOR_REFACTOR/scripts/benchmark_024b_cpu.jl
}

# Keep the longest case last so every earlier result survives a wall-time limit.
for n in 1000 3162 10000 31623 100000 316228; do
  run_case 1 "$n"
  run_case 64 "$n"
done
run_case 64 1000000
run_case 1 1000000
