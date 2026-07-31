#!/bin/bash
#SBATCH --job-name=fm024bref
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -euo pipefail
module load julia

WORKDIR="${FM024B_DIR:-$HOME/FastMultipole-024b}"
ENVDIR="${FM024B_ENV:-$HOME/fm024env}"
OUTDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

echo "=== task 024b direct-reference environment"
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK job=$SLURM_JOB_ID"
lscpu
julia --version

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  FM024B_DIRECT_THREADS=64 FM024B_REFERENCE_DIR="$OUTDIR" \
  julia -t 64 --project="$ENVDIR" \
    MATRIX_OPERATOR_REFACTOR/scripts/prepare_024b_direct_references.jl

cd "$OUTDIR"
shasum -a 256 -c direct_reference_checksums.sha256

