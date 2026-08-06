#!/bin/bash
#SBATCH --job-name=fm032a
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Task 032a Stage C H200 job:
#   1. device correctness preflight — the shipped interface + lifecycle CUDA
#      tests plus the NEW stage-C binned-nearfield test
#      (cuda_radix_nearfield_binning_test.jl: mechanism parity at P=4/P=8 and
#      both precisions, TwoPass pass-2 device mirror, counters, allocation
#      stability, homogeneity diagnostics, validation paths);
#   2. cuda_032a_stagec_benchmark.jl — the §6.3 mechanism-selection matrix
#      (timings + achieved warp homogeneity + sampled-direct error) at the
#      three operating points.
# Environment pattern from cuda_032_run.sh; julia pinned to 1.11.7 (1.12.6
# segfaults JIT-compiling the device step, job 13058191).
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${FM032A_DIR:-$HOME/FastMultipole-023}"
ENVDIR="${FM032A_ENV:-$HOME/fm023env}"
OUTDIR="${FM032A_OUTDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/split_nearfield}"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
export JULIA_NUM_THREADS=8

echo "=== preflight: CUDA interface tests (032 stages 1-3) ==="
julia --project="$ENVDIR" test/cuda_radix_interface_test.jl
echo "=== preflight: shipped lifecycle regression test ==="
julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
echo "=== preflight: stage C binned-nearfield test (NEW) ==="
julia --project="$ENVDIR" test/cuda_radix_nearfield_binning_test.jl

echo "=== stage C mechanism-selection benchmark ==="
FM032A_OUTDIR="$OUTDIR" julia --project="$ENVDIR" \
    MATRIX_OPERATOR_REFACTOR/scripts/cuda_032a_stagec_benchmark.jl

echo "fm032a job complete"
