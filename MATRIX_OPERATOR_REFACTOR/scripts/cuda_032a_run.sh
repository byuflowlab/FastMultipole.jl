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

if [ "${FM032A_RUN_STAGEC:-0}" = "1" ]; then
    echo "=== stage C mechanism-selection benchmark ==="
    FM032A_OUTDIR="$OUTDIR" julia --project="$ENVDIR" \
        MATRIX_OPERATOR_REFACTOR/scripts/cuda_032a_stagec_benchmark.jl
fi

echo "=== stage D A/B ladder (cube + wake, three strategies, rho_t lever) ==="
FM032A_OUTDIR="$OUTDIR" FM032A_SENTINELS="${FM032A_SENTINELS:-0}" \
    FM032A_CASES="${FM032A_CASES:-}" \
    julia --project="$ENVDIR" \
    MATRIX_OPERATOR_REFACTOR/scripts/cuda_032a_staged_ab.jl

echo "=== stage D: scalar no-regression (028/030 verdict config, unchanged harness) ==="
REFDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references"
COMMON=(FM028_P=3 FM028_STRAT=dense FM028_LH=0 FM028_K=full
        FM028_REPS=9 FM028_STEPS=0 FM028_STALE=0
        FM028_BOUND=ab FM028_SYMMETRIC=0
        FM028_M2L_THREADS=64 FM028_M2L_BLOCK_CAP=65536 FM028_COUNTING_SORT=1
        FM028_REFDIR="$REFDIR")
for cfg in "Float32 fp16" "Float64 off"; do
    read -r tf fmt <<< "$cfg"
    out="$OUTDIR/cuda032a_nr_sched6_5_5_5_${fmt}_n1000000_$(hostname)_${SLURM_JOB_ID:-manual}.csv"
    echo "=== no-regression case tf=$tf fmt=$fmt"
    env "${COMMON[@]}" \
        FM028_N=1000000 FM028_ELL=5 FM028_POLICY=sched6-5-5-5 \
        FM028_TF="$tf" FM028_TENSOR_FORMAT="$fmt" FM028_OUT="$out" \
        julia --project="$ENVDIR" \
        MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
done

echo "fm032a job complete"
