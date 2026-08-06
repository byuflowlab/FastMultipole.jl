#!/bin/bash
#SBATCH --job-name=fm032
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Task 032 H200 job (stage 4 form; the stage-2 benchmarks ran as job 13058240):
#   1. device correctness preflight — the stage 1+2+3 CUDA interface tests
#      (generalized packing, vortex B2M + LH, 13-row hessian, direct-kernel
#      functors, RegularizedVortex parity, adequacy gate, persistent device
#      buffers, device recenter!) plus the shipped lifecycle regression test;
#   2. cuda_032_validation.jl — device-resident regularized-vortex consumer:
#      accuracy gate (Float64 u_rel_rms <= 1e-3, J diagnostic), 023 counter
#      contract, steady-state allocation probe, at n = 1e5 (auto ell) and
#      n = 1e6;
#   3. scalar no-regression — benchmark_028_feasibility.jl UNCHANGED at the
#      shipped 028/030 verdict configuration (sched6-5-5-5, n = 1e6, fp16/F32
#      and off/F64), compared offline against the 030 sweep rows of record
#      (cuda030_sweep_sched6_5_5_5_{fp16,off}_n1000000_*, verdict 9.591 ms
#      F32);
#   4. optionally (FM032_RUN_NEARFIELD=1) the stage-2 nearfield benchmarks.
# Environment pattern from cuda_030_run.sh: the fm023env project carries CUDA
# (local-toolkit preferences; compute nodes have no internet).
# no -u: /etc/profile.d scripts reference unset vars on the cluster
set -eo pipefail
source /etc/profile
# julia pinned to 1.11.7: the module default moved to 1.12.6 after the 030 runs
# (2026-08-03) and 1.12.6 segfaults in host LLVM while JIT-compiling the device
# step (job 13058191); 1.11.7 is the toolchain of record for every H200 result
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${FM032_DIR:-$HOME/FastMultipole-023}"
ENVDIR="${FM032_ENV:-$HOME/fm023env}"
OUTDIR="${FM032_OUTDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms}"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
export JULIA_NUM_THREADS=8

echo "=== preflight: CUDA interface tests (stages 1-3) ==="
julia --project="$ENVDIR" test/cuda_radix_interface_test.jl
echo "=== preflight: shipped lifecycle regression test ==="
julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl

# q ladder: q=12 measured 1.088e-3 > the 1e-3 gate (job 13058532, truncation-
# dominated at P=4); q=16 is the modeled fix (~6.5e-4), q=20 the measured-safe
# fallback (~1.9e-4 modeled, 389-offset near set). Try 16, fall back to 20.
run_validation () {
    local n="$1"
    for q in 16 20; do
        echo "=== stage 4: device-resident vortex validation (n=$n, q=$q) ==="
        if FM032V_OUTDIR="$OUTDIR" FM032V_N="$n" FM032V_Q="$q" \
            julia --project="$ENVDIR" \
            MATRIX_OPERATOR_REFACTOR/scripts/cuda_032_validation.jl; then
            return 0
        fi
        echo "=== validation failed at n=$n q=$q"
    done
    echo "=== validation failed at n=$n for every q in the ladder"
    return 1
}
run_validation 100000
run_validation 1000000

echo "=== stage 4: scalar no-regression (028/030 verdict config, unchanged harness) ==="
REFDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references"
COMMON=(FM028_P=3 FM028_STRAT=dense FM028_LH=0 FM028_K=full
        FM028_REPS=9 FM028_STEPS=0 FM028_STALE=0
        FM028_BOUND=ab FM028_SYMMETRIC=0
        FM028_M2L_THREADS=64 FM028_M2L_BLOCK_CAP=65536 FM028_COUNTING_SORT=1
        FM028_REFDIR="$REFDIR")
for cfg in "Float32 fp16" "Float64 off"; do
    read -r tf fmt <<< "$cfg"
    out="$OUTDIR/cuda032nr_sched6_5_5_5_${fmt}_n1000000_$(hostname)_${SLURM_JOB_ID:-manual}.csv"
    echo "=== no-regression case tf=$tf fmt=$fmt"
    env "${COMMON[@]}" \
        FM028_N=1000000 FM028_ELL=5 FM028_POLICY=sched6-5-5-5 \
        FM028_TF="$tf" FM028_TENSOR_FORMAT="$fmt" FM028_OUT="$out" \
        julia --project="$ENVDIR" \
        MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
done

if [ "${FM032_RUN_NEARFIELD:-0}" = "1" ]; then
    echo "=== stage 2 nearfield benchmarks (rerun) ==="
    FM032_OUTDIR="$OUTDIR" julia --project="$ENVDIR" \
        MATRIX_OPERATOR_REFACTOR/scripts/benchmark_032_nearfield.jl
fi

echo "fm032 job complete"
