#!/bin/bash
#SBATCH --job-name=fmm_treebuild
#SBATCH --qos=eng
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# 4-way tree-build benchmark (CPU / Metal-KA / CUDA-KA / CUDA-native) for
# FastMultipole's ka-migration side-track. Pattern: FLOWVPM-kabench's
# ka_cuda_bench_run.sh, trimmed for this repo's layout.
#
# n=1e6 tree-build timing showed IQR 5-8x the median even at 100 trials
# (jobs 13506026/13506031). An --exclusive rerun (13506032) was queued to rule
# out shared-node contention but sat pending on resources too long to be
# worth waiting on. Instead the co-tenancy log below now also samples SM
# clock, power draw, and throttle-reason flags, which tests the clock-
# throttling hypothesis directly without needing a whole-node reservation;
# memory.used/utilization.gpu still catch co-tenant activity if any is there.
# cpus-per-task raised 1 -> 8. Jobs 13508353 (1 CPU) vs 13508376 (8 CPUs), same
# isolated n=1e6 case, showed the whole n>=1e5 variance was host CPU starvation:
# slow trials 38/100 -> 2/100 (KA), IQR ±31.99ms -> ±0.24ms, and the process
# stopped sitting pinned at proc_cpu/wall ~1.0. EVERY earlier job in this
# investigation ran at 1 CPU, including the ones behind the "KA is 2.4-3.5x
# slower at n<=1e4" reading -- and KA issues more, smaller kernels than the
# native arm, so CPU starvation penalizes it disproportionately. That number has
# to be re-measured here before it can be trusted.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"
WORKDIR="${FMM_KABENCH_DIR:-$HOME/FastMultipole-kabench}"
ENVDIR="${FMM_KABENCH_ENV:-$HOME/fm_kabench_env}"
cd "$WORKDIR"
LOG="$WORKDIR/tree_build_bench_${SLURM_JOB_ID}.log"
PROV="$WORKDIR/tree_build_bench_${SLURM_JOB_ID}.provenance"
GPULOG="$WORKDIR/tree_build_bench_${SLURM_JOB_ID}.gpu.csv"

# Background GPU sample, 1s cadence, for the duration of the julia run:
# utilization/memory catch co-tenant activity; clocks/power/throttle-reasons
# catch thermal/power throttling independent of any other tenant.
# clocks.mem/temperature.memory added because the SM-clock check alone
# (13506267) didn't rule out HBM-specific throttling, which would explain
# the n>=1e5 staircase hitting every kernel type uniformly.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,\
clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,temperature.memory,\
power.draw,power.limit,clocks_event_reasons.active \
    --format=csv -l 1 > "$GPULOG" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null' EXIT

echo "=== 4-way tree-build benchmark: test/metal_env/tree_build_benchmark_4way.jl ==="
julia --project="$ENVDIR" test/metal_env/tree_build_benchmark_4way.jl 2>&1 | tee "$LOG"

kill "$GPU_MONITOR_PID" 2>/dev/null
trap - EXIT

{
  echo "julia=$(julia --version)"
  echo "cuda_module=$(module list 2>&1 | tr '\n' ' ')"
  echo "device=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
  echo "fmm_tree_sha256=$(find "$WORKDIR/src" "$WORKDIR/test/metal_env" -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  echo "manifest_sha256=$(shasum -a 256 "$ENVDIR/Manifest.toml" | awk '{print $1}')"
  echo "raw_log=$LOG"
  echo "raw_log_sha256=$(shasum -a 256 "$LOG" | awk '{print $1}')"
  echo "gpu_log=$GPULOG"
} > "$PROV"
cat "$PROV"
echo "tree_build_bench job complete"
