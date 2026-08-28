#!/bin/bash
#SBATCH --job-name=fmm_treebuild_bisect
#SBATCH --qos=eng
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out
# Combined job (per CLAUDE.md's Cluster Jobs guidance: stage ready GPU workloads
# into one sbatch script rather than queueing separately) covering two checks
# after the KAAdaptiveTreeContext preallocated-buffer fix (commit b5f1e6a):
#
# 1. tree_build_benchmark_4way.jl (commit 9ea8d15 adds GC.gc()/CUDA.reclaim()
#    between test cases) -- job 13506207 showed memory.used growing
#    monotonically 0->41GB across all 8 cases with no drops between them
#    (stale actx/RadixFMMCache buffers from earlier cases never freed), and
#    n>=1e5 timing got WORSE than prior runs despite the allocation fix.
#    Verifies whether reclaiming between cases resolves that regression.
# 2. ka_alloc_bisect.jl (commit 9ea8d15 adds Part 3) -- localizes the residual
#    ~18.1KB/trial CUDA-KA allocation at n=1e6 (down from ~1.95GB/trial, but
#    still above CUDA-native's ~3.3KB/trial) to production code's
#    sortperm!(view, view) calls (ka_build_adaptive_tree!/ka_adaptive_balance!/
#    ka_adaptive_finalize!) vs. the original bisect's plain-CuArray test
#    (0.0MB), by comparing sortperm! on views vs. plain arrays directly.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"
WORKDIR="${FMM_KABENCH_DIR:-$HOME/FastMultipole-kabench}"
ENVDIR="${FMM_KABENCH_ENV:-$HOME/fm_kabench_env}"
cd "$WORKDIR"
TREE_LOG="$WORKDIR/tree_build_bench_${SLURM_JOB_ID}.log"
TREE_GPULOG="$WORKDIR/tree_build_bench_${SLURM_JOB_ID}.gpu.csv"
BISECT_LOG="$WORKDIR/ka_alloc_bisect_${SLURM_JOB_ID}.log"
PROV="$WORKDIR/tree_build_and_bisect_${SLURM_JOB_ID}.provenance"

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,\
clocks.sm,clocks.max.sm,power.draw,power.limit,clocks_event_reasons.active \
    --format=csv -l 1 > "$TREE_GPULOG" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null' EXIT

echo "=== 4-way tree-build benchmark: test/metal_env/tree_build_benchmark_4way.jl ==="
julia --project="$ENVDIR" test/metal_env/tree_build_benchmark_4way.jl 2>&1 | tee "$TREE_LOG"

kill "$GPU_MONITOR_PID" 2>/dev/null
trap - EXIT

echo "=== KA alloc bisect: test/metal_env/ka_alloc_bisect.jl ==="
julia --project="$ENVDIR" test/metal_env/ka_alloc_bisect.jl 2>&1 | tee "$BISECT_LOG"

{
  echo "julia=$(julia --version)"
  echo "cuda_module=$(module list 2>&1 | tr '\n' ' ')"
  echo "device=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
  echo "fmm_tree_sha256=$(find "$WORKDIR/src" "$WORKDIR/ext" "$WORKDIR/test/metal_env" -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  echo "manifest_sha256=$(shasum -a 256 "$ENVDIR/Manifest.toml" | awk '{print $1}')"
  echo "tree_build_log=$TREE_LOG"
  echo "tree_build_log_sha256=$(shasum -a 256 "$TREE_LOG" | awk '{print $1}')"
  echo "tree_build_gpu_log=$TREE_GPULOG"
  echo "bisect_log=$BISECT_LOG"
  echo "bisect_log_sha256=$(shasum -a 256 "$BISECT_LOG" | awk '{print $1}')"
} > "$PROV"
cat "$PROV"
echo "tree_build_and_bisect job complete"
