#!/bin/bash
#SBATCH --job-name=fm_treebuild_isolate
#SBATCH --gpus=h200:1
#SBATCH --qos=eng
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# Isolated single n=1e6 case (own fresh process), vs. the 8-cases-per-process
# tree_build_bench_run.sh. Last untried lever for the n>=1e5 staircase anomaly.
#
# Job 13506486 showed the anomaly reproduces in isolation as a smooth ramp
# starting ~trial 50-55/100 in both arms. The script now samples NVML
# clocks/power/temp/pstate/throttle-reasons IN-PROCESS, once per trial, aligned
# to trial index — every prior telemetry check (13506034/13506267/13506362)
# used a 1 Hz background nvidia-smi loop, too coarse and unaligned to see a
# transition landing at a specific trial. No background GPU log here: the
# per-trial NVML samples supersede it.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"
BASEDIR="${FM_TREEBUILD_BENCH_DIR:-$HOME/archive/fastmultipole-ka-cuda-bench}"
ENVDIR="$BASEDIR/env"
RESULTS="$BASEDIR/results"
cd "$RESULTS"
LOG="$RESULTS/tree_build_bench_isolate_${SLURM_JOB_ID}.log"
PROV="$RESULTS/tree_build_bench_isolate_${SLURM_JOB_ID}.provenance"
echo "=== Isolated single-case tree-build benchmark: test/metal_env/tree_build_benchmark_isolate_1e6.jl ==="
julia --project="$ENVDIR" "$BASEDIR/FastMultipole/test/metal_env/tree_build_benchmark_isolate_1e6.jl" 2>&1 | tee "$LOG"
{
  echo "julia=$(julia --version)"
  echo "cuda_module=$(module list 2>&1 | tr '\n' ' ')"
  echo "device=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
  echo "fastmultipole_tree_sha256=$({ find "$BASEDIR/FastMultipole/src" "$BASEDIR/FastMultipole/ext" -type f -print0; } | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  echo "manifest_sha256=$(shasum -a 256 "$ENVDIR/Manifest.toml" | awk '{print $1}')"
  echo "raw_log=$LOG"
  echo "raw_log_sha256=$(shasum -a 256 "$LOG" | awk '{print $1}')"
} > "$PROV"
cat "$PROV"
echo "fm_treebuild_isolate job complete"
