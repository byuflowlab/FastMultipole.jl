#!/bin/bash
#SBATCH --job-name=fmm_ka_alloc_bisect
#SBATCH --qos=eng
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
# Bisect job 13506036's ~1.95GB/trial CUDA-KA allocation between sortperm!
# and accumulate! (see project_fastmultipole_ka_migration memory). Cheap,
# non-exclusive -- no need for the co-tenancy/clock diagnostics used by
# tree_build_bench_run.sh.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
WORKDIR="${FMM_KABENCH_DIR:-$HOME/FastMultipole-kabench}"
ENVDIR="${FMM_KABENCH_ENV:-$HOME/fm_kabench_env}"
cd "$WORKDIR"
LOG="$WORKDIR/ka_alloc_bisect_${SLURM_JOB_ID}.log"
PROV="$WORKDIR/ka_alloc_bisect_${SLURM_JOB_ID}.provenance"

echo "=== KA alloc bisect: test/metal_env/ka_alloc_bisect.jl ==="
julia --project="$ENVDIR" test/metal_env/ka_alloc_bisect.jl 2>&1 | tee "$LOG"

{
  echo "julia=$(julia --version)"
  echo "device=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
  echo "fmm_tree_sha256=$(find "$WORKDIR/src" "$WORKDIR/ext" "$WORKDIR/test/metal_env" -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  echo "manifest_sha256=$(shasum -a 256 "$ENVDIR/Manifest.toml" | awk '{print $1}')"
  echo "raw_log=$LOG"
  echo "raw_log_sha256=$(shasum -a 256 "$LOG" | awk '{print $1}')"
} > "$PROV"
cat "$PROV"
echo "ka_alloc_bisect job complete"
