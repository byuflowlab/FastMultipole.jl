#!/bin/bash
#SBATCH --job-name=fmm_ka_correct
#SBATCH --qos=eng
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
# CUDA/H200 correctness gate for the KA adaptive-tree suites.
#
# Steps (i)-(iv) of the dispatch-wiring plan ran under a Metal-only gate: every
# ka_*_correctness.jl hardcoded Metal.MtlArray/Metal.MetalBackend(). Step (v)
# hands actx.grid to a DeviceResidentRadixState, whose lifecycle is CUDA-only,
# so from here on correctness has to be demonstrated on real CUDA hardware.
# test/metal_env/ka_backend.jl makes the suites backend-agnostic (Metal on
# Apple, CUDA elsewhere) and this job is the CUDA arm.
#
# Only the tree-build phases have been exercised on CUDA before, and only for
# timing (the KA arm of tree_build_benchmark_4way.jl) -- never for correctness,
# and Phases C/D/E/F/G never on CUDA at all. A failure here is therefore
# informative, not just a rubber stamp: KA's device-side semantics (sortperm!,
# scan/compaction, atomics, UInt64 key handling) are backend-dependent in ways
# the Metal runs cannot cover.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"
WORKDIR="${FMM_KABENCH_DIR:-$HOME/FastMultipole-kabench}"
ENVDIR="${FMM_KABENCH_ENV:-$HOME/fm_kabench_env}"
cd "$WORKDIR"
LOG="$WORKDIR/ka_correctness_cuda_${SLURM_JOB_ID}.log"
PROV="$WORKDIR/ka_correctness_cuda_${SLURM_JOB_ID}.provenance"

SUITES="ka_tree_leaves_correctness.jl ka_tree_balance_correctness.jl \
ka_tree_finalize_correctness.jl ka_tree_sigma_sweep_correctness.jl \
ka_tree_build_correctness.jl ka_tree_lists_correctness.jl \
ka_radix_state_correctness.jl ka_radix_lists_wiring_correctness.jl"
FAILED=""
: > "$LOG"
for s in $SUITES; do
  echo "=== $s ===" | tee -a "$LOG"
  # each suite is a separate julia process: a device fault in one phase must
  # not mask the phases after it
  if julia --project="$ENVDIR" "test/metal_env/$s" 2>&1 | tee -a "$LOG"; then
    echo "--- $s: exit 0" | tee -a "$LOG"
  else
    echo "--- $s: FAILED" | tee -a "$LOG"
    FAILED="$FAILED $s"
  fi
done

{
  echo "julia=$(julia --version)"
  echo "cuda_module=$(module list 2>&1 | tr '\n' ' ')"
  echo "device=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
  echo "fmm_tree_sha256=$(find "$WORKDIR/src" "$WORKDIR/ext" "$WORKDIR/test/metal_env" -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  echo "manifest_sha256=$(shasum -a 256 "$ENVDIR/Manifest.toml" | awk '{print $1}')"
  echo "raw_log=$LOG"
  echo "raw_log_sha256=$(shasum -a 256 "$LOG" | awk '{print $1}')"
  echo "failed_suites=${FAILED:-none}"
} > "$PROV"
cat "$PROV"
echo "=== summary ==="
grep -E "cases passed|✓✓✓|^--- " "$LOG" || true
[ -z "$FAILED" ] || { echo "FAILING SUITES:$FAILED"; exit 1; }
echo "ka_correctness_cuda job complete: all suites passed"
