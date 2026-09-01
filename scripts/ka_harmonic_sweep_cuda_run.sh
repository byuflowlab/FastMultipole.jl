#!/bin/bash
#SBATCH --job-name=fmm_ka_harmsweep
#SBATCH --qos=eng
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# H200 gate for the session-41 harmonic-sweep rewrite of the KA B2M and L2B
# kernels (ext/FastMultipoleKAExt.jl: ka_vortex_q3 / ka_local_eval_flat).
#
# The rewrite is KA-only -- src/translate_batched_resident.jl and
# src/translate_batched_cuda.jl are untouched -- and it was developed and gated
# entirely on Metal. Three things need real CUDA hardware to establish:
#
#   Arm 0  the new harmonic walk returns BIT-IDENTICAL coefficients to the
#          shared `_resident_vortex_q` it replaces. Host-side, so this is a
#          portability check on the arithmetic, not the device.
#   Arm 1  the rewritten KA kernels are correct on CUDA, scored against
#          FastMultipole's OWN CPU host kernels (the suites' oracles). Metal
#          green says nothing about CUDA -- different codegen, different
#          fast-math, different reduction hardware.
#   Arm 2  the CUDA path itself did not move. It cannot have, since no file it
#          reads was edited, but test/cuda/runtests.jl is cheap insurance and
#          the claim is worth a green light rather than an argument.
#
# Each arm runs as its own julia process so a device fault in one cannot mask
# the others. Submit via scripts/ka_harmonic_sweep_cuda_submit.sh.
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"

WORKDIR="${FMM_KABENCH_DIR:-$HOME/FastMultipole-kabench}"
ENVDIR="${FMM_KABENCH_ENV:-$HOME/fm_kabench_env}"
CUDAENV="${FMM_CUDATEST_ENV:-$HOME/fm_cudatest_env}"
cd "$WORKDIR"
LOG="$WORKDIR/ka_harmsweep_cuda_${SLURM_JOB_ID}.log"
PROV="$WORKDIR/ka_harmsweep_cuda_${SLURM_JOB_ID}.provenance"
: > "$LOG"
FAILED=""

run_arm () {  # run_arm <label> <project> <script...>
  local label="$1"; shift
  local proj="$1"; shift
  echo "=== $label ===" | tee -a "$LOG"
  if julia --project="$proj" "$@" 2>&1 | tee -a "$LOG"; then
    echo "--- $label: exit 0" | tee -a "$LOG"
  else
    echo "--- $label: FAILED" | tee -a "$LOG"
    FAILED="$FAILED $label"
  fi
}

# ---- Arm 0: host-side bit-exactness of the new harmonic walk
run_arm "q3_bit_exact" "$ENVDIR" test/metal_env/_probe_q3_exact.jl

# ---- Arm 1: the KA suites the rewrite can affect, plus the stages downstream
# of B2M/L2B that consume their output end to end.
SUITES="ka_b2m_correctness.jl \
ka_l2b_correctness.jl \
ka_m2m_correctness.jl \
ka_m2l_correctness.jl \
ka_l2l_correctness.jl \
ka_m2t_correctness.jl \
ka_s2l_correctness.jl \
ka_nearfield_correctness.jl \
ka_stage_groups_correctness.jl \
ka_device_cache_correctness.jl \
ka_lifecycle_body_correctness.jl \
ka_production_driver_correctness.jl"
for s in $SUITES; do
  run_arm "$s" "$ENVDIR" "test/metal_env/$s"
done

# ---- Arm 2: CUDA-path regression (must be untouched by a KA-only change)
run_arm "cuda_runtests" "$CUDAENV" test/cuda/runtests.jl

{
  echo "julia=$(julia --version)"
  echo "cuda_module=$(module list 2>&1 | tr '\n' ' ')"
  echo "device=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)"
  echo "fmm_tree_sha256=$(find "$WORKDIR/src" "$WORKDIR/ext" "$WORKDIR/test" -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
  echo "ext_sha256=$(shasum -a 256 "$WORKDIR/ext/FastMultipoleKAExt.jl" | awk '{print $1}')"
  echo "manifest_sha256=$(shasum -a 256 "$ENVDIR/Manifest.toml" | awk '{print $1}')"
  echo "raw_log=$LOG"
  echo "failed_arms=${FAILED:-none}"
} > "$PROV"
cat "$PROV"

echo "=== summary ==="
grep -E "PASS|FAIL|passed|bit-identical|^--- " "$LOG" | tail -60 || true
[ -z "$FAILED" ] || { echo "FAILING ARMS:$FAILED"; exit 1; }
echo "ka_harmsweep_cuda job complete: all arms passed"
