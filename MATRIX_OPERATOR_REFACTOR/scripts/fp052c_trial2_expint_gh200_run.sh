#!/usr/bin/env bash
# 052c TRIAL 2 (Ryan-ruled 2026-09-05): exponential integrator INSTEAD of
# sigma/dtZ clamps. WAKE_EXPINT=true routes the wake through FLOWVPM
# euler_exp (frozen-gradient geometric update, sigma > 0 by construction);
# SIGMA_DTZ_CAP/SIGMA_FLOOR_FRAC are UNSET (guard=off — euler_exp rejects a
# non-empty sigma_guard anyway). Requires the 026 Phase 1b Task 1 GPU
# broadcast port (_euler_exp_broadcast! + _corespreading_eulerexp_broadcast!)
# in the -gh200 silo FLOWVPM (installed 2026-09-05, .bak-preexp backups).
# Two stages, one allocation:
#   (1) 36-step mature gate vs pinned CPU EULER reference — INFORMATIONAL
#       ONLY (a different integrator legitimately shifts the fingerprint);
#       gate result is recorded but does not abort the chain. GPU-routing
#       source gates remain FATAL.
#   (2) full 1080-step stage-d acceptance, run_name fm052d_gpu_1080_t2exp
#       (t1 guarded run and unguarded baseline both PRESERVED).
# Readout: does the historical step-~1015 sigma collapse resolve without
# clamps? Watch min_sigma trajectory + the euler_exp substep-budget error
# (a throw there = instability persists, informative not catastrophic).
# Ledger: FastMultipole MATRIX_OPERATOR_REFACTOR/052c-sigma-experiments-2026-08-26.md
set -euo pipefail
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: gh200 launcher requires aarch64 node (got $(uname -m))"; exit 2; }
JULIA_BIN=$HOME/julia/julia-1.11.7/bin/julia
[[ -x "$JULIA_BIN" ]] || { echo "ERROR: ARM julia not found at $JULIA_BIN"; exit 2; }
export JULIA_DEPOT_PATH=$HOME/fm052depot-gh200
export PATH="$HOME/julia/julia-1.11.7/bin:$PATH"   # fm052_gate.sh calls bare `julia`
export FP052_JULIA_BIN="$JULIA_BIN"                 # fm052_provenance.sh override
FPDIR=$HOME/FLOWPanel-052-gh200
VPMDIR=$HOME/FLOWVPM-052-gh200
ENVDIR=$HOME/fm052env-gh200
TOLERANCE=$HOME/projects/FLOWPanel.jl/data/fm052_campaign_lock/fm052_locked_tolerances.toml
CPU_RUN=$HOME/projects/FLOWPanel.jl/data/fm052r_cpu_mature_pinned
export FP052_COUNT_TOL=${FP052_COUNT_TOL:-16}
test -s "$TOLERANCE" || { echo "locked tolerance missing: $TOLERANCE"; exit 1; }
test -d "$CPU_RUN" || { echo "pinned CPU mature reference missing: $CPU_RUN"; exit 1; }
source "$VPMDIR/scripts/fm052_common.sh"
fm052_preflight_checkpoint
THREADS=${SLURM_CPUS_PER_TASK:-72}
export JULIA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS \
  OPENBLAS_NUM_THREADS=$THREADS BLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
GUARD_ENV=(WAKE_EXPINT=true)
cd "$FPDIR"
nvidia-smi -L

# ---------- (1) mature gate, guard armed ----------
run_name=fm052c_gpu_mature_t2exp_${SLURM_JOB_ID}
mkdir -p "data/$run_name"
find "data/$run_name" -maxdepth 1 -name '*.pvd' -delete
env "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" "${GUARD_ENV[@]}" \
  NREVS=19.5 RESTART_STEP="$FM052_RESTART_STEP" RESTART_NAME="$FM052_RESTART_NAME" \
  RESTART_PATH="$FM052_CHECKPOINT_ROOT" RHPC_SOLVER_S_GPU_SAMPLE_INTERVAL=1 \
  FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true \
  RUN_NAME="$run_name" \
  "$JULIA_BIN" --project="$ENVDIR" --threads="$THREADS" examples/rotor_hover_pressure_comparison.jl \
  2>&1 | tee "data/${run_name}.log"
gpu_s_count=$(grep -c source_influence_s_gpu_gemv "data/${run_name}.log" || true)
cpu_s_count=$(grep -c source_influence_s_gemv "data/${run_name}.log" || true)
test "$gpu_s_count" -eq 36 && test "$cpu_s_count" -eq 0 || {
  echo "source gate FAILED: gpu_s=$gpu_s_count (want 36) cpu_s=$cpu_s_count (want 0)"; exit 1; }
"$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" verify \
  "data/$run_name" 720 755
if FP052_VPMDIR="$VPMDIR" FP052_ENV="$ENVDIR" \
  bash "$VPMDIR/scripts/fm052_gate.sh" "$TOLERANCE" "$CPU_RUN" \
  "data/$run_name" "data/fm052c_mature_gate_t2exp_${SLURM_JOB_ID}"; then
  echo "052c trial-2 stage 1: mature gate PASSED vs euler CPU reference (expint)"
else
  echo "052c trial-2 stage 1: mature gate DIFFERS vs euler CPU reference (EXPECTED-POSSIBLE with expint; informational, continuing)"
fi

# ---------- (2) 1080-step acceptance, guard armed ----------
run_name=fm052d_gpu_1080_t2exp
mkdir -p "data/$run_name"
find "data/$run_name" -maxdepth 1 -name '*.pvd' -delete
"$VPMDIR/scripts/fm052_provenance.sh" \
  "data/$run_name/${run_name}_provenance.toml" "$FM052_CHECKPOINT_ROOT"
start_ns=$(date +%s%N)
env "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" "${GUARD_ENV[@]}" \
  NREVS=28.5 \
  FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true \
  RUN_NAME="$run_name" \
  "$JULIA_BIN" --project="$ENVDIR" --threads="$THREADS" examples/rotor_hover_pressure_comparison.jl \
  2>&1 | tee "data/${run_name}.log"
end_ns=$(date +%s%N)
printf '%d.%09d\n' "$(( (end_ns - start_ns) / 1000000000 ))" \
  "$(( (end_ns - start_ns) % 1000000000 ))" > "data/$run_name/${run_name}_process_wall_s.txt"
gpu_s_count=$(grep -c source_influence_s_gpu_gemv "data/${run_name}.log" || true)
cpu_s_count=$(grep -c source_influence_s_gemv "data/${run_name}.log" || true)
test "$gpu_s_count" -eq 1080 && test "$cpu_s_count" -eq 0 || {
  echo "source gate FAILED: gpu_s=$gpu_s_count (want 1080) cpu_s=$cpu_s_count (want 0)"; exit 1; }
grep -q '^n_steps = 1080$' "data/$run_name/${run_name}_case_metadata.toml"
"$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" report \
  "data/$run_name" "data/${run_name}.log"
"$JULIA_BIN" --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" verify \
  "data/$run_name" 0 1079
echo "052c trial-2 (expint) stage-d acceptance COMPLETE: data/$run_name"
