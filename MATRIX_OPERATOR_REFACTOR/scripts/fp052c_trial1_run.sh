#!/usr/bin/env bash
# 052c sigma-guard TRIAL 1 (Ryan-ruled 2026-08-26): cap dt*Z at 0.5 +
# absolute sigma floor at 1% sigma_shed, kwarg-gated (SIGMA_DTZ_CAP /
# SIGMA_FLOOR_FRAC -> simulate! sigma_guard). Two stages, one allocation:
#   (1) 36-step mature gate vs pinned CPU reference with the guard ARMED
#       (expected: guard never engages in the mature window -> gates
#       bit-identical; set -e aborts the chain if not).
#   (2) full 1080-step stage-d acceptance with the guard ARMED, run_name
#       fm052d_gpu_1080_t1 (the unguarded baseline fm052d_gpu_1080 is
#       PRESERVED for comparison) + canonical stage-d checks.
# Ledger: FastMultipole MATRIX_OPERATOR_REFACTOR/052c-sigma-experiments-2026-08-26.md
set -euo pipefail
module load cuda julia/1.11.7-6bmogfl
FPDIR=$HOME/FLOWPanel-052-h200
VPMDIR=$HOME/FLOWVPM-052-h200
ENVDIR=$HOME/fm052env-h200
TOLERANCE=$HOME/FLOWPanel-052/data/fm052_campaign_lock/fm052_locked_tolerances.toml
CPU_RUN=$HOME/FLOWPanel-052/data/fm052r_cpu_mature_pinned
export FP052_COUNT_TOL=${FP052_COUNT_TOL:-16}
test -s "$TOLERANCE" || { echo "locked tolerance missing: $TOLERANCE"; exit 1; }
test -d "$CPU_RUN" || { echo "pinned CPU mature reference missing: $CPU_RUN"; exit 1; }
source "$VPMDIR/scripts/fm052_common.sh"
fm052_preflight_checkpoint
THREADS=${SLURM_CPUS_PER_TASK:-64}
export JULIA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS \
  OPENBLAS_NUM_THREADS=$THREADS BLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
GUARD_ENV=(SIGMA_DTZ_CAP=0.5 SIGMA_FLOOR_FRAC=0.01)
cd "$FPDIR"
nvidia-smi -L

# ---------- (1) mature gate, guard armed ----------
run_name=fm052c_gpu_mature_t1_${SLURM_JOB_ID}
mkdir -p "data/$run_name"
find "data/$run_name" -maxdepth 1 -name '*.pvd' -delete
env "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" "${GUARD_ENV[@]}" \
  NREVS=19.5 RESTART_STEP="$FM052_RESTART_STEP" RESTART_NAME="$FM052_RESTART_NAME" \
  RESTART_PATH="$FM052_CHECKPOINT_ROOT" RHPC_SOLVER_S_GPU_SAMPLE_INTERVAL=1 \
  FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true \
  RUN_NAME="$run_name" \
  julia --project="$ENVDIR" --threads="$THREADS" examples/rotor_hover_pressure_comparison.jl \
  2>&1 | tee "data/${run_name}.log"
gpu_s_count=$(grep -c source_influence_s_gpu_gemv "data/${run_name}.log" || true)
cpu_s_count=$(grep -c source_influence_s_gemv "data/${run_name}.log" || true)
test "$gpu_s_count" -eq 36 && test "$cpu_s_count" -eq 0 || {
  echo "source gate FAILED: gpu_s=$gpu_s_count (want 36) cpu_s=$cpu_s_count (want 0)"; exit 1; }
julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" verify \
  "data/$run_name" 720 755
FP052_VPMDIR="$VPMDIR" FP052_ENV="$ENVDIR" \
  bash "$VPMDIR/scripts/fm052_gate.sh" "$TOLERANCE" "$CPU_RUN" \
  "data/$run_name" "data/fm052c_mature_gate_t1_${SLURM_JOB_ID}"
echo "052c trial-1 stage 1: mature gate PASSED with guard armed (run data/$run_name)"

# ---------- (2) 1080-step acceptance, guard armed ----------
run_name=fm052d_gpu_1080_t1
mkdir -p "data/$run_name"
find "data/$run_name" -maxdepth 1 -name '*.pvd' -delete
"$VPMDIR/scripts/fm052_provenance.sh" \
  "data/$run_name/${run_name}_provenance.toml" "$FM052_CHECKPOINT_ROOT"
start_ns=$(date +%s%N)
env "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" "${GUARD_ENV[@]}" \
  NREVS=28.5 \
  FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true \
  RUN_NAME="$run_name" \
  julia --project="$ENVDIR" --threads="$THREADS" examples/rotor_hover_pressure_comparison.jl \
  2>&1 | tee "data/${run_name}.log"
end_ns=$(date +%s%N)
printf '%d.%09d\n' "$(( (end_ns - start_ns) / 1000000000 ))" \
  "$(( (end_ns - start_ns) % 1000000000 ))" > "data/$run_name/${run_name}_process_wall_s.txt"
gpu_s_count=$(grep -c source_influence_s_gpu_gemv "data/${run_name}.log" || true)
cpu_s_count=$(grep -c source_influence_s_gemv "data/${run_name}.log" || true)
test "$gpu_s_count" -eq 1080 && test "$cpu_s_count" -eq 0 || {
  echo "source gate FAILED: gpu_s=$gpu_s_count (want 1080) cpu_s=$cpu_s_count (want 0)"; exit 1; }
grep -q '^n_steps = 1080$' "data/$run_name/${run_name}_case_metadata.toml"
julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" report \
  "data/$run_name" "data/${run_name}.log"
julia --project="$ENVDIR" "$VPMDIR/scripts/fm052_compare.jl" verify \
  "data/$run_name" 0 1079
echo "052c trial-1 stage-d acceptance COMPLETE: data/$run_name"
