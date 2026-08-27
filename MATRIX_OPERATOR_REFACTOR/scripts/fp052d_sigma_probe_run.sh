#!/usr/bin/env bash
# 052d sigma-collapse diagnostic probe (option 4, Ryan-approved 2026-08-26):
# warm restart of the FAILED acceptance 13497273 (rVPM core collapse:
# min_sigma crosses zero at step 1015, monitor rootfinder crash ~1071)
# from its step-990 snapshot, run 42 steps through step ~1031 with
# WAKE_HEALTH_DTZ=true (max per-step dt*Z contraction) and
# WAKE_HEALTH_ATTRIBUTION=true (p1 sigma percentile + argmin position):
# discriminates population contraction vs few-outlier collapse and shows
# whether the same region persists through the crossing.
# NOTE: restart source is the f32 single-series -> not replay-exact; the
# crossing step may shift slightly. Fine for diagnosis.
set -euo pipefail
module load cuda julia/1.11.7-6bmogfl
FPDIR=$HOME/FLOWPanel-052-h200
VPMDIR=$HOME/FLOWVPM-052-h200
ENVDIR=$HOME/fm052env-h200
FAILED_RUN=$FPDIR/data/fm052d_gpu_1080
source "$VPMDIR/scripts/fm052_common.sh"
THREADS=${SLURM_CPUS_PER_TASK:-64}
export JULIA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS \
  OPENBLAS_NUM_THREADS=$THREADS BLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
cd "$FPDIR"
nvidia-smi -L
test -f "$FAILED_RUN/fm052d_gpu_1080_wake1_particles/fm052d_gpu_1080_wake1_particles.990.vtp" || {
  echo "step-990 snapshot missing"; exit 1; }
probe_name=fm052d_sigma_probe_${SLURM_JOB_ID}
mkdir -p "data/$probe_name"
env "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" \
  NREVS=27.167 RESTART_STEP=990 RESTART_NAME=fm052d_gpu_1080 \
  RESTART_PATH="$FAILED_RUN" \
  WAKE_HEALTH_DTZ=true WAKE_HEALTH_ATTRIBUTION=true \
  FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true \
  RUN_NAME="$probe_name" \
  julia --project="$ENVDIR" --threads="$THREADS" examples/rotor_hover_pressure_comparison.jl \
  2>&1 | tee "data/${probe_name}.log" || echo "probe exited nonzero (expected if the monitor crash reproduces past the crossing) — wake_health rows up to the crash are the deliverable"
echo "sigma probe done: data/$probe_name"
tail -5 "data/$probe_name/monitors/${probe_name}_monitor04_wake_health_system1.csv" || true
