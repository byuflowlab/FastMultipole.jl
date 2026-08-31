#!/usr/bin/env bash
#SBATCH --job-name=fp052d5
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# 052d Step-5 combined GPU validation (ONE job per the cluster house rule;
# submission requires Ryan's authorization).
#
#   1. p34 Stage-A oracle  — device producers vs untouched host CrossStencil
#                            (3 configs, list bit-compare)
#   2. p35 Stage-B oracle  — device panel B2M + M2M vs host reference,
#                            incl. TE-wake case 3 (host ref = the actual
#                            RigidWakeBody overload; REQUIRES FLOWPanel env)
#   3. p36 Stage-C oracle  — device cross-M2L locals vs production host M2L
#   4. p37 Stage-D oracle  — END-TO-END B2M→M2M→M2L→L2L→L2B vs host classic
#                            evaluate_local (relRMS 1e-10)
#   5. p38 Stage-E oracle  — block-sparse near field vs host block reference
#                            (wake arm, ring + doublet tags; timing vs the
#                            ~0.039 s step-472 target)
#   6. Stage-F XVERIFY smoke — short production-shape FLOWPanel run with
#                            PANEL_INFLUENCE_FMM=1 + DEVICE + XVERIFY:
#                            prints per-step device-vs-hostFMM relU.
#                            EXPECTATION: relU ~ 1e-5 (sum of the two
#                            independent approximation errors — host tree-FMM
#                            p8/theta0.4 vs cross stencil q12/P6; the cross
#                            stencil itself is certified 7.2e-6 vs dense,
#                            p32g). Values >> 1e-4 indicate a port bug.
#
# Prereqs on the cluster:
#   - FastMultipole checkout with this branch (incl.
#     MATRIX_OPERATOR_REFACTOR/prototypes/052d_cross_stencil/)
#   - FLOWPanel checkout with the Stage-F seam edits
#   - env with both dev'd (fm052env pattern)
#   - snapshot472 bins rsynced; point SNAPDIR at them (REQUIRED — the local
#     scratchpad path baked into the oracle defaults does not exist here)
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"; nvidia-smi -L

FMDIR="${FP052D_FMDIR:-$HOME/FastMultipole-052-h200}"
FPDIR="${FP052D_FPDIR:-$HOME/FLOWPanel-052-h200}"
ENVDIR="${FP052D_ENV:-$HOME/fm052env-h200}"
SNAPDIR="${SNAPDIR:?set SNAPDIR to the rsynced snapshot472 directory}"
test -f "$SNAPDIR/panel_offsets_i64.bin" || { echo "snapshot472 missing at $SNAPDIR"; exit 1; }

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export SNAPDIR
ORACLES="$FMDIR/MATRIX_OPERATOR_REFACTOR/prototypes/052d_cross_stencil"
cd "$FMDIR"

fail=0
for stage in p34_stageA_oracle p35_stageB_oracle p36_stageC_oracle \
             p37_stageD_oracle p38_stageE_oracle; do
  echo "=== $stage ==="
  LOG="fp052d5_${stage}_${SLURM_JOB_ID}.log"
  if julia --project="$ENVDIR" "$ORACLES/$stage.jl" 2>&1 | tee "$LOG"; then
    echo "$stage PASS"
  else
    echo "$stage FAIL (log: $LOG)"; fail=1
  fi
done

echo "=== stage 6: Stage-F XVERIFY smoke (production shape, few steps) ==="
# Short warm-restart at the certified step-472 state if available; otherwise a
# short cold run. Either way the deliverable is the per-step
# `panel_cross_xverify ... relU=` lines and the route counters.
cd "$FPDIR"
XLOG="fp052d5_xverify_${SLURM_JOB_ID}.log"
source "$HOME/FLOWVPM-052-h200/scripts/fm052_common.sh" || true
env "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" \
  PANEL_INFLUENCE_FMM=1 PANEL_INFLUENCE_FMM_DEVICE=1 \
  PANEL_INFLUENCE_FMM_XVERIFY=1 \
  NREVS=0.3 RUN_NAME="fp052d5_xverify_${SLURM_JOB_ID}" \
  FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true \
  julia --project="$ENVDIR" --threads="$JULIA_NUM_THREADS" \
  examples/rotor_hover_pressure_comparison.jl 2>&1 | tee "$XLOG" || fail=1
# device-route hit counter: exactly one xverify line prints per successful
# device-route call (right before _gpu_route_hit!), so the line count IS the
# cross-route hit count for this run — require it to be a positive integer
XHITS=$(grep -c "panel_cross_xverify" "$XLOG" || true)
echo "cross-route hits (xverify lines): ${XHITS:-0}"
if ! [ "${XHITS:-0}" -gt 0 ] 2>/dev/null; then
  echo "device-route hit count is zero — cross route did not engage"; fail=1
fi
grep "panel_cross_xverify" "$XLOG" | tail -5
# enforce the stated accuracy bound: EVERY step's relU must parse and sit at
# or below 1e-4 (values >> 1e-4 indicate a port bug; NaN also fails)
if ! awk '
  /panel_cross_xverify/ {
    found = 0
    for (i = 1; i <= NF; i++) if ($i ~ /^relU=/) {
      found = 1; v = substr($i, 6) + 0
      if (v != v || v > 1e-4) { printf "BAD relU %s on: %s\n", substr($i, 6), $0; bad = 1 }
    }
    if (!found) { printf "xverify line without relU: %s\n", $0; bad = 1 }
  }
  END { exit bad ? 1 : 0 }' "$XLOG"; then
  echo "xverify relU threshold (1e-4) FAIL"; fail=1
else
  echo "xverify relU threshold (1e-4) PASS over ${XHITS:-0} steps"
fi

echo "=== provenance ==="
echo "fastmultipole_sha=$(git -C "$FMDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
echo "flowpanel_sha=$(git -C "$FPDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
echo "snapdir=$SNAPDIR"
echo "julia=$(julia --version)"

[ "$fail" -eq 0 ] && echo "fp052d step-5 job complete: ALL STAGES PASS" || { echo "fp052d step-5 job complete: FAILURES above"; exit 1; }
