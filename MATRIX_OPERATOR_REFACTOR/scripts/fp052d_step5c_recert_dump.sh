#!/usr/bin/env bash
#SBATCH --job-name=fp052d5c
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# 052d Step-5c (ONE combined job per the cluster house rule; submission
# authorized by Ryan 2026-08-29):
#   1. p34-p38 oracle recert (p37 host-ref tag-3 fix landed after job 13508865;
#      the other four were green there — rerun all five for one clean log).
#   2. Short xverify run with PANEL_FMM_DUMP: at np thresholds 3500/12500/28000
#      dump the exact per-step state (srcmat/wakemat/cent/positions) plus BOTH
#      route results (deviceU = cross pass, hostU = host fmm!). The relU
#      attribution then only needs a CPU dense reference on these states —
#      no trajectory replay. Bounded by `timeout`: the run needs only ~56 steps
#      to reach np~28k; exit 124 after all dumps land is SUCCESS.
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
  LOG="fp052d5c_${stage}_${SLURM_JOB_ID}.log"
  if julia --project="$ENVDIR" "$ORACLES/$stage.jl" 2>&1 | tee "$LOG"; then
    echo "$stage PASS"
  else
    echo "$stage FAIL (log: $LOG)"; fail=1
  fi
done
if grep -hE "oracle: [0-9]+ PASS, [0-9]+ FAIL" fp052d5c_p3?_*_${SLURM_JOB_ID}.log \
    | grep -vE ", 0 FAIL"; then
  echo "one or more oracles reported FAIL lines above"; fail=1
else
  echo "ALL ORACLES GREEN"
fi

echo "=== stage 2: xverify state dumps (np 3500/12500/28000) ==="
cd "$FPDIR"
DUMPDIR="$FMDIR/relU_dumps_${SLURM_JOB_ID}"
XLOG="fp052d5c_xverify_${SLURM_JOB_ID}.log"
source "$HOME/FLOWVPM-052-h200/scripts/fm052_common.sh" || true
set +e
timeout 60m env "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" \
  PANEL_INFLUENCE_FMM=1 PANEL_INFLUENCE_FMM_DEVICE=1 \
  PANEL_INFLUENCE_FMM_XVERIFY=1 \
  PANEL_FMM_DUMP_DIR="$DUMPDIR" PANEL_FMM_DUMP_NP=3500,12500,28000 \
  NREVS=0.3 RUN_NAME="fp052d5c_xverify_${SLURM_JOB_ID}" \
  FLOWPANEL_GPU_TIMERS=true FLOWPANEL_STEP_TIMERS=true \
  julia --project="$ENVDIR" --threads="$JULIA_NUM_THREADS" \
  examples/rotor_hover_pressure_comparison.jl > "$XLOG" 2>&1
xrc=$?
set -e
echo "xverify run exit code: $xrc (124 = timeout, expected)"
grep -E "panel_cross_xverify|panel_fmm_dump|panel_cross config" "$XLOG" | tail -12
NDUMP=$(ls "$DUMPDIR"/dump_np*_meta.txt 2>/dev/null | wc -l)
echo "dumps present: $NDUMP / 3"
ls -la "$DUMPDIR" 2>/dev/null || true
if [ "$NDUMP" -lt 3 ]; then echo "MISSING DUMPS"; fail=1; fi

echo "=== provenance ==="
echo "fastmultipole_sha=$(git -C "$FMDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
echo "flowpanel_sha=$(git -C "$FPDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
echo "julia=$(julia --version)"

[ "$fail" -eq 0 ] && echo "fp052d step-5c job complete: ALL STAGES PASS" || { echo "fp052d step-5c job complete: FAILURES above"; exit 1; }
