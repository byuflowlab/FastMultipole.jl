#!/usr/bin/env bash
#SBATCH --job-name=fp052d5e
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
# 052d Step-5e: eval-time state capture. Rerun the xverify sim briefly (to
# np>=3500 only) with the new _cross_run_body! eval-dump hook so we can compare
# the state the device pass ACTUALLY uploaded against the state the dump hook
# captured moments later — settles the production-vs-dump input question.
# (Investigation authorized by Ryan 2026-08-29.)
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"; nvidia-smi -L

FMDIR="${FP052D_FMDIR:-$HOME/FastMultipole-052-h200}"
FPDIR="${FP052D_FPDIR:-$HOME/FLOWPanel-052-h200}"
ENVDIR="${FP052D_ENV:-$HOME/fm052env-h200}"
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

cd "$FPDIR"
DUMPDIR="$FMDIR/relU_dumps_5e_${SLURM_JOB_ID}"
XLOG="fp052d5e_xverify_${SLURM_JOB_ID}.log"
source "$HOME/FLOWVPM-052-h200/scripts/fm052_common.sh" || true
set +e
timeout 25m env "${FM052_PRODUCTION_ENV[@]}" "${FM052_GPU_ENV[@]}" \
  PANEL_INFLUENCE_FMM=1 PANEL_INFLUENCE_FMM_DEVICE=1 \
  PANEL_INFLUENCE_FMM_XVERIFY=1 \
  PANEL_FMM_DUMP_DIR="$DUMPDIR" PANEL_FMM_DUMP_NP=3500 \
  NREVS=0.05 RUN_NAME="fp052d5e_xverify_${SLURM_JOB_ID}" \
  julia --project="$ENVDIR" --threads="$JULIA_NUM_THREADS" \
  examples/rotor_hover_pressure_comparison.jl > "$XLOG" 2>&1
xrc=$?
set -e
echo "xverify run exit code: $xrc (124 = timeout, OK if dumps landed)"
grep -E "panel_cross_xverify|panel_fmm_dump|panel_cross_eval_dump|panel_cross config" "$XLOG" | tail -12
ls -la "$DUMPDIR" 2>/dev/null || true
N1=$(ls "$DUMPDIR"/ateval_np*_meta.txt 2>/dev/null | wc -l)
N2=$(ls "$DUMPDIR"/dump_np*_meta.txt 2>/dev/null | wc -l)
echo "eval dumps: $N1, post dumps: $N2"
[ "$N1" -ge 1 ] && [ "$N2" -ge 1 ] && echo "fp052d step-5e complete: DUMPS PRESENT" \
  || { echo "fp052d step-5e: MISSING DUMPS"; tail -30 "$XLOG"; exit 1; }
