#!/usr/bin/env bash
#SBATCH --job-name=fp052d5b
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# 052d Step-5b: oracles-only rerun (ONE job per the cluster house rule;
# submission requires Ryan's authorization). Reruns p34-p38 after the
# 2026-08-29 fixes (ulp containment tolerance, zero-node guard in
# finish_cross_locals!, p36 host-ref tag-3 mapping). Stage-F XVERIFY is
# deliberately omitted: the relU gate is being recalibrated via a CPU dense
# spot-check first (see the 052d plan of 2026-08-29).
#
# Prereqs identical to fp052d_step5_oracles_run.sh.
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
  LOG="fp052d5b_${stage}_${SLURM_JOB_ID}.log"
  if julia --project="$ENVDIR" "$ORACLES/$stage.jl" 2>&1 | tee "$LOG"; then
    echo "$stage PASS"
  else
    echo "$stage FAIL (log: $LOG)"; fail=1
  fi
done

# oracle scripts print "N PASS, M FAIL" but exit 0 either way; enforce M == 0
if grep -hE "oracle: [0-9]+ PASS, [0-9]+ FAIL" fp052d5b_p3?_*_${SLURM_JOB_ID}.log \
    | grep -vE ", 0 FAIL"; then
  echo "one or more oracles reported FAIL lines above"; fail=1
fi

echo "=== provenance ==="
echo "fastmultipole_sha=$(git -C "$FMDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
echo "flowpanel_sha=$(git -C "$FPDIR" rev-parse HEAD 2>/dev/null || echo rsync-no-git)"
echo "snapdir=$SNAPDIR"
echo "julia=$(julia --version)"

[ "$fail" -eq 0 ] && echo "fp052d step-5b job complete: ALL ORACLES PASS" || { echo "fp052d step-5b job complete: FAILURES above"; exit 1; }
