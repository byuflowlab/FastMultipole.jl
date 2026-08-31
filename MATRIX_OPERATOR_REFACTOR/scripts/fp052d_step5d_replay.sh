#!/usr/bin/env bash
#SBATCH --job-name=fp052d5d
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# 052d Step-5d: device-side dense + cross-pass replay on job 13509236's
# dumped production states (relU attribution follow-up; authorized by Ryan
# 2026-08-29 "go ahead and investigate" after endorsing the device dense solve).
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"; nvidia-smi -L

FMDIR="${FP052D_FMDIR:-$HOME/FastMultipole-052-h200}"
ENVDIR="${FP052D_ENV:-$HOME/fm052env-h200}"
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export DUMPDIR="$FMDIR/relU_dumps_13509236"
test -f "$DUMPDIR/dump_np3544_meta.txt" || { echo "dumps missing at $DUMPDIR"; exit 1; }

cd "$FMDIR"
julia --project="$ENVDIR" \
  MATRIX_OPERATOR_REFACTOR/prototypes/052d_cross_stencil/p47_device_replay.jl
echo "fp052d step-5d complete"
