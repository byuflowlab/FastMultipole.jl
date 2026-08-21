#!/bin/bash
#SBATCH --job-name=fm023fsw
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Task 023f optimize phase: dense whole-pass chunk-width sweep (production default
# 2^14). Restricted config set; production code unchanged during measurement
# (FM023F_CHUNK overrides DENSE_CUDA_CHUNK). Pattern: cuda_023d_sweep.sh.
source /etc/profile
set -o pipefail
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L

WORKDIR="${FM023F_DIR:-$HOME/FastMultipole-023}"
ENVDIR="$HOME/fm023env"
cd "$WORKDIR"

status=0
for chunk in 4096 16384 65536; do
    echo "=== chunk=$chunk"
    FM023F_CHUNK=$chunk FM023F_N=20000 FM023F_P=8,12 FM023F_F32=1 FM023F_REPS=7 \
        FM023F_OUT="MATRIX_OPERATOR_REFACTOR/data/dense_translation_m2l_cuda/chunk_sweep_${chunk}_$(hostname).csv" \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023f_dense_m2l_cuda.jl
    rc=$?
    echo "CHUNK_${chunk}_EXIT=$rc"
    status=$(( status || rc ))
done
exit $status
