#!/bin/bash
#SBATCH --job-name=fm023dsw
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Task 023d optimize phase: whole-pass chunk-width sweep for the precomputed-y
# device M2L (production default 2^14). Restricted config set; production code
# unchanged during measurement (FM023D_CHUNK overrides both whole-pass Refs).
source /etc/profile
set -o pipefail
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L

WORKDIR="${FM023D_DIR:-$HOME/FastMultipole-023}"
ENVDIR="$HOME/fm023env"
cd "$WORKDIR"

status=0
for chunk in 4096 16384 65536; do
    echo "=== chunk=$chunk"
    FM023D_CHUNK=$chunk FM023D_N=20000 FM023D_P=8,12 FM023D_F32=0 FM023D_REPS=7 \
        FM023D_OUT="MATRIX_OPERATOR_REFACTOR/data/precomputed_y_resident_m2l_cuda/chunk_sweep_${chunk}_$(hostname).csv" \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023d_precomputed_y_cuda.jl
    rc=$?
    echo "CHUNK_${chunk}_EXIT=$rc"
    status=$(( status || rc ))
done
exit $status
