#!/bin/bash
#SBATCH --job-name=fm035-analysis
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
FMDIR="${FM035_FMDIR:-$HOME/FastMultipole-034}"
ENVDIR="${FM035_ENV:-$HOME/fm034env}"
DATADIR="$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign"
mkdir -p "$DATADIR"
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1 JULIA_NUM_THREADS=8 FM035_FMDIR="$FMDIR"

cd "$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/references"
sha256sum -c direct_reference_checksums.sha256
cd "$FMDIR"
export FM035D_OUT="$DATADIR/${FM035D_OUTNAME:-fm035_error_decomposition.csv}"
# Optional config-file-driven run (037b): FM035D_CONFIGNAME names a file in
# MATRIX_OPERATOR_REFACTOR/scripts/ that becomes FM035D_CONFIG_FILE.
if [ -n "${FM035D_CONFIGNAME:-}" ]; then
    export FM035D_CONFIG_FILE="$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/$FM035D_CONFIGNAME"
fi
julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/benchmark_035_error_decomposition.jl
