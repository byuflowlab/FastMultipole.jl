#!/bin/bash
#SBATCH --job-name=fm035-ncu
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=fm035-ncu-%A_%a.out
set -eo pipefail
source /etc/profile
module load cuda julia/1.11.7-6bmogfl
FMDIR="${FM035_FMDIR:-$HOME/FastMultipole-034}"
ENVDIR="${FM035_ENV:-$HOME/fm034env}"
DATADIR="$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign"
mkdir -p "$DATADIR"
export FASTMULTIPOLE_FORCE_CUDA_LOAD=1 JULIA_NUM_THREADS=8 FM035_FMDIR="$FMDIR"
CASES=(cube cube wake wake)
TFS=(Float32 Float64 Float32 Float64)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
TF="${TFS[$SLURM_ARRAY_TASK_ID]}"
STEM="fm035_ncu_${CASE}_n1000000_${TF}"
NCU=/apps/cudatoolkit/12.8.1/bin/ncu
cd "$FMDIR"

# CUDA profiler API limits both application runs to one warmed nearfield launch.
# Setup/classification gets a low-replay basic pass; the three pair buckets get
# the detailed analysis pass used for the bottleneck verdict.
"$NCU" --profile-from-start off --replay-mode kernel --set basic \
    --kernel-name-base function \
    --kernel-name 'regex:_cuda_(cell_sigma|nf_scalars|nearfield_bin)_kernel' \
    --export "$DATADIR/${STEM}_setup" --force-overwrite \
    julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/profile_035_nearfield_ncu.jl "$CASE" "$TF"
"$NCU" --import "$DATADIR/${STEM}_setup.ncu-rep" --csv --page raw > "$DATADIR/${STEM}_setup.csv"

"$NCU" --profile-from-start off --replay-mode kernel --set detailed \
    --kernel-name-base function \
    --kernel-name 'regex:_cuda_direct_pairs_bucket_kernel' \
    --export "$DATADIR/${STEM}_buckets" --force-overwrite \
    julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/profile_035_nearfield_ncu.jl "$CASE" "$TF"
"$NCU" --import "$DATADIR/${STEM}_buckets.ncu-rep" --csv --page raw > "$DATADIR/${STEM}_buckets.csv"
