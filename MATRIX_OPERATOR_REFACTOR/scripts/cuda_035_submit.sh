#!/bin/bash
# Task 035 local driver: refresh the task-034-owned cluster trees
# (~/FLOWVPM-034 gpu-full, ~/FastMultipole-034 matrix-ops — the 034 layout,
# reused by 035 per the task handoff; NOT the 023/029p2/033 trees) to the
# current local working trees, then submit the H200 tuning-sweep job.
# Pattern: FLOWVPM scripts/cuda_034_submit.sh (env recipe unchanged, env
# ~/fm034env already instantiated; a plain instantiate refresh is run in case
# the dev'ed trees changed deps).
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_035_submit.sh   # from FastMultipole repo root
# Requires a live ssh master session to `orc`.
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-034
FMDIR=FastMultipole-034
ENVDIR='$HOME/fm034env'
VPMLOCAL=../FLOWVPM.jl

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR/MATRIX_OPERATOR_REFACTOR"

# FLOWVPM tree (gpu-full working tree, includes the 035 settings extension)
rsync -az --delete --exclude .git \
    "$VPMLOCAL/src" "$VPMLOCAL/ext" "$VPMLOCAL/test" "$VPMLOCAL/scripts" \
    "$VPMLOCAL/Project.toml" \
    "$REMOTE:$VPMDIR/"

# FastMultipole tree (matrix-ops working tree at the current tip — the -034
# copy predates the 032a shipped defaults and must be refreshed)
rsync -az --delete --exclude .git --exclude '*.mem' \
    src test Project.toml \
    "$REMOTE:$FMDIR/"
rsync -az --delete --exclude '*.mem' \
    MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/"

# 033 checksummed sampled-direct references (accuracy instrument) + the 024b
# scalar references (the FM035_NOREG scalar no-regression stage)
ssh "$REMOTE" "mkdir -p $FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline $FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign $FMDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling"
rsync -az --delete \
    MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/references \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/"
rsync -az --delete \
    MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/"

# env refresh (dev paths already registered by cuda_034_submit.sh)
ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && export JULIA_PKG_PRECOMPILE_AUTO=0 \
  && julia --project=$ENVDIR -e \"using Pkg; Pkg.instantiate()\" \
  && cd $FMDIR \
  && sbatch --export=ALL,FM035_PREFLIGHT=${FM035_PREFLIGHT:-1},FM035_CASEFILE=${FM035_CASEFILE:-fm035_cases_initial.txt},FM035_OUTNAME=${FM035_OUTNAME:-fm035_sweep.csv},FM035_NOREG=${FM035_NOREG:-0},FM035_NSYS=${FM035_NSYS:-0} MATRIX_OPERATOR_REFACTOR/scripts/cuda_035_run.sh'"

echo "Submitted. Poll with:  ssh orc 'bash -lc \"squeue -u \\\$USER\"'"
