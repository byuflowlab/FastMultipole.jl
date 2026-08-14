#!/bin/bash
# Tasks 037e/037f local driver: refresh the 034-layout cluster trees to the
# current local working trees (identical rsync set to cuda_035_submit.sh),
# then submit the two back-to-back H200 jobs (stage e: preflight + scoping +
# 037e screen; stage f: 037f screen + oracle).
#   bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_037ef_submit.sh
# Requires a live ssh master session to `orc`.
set -euo pipefail
REMOTE=orc
VPMDIR=FLOWVPM-034
FMDIR=FastMultipole-034
ENVDIR='$HOME/fm034env'
VPMLOCAL=../FLOWVPM.jl

ssh "$REMOTE" "mkdir -p $VPMDIR $FMDIR/MATRIX_OPERATOR_REFACTOR"

rsync -az --delete --exclude .git \
    "$VPMLOCAL/src" "$VPMLOCAL/ext" "$VPMLOCAL/test" "$VPMLOCAL/scripts" \
    "$VPMLOCAL/Project.toml" \
    "$REMOTE:$VPMDIR/"

rsync -az --delete --exclude .git --exclude '*.mem' \
    src test Project.toml \
    "$REMOTE:$FMDIR/"
rsync -az --delete --exclude '*.mem' \
    MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/"

ssh "$REMOTE" "mkdir -p $FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline $FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign $FMDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling"
rsync -az --delete \
    MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/references \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/"
rsync -az --delete \
    MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/"
rsync -az --delete --exclude '*.out' \
    MATRIX_OPERATOR_REFACTOR/data/rotor_wake \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/data/"

ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && export JULIA_PKG_PRECOMPILE_AUTO=0 \
  && julia --project=$ENVDIR -e \"using Pkg; Pkg.instantiate()\" \
  && cd $FMDIR \
  && sbatch --export=ALL,FM037EF_STAGE=e MATRIX_OPERATOR_REFACTOR/scripts/cuda_037ef_run.sh \
  && sbatch --export=ALL,FM037EF_STAGE=f MATRIX_OPERATOR_REFACTOR/scripts/cuda_037ef_run.sh'"

echo "Submitted stage-e and stage-f jobs. Poll with:  ssh orc 'bash -lc \"squeue -u \\\$USER\"'"
