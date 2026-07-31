#!/bin/bash
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-024b
STAGE=staging_024b_cuda
REF_JOB="${1:-}"
REFDIR=MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references

if [[ -z "$REF_JOB" ]]; then
  for n in 1000 3162 10000 31623 100000 316228 1000000; do
    test -s "$REFDIR/direct_reference_n${n}.csv"
  done
  (cd "$REFDIR" && shasum -a 256 -c direct_reference_checksums.sha256)
fi

ssh "$REMOTE" "mkdir -p $RDIR/$STAGE/references"
rsync -az --delete --exclude .git src test Project.toml Manifest.toml \
  MATRIX_OPERATOR_REFACTOR/scripts "$REMOTE:$RDIR/$STAGE/"
if [[ -z "$REF_JOB" ]]; then
  rsync -az --delete "$REFDIR/" "$REMOTE:$RDIR/$STAGE/references/"
fi

dependency_arg=""
if [[ -n "$REF_JOB" ]]; then
  dependency_arg="--dependency=afterok:$REF_JOB"
fi

ssh "$REMOTE" "cd $RDIR \
  && mkdir -p MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references \
  && rsync -a --delete $STAGE/src/ src/ \
  && rsync -a --delete $STAGE/test/ test/ \
  && rsync -a --delete $STAGE/scripts/ MATRIX_OPERATOR_REFACTOR/scripts/ \
  && cp $STAGE/Project.toml $STAGE/Manifest.toml . \
  $([[ -z "$REF_JOB" ]] && printf '%s' "&& rsync -a --delete $STAGE/references/ MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references/") \
  && bash -lc 'module load julia \
    && julia --project=\$HOME/fm024env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
    && sbatch $dependency_arg MATRIX_OPERATOR_REFACTOR/scripts/cuda_024b_run.sh'"
