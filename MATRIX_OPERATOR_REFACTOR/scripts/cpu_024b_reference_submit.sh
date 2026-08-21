#!/bin/bash
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-024b
STAGE=staging_024b_reference

ssh "$REMOTE" "mkdir -p $RDIR/$STAGE/matrix_scripts"
rsync -az --delete --exclude .git src test Project.toml Manifest.toml \
  "$REMOTE:$RDIR/$STAGE/"
rsync -az --delete MATRIX_OPERATOR_REFACTOR/scripts/ \
  "$REMOTE:$RDIR/$STAGE/matrix_scripts/"
ssh "$REMOTE" "cd $RDIR \
  && mkdir -p MATRIX_OPERATOR_REFACTOR/scripts \
  && rsync -a --delete $STAGE/src/ src/ \
  && rsync -a --delete $STAGE/test/ test/ \
  && rsync -a --delete $STAGE/matrix_scripts/ MATRIX_OPERATOR_REFACTOR/scripts/ \
  && cp $STAGE/Project.toml $STAGE/Manifest.toml . \
  && bash -lc 'module load julia \
    && julia --project=\$HOME/fm024env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
    && sbatch MATRIX_OPERATOR_REFACTOR/scripts/cpu_024b_reference_run.sh'"
