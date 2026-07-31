#!/bin/bash
# Synchronize the exact task-026 source snapshot and submit the ORC campaign.
set -euo pipefail

REMOTE="${FM026_REMOTE:-orc}"
RDIR="${FM026_REMOTE_DIR:-FastMultipole-026}"
STAGE="${FM026_STAGE:-staging_026_cpu}"
PHASE="${FM026_PHASE:-all}"
SELECTED_WINDOW="${FM026_SELECTED_WINDOW:-}"
STRATEGIES="${FM026_STRATEGIES:-}"   # comma-separated, no spaces
manifest=$(mktemp)
trap 'rm -f "$manifest"' EXIT

find src test MATRIX_OPERATOR_REFACTOR/scripts -type f -print0 |
  sort -z | xargs -0 shasum -a 256 > "$manifest"
manifest_sha=$(shasum -a 256 "$manifest" | awk '{print $1}')

ssh "$REMOTE" "mkdir -p $RDIR/$STAGE"
rsync -az --delete --exclude .git src test Project.toml Manifest.toml \
  MATRIX_OPERATOR_REFACTOR/scripts "$REMOTE:$RDIR/$STAGE/"
scp "$manifest" "$REMOTE:$RDIR/$STAGE/source_manifest.sha256"
ssh "$REMOTE" "cd $RDIR \
  && rsync -a --delete $STAGE/src/ src/ \
  && rsync -a --delete $STAGE/test/ test/ \
  && rsync -a --delete $STAGE/scripts/ MATRIX_OPERATOR_REFACTOR/scripts/ \
  && cp $STAGE/Project.toml $STAGE/Manifest.toml . \
  && cp $STAGE/source_manifest.sha256 MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_host/ \
  && bash -lc 'module load julia \
    && julia --project=\$HOME/fm026env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
    && FM026_SOURCE_HASH=$manifest_sha FM026_PHASE=$PHASE \
       FM026_SELECTED_WINDOW=$SELECTED_WINDOW FM026_STRATEGIES=$STRATEGIES \
       sbatch --export=ALL,FM026_SOURCE_HASH,FM026_PHASE,FM026_SELECTED_WINDOW,FM026_STRATEGIES \
       MATRIX_OPERATOR_REFACTOR/scripts/cpu_026_run.sh'"
