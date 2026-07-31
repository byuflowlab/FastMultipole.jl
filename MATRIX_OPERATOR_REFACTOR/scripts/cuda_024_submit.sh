#!/bin/bash
# Synchronize the exact task-024 snapshot and submit the H200 campaign.
# Run only after the user has explicitly approved cuda_024_run.sh.
set -euo pipefail
REMOTE=orc
RDIR=FastMultipole-024
STAGE=staging_024_cuda

commit=$(git rev-parse HEAD)
tree=$(git rev-parse 'HEAD^{tree}')
if git diff --quiet && git diff --cached --quiet; then worktree=clean; else worktree=dirty; fi
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
  && mkdir -p MATRIX_OPERATOR_REFACTOR/data/operator_ab_benchmark/raw \
  && rsync -a --delete $STAGE/src/ src/ \
  && rsync -a --delete $STAGE/test/ test/ \
  && rsync -a --delete $STAGE/scripts/ MATRIX_OPERATOR_REFACTOR/scripts/ \
  && cp $STAGE/Project.toml $STAGE/Manifest.toml . \
  && cp $STAGE/source_manifest.sha256 MATRIX_OPERATOR_REFACTOR/data/operator_ab_benchmark/ \
  && bash -lc 'module load julia \
    && julia --project=\$HOME/fm024env -e \"using Pkg; Pkg.develop(path=\\\".\\\"); Pkg.instantiate()\" \
    && FM024_GIT_COMMIT=$commit FM024_GIT_TREE=$tree FM024_GIT_WORKTREE=$worktree \
       FM024_SOURCE_MANIFEST=$manifest_sha \
       sbatch --export=ALL,FM024_GIT_COMMIT,FM024_GIT_TREE,FM024_GIT_WORKTREE,FM024_SOURCE_MANIFEST \
       MATRIX_OPERATOR_REFACTOR/scripts/cuda_024_run.sh'"
