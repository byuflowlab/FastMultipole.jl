#!/usr/bin/env bash
# Ryan-only optional B200 wrapper. Run only after inspecting a passing B200 probe.
set -euo pipefail

PROBE_JOB=${1:?usage: bash fm052_b200_smoke_submit.sh PROBE_JOB}
[[ "$PROBE_JOB" =~ ^[0-9]+$ ]] || { echo "PROBE_JOB must be numeric" >&2; exit 64; }

ssh orc "bash -lc 'cd \"\$HOME/FLOWVPM-052-b200\" && sbatch --parsable --nodes=1 \
  --partition=cs3 --qos=standby --no-requeue --gpus=b200:1 \
  --cpus-per-task=64 --mem=192G --time=02:00:00 \
  --job-name=fp052-b200-smoke \
  --output=\"\$HOME/FLOWPanel-052-b200/data/fm052_multiarch/b200/slurm/fp052-b200-smoke-%j.out\" \
  --error=\"\$HOME/FLOWPanel-052-b200/data/fm052_multiarch/b200/slurm/fp052-b200-smoke-%j.out\" \
  --export=ALL,FP052_ARCH=b200,FP052_STAGE=smoke,FP052_GPU_GRES=b200,FP052_PARTITION=cs3,FP052_PROBE_JOB=$PROBE_JOB \
  scripts/fm052_arch_run.sh'"
