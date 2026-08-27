#!/usr/bin/env bash
# Ryan-only submission wrapper. Run only after inspecting a passing H100 probe.
set -euo pipefail

PROBE_JOB=${1:?usage: bash fm052_h100_smoke_submit.sh PROBE_JOB}
[[ "$PROBE_JOB" =~ ^[0-9]+$ ]] || { echo "PROBE_JOB must be numeric" >&2; exit 64; }

ssh orc "bash -lc 'cd \"\$HOME/FLOWVPM-052-h100\" && sbatch --parsable --nodes=1 \
  --partition=cs2 --qos=standby --no-requeue --gpus=h100:1 \
  --cpus-per-task=64 --mem=192G --time=02:00:00 \
  --job-name=fp052-h100-smoke \
  --output=\"\$HOME/FLOWPanel-052-h100/data/fm052_multiarch/h100/slurm/fp052-h100-smoke-%j.out\" \
  --error=\"\$HOME/FLOWPanel-052-h100/data/fm052_multiarch/h100/slurm/fp052-h100-smoke-%j.out\" \
  --export=ALL,FP052_ARCH=h100,FP052_STAGE=smoke,FP052_GPU_GRES=h100,FP052_PARTITION=cs2,FP052_PROBE_JOB=$PROBE_JOB \
  scripts/fm052_arch_run.sh'"
