#!/usr/bin/env bash
# Ryan-only L40S probe wrapper. No later L40S stage is provided unless this probe passes every unchanged memory gate.
set -euo pipefail

ssh orc 'bash -lc '\''cd "$HOME/FLOWVPM-052-l40s" && sbatch --parsable --nodes=1 \
  --partition=m13l --no-requeue --gpus=l40s:1 \
  --cpus-per-task=64 --mem=192G --time=01:00:00 \
  --job-name=fp052-l40s-probe \
  --output="$HOME/FLOWPanel-052-l40s/data/fm052_multiarch/l40s/slurm/fp052-l40s-probe-%j.out" \
  --error="$HOME/FLOWPanel-052-l40s/data/fm052_multiarch/l40s/slurm/fp052-l40s-probe-%j.out" \
  --export=ALL,FP052_ARCH=l40s,FP052_STAGE=probe,FP052_GPU_GRES=l40s,FP052_PARTITION=m13l \
  scripts/fm052_arch_run.sh'\'''
