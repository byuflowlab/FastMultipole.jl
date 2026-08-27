#!/usr/bin/env bash
# Ryan-only supplemental H200 probe wrapper. It does not replace or modify the protected canonical chain.
set -euo pipefail

ssh orc 'bash -lc '\''cd "$HOME/FLOWVPM-052-h200" && sbatch --parsable --nodes=1 \
  --partition=eng --qos=eng --no-requeue --gpus=h200:1 \
  --cpus-per-task=64 --mem=192G --time=01:00:00 \
  --job-name=fp052-h200-probe \
  --output="$HOME/FLOWPanel-052-h200/data/fm052_multiarch/h200/slurm/fp052-h200-probe-%j.out" \
  --error="$HOME/FLOWPanel-052-h200/data/fm052_multiarch/h200/slurm/fp052-h200-probe-%j.out" \
  --export=ALL,FP052_ARCH=h200,FP052_STAGE=probe,FP052_GPU_GRES=h200,FP052_PARTITION=eng \
  scripts/fm052_arch_run.sh'\'''
