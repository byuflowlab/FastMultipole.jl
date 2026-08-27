#!/usr/bin/env bash
# Ryan-only optional B200 submission wrapper. Agents must not execute this file.
set -euo pipefail

ssh orc 'bash -lc '\''cd "$HOME/FLOWVPM-052-b200" && sbatch --parsable --nodes=1 \
  --partition=cs3 --qos=standby --no-requeue --gpus=b200:1 \
  --cpus-per-task=64 --mem=192G --time=02:00:00 \
  --job-name=fp052-b200-probe \
  --output="$HOME/FLOWPanel-052-b200/data/fm052_multiarch/b200/slurm/fp052-b200-probe-%j.out" \
  --error="$HOME/FLOWPanel-052-b200/data/fm052_multiarch/b200/slurm/fp052-b200-probe-%j.out" \
  --export=ALL,FP052_ARCH=b200,FP052_STAGE=probe,FP052_GPU_GRES=b200,FP052_PARTITION=cs3 \
  scripts/fm052_arch_run.sh'\'''
