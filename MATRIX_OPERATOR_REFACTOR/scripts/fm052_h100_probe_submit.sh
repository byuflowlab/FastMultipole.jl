#!/usr/bin/env bash
# Ryan-only submission wrapper. Agents must not execute this file.
set -euo pipefail

ssh orc 'bash -lc '\''cd "$HOME/FLOWVPM-052-h100" && sbatch --parsable --nodes=1 \
  --partition=cs2 --qos=standby --no-requeue --gpus=h100:1 \
  --cpus-per-task=64 --mem=192G --time=02:00:00 \
  --job-name=fp052-h100-probe \
  --output="$HOME/FLOWPanel-052-h100/data/fm052_multiarch/h100/slurm/fp052-h100-probe-%j.out" \
  --error="$HOME/FLOWPanel-052-h100/data/fm052_multiarch/h100/slurm/fp052-h100-probe-%j.out" \
  --export=ALL,FP052_ARCH=h100,FP052_STAGE=probe,FP052_GPU_GRES=h100,FP052_PARTITION=cs2 \
  scripts/fm052_arch_run.sh'\'''
