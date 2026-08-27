#!/usr/bin/env bash
# Submits the combined B200 probe->smoke->mature chain as ONE job (user
# directive 2026-08-25: combine stages to pay the queue wait once). Optional
# arm: cancel if still pending after ~1 day.
set -euo pipefail

ssh orc 'bash -lc '\''cd "$HOME/FLOWVPM-052-b200" && sbatch --parsable --nodes=1 \
  --partition=cs3 --qos=standby --no-requeue --gpus=b200:1 \
  --cpus-per-task=64 --mem=192G --time=03:00:00 \
  --job-name=fp052-b200-chain \
  --output="$HOME/FLOWPanel-052-b200/data/fm052_multiarch/b200/slurm/fp052-b200-chain-%j.out" \
  --error="$HOME/FLOWPanel-052-b200/data/fm052_multiarch/b200/slurm/fp052-b200-chain-%j.out" \
  --export=ALL,FP052_ARCH=b200,FP052_GPU_GRES=b200,FP052_PARTITION=cs3 \
  scripts/fm052_arch_chain.sh'\'''
