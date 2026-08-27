#!/usr/bin/env bash
# Submits the combined H100 probe->smoke->mature chain as ONE job (user
# directive 2026-08-25: combine stages to pay the queue wait once). Internal
# manifest gating in fm052_arch_chain.sh replaces the between-job inspections.
set -euo pipefail

ssh orc 'bash -lc '\''cd "$HOME/FLOWVPM-052-h100" && sbatch --parsable --nodes=1 \
  --partition=cs2 --qos=standby --no-requeue --gpus=h100:1 \
  --cpus-per-task=64 --mem=192G --time=03:00:00 \
  --job-name=fp052-h100-chain \
  --output="$HOME/FLOWPanel-052-h100/data/fm052_multiarch/h100/slurm/fp052-h100-chain-%j.out" \
  --error="$HOME/FLOWPanel-052-h100/data/fm052_multiarch/h100/slurm/fp052-h100-chain-%j.out" \
  --export=ALL,FP052_ARCH=h100,FP052_GPU_GRES=h100,FP052_PARTITION=cs2 \
  scripts/fm052_arch_chain.sh'\'''
