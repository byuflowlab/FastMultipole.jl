#!/usr/bin/env bash
# Submits the combined protected H200 arm (stages a b c + inline mature gate +
# stage d) as ONE job on qos=eng (user directive 2026-08-25: shorter H200 line;
# combine stages to pay the queue wait once).
set -euo pipefail

ssh orc 'bash -lc '\''cd "$HOME/FLOWVPM-046" && sbatch --parsable --qos=eng \
  --export=ALL scripts/fm052_chain_run.sh'\'''
