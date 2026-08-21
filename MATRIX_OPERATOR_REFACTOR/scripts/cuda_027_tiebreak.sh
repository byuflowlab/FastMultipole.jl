#!/bin/bash
#SBATCH --job-name=fm027tb
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
# Task 027 Checkpoint D tiebreaker.
#
# Job 12993753 left exactly one cell unresolved: flat|concat|P=4|n=2000, whose
# new/old ratio was 0.799 in round 1 and 1.401 in round 2 while all 15 other cells
# stayed within +-1.4% in both. It is the fastest cell in the matrix (~0.3 ms) and
# therefore launch-latency dominated. Two rounds of 7 samples cannot adjudicate it.
#
# This job runs THAT CONFIGURATION ONLY, alternating old/new sources over many short
# rounds, so between-round environmental drift averages out instead of aliasing onto
# one side.
source /etc/profile
set -o pipefail
module load cuda julia
echo "=== node: $(hostname)"
nvidia-smi -L

WORKDIR="${FM027_DIR:-$HOME/FastMultipole-023}"
ENVDIR="$HOME/fm023env"
OLDSRC="${FM026_OLD_SRC:-$HOME/FastMultipole-026/staging_024b_reference/src}"
cd "$WORKDIR"

TB=MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_cuda/tiebreak_${SLURM_JOB_ID}
mkdir -p "$TB"

NEWSRC_BACKUP="$(mktemp -d)/src_new"
cp -r src "$NEWSRC_BACKUP"
restore_new() { rm -rf src && cp -r "$NEWSRC_BACKUP" src; }
trap restore_new EXIT

# 8 alternating rounds x 15 samples = 120 samples per side for the single cell.
for round in 1 2 3 4 5 6 7 8; do
    restore_new
    FM026G_OUT="$TB/new_r$round" FM026G_LABEL="new_r$round" FM026G_REPS=15 \
        FM026G_ONLY_STRATEGY=concat FM026G_ONLY_P=4 FM026G_ONLY_N=2000 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/gate_026_cases.jl

    rm -rf src && cp -r "$OLDSRC" src
    FM026G_OUT="$TB/old_r$round" FM026G_LABEL="old_r$round" FM026G_REPS=15 \
        FM026G_ONLY_STRATEGY=concat FM026G_ONLY_P=4 FM026G_ONLY_N=2000 \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/gate_026_cases.jl
done
restore_new
echo "TIEBREAK_EXIT=$?"
echo "TB_DIR=$TB"
