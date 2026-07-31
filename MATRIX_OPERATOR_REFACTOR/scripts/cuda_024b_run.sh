#!/bin/bash
#SBATCH --job-name=fm024bh200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -euo pipefail
module load cuda julia

WORKDIR="${FM024B_DIR:-$HOME/FastMultipole-024b}"
ENVDIR="${FM024B_ENV:-$HOME/fm024env}"
GPU_DEPOT="${FM024B_GPU_DEPOT:-$WORKDIR/.julia_gpu}"
OUTDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling"
REFDIR="$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references"
cd "$WORKDIR"
mkdir -p "$OUTDIR" "$GPU_DEPOT"
export JULIA_DEPOT_PATH="$GPU_DEPOT:${JULIA_DEPOT_PATH:-$HOME/.julia}"
out="$OUTDIR/gpu_$(hostname)_${SLURM_JOB_ID}.csv"
failures="$OUTDIR/gpu_failures_$(hostname)_${SLURM_JOB_ID}.csv"
failed_cases=0

echo "=== task 024b H200 environment"
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK job=$SLURM_JOB_ID"
nvidia-smi
julia --version
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()'

for n in 1000 3162 10000 31623 100000 316228 1000000; do
  test -s "$REFDIR/direct_reference_n${n}.csv"
done
(cd "$REFDIR" && shasum -a 256 -c direct_reference_checksums.sha256)

ells_for_n () {
  case "$1" in
    1000|3162) echo "2 3 4" ;;
    10000|31623) echo "3 4 5" ;;
    100000|316228) echo "4 5 6" ;;
    1000000) echo "4 5 6 7" ;;
    *) echo "unsupported n=$1" >&2; return 1 ;;
  esac
}

# Timing rows and failure-ledger rows share the "gpu,<tf>,<n>,<ell>," prefix, so
# the timing files must be selected explicitly rather than by a gpu_*.csv glob.
timing_csvs () {
  find "$OUTDIR" -maxdepth 1 -name 'gpu_*.csv' ! -name 'gpu_failures_*.csv'
}

case_done () {
  local n=$1 ell=$2 tf=$3 f
  while read -r f; do
    [[ -n "$f" ]] || continue
    grep -q "^gpu,${tf},${n},${ell}," "$f" && return 0
  done < <(timing_csvs)
  return 1
}

# A case that exhausted device memory (directly, or through the prescribed
# precomputed-y fallback) is a hardware-capacity result and belongs in the
# published ledger; anything else is an unexplained failure and must not be
# silently recorded as one.
classify_failure () {
  local log=$1
  if grep -qiE 'out of (gpu )?memory|OutOfMemoryError|exceeds the free-memory budget|max_persistent_bytes|estimated device peak' "$log"; then
    echo "failed_after_fallback"
  else
    echo "failed_other"
  fi
}

for n in 1000 3162 10000 31623 100000 316228 1000000; do
  for ell in $(ells_for_n "$n"); do
    for tf in Float64 Float32; do
      if case_done "$n" "$ell" "$tf"; then
        echo "=== skip completed GPU tf=$tf n=$n ell=$ell"
        continue
      fi
      echo "=== GPU tf=$tf n=$n ell=$ell"
      caselog="$(mktemp)"
      if ! FM024B_N="$n" FM024B_ELL="$ell" FM024B_TF="$tf" FM024B_OUT="$out" \
          FM024B_REFERENCE_DIR="$REFDIR" \
          julia --project="$ENVDIR" \
            MATRIX_OPERATOR_REFACTOR/scripts/benchmark_024b_gpu.jl \
            2>&1 | tee "$caselog"; then
        status="$(classify_failure "$caselog")"
        if [[ ! -s "$failures" ]]; then
          echo "mode,tf,n,ell,status,job_id" > "$failures"
        fi
        echo "gpu,$tf,$n,$ell,$status,$SLURM_JOB_ID" >> "$failures"
        failed_cases=$((failed_cases + 1))
      fi
      rm -f "$caselog"
    done
  done
done

if (( failed_cases > 0 )); then
  echo "024b GPU completed feasible cases with $failed_cases unresolved failures" >&2
  exit 1
fi
