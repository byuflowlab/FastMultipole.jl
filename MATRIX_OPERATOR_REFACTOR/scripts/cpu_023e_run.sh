#!/bin/bash
#SBATCH --job-name=fm023ecpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
source /etc/profile
set -uo pipefail
module load julia

cd "$HOME/FastMultipole-023"
ENVDIR="$HOME/fm023env"
OUTDIR="MATRIX_OPERATOR_REFACTOR/data/dense_translation_m2l_host"
STAMP=$(date +%Y%m%d-%H%M%S)
LABEL=${FM023E_LABEL:-functional_baseline}

echo "=== task 023e environment"
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK job=$SLURM_JOB_ID label=$LABEL"
uname -a
lscpu | grep -E "Model name|Socket|Core|Thread|CPU\(s\)" || true
julia --version
echo "git_commit=${FM023E_GIT_COMMIT:-unknown}"
echo "git_tree=${FM023E_GIT_TREE:-unknown}"
echo "worktree=${FM023E_GIT_WORKTREE:-unknown}"

echo "=== focused 023e tests"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=test test/dense_translation_m2l_test.jl || exit 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=test test/resident_m2m_gemm_test.jl || exit 1

mkdir -p "$OUTDIR"
status=0
for bt in ${FM023E_BLAS_THREAD_LIST:-1 $SLURM_CPUS_PER_TASK}; do
  for tf in ${FM023E_TFS:-Float32 Float64}; do
    for lh in ${FM023E_LHS:-false true}; do
      for p in ${FM023E_PS:-4 8 12}; do
        for n in ${FM023E_NS:-150 2000 20000}; do
          for variant in ${FM023E_VARIANTS:-dense materialized_concat factored precomputed_y}; do
            out="$OUTDIR/${LABEL}_host_$(hostname)_blas${bt}_${tf}_lh${lh}_p${p}_n${n}_${variant}_$STAMP.csv"
            echo "=== $out"
            OPENBLAS_NUM_THREADS=$bt OMP_NUM_THREADS=$bt \
              FM023E_BLAS_THREADS=$bt FM023E_TF=$tf FM023E_LH=$lh \
              FM023E_P=$p FM023E_N=$n FM023E_VARIANT=$variant FM023E_LABEL=$LABEL \
              FM023E_APPLY_CHUNK=${FM023E_APPLY_CHUNK:-0} \
              FM023E_BUILD_CHUNK=${FM023E_BUILD_CHUNK:-0} \
              FM023E_MAX_PERSISTENT_BYTES=${FM023E_MAX_PERSISTENT_BYTES:-4294967296} \
              FM023E_OUT="$out" julia --project="$ENVDIR" \
              MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023e_dense_m2l_host.jl || status=1
          done
        done
      done
    done
  done
done
exit $status
