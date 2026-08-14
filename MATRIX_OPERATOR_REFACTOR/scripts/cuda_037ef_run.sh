#!/bin/bash
#SBATCH --job-name=fm037ef
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Tasks 037e/037f pre-registered H200 screens. Pattern: cuda_035_run.sh
# (env recipe unchanged; 034 trees + fm034env).
#
# FM037EF_STAGE=e (default):
#   1-4. the cuda_035_run.sh preflight test stages (the binning and
#        device-interface files now carry the 037e/037f testsets, so this is
#        the first on-hardware run of both mechanisms' correctness gates);
#   5. sha256 reference integrity + 033 refcheck at shipped defaults;
#   6. 037e E0 full-n scoping census (host arrays, CPU only);
#   7. the 24-row fm037e screen -> fm037e_screen.csv.
# FM037EF_STAGE=f:
#   5. sha256 reference integrity;
#   8. the 48-row fm037f screen -> fm037f_screen.csv;
#   9. the 30-config fm037f error-decomposition oracle ->
#      fm037f_decomposition.csv (budget-note per-case pass criterion).
# no -u: /etc/profile.d scripts reference unset vars on the cluster
set -eo pipefail
source /etc/profile
# julia pinned to 1.11.7: 1.12.6 segfaults in host LLVM JIT (job 13058191)
module load cuda julia/1.11.7-6bmogfl
echo "=== node: $(hostname)"
nvidia-smi -L

WORKDIR="${FM035_VPMDIR:-$HOME/FLOWVPM-034}"
FMDIR="${FM035_FMDIR:-$HOME/FastMultipole-034}"
ENVDIR="${FM035_ENV:-$HOME/fm034env}"
DATADIR="$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign"
STAGE="${FM037EF_STAGE:-e}"
mkdir -p "$DATADIR"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
export JULIA_NUM_THREADS=8

if [ "$STAGE" == "e" ] && [ "${FM037EF_PREFLIGHT:-1}" == "1" ]; then
    cd "$FMDIR"
    echo "=== preflight 1: FastMultipole CUDA interface tests ==="
    julia --project="$ENVDIR" test/cuda_radix_interface_test.jl
    echo "=== preflight 1b: FastMultipole CUDA lifecycle tests ==="
    julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
    echo "=== preflight 2: host device-system interface tests (incl. 037f modes) ==="
    julia --project="$ENVDIR" test/device_system_interface_test.jl
    echo "=== preflight 3: CUDA nearfield-binning tests (incl. 037e + 037f CUDA testsets) ==="
    julia --project="$ENVDIR" test/cuda_radix_nearfield_binning_test.jl
    cd "$WORKDIR"
    echo "=== preflight 4: FLOWVPM radix coupling tests ==="
    julia --project="$ENVDIR" test/runtests_gpu_fmm.jl
fi

echo "=== 033 reference integrity ==="
cd "$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/references"
sha256sum -c direct_reference_checksums.sha256

if [ "$STAGE" == "e" ]; then
    if [ "${FM037EF_REFCHECK:-1}" == "1" ]; then
        echo "=== 033 refcheck at shipped coupling defaults ==="
        cd "$WORKDIR"
        julia --project="$ENVDIR" scripts/cuda_034_refcheck.jl "$FMDIR" 10000 100000
    fi

    echo "=== 037e E0 scoping census (full n, host) ==="
    cd "$FMDIR"
    FM037E_OUT="$DATADIR/fm037e_scoping_${SLURM_JOB_ID:-manual}.csv" \
        julia --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/fm037e_scoping.jl

    echo "=== 037e pair-AABB screen ==="
    cd "$WORKDIR"
    export FM035_FMDIR="$FMDIR"
    export FM035_CASES="$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/fm037e_cases_screen.txt"
    export FM035_OUT="$DATADIR/fm037e_screen.csv"
    julia --project="$ENVDIR" "$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/benchmark_035_gpu.jl"
else
    echo "=== 037f cheapened-g/h screen ==="
    cd "$WORKDIR"
    export FM035_FMDIR="$FMDIR"
    export FM035_CASES="$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/fm037f_cases_screen.txt"
    export FM035_OUT="$DATADIR/fm037f_screen.csv"
    julia --project="$ENVDIR" "$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/benchmark_035_gpu.jl"

    echo "=== 037f error-decomposition oracle ==="
    cd "$WORKDIR"
    FM035D_CONFIG_FILE="$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/fm037f_cutoff_configs.txt" \
    FM035D_OUT="$DATADIR/fm037f_decomposition.csv" \
        julia --project="$ENVDIR" \
        "$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/benchmark_035_error_decomposition.jl"
fi

echo "fm037ef stage $STAGE complete"
