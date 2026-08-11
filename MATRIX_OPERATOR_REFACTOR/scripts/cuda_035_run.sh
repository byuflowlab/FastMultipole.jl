#!/bin/bash
#SBATCH --job-name=fm035
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Task 035 H200 tuning-sweep job. Pattern: FLOWVPM scripts/cuda_034_run.sh.
#
#   1. preflight: FastMultipole CUDA device-interface tests (032 surface);
#   2. preflight: FastMultipole host device-system interface tests — includes
#      the 032a `shipped nearfield defaults` assertions (first on-hardware run
#      of the shipped-default tests, per the 032a approval note);
#   3. preflight: FastMultipole CUDA nearfield-binning tests — asserts the
#      shipped CUDA stream-mechanism Ref defaults on hardware;
#   4. preflight: FLOWVPM radix coupling tests (Part A host incl. the new
#      035 tuning-settings testset; Part B device-resident);
#   5. sha256-verify the synced 033 sampled-direct references;
#   6. the pre-registered 035 tuning sweep (benchmark_035_gpu.jl).
#
# FM035_PREFLIGHT=0 skips stages 1-4 for iteration. FM035_CASEFILE selects the
# config file (relative to MATRIX_OPERATOR_REFACTOR/scripts).
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
mkdir -p "$DATADIR"

export FASTMULTIPOLE_FORCE_CUDA_LOAD=1
export FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1
export JULIA_NUM_THREADS=8

if [ "${FM035_PREFLIGHT:-1}" == "1" ]; then
    cd "$FMDIR"
    echo "=== preflight 1: FastMultipole CUDA interface tests (032 surface) ==="
    julia --project="$ENVDIR" test/cuda_radix_interface_test.jl
    echo "=== preflight 2: host device-system interface tests (032a shipped defaults) ==="
    julia --project="$ENVDIR" test/device_system_interface_test.jl
    echo "=== preflight 3: CUDA nearfield-binning tests (032a mechanism defaults) ==="
    julia --project="$ENVDIR" test/cuda_radix_nearfield_binning_test.jl

    cd "$WORKDIR"
    echo "=== preflight 4: FLOWVPM radix coupling tests (Part A + Part B) ==="
    julia --project="$ENVDIR" test/runtests_gpu_fmm.jl
fi

echo "=== 033 reference integrity ==="
cd "$FMDIR/MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/references"
sha256sum -c direct_reference_checksums.sha256

echo "=== 035 tuning sweep ==="
cd "$WORKDIR"
export FM035_FMDIR="$FMDIR"
export FM035_CASES="$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/${FM035_CASEFILE:-fm035_cases_initial.txt}"
# fixed filename so completed-label resume works across jobs (job/host are
# provenance columns inside the CSV)
export FM035_OUT="$DATADIR/fm035_sweep.csv"
julia --project="$ENVDIR" "$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/benchmark_035_gpu.jl"

echo "fm035 job complete"
