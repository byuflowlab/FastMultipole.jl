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
    echo "=== preflight 1b: FastMultipole CUDA lifecycle tests (scalar B2M parity) ==="
    julia --project="$ENVDIR" test/cuda_radix_lifecycle_test.jl
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

if [ "${FM035_REFCHECK:-1}" == "1" ]; then
    # 033-checksummed-reference gate at the SHIPPED coupling defaults
    # (cuda_034_refcheck.jl: cube+wake x n=1e4+1e5, Float64 gated at 1e-3,
    # Float32 reported, 023 counters asserted per solve)
    echo "=== 033 refcheck at shipped coupling defaults ==="
    cd "$WORKDIR"
    julia --project="$ENVDIR" scripts/cuda_034_refcheck.jl "$FMDIR" 10000 100000
fi

if [ "${FM035_NOREG:-0}" == "1" ]; then
    # scalar 028/030 no-regression (B2M is shared with the scalar path):
    # the unchanged 028 harness at the verdict config, both precisions
    echo "=== scalar no-regression (028/030 verdict config) ==="
    cd "$FMDIR"
    REFDIR="$FMDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references"
    COMMON=(FM028_P=3 FM028_STRAT=dense FM028_LH=0 FM028_K=full
            FM028_REPS=9 FM028_STEPS=0 FM028_STALE=0
            FM028_BOUND=ab FM028_SYMMETRIC=0
            FM028_M2L_THREADS=64 FM028_M2L_BLOCK_CAP=65536 FM028_COUNTING_SORT=1
            FM028_REFDIR="$REFDIR")
    for cfg in "Float32 fp16" "Float64 off"; do
        read -r tf fmt <<< "$cfg"
        out="$DATADIR/fm035_nr_sched6_5_5_5_${fmt}_n1000000_${SLURM_JOB_ID:-manual}.csv"
        echo "=== no-regression case tf=$tf fmt=$fmt"
        env "${COMMON[@]}" \
            FM028_N=1000000 FM028_ELL=5 FM028_POLICY=sched6-5-5-5 \
            FM028_TF="$tf" FM028_TENSOR_FORMAT="$fmt" FM028_OUT="$out" \
            julia --project="$ENVDIR" \
            MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl
    done
fi

if [ "${FM035_NSYS:-0}" == "1" ]; then
    # Cycle 3D profiling rider (counter-free): Nsight Systems timeline of five
    # production U/J solves per case at n=1e6 (graph capture + overlap ON),
    # plus exact nearfield body-pair counts for the analytic roofline. nsys
    # needs no GPU performance-counter privilege (NCU remains blocked by
    # ERR_NVGPUCTRPERM, jobs 13157746). Skips gracefully if nsys is absent.
    echo "=== 035 nsys production-timeline rider ==="
    cd "$WORKDIR"
    export FM035_FMDIR="$FMDIR"
    if command -v nsys >/dev/null 2>&1; then
        for cfg in "cube Float32" "wake Float32" "cube Float64" "wake Float64"; do
            read -r ncase ntf <<< "$cfg"
            rep="$DATADIR/fm035_nsys_${ncase}_${ntf}_${SLURM_JOB_ID:-manual}"
            echo "=== nsys case=$ncase tf=$ntf"
            nsys profile -o "$rep" --force-overwrite=true \
                --trace=cuda --sample=none --cpuctxsw=none \
                --capture-range=cudaProfilerApi --capture-range-end=stop \
                julia --project="$ENVDIR" \
                "$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/profile_035_nsys.jl" \
                "$ncase" "$ntf"
            nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace \
                --format csv -o "$rep" "$rep.nsys-rep" || true
        done
    else
        echo "nsys not found on PATH; rider skipped (record in the work log)"
    fi
fi

echo "=== 035 tuning sweep ==="
cd "$WORKDIR"
export FM035_FMDIR="$FMDIR"
export FM035_CASES="$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/${FM035_CASEFILE:-fm035_cases_initial.txt}"
# fixed filename so completed-label resume works across jobs (job/host are
# provenance columns inside the CSV); FM035_OUTNAME switches campaign files
export FM035_OUT="$DATADIR/${FM035_OUTNAME:-fm035_sweep.csv}"
julia --project="$ENVDIR" "$FMDIR/MATRIX_OPERATOR_REFACTOR/scripts/benchmark_035_gpu.jl"

echo "fm035 job complete"
