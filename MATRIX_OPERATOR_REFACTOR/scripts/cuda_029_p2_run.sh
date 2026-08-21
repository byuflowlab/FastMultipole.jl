#!/bin/bash
#SBATCH --job-name=fm029p2
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:30:00
#SBATCH --output=%x-%j.out
# Task 029 prototype P2: 2-GPU octant-decomposition feasibility slice on one
# m13h node (2x H200, NVLink). Sections:
#   1. environment + 2-GPU topology record (nvidia-smi topo -m)
#   2. preflight: cuda_radix_graph_test (single-GPU machinery this prototype
#      leans on) + the NEW test/cuda_radix_twogpu_test.jl distributed gates
#   3. 024b reference integrity gate (n=1e6)
#   4. prototype_029_p2_twogpu.jl — the measured 2-GPU verdict
set -o pipefail
source /etc/profile
# julia pinned: 1.12.6 segfaults host LLVM JIT on the device step (job 13058336)
module load cuda julia/1.11.7-6bmogfl

echo "=== node: $(hostname)"
nvidia-smi -L
echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "=== nvidia-smi power/clock record"
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE | grep -E \
  "Power Limit|Persistence|Performance State|Clocks Event|SM *:|Memory *:|Applications" | head -60
echo "=== nvidia-smi topology (2 GPUs)"
nvidia-smi topo -m

WORKDIR="${FM029_DIR:-$HOME/FastMultipole-023}"
ENVDIR="${FM029_ENV:-$HOME/fm023env}"
OUTDIR="${FM029_OUTDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms}"
REFDIR="${FM029_REFDIR:-$WORKDIR/MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/references}"
cd "$WORKDIR"
mkdir -p "$OUTDIR"

echo "=== source manifest"
julia --project="$ENVDIR" -e '
    using SHA
    files = sort(filter(f -> endswith(f, ".jl"), readdir("src")))
    ctx = SHA.SHA256_CTX()
    for f in files
        SHA.update!(ctx, codeunits(f)); SHA.update!(ctx, read(joinpath("src", f)))
    end
    println("SOURCE_MANIFEST=", bytes2hex(SHA.digest!(ctx))[1:16])'

echo "=== CUDA.jl versioninfo"
julia --project="$ENVDIR" -e 'using CUDA; CUDA.versioninfo()' || { echo "CUDA_JL_LOAD_FAIL"; exit 1; }

preflight_fail=0
echo "=== test/cuda_radix_graph_test.jl (single-GPU graph machinery)"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia --project="$ENVDIR" test/cuda_radix_graph_test.jl
s=$?; echo "GRAPH_TEST_EXIT=$s"; (( s )) && preflight_fail=1

echo "=== test/cuda_radix_twogpu_test.jl (P2 distributed gates)"
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 FASTMULTIPOLE_REQUIRE_TWOGPU_TESTS=1 \
  timeout 2400 julia -t 6 --project="$ENVDIR" test/cuda_radix_twogpu_test.jl
s=$?; echo "TWOGPU_TEST_EXIT=$s"; (( s )) && preflight_fail=1

if (( preflight_fail )); then
    echo "PREFLIGHT_EXIT=1"
    exit 1
fi
echo "PREFLIGHT_EXIT=0"

echo "=== 024b reference integrity (n=1e6)"
if [ ! -s "$REFDIR/direct_reference_n1000000.csv" ]; then
    echo "MISSING_REFERENCE n=1000000 in $REFDIR" >&2
    echo "REFERENCE_GATE_EXIT=1"
    exit 1
fi
(cd "$REFDIR" && shasum -a 256 -c direct_reference_checksums.sha256) || {
    echo "REFERENCE_GATE_EXIT=1"; exit 1; }
echo "REFERENCE_GATE_EXIT=0"

echo "=== P2 two-GPU verdict measurement"
FM029P2_N=1000000 FM029P2_REPS="${FM029P2_REPS:-25}" FM029P2_STEPS=5 \
FM029P2_DT=1e-5 FM029P2_ELL=5 FM029P2_P=3 FM029P2_POLICY=sched6-5-4-4 \
FM029P2_TF=Float32 FM029P2_TENSOR=fp16 FM029P2_REFDIR="$REFDIR" \
FM029P2_OUT="$OUTDIR/cuda029p2_sched6_5_4_4_fp16_n1000000_$(hostname)_${SLURM_JOB_ID:-manual}.csv" \
  timeout 3600 julia -t 6 --project="$ENVDIR" MATRIX_OPERATOR_REFACTOR/scripts/prototype_029_p2_twogpu.jl
s=$?
echo "P2_BENCH_EXIT=$s"
if (( s )); then
  echo "P2_EXIT=1"
  exit 1
fi
echo "P2_EXIT=0"
