#!/bin/bash
# Local driver for the session-41 harmonic-sweep H200 gate: sync this
# FastMultipole working tree to orc, refresh the two envs, submit
# scripts/ka_harmonic_sweep_cuda_run.sh.
#
#   bash scripts/ka_harmonic_sweep_cuda_submit.sh   # from the FastMultipole repo root
#
# Requires a live ssh master session to `orc`. Pattern: FLOWVPM's
# scripts/ka_cuda_bench_submit.sh.
#
# Two envs on purpose: fm_kabench_env (FastMultipole + CUDA + KA, drives the KA
# suites) and fm_cudatest_env (test/cuda/Project.toml's deps, drives the CUDA
# regression). Pkg.resolve() runs on both because rsyncing src/ext/test never
# updates an env manifest's cached dep list, which is what breaks extension
# precompilation after a dep changes.
set -euo pipefail
REMOTE=orc
FMDIR=FastMultipole-kabench
ENVDIR='$HOME/fm_kabench_env'
CUDAENV='$HOME/fm_cudatest_env'

ssh "$REMOTE" "mkdir -p $FMDIR"

rsync -az --delete --exclude .git \
    src ext test scripts Project.toml \
    "$REMOTE:$FMDIR/"

# test/cuda_radix_interface_test.jl includes ../MATRIX_OPERATOR_REFACTOR/scripts/
# fm028_device_system.jl. Its own rsync call, not another source in the list
# above: without -R rsync strips the leading path and it lands in scripts/.
ssh "$REMOTE" "mkdir -p $FMDIR/MATRIX_OPERATOR_REFACTOR"
rsync -az --delete MATRIX_OPERATOR_REFACTOR/scripts \
    "$REMOTE:$FMDIR/MATRIX_OPERATOR_REFACTOR/"

# compute nodes have no internet: pin the CUDA JLLs to the system toolkit
ssh "$REMOTE" "mkdir -p fm_kabench_env fm_cudatest_env
for d in fm_kabench_env fm_cudatest_env; do cat > \$d/LocalPreferences.toml <<PREFS
[CUDA_Compiler_jll]
local = \"true\"

[CUDA_Driver_jll]
local = \"true\"

[CUDA_Runtime_jll]
local = \"true\"
PREFS
done"

ssh "$REMOTE" "bash -lc 'module load julia/1.11.7-6bmogfl \
  && export JULIA_PKG_PRECOMPILE_AUTO=0 \
  && julia --project=$ENVDIR -e \"using Pkg; Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); Pkg.add([\\\"CUDA\\\",\\\"KernelAbstractions\\\",\\\"GPUArraysCore\\\",\\\"StaticArrays\\\",\\\"Random\\\"]); Pkg.resolve(); Pkg.instantiate()\" \
  && julia --project=$CUDAENV -e \"using Pkg; Pkg.develop(path=\\\"\$HOME/$FMDIR\\\"); Pkg.add([\\\"CUDA\\\",\\\"Test\\\"]); Pkg.resolve(); Pkg.instantiate()\" \
  && cd $FMDIR \
  && sbatch --qos=eng scripts/ka_harmonic_sweep_cuda_run.sh'"

echo "Submitted. Poll with:  ssh orc 'bash -lc \"squeue -u \\\$USER -o \\\"%.10i %.9T %.8q %.14R\\\"\"'"
echo "Results: ~/$FMDIR/ka_harmsweep_cuda_<jobid>.{log,provenance} and fmm_ka_harmsweep-<jobid>.out on orc."
