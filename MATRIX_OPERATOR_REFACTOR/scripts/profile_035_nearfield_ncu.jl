# Task 035: one profiler-delimited P5 nearfield launch for Nsight Compute.
# Usage: julia ... profile_035_nearfield_ncu.jl CASE TF

import CUDA
CUDA.functional() || error("CUDA is not functional")
case = ARGS[1]
TF = ARGS[2] == "Float32" ? Float32 : ARGS[2] == "Float64" ? Float64 :
    error("precision must be Float32 or Float64")
const N = 1_000_000
const ORDER = 4                 # literature P=5
const RHO_T = 3.668

fmdir = abspath(get(ENV, "FM035_FMDIR", get(ENV, "HOME", "") * "/FastMultipole-034"))
include(joinpath(fmdir, "MATRIX_OPERATOR_REFACTOR", "scripts", "benchmark_033_common.jl"))
const FM = vpm.fmm

ell, q = case == "cube" ? (5, 12) : case == "wake" ? (6, 6) :
    error("case must be cube or wake")
cpu = fm033_build(case, N)
gpu = vpm.ParticleField(N, Float64;
    formulation=vpm.rVPM, kernel=vpm.gaussianerf, viscous=vpm.Inviscid(),
    SFS=vpm.noSFS, transposed=true, integration=vpm.rungekutta3,
    UJ=vpm.UJ_fmm, fmm=fm033_settings(), arraytype=CUDA.CuArray)
gpu.np = N
gpu.particles .= CUDA.CuArray{Float64}(Array(cpu.particles)[:, 1:N])
vpm.radix_fmm_settings!(gpu; expansion_order=ORDER, ell,
    near_radius2=q, precision=TF, direct_kernel=:partitioned, rho_t=RHO_T,
    m2l_strategy=:dense)

# Build, JIT, and refresh outside the profiler API range.
vpm.UJ_fmm(gpu); vpm.UJ_fmm(gpu); CUDA.synchronize()
st = vpm._radix_fmm_couplings[gpu]
FM.update_cuda_radix_state!(st.cache, (gpu,)); CUDA.synchronize()
FM.CUDA_GRAPH_LIFECYCLE[] = false
FM.CUDA_OVERLAP_NEARFIELD[] = false
println("NCU_READY case=$case n=$N tf=$TF literature_P=5 expansion_order=$ORDER " *
    "ell=$ell q=$q rho_t=$RHO_T direct_pairs=$(st.cache.state.counts.n_direct)")
flush(stdout)

CUDA.Profile.start()
FM._launch_cuda_nearfield_kernel!(st.cache.state)
CUDA.synchronize()
CUDA.Profile.stop()
println("NCU_DONE")
