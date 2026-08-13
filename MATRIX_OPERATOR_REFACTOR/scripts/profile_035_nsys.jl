# Task 035 cycle 3D profiling rider: production-timeline U/J solves under
# Nsight Systems (timeline tracing needs no GPU performance-counter privilege,
# unlike Nsight Compute — job 13157746 was blocked by ERR_NVGPUCTRPERM).
# Graph capture and nearfield/far-field overlap stay ON: this records the
# production overlap structure, not isolated launches.
#
# Also prints the exact nearfield body-pair total (sum over direct routes of
# |target cell| x |source cell|) for the counter-free analytic roofline:
# achieved GFLOP/s and GB/s are computed offline from these counts and the
# CUDA-event nearfield stage medians of the same job's sweep rows.
#
# Usage: julia ... profile_035_nsys.jl CASE TF   (shipped coupling defaults)

import CUDA
CUDA.functional() || error("CUDA is not functional")
case = ARGS[1]
TF = ARGS[2] == "Float32" ? Float32 : ARGS[2] == "Float64" ? Float64 :
    error("precision must be Float32 or Float64")
const N = 1_000_000

fmdir = abspath(get(ENV, "FM035_FMDIR", get(ENV, "HOME", "") * "/FastMultipole-034"))
include(joinpath(fmdir, "MATRIX_OPERATOR_REFACTOR", "scripts", "benchmark_033_common.jl"))
const FM = vpm.fmm

cpu = fm033_build(case, N)
gpu = vpm.ParticleField(N, Float64;
    formulation=vpm.rVPM, kernel=vpm.gaussianerf, viscous=vpm.Inviscid(),
    SFS=vpm.noSFS, transposed=true, integration=vpm.rungekutta3,
    UJ=vpm.UJ_fmm, fmm=fm033_settings(), arraytype=CUDA.CuArray)
gpu.np = N
gpu.particles .= CUDA.CuArray{Float64}(Array(cpu.particles)[:, 1:N])
# Shipped cycle-3D coupling defaults; only precision is pinned per run.
vpm.radix_fmm_settings!(gpu; precision=TF)

# Build + JIT outside the profiler range.
vpm.UJ_fmm(gpu); vpm.UJ_fmm(gpu); CUDA.synchronize()
st = vpm._radix_fmm_couplings[gpu]
state = st.cache.state
counts = state.counts

# Exact nearfield body-pair total from the (construction-only) direct route
# list and per-cell body ranges. cell_ranges is column-per-cell
# (first_body, count) and both direct route arrays hold cell indices — see
# _host_direct_pairs_functor_kernel!. Falls back to route count only if the
# index sanity check fails.
cr = Array(state.grid.cell_ranges)
sizes = Int64.(cr[2, :])
dt = Array(state.direct_targets)[1:counts.n_direct]
ds = Array(state.direct_sources)[1:counts.n_direct]
pair_total = try
    ok = all(1 .<= dt .<= length(sizes)) && all(1 .<= ds .<= length(sizes))
    ok || error("direct route indices outside cell range " *
        "(dt in $(extrema(dt)), ds in $(extrema(ds)), n_cells=$(length(sizes)))")
    sum(sizes[t] * sizes[s] for (t, s) in zip(dt, ds))
catch e
    @warn "pair-total computation failed; roofline must fall back to route stats" e
    -1
end
println("NSYS_COUNTS case=$case n=$N tf=$TF n_cells=$(counts.n_cells) " *
    "n_nodes=$(counts.n_nodes) n_routes=$(counts.n_routes) " *
    "n_direct_routes=$(counts.n_direct) body_pairs=$pair_total " *
    "min_cell=$(minimum(sizes)) max_cell=$(maximum(sizes)) " *
    "mean_cell=$(round(sum(sizes) / length(sizes); digits=2))")
flush(stdout)

# Profiled region: five production U/J solves (graph capture + overlap ON).
CUDA.Profile.start()
for _ in 1:5
    vpm.UJ_fmm(gpu)
end
CUDA.synchronize()
CUDA.Profile.stop()
println("NSYS_DONE")
