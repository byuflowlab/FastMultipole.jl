# Task 028 cycle 4: residual kernel attribution at the current verdict point.
#
# This is intentionally benchmark-only.  Script-local racing-store/no-store
# kernels produce invalid numerical output and exist solely to price atomics and
# retained arithmetic at fixed work.  The production lifecycle and convection
# correctness gates run before this script in cuda_028_run.sh.

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using LinearAlgebra
using Dates
using Printf

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

include(joinpath(@__DIR__, "fm028_device_system.jl"))
const FM = FastMultipole

const N = parse(Int, get(ENV, "FM028_N", "1000000"))
const P = parse(Int, get(ENV, "FM028_P", "3"))
const ELL = parse(Int, get(ENV, "FM028_ELL", "5"))
const K = parse(Int, get(ENV, "FM028_K", "1740"))
const REPS = parse(Int, get(ENV, "FM028_REPS", "7"))
const SEED = 24025
const BOX_MIN = SVector(-0.01, -0.01, -0.01)
const BOX_SIZE = 1.02
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const OUTDIR = get(ENV, "FM028_OUTDIR", joinpath(@__DIR__, "..", "data",
    "feasibility_1m_10ms"))
const PREFIX = "attribution_$(gethostname())_$(STAMP)"
const JOBID = get(ENV, "SLURM_JOB_ID", "")

function median_gpu_ms(f, reps=REPS)
    f()
    CUDA.synchronize()
    samples = Float64[]
    for _ in 1:reps
        push!(samples, Float64(CUDA.@elapsed f()) * 1e3)
    end
    return median(samples), minimum(samples), maximum(samples)
end

function write_csv(path, rows)
    isempty(rows) && return
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join(string.(values(row)), ','))
        end
    end
end

# Same tiled computation as the production kernel, parameterized only by the
# terminal store. MODE=:atomic is numerically valid; :store and :nostore are
# timing discriminators. The Inf guard keeps the no-store arithmetic live.
function attr_tiled_kernel!(loc_phi, loc_chi, ops, route_class,
        route_sources, route_targets, mp_phi, mp_chi, phi_flat_idx, chi_flat_idx,
        ndof_phi, src_scale, tgt_scale, lcol, n_routes, ::Val{LH},
        ::Val{MODE}) where {LH,MODE}
    T = eltype(ops)
    D = size(ops, 1)
    tid = threadIdx().x
    nthreads = blockDim().x
    lane = Int((tid - Int32(1)) % Int32(32))
    w = Int((tid - Int32(1)) ÷ Int32(32))
    nwarps = Int(nthreads ÷ Int32(32))
    tile = CUDA.CuDynamicSharedArray(T, D * D)
    mp_buf = CUDA.CuDynamicSharedArray(T, (D, nwarps), D * D * sizeof(T))
    chunk = cld(n_routes, gridDim().x)
    j0 = (blockIdx().x - 1) * chunk + 1
    hi = min(j0 + chunk - 1, n_routes)
    @inbounds while j0 <= hi
        k = Int(route_class[j0])
        slo = j0
        shi = hi
        while slo < shi
            mid = (slo + shi + 1) >> 1
            if Int(route_class[mid]) == k
                slo = mid
            else
                shi = mid - 1
            end
        end
        je = slo
        idx = Int(tid)
        while idx <= D * D
            r = (idx - 1) % D + 1
            i = (idx - 1) ÷ D + 1
            tile[idx] = tgt_scale[r, lcol] * ops[r, i, k] * src_scale[i, lcol]
            idx += nthreads
        end
        CUDA.sync_threads()
        j = j0 + w
        while j <= je
            src_col = route_sources[j]
            tgt_col = route_targets[j]
            i = lane + 1
            while i <= D
                if i <= ndof_phi
                    mp_buf[i, w + 1] = mp_phi[phi_flat_idx[i], src_col]
                elseif LH
                    mp_buf[i, w + 1] = mp_chi[chi_flat_idx[i - ndof_phi], src_col]
                else
                    mp_buf[i, w + 1] = zero(T)
                end
                i += 32
            end
            CUDA.sync_warp()
            r = lane + 1
            while r <= D
                acc = zero(T)
                for i in 1:D
                    acc += tile[r + (i - 1) * D] * mp_buf[i, w + 1]
                end
                if r <= ndof_phi
                    if MODE === :atomic
                        CUDA.@atomic loc_phi[phi_flat_idx[r], tgt_col] += acc
                    elseif MODE === :store
                        loc_phi[phi_flat_idx[r], tgt_col] = acc
                    else
                        acc == T(Inf) && (loc_phi[phi_flat_idx[r], tgt_col] = acc)
                    end
                elseif LH
                    if MODE === :atomic
                        CUDA.@atomic loc_chi[chi_flat_idx[r - ndof_phi], tgt_col] += acc
                    elseif MODE === :store
                        loc_chi[chi_flat_idx[r - ndof_phi], tgt_col] = acc
                    else
                        acc == T(Inf) &&
                            (loc_chi[chi_flat_idx[r - ndof_phi], tgt_col] = acc)
                    end
                end
                r += 32
            end
            CUDA.sync_warp()
            j += nwarps
        end
        CUDA.sync_threads()
        j0 = je + 1
    end
    return nothing
end

# Same warp-per-pair arithmetic as production with a selectable terminal store.
function attr_direct_kernel!(output, source_bodies, cell_ranges, direct_targets,
        direct_sources, npairs, ::Val{MODE}) where MODE
    T = eltype(output)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    pair_i = (blockIdx().x - 1) * warps_per_block + warp_in_block + 1
    warp_stride = gridDim().x * warps_per_block
    c = inv(T(4) * T(π))
    @inbounds while pair_i <= npairs
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + lane
        while i <= tlast
            xi = source_bodies[1, i]
            yi = source_bodies[2, i]
            zi = source_bodies[3, i]
            u = zero(T)
            gx = zero(T)
            gy = zero(T)
            gz = zero(T)
            for j in sfirst:slast
                i == j && continue
                dx = xi - source_bodies[1, j]
                dy = yi - source_bodies[2, j]
                dz = zi - source_bodies[3, j]
                r2 = dx * dx + dy * dy + dz * dz
                if r2 > zero(r2)
                    invr = FM._cuda_fast_rsqrt(r2)
                    q = source_bodies[5, j] * c
                    u += q * invr
                    invr3 = invr * invr * invr
                    gx -= q * dx * invr3
                    gy -= q * dy * invr3
                    gz -= q * dz * invr3
                end
            end
            if MODE === :atomic
                CUDA.@atomic output[1, i] += u
                CUDA.@atomic output[2, i] += gx
                CUDA.@atomic output[3, i] += gy
                CUDA.@atomic output[4, i] += gz
            elseif MODE === :store
                output[1, i] = u
                output[2, i] = gx
                output[3, i] = gy
                output[4, i] = gz
            else
                u == T(Inf) && (output[1, i] = u + gx + gy + gz)
            end
            i += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

function build_target()
    bodies = fm028_body_matrix(SEED, N)
    sys = FM028DeviceSystem{Float32}(bodies)
    opts = CUDARadixLifecycleOptions(; precision=Float32,
        operator=MaterializedYRotationM2L(),
        m2l_strategy=DenseTranslationM2L())
    cache = RadixFMMCache(sys; expansion_order=P, ell=ELL, max_n_bodies=N,
        bounds=(BOX_MIN, BOX_SIZE), lamb_helmholtz=false, device=true,
        options=opts, near_radius2=12, window_classes=K)
    fmm!(sys, cache; scalar_potential=true, gradient=true)
    CUDA.synchronize()
    return sys, cache
end

mkpath(OUTDIR)
println("building target n=$N P=$P ell=$ELL q=12 K=$K Float32")
sys, cache = build_target()
state = cache.state
hctx = cache.device_ctx.hierarchical_ctx
ws = state.scratch
plan = hctx.apply_plan
plan isa FM.ResidentM2LDenseCUDAPlan || error("expected dense CUDA plan")

# Capture the whole leaf level in one window (K=1740 at q=12).
hctx.noffsets <= hctx.window_classes ||
    error("attribution requires one leaf window; noffsets=$(hctx.noffsets), K=$(hctx.window_classes)")
n_routes = FM._cuda_hier_generate_window!(state, hctx, plan.route_class, ELL,
    1, hctx.noffsets, 0)
FM._cuda_hier_refresh_dense_window!(plan, hctx, 1, hctx.noffsets, n_routes)
state.counts.n_routes = n_routes

common = (plan.operators, plan.route_class, state.route_sources,
    state.route_targets, state.multipoles.phi, state.multipoles.chi,
    ws.phi_flat_idx, ws.chi_flat_idx, plan.ndof_phi, hctx.source_scale,
    hctx.target_scale, ELL - 1, n_routes, Val(false))

m2l_rows = NamedTuple[]
for threads in (32, 64, 128, 256), cap in (512, 2048, 8192, 16384, 65536)
    nwarps = threads ÷ 32
    blocks = min(cld(n_routes, nwarps), cap)
    shmem = (plan.ndof * plan.ndof + nwarps * plan.ndof) * sizeof(Float32)
    launch() = CUDA.@cuda threads=threads blocks=blocks shmem=shmem attr_tiled_kernel!(
        state.locals.phi, state.locals.chi, common..., Val(:atomic))
    med, lo, hi = median_gpu_ms(launch)
    push!(m2l_rows, (; job=JOBID, kernel="m2l_tiled", mode="atomic",
        precision="Float32", n=N, ell=ELL, q=12, routes=n_routes,
        ndof=plan.ndof, threads, warps=threads ÷ 32, block_cap=cap,
        blocks, median_ms=med, min_ms=lo, max_ms=hi))
    @printf("M2L sweep threads=%3d cap=%5d blocks=%5d  %.3f ms\n",
        threads, cap, blocks, med)
end

for mode in (:atomic, :store, :nostore)
    threads = 128
    cap = 16384
    blocks = min(cld(n_routes, threads ÷ 32), cap)
    shmem = (plan.ndof * plan.ndof + (threads ÷ 32) * plan.ndof) * sizeof(Float32)
    launch() = CUDA.@cuda threads=threads blocks=blocks shmem=shmem attr_tiled_kernel!(
        state.locals.phi, state.locals.chi, common..., Val(mode))
    med, lo, hi = median_gpu_ms(launch)
    push!(m2l_rows, (; job=JOBID, kernel="m2l_tiled", mode=string(mode),
        precision="Float32", n=N, ell=ELL, q=12, routes=n_routes,
        ndof=plan.ndof, threads, warps=threads ÷ 32, block_cap=cap,
        blocks, median_ms=med, min_ms=lo, max_ms=hi))
    @printf("M2L A/B %-7s %.3f ms\n", mode, med)
end

npairs = state.counts.n_direct
direct_rows = NamedTuple[]
for threads in (64, 128, 256, 512), cap in (512, 2048, 8192, 16384, 65536)
    blocks = min(cld(npairs, threads ÷ 32), cap)
    launch() = CUDA.@cuda threads=threads blocks=blocks attr_direct_kernel!(
        state.output, state.source_bodies, state.cell_ranges,
        state.direct_targets, state.direct_sources, npairs, Val(:atomic))
    med, lo, hi = median_gpu_ms(launch)
    push!(direct_rows, (; job=JOBID, kernel="nearfield", mode="atomic",
        precision="Float32", n=N, ell=ELL, q=12, pairs=npairs, threads,
        warps=threads ÷ 32, block_cap=cap, blocks, median_ms=med,
        min_ms=lo, max_ms=hi))
    @printf("near sweep threads=%3d cap=%5d blocks=%5d  %.3f ms\n",
        threads, cap, blocks, med)
end

for mode in (:atomic, :store, :nostore)
    threads = 128
    cap = 16384
    blocks = min(cld(npairs, threads ÷ 32), cap)
    launch() = CUDA.@cuda threads=threads blocks=blocks attr_direct_kernel!(
        state.output, state.source_bodies, state.cell_ranges,
        state.direct_targets, state.direct_sources, npairs, Val(mode))
    med, lo, hi = median_gpu_ms(launch)
    push!(direct_rows, (; job=JOBID, kernel="nearfield", mode=string(mode),
        precision="Float32", n=N, ell=ELL, q=12, pairs=npairs, threads,
        warps=threads ÷ 32, block_cap=cap, blocks, median_ms=med,
        min_ms=lo, max_ms=hi))
    @printf("near A/B %-7s %.3f ms\n", mode, med)
end

write_csv(joinpath(OUTDIR, PREFIX * "_m2l.csv"), m2l_rows)
write_csv(joinpath(OUTDIR, PREFIX * "_nearfield.csv"), direct_rows)

# Timeline metadata for the exact current production launches.  CUPTI may be
# restricted on a shared node, so attribution A/B data remains load-bearing.
try
    println("\n--- CUDA trace: production tiled leaf M2L")
    m2l_trace = CUDA.@profile trace=true FM._cuda_hier_dense_apply_window!(
        state, ws, plan, hctx, ELL)
    show(stdout, MIME"text/plain"(), m2l_trace)
    println("\n--- CUDA trace: production nearfield")
    near_trace = CUDA.@profile trace=true begin
        threads = 128
        blocks = min(cld(npairs, threads ÷ 32), FM.DIRECT_CUDA_MAX_BLOCKS[])
        CUDA.@cuda threads=threads blocks=blocks FM._cuda_direct_pairs_output_kernel!(
            state.output, state.source_bodies, state.cell_ranges,
            state.direct_targets, state.direct_sources, npairs)
    end
    show(stdout, MIME"text/plain"(), near_trace)
    println()
catch err
    println("CUPTI trace unavailable: ", first(sprint(showerror, err), 500))
end

println("wrote attribution CSVs with prefix ", PREFIX)
