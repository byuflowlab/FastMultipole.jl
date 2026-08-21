# Task 029 cycle 1, P3 rider: ISOLATED fixed-work A/B of nearfield ILP
# mechanisms. NO production change — script-local kernel variants run on the
# direct-pair list captured from a real cache at the baseline geometries, and
# every variant is parity-checked against the script-local baseline kernel
# (identical math to production `_cuda_direct_pairs_output_kernel!`).
#
# Mechanisms under test (028 closed launch-shape/atomic items respected: same
# 128-thread warp-per-pair launch, same `_cuda_fast_rsqrt`, same atomics):
#   base     exact copy of the production kernel body
#   unroll2  source-loop unrolled x2 with independent accumulator chains (ILP)
#   unroll4  source-loop unrolled x4
#   shm      per-warp shared-memory source tile (32-body chunks staged
#            cooperatively; the inner loop reads shared instead of global)
#   shm_u2   shm tile + x2 accumulator chains
#
# Falsification (step-2 P3 design): best-variant gain < 20% closes nearfield
# micro-optimization; the only remaining nearfield lever is then Stage-9a
# macrocell ownership with its own pair-coverage evidence.
#
# Env: FM029N_POLICY (default "sched6-5-4-4"), FM029N_N (default 1000000),
#      FM029N_REPS (default 25), FM029N_OUT (csv path)

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using Dates
using Printf
using SHA

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

include(joinpath(@__DIR__, "fm028_device_system.jl"))

const FM = FastMultipole
const POLICY = get(ENV, "FM029N_POLICY", "sched6-5-4-4")
const N = parse(Int, get(ENV, "FM029N_N", "1000000"))
const REPS = parse(Int, get(ENV, "FM029N_REPS", "25"))
const ELL = 5
const P = 3
const SEED = 24025
const BOX_MIN = SVector(-0.01, -0.01, -0.01)
const BOX_SIZE = 1.02
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const OUT = get(ENV, "FM029N_OUT", joinpath(@__DIR__, "..", "data",
    "performance_high_score_1m_1ms",
    "cuda029_nearfield_ilp_$(gethostname())_$(STAMP).csv"))

function _source_manifest()
    srcdir = joinpath(REPO, "src")
    files = sort(filter(f -> endswith(f, ".jl"), readdir(srcdir)))
    ctx = SHA.SHA256_CTX()
    for f in files
        SHA.update!(ctx, codeunits(f))
        SHA.update!(ctx, read(joinpath(srcdir, f)))
    end
    return bytes2hex(SHA.digest!(ctx))[1:16]
end

function _schedule_kwargs(policy, ell, ::Type{TF}) where TF
    qs = parse.(Int, split(policy[6:end], '-'))
    length(qs) == ell - 1 || error("policy $policy needs $(ell - 1) entries")
    q = last(qs)
    h0 = TF(BOX_SIZE / 2)
    eps = rigid_stencil_epsilon(P, h0, ell, q; lamb_helmholtz=false, TF)
    K = length(union((Set((j == 1 ? RigidHierarchicalTables(x) :
        FM._rigid_transition_tables(qs[j - 1], x)).push_offsets)
        for (j, x) in enumerate(qs))...))
    base = HierarchicalRigidStencil(ConstantPStencilConfig(P, eps;
        lamb_helmholtz=false); near_radius2=q, window_classes=K)
    return (; policy=FM._hierarchical_stencil_with_schedule(base, qs))
end

# ---- kernel variants ---------------------------------------------------------

# base: verbatim production math (warp-per-pair, lane-per-target, serial source
# chain, 4 atomics per target)
function _nf_base_kernel!(output, source_bodies, cell_ranges,
        direct_targets, direct_sources, npairs)
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
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
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
            CUDA.@atomic output[1, i] += u
            CUDA.@atomic output[2, i] += gx
            CUDA.@atomic output[3, i] += gy
            CUDA.@atomic output[4, i] += gz
            i += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

# one source-interaction evaluation, shared by the unrolled variants
@inline function _nf_accum(xi, yi, zi, sx, sy, sz, q, u, gx, gy, gz)
    dx = xi - sx
    dy = yi - sy
    dz = zi - sz
    r2 = dx * dx + dy * dy + dz * dz
    if r2 > zero(r2)
        invr = FM._cuda_fast_rsqrt(r2)
        invr3 = invr * invr * invr
        u += q * invr
        gx -= q * dx * invr3
        gy -= q * dy * invr3
        gz -= q * dz * invr3
    end
    return u, gx, gy, gz
end

# unroll2/unroll4: independent accumulator chains break the serial
# rsqrt->fma dependency chain of the source loop (ILP), everything else equal
function _nf_unroll_kernel!(output, source_bodies, cell_ranges,
        direct_targets, direct_sources, npairs, ::Val{U}) where U
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
            u1 = zero(T); gx1 = zero(T); gy1 = zero(T); gz1 = zero(T)
            u2 = zero(T); gx2 = zero(T); gy2 = zero(T); gz2 = zero(T)
            u3 = zero(T); gx3 = zero(T); gy3 = zero(T); gz3 = zero(T)
            u4 = zero(T); gx4 = zero(T); gy4 = zero(T); gz4 = zero(T)
            j = sfirst
            while j + (U - 1) <= slast
                if i != j
                    u1, gx1, gy1, gz1 = _nf_accum(xi, yi, zi,
                        source_bodies[1, j], source_bodies[2, j],
                        source_bodies[3, j], source_bodies[5, j] * c,
                        u1, gx1, gy1, gz1)
                end
                if i != j + 1
                    u2, gx2, gy2, gz2 = _nf_accum(xi, yi, zi,
                        source_bodies[1, j + 1], source_bodies[2, j + 1],
                        source_bodies[3, j + 1], source_bodies[5, j + 1] * c,
                        u2, gx2, gy2, gz2)
                end
                if U == 4
                    if i != j + 2
                        u3, gx3, gy3, gz3 = _nf_accum(xi, yi, zi,
                            source_bodies[1, j + 2], source_bodies[2, j + 2],
                            source_bodies[3, j + 2], source_bodies[5, j + 2] * c,
                            u3, gx3, gy3, gz3)
                    end
                    if i != j + 3
                        u4, gx4, gy4, gz4 = _nf_accum(xi, yi, zi,
                            source_bodies[1, j + 3], source_bodies[2, j + 3],
                            source_bodies[3, j + 3], source_bodies[5, j + 3] * c,
                            u4, gx4, gy4, gz4)
                    end
                end
                j += U
            end
            while j <= slast
                if i != j
                    u1, gx1, gy1, gz1 = _nf_accum(xi, yi, zi,
                        source_bodies[1, j], source_bodies[2, j],
                        source_bodies[3, j], source_bodies[5, j] * c,
                        u1, gx1, gy1, gz1)
                end
                j += 1
            end
            CUDA.@atomic output[1, i] += (u1 + u2) + (u3 + u4)
            CUDA.@atomic output[2, i] += (gx1 + gx2) + (gx3 + gx4)
            CUDA.@atomic output[3, i] += (gy1 + gy2) + (gy3 + gy4)
            CUDA.@atomic output[4, i] += (gz1 + gz2) + (gz3 + gz4)
            i += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

# shm / shm_u2: the warp cooperatively stages a 32-source chunk (x, y, z, q*c)
# into its shared-memory slice, then every lane's inner loop reads the tile —
# global source traffic drops from (targets x sources) loads to (sources) loads
# per pair. `::Val{U}` selects 1 or 2 accumulator chains over the tile.
function _nf_shm_kernel!(output, source_bodies, cell_ranges,
        direct_targets, direct_sources, npairs, ::Val{U}) where U
    T = eltype(output)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    warps_per_block = blockDim().x ÷ Int32(32)
    warp_in_block = (threadIdx().x - Int32(1)) ÷ Int32(32)
    tile = CUDA.CuDynamicSharedArray(T, (4, 32 * warps_per_block))
    toff = Int(warp_in_block) * 32
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
        active = i <= tlast
        xi = active ? source_bodies[1, i] : zero(T)
        yi = active ? source_bodies[2, i] : zero(T)
        zi = active ? source_bodies[3, i] : zero(T)
        u1 = zero(T); gx1 = zero(T); gy1 = zero(T); gz1 = zero(T)
        u2 = zero(T); gx2 = zero(T); gy2 = zero(T); gz2 = zero(T)
        s0 = sfirst
        while s0 <= slast
            nchunk = min(32, slast - s0 + 1)
            # cooperative stage (sync_warp brackets the tile reuse)
            if lane < nchunk
                sj = s0 + lane
                col = toff + lane + 1
                tile[1, col] = source_bodies[1, sj]
                tile[2, col] = source_bodies[2, sj]
                tile[3, col] = source_bodies[3, sj]
                tile[4, col] = source_bodies[5, sj] * c
            end
            CUDA.sync_warp()
            if active
                t = 1
                while t + (U - 1) <= nchunk
                    col = toff + t
                    if i != s0 + t - 1
                        u1, gx1, gy1, gz1 = _nf_accum(xi, yi, zi,
                            tile[1, col], tile[2, col], tile[3, col],
                            tile[4, col], u1, gx1, gy1, gz1)
                    end
                    if U == 2 && i != s0 + t
                        u2, gx2, gy2, gz2 = _nf_accum(xi, yi, zi,
                            tile[1, col + 1], tile[2, col + 1],
                            tile[3, col + 1], tile[4, col + 1],
                            u2, gx2, gy2, gz2)
                    end
                    t += U
                end
                while t <= nchunk
                    col = toff + t
                    if i != s0 + t - 1
                        u1, gx1, gy1, gz1 = _nf_accum(xi, yi, zi,
                            tile[1, col], tile[2, col], tile[3, col],
                            tile[4, col], u1, gx1, gy1, gz1)
                    end
                    t += 1
                end
            end
            CUDA.sync_warp()
            s0 += 32
        end
        if active
            CUDA.@atomic output[1, i] += u1 + u2
            CUDA.@atomic output[2, i] += gx1 + gx2
            CUDA.@atomic output[3, i] += gy1 + gy2
            CUDA.@atomic output[4, i] += gz1 + gz2
        end
        # NOTE: targets beyond 32 per cell are impossible at the measured
        # geometries (max ~35? guard anyway): handle the tail serially
        i2 = tfirst + lane + 32
        while i2 <= tlast
            xj = source_bodies[1, i2]
            yj = source_bodies[2, i2]
            zj = source_bodies[3, i2]
            ut = zero(T); gxt = zero(T); gyt = zero(T); gzt = zero(T)
            for j in sfirst:slast
                i2 == j && continue
                ut, gxt, gyt, gzt = _nf_accum(xj, yj, zj,
                    source_bodies[1, j], source_bodies[2, j],
                    source_bodies[3, j], source_bodies[5, j] * c,
                    ut, gxt, gyt, gzt)
            end
            CUDA.@atomic output[1, i2] += ut
            CUDA.@atomic output[2, i2] += gxt
            CUDA.@atomic output[3, i2] += gyt
            CUDA.@atomic output[4, i2] += gzt
            i2 += 32
        end
        pair_i += warp_stride
    end
    return nothing
end

# ---- measurement -------------------------------------------------------------

function main()
    println("SOURCE_MANIFEST=", _source_manifest())
    println("julia=", VERSION, " CUDA_runtime=", CUDA.runtime_version(),
        " gpu=", CUDA.name(CUDA.device()), " host=", gethostname(),
        " job=", get(ENV, "SLURM_JOB_ID", ""), " policy=", POLICY, " n=", N,
        " reps=", REPS)
    TF = Float32
    bodies = fm028_body_matrix(SEED, N)
    sys = FM028DeviceSystem{TF}(bodies)
    opts = CUDARadixLifecycleOptions(; precision=TF,
        operator=MaterializedYRotationM2L(), m2l_strategy=DenseTranslationM2L())
    cache = RadixFMMCache(sys; expansion_order=P, ell=ELL, max_n_bodies=N,
        bounds=(BOX_MIN, BOX_SIZE), lamb_helmholtz=false, device=true,
        options=opts, _schedule_kwargs(POLICY, ELL, TF)...)
    CUDA.synchronize()
    state = cache.state
    npairs = state.counts.n_direct
    threads = 128
    blocks = min(cld(npairs, threads ÷ 32), FM.DIRECT_CUDA_MAX_BLOCKS[])
    shmem = 4 * 32 * (threads ÷ 32) * sizeof(TF)
    out = CUDA.zeros(TF, 4, N)
    println("fixed work: npairs=", npairs, " blocks=", blocks)

    launch_base!() = CUDA.@cuda threads=threads blocks=blocks _nf_base_kernel!(
        out, state.source_bodies, state.cell_ranges, state.direct_targets,
        state.direct_sources, npairs)
    launch_u2!() = CUDA.@cuda threads=threads blocks=blocks _nf_unroll_kernel!(
        out, state.source_bodies, state.cell_ranges, state.direct_targets,
        state.direct_sources, npairs, Val(2))
    launch_u4!() = CUDA.@cuda threads=threads blocks=blocks _nf_unroll_kernel!(
        out, state.source_bodies, state.cell_ranges, state.direct_targets,
        state.direct_sources, npairs, Val(4))
    launch_shm!() = CUDA.@cuda threads=threads blocks=blocks shmem=shmem _nf_shm_kernel!(
        out, state.source_bodies, state.cell_ranges, state.direct_targets,
        state.direct_sources, npairs, Val(1))
    launch_shm_u2!() = CUDA.@cuda threads=threads blocks=blocks shmem=shmem _nf_shm_kernel!(
        out, state.source_bodies, state.cell_ranges, state.direct_targets,
        state.direct_sources, npairs, Val(2))
    variants = [("base", launch_base!), ("base_repeat", launch_base!),
        ("unroll2", launch_u2!), ("unroll4", launch_u4!),
        ("shm", launch_shm!), ("shm_u2", launch_shm_u2!)]

    # parity reference
    fill!(out, zero(TF)); launch_base!(); CUDA.synchronize()
    ref = Array(out)
    refscale = max(1.0, maximum(abs.(ref)))

    rows = NamedTuple[]
    base_ms = NaN
    for (name, launch!) in variants
        fill!(out, zero(TF)); launch!(); CUDA.synchronize()   # JIT warm
        got = Array(out)
        maxdev = maximum(abs.(got .- ref)) / refscale
        ok = maxdev <= 2e-4    # atomic/reassociation tolerance at Float32
        ts = Float64[]
        for _ in 1:REPS
            push!(ts, 1e3 * Float64(CUDA.@elapsed launch!()))
        end
        ms = median(ts)
        name == "base" && (base_ms = ms)
        speedup = base_ms / ms
        @printf("%-12s  %8.3f ms  [%8.3f, %8.3f]  x%5.3f vs base  parity %s (maxdev %.2e)\n",
            name, ms, minimum(ts), maximum(ts), speedup, ok ? "OK" : "FAIL", maxdev)
        push!(rows, (; policy=POLICY, n=N, npairs, variant=name, ms,
            min_ms=minimum(ts), max_ms=maximum(ts), speedup,
            parity_ok=ok, max_rel_dev=maxdev, reps=REPS))
        ok || @warn "parity FAILED for $name"
    end
    best = argmin([r.ms for r in rows[3:end]]) + 2
    gain = 1 - rows[best].ms / base_ms
    @printf("best variant: %s  gain %.1f%%  -> %s\n", rows[best].variant,
        100 * gain, gain >= 0.20 ? "MECHANISM CONFIRMED (>=20%)" :
        "FALSIFIED (<20% — close nearfield micro-opt per step-2 P3 rule)")

    mkpath(dirname(OUT))
    open(OUT, "w") do io
        println(io, join(string.(keys(rows[1])), ','))
        for r in rows
            println(io, join(string.(values(r)), ','))
        end
    end
    println("WROTE ", OUT)
    println("NEARFIELD_ILP_EXIT=0")
end

main()
