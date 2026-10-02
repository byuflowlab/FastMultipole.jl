# Regression tests for the KernelAbstractions extension that need no GPU: each
# runs on the KA CPU backend (or on plain host scalars) and covers one defect
# that was fixed in ext/ka.
#
#   julia --project=test/gpu -e 'using Test; include("test/ka_cpu_regression_test.jl")'
using FastMultipole, KernelAbstractions, GPUArraysCore, LinearAlgebra, Random, Test
using FastMultipole.StaticArrays
const KA = KernelAbstractions
const FM = FastMultipole
const KAExt = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

@isdefined(VortexParticles) || include(joinpath(@__DIR__, "vortex.jl"))
@isdefined(Gravitational) || include(joinpath(@__DIR__, "gravitational.jl"))
@isdefined(SmoothedVortex) || include(joinpath(@__DIR__, "interface_test_systems.jl"))

# Systems owned by this file, so the `device_backend`/`source_revision` methods
# below do not leak onto types other test files use.
#   KACPUVortex      a PartitionedSmoothedVortex resident on the KA CPU backend
#   KACPUTreeSource  the same bodies as a tree source with a fixed revision
struct KACPUVortex{TF}
    inner::PartitionedSmoothedVortex{TF}
end
struct KACPUTreeSource{TF}
    inner::PartitionedSmoothedVortex{TF}
end
for T in (:KACPUVortex, :KACPUTreeSource)
    @eval begin
        FM.source_system_to_buffer!(buffer, i_buffer, s::$T, i_body) =
            FM.source_system_to_buffer!(buffer, i_buffer, s.inner, i_body)
        FM.data_per_body(::$T) = 8
        FM.get_position(s::$T, i) = FM.get_position(s.inner, i)
        FM.strength_dims(::$T) = 3
        FM.get_n_bodies(s::$T) = FM.get_n_bodies(s.inner)
        FM.has_vector_potential(::$T) = true
        FM.body_type(::$T) = Point{Vortex}
        FM.direct_kernel(::$T) = PartitionedVortex(; sigma_row=8)
        FM.buffer_to_target_system!(s::$T, i_target, switch, buffer, i_buffer) =
            FM.buffer_to_target_system!(s.inner, i_target, switch, buffer, i_buffer)
    end
end
FM.device_backend(::KACPUVortex) = KA.CPU()
FM.source_revision(::KACPUTreeSource) = 1

function ka_cpu_smoothed(seed, n, sigma; scale=1.0, shift=0.0)
    Random.seed!(seed)
    base = VortexParticles(shift .+ scale .* rand(3, n), randn(3, n) ./ n, zeros(n);
        potential=zeros(13, n), gradient_stretching=zeros(6, n))
    return KACPUVortex(PartitionedSmoothedVortex(SmoothedVortex(base, fill(sigma, n))))
end

ka_cpu_cache(sys; ell=3) = RadixFMMCache(sys; expansion_order=4, ell, window_classes=64,
    hessian=true, device=true,
    options=FM.RadixLifecycleOptions(; precision=Float64,
        m2l_strategy=FM.ConcatenatedFixedZM2L(), body_type=FM.Point{FM.Vortex}))

@testset "KA regression: Float64 _ka_invsqrt outside the Float32 range" begin
    for r2 in (1e-50, 1e-46, 1e-40, 1e-30, 0.37, 2.0, 1e30, 3e38, 1e39, 1e100)
        want = inv(sqrt(r2))
        @test isapprox(KAExt._ka_invsqrt(r2), want; rtol=4eps())
        @test isapprox(KAExt._ka_invsqrt(r2, Val(false)), want; rtol=4eps())
    end
end

@testset "KA regression: checked keys raise ArgumentError for NaN/huge positions" begin
    be = KA.CPU()
    x_min = SVector(0.0, 0.0, 0.0); box = SVector(1.0, 1.0, 1.0)
    for bad in (NaN, 1e300, -1e300, Inf)
        pos = rand(3, 8); pos[2, 5] = bad
        keys = zeros(UInt64, 8)
        @test_throws ArgumentError KAExt.ka_radix_keys_checked!(keys, zeros(Int32, 1),
            zeros(Int32, 1), pos, x_min, box, 0.5, 3)
    end
    # in-box bodies still pass
    keys = zeros(UInt64, 8)
    @test KAExt.ka_radix_keys_checked!(keys, zeros(Int32, 1), zeros(Int32, 1),
        rand(3, 8), x_min, box, 0.5, 3) === keys
end

@testset "KA regression: device RegularizedVortex regularizes every pair" begin
    rk = RegularizedVortex(; sigma_row=8)
    pk = PartitionedVortex(; sigma_row=8)
    dk = KAExt._ka_device_direct_kernel(rk, Float64)
    dpk = KAExt._ka_device_direct_kernel(pk, Float64)
    rng = MersenneTwister(11)
    src = zeros(8, 1)
    nbeyond = 0
    for trial in 1:200
        sigma = 0.01 + 0.09 * rand(rng)
        rho = (0.2 + 9.0 * rand(rng))          # inside and beyond both cutoffs
        u = normalize(randn(rng, 3))
        dx, dy, dz = u .* (rho * sigma)
        r2 = dx^2 + dy^2 + dz^2; invr = inv(sqrt(r2))
        src[5:7, 1] .= randn(rng, 3); src[8, 1] = sigma
        host = FM._direct_pair_ugh(rk, dx, dy, dz, r2, invr, src, 1)
        dev = FM._direct_pair_ugh(dk, dx, dy, dz, r2, invr, src, 1)
        @test all(isapprox.(dev, host; rtol=1e-12, atol=1e-14 * maximum(abs, host)))
        @test all(isapprox.(FM._direct_pair_ug(dk, dx, dy, dz, r2, invr, src, 1),
            FM._direct_pair_ug(rk, dx, dy, dz, r2, invr, src, 1); rtol=1e-12))
        # the partitioned mirror still goes singular beyond its own rho_t
        if pk.rho_t < rho < rk.rho_t
            nbeyond += 1
            @test FM._direct_pair_ugh(dpk, dx, dy, dz, r2, invr, src, 1) !=
                FM._direct_pair_ugh(dk, dx, dy, dz, r2, invr, src, 1)
        end
    end
    @test nbeyond > 0
end

@testset "KA regression: refresh without the within-cell subsort (skipped on the CPU backend)" begin
    # the subsort's barrier loop cannot be lowered by the KA CPU backend, so the
    # refresh skips it there; the body sort alone must still give a permutation
    sys = ka_cpu_smoothed(21, 300, 0.002)
    cache = ka_cpu_cache(sys)
    @test cache.ell == 3
    KAExt.ka_update_radix_state!(cache, (sys,))
    grid = cache.device_ctx.grid
    n = grid.n_bodies
    @test sort(grid.perm[1:n]) == 1:n
    @test grid.invperm[grid.perm[1:n]] == 1:n
end

@testset "KA regression: all-direct demotion refresh on a device cache" begin
    sys = ka_cpu_smoothed(22, 300, 0.2)   # cutoff rho_t*sigma ~ 0.85: no stencil is adequate
    cache = @test_logs (:warn, r"near-set adequacy failed") match_mode=:any ka_cpu_cache(sys)
    KAExt.ka_update_radix_state!(cache, (sys,))
    @test cache.ell == 2
    @test isempty(cache.accepted_offsets)
    hctx = cache.device_ctx.hierarchical_ctx
    @test hctx.win_valid && hctx.total_routes == 0
    @test cache.state.counts.n_routes == 0
    # every cell pair is direct at the degenerate geometry
    nc = cache.state.counts.n_cells
    @test cache.state.counts.n_direct == nc * nc
end

@testset "KA regression: extra-tree cache is keyed by identity and evicts old epochs" begin
    sys = ka_cpu_smoothed(23, 200, 0.002)
    cache = ka_cpu_cache(sys)
    KAExt.ka_update_radix_state!(cache, (sys,))
    ctx = cache.device_ctx
    ex = KACPUTreeSource(ka_cpu_smoothed(24, 40, 0.002).inner)
    other = KACPUTreeSource(ka_cpu_smoothed(25, 40, 0.002).inner)
    epoch = ctx.epoch_id[]
    kernel = FM.direct_kernel(ex)
    # an entry left under ex's objectid by a different system (objectid reuse)
    bogus = (; marker = :bogus)
    empty!(ctx.extra_tree_cache)
    ctx.extra_tree_cache[objectid(ex)] = (; system = other, revision = 1, epoch,
        kernel, prepared = bogus, used = 0)
    ctx.extra_tree_cache[UInt(12345)] = (; system = other, revision = 1,
        epoch = epoch - 1, kernel, prepared = bogus, used = 0)
    misses = ctx.extra_tree_misses[]
    p = KAExt._ka_extra_tree_prepared!(cache, ex)
    @test p !== bogus
    @test ctx.extra_tree_misses[] == misses + 1
    @test !haskey(ctx.extra_tree_cache, UInt(12345))      # stale epoch evicted
    hits = ctx.extra_tree_hits[]
    @test KAExt._ka_extra_tree_prepared!(cache, ex) === p  # same system: a hit
    @test ctx.extra_tree_hits[] == hits + 1
end

@testset "KA regression: extra-tree cache keeps a working set above 64 sources" begin
    sys = ka_cpu_smoothed(27, 200, 0.002)
    cache = ka_cpu_cache(sys)
    KAExt.ka_update_radix_state!(cache, (sys,))
    ctx = cache.device_ctx
    srcs = [KACPUTreeSource(ka_cpu_smoothed(1000 + k, 4, 0.002).inner) for k in 1:100]
    empty!(ctx.extra_tree_cache)
    first_call = [KAExt._ka_extra_tree_prepared!(cache, s) for s in srcs]
    hits = ctx.extra_tree_hits[]
    second_call = [KAExt._ka_extra_tree_prepared!(cache, s) for s in srcs]
    @test ctx.extra_tree_hits[] == hits + 100
    @test all(second_call .=== first_call)
    # past the cap, the least recently used entry is the one evicted
    cap = KAExt._KA_EXTRA_TREE_CACHE_MAX
    more = [KACPUTreeSource(ka_cpu_smoothed(2000 + k, 4, 0.002).inner) for k in 1:(cap - 100)]
    foreach(s -> KAExt._ka_extra_tree_prepared!(cache, s), more)
    @test length(ctx.extra_tree_cache) == cap
    KAExt._ka_extra_tree_prepared!(cache, srcs[1])          # refresh srcs[1]
    KAExt._ka_extra_tree_prepared!(cache, KACPUTreeSource(ka_cpu_smoothed(3000, 4, 0.002).inner))
    @test length(ctx.extra_tree_cache) == cap
    @test haskey(ctx.extra_tree_cache, objectid(srcs[1]))
    @test !haskey(ctx.extra_tree_cache, objectid(srcs[2]))
end

@testset "KA regression: nearfield workgroup must hold whole lane teams" begin
    sys = ka_cpu_smoothed(26, 300, 0.002)
    cache = ka_cpu_cache(sys)
    KAExt.ka_update_radix_state!(cache, (sys,))
    state = cache.state
    lanes = KAExt._nf_config(KA.CPU(), Float64).lanes
    @test_throws ArgumentError KAExt.ka_launch_nearfield!(state; workgroup=lanes + lanes ÷ 2)
    @test_throws ArgumentError KAExt.ka_launch_nearfield!(state; workgroup=-1)
    @test_throws ArgumentError KAExt.ka_launch_l2b!(state; workgroup=-1)
end

# The full device lifecycle does not run on the KA CPU backend (its per-cell
# and per-pair team kernels use group indices the CPU backend cannot lower), so
# the end-to-end counterparts of the checks below -- metadata delivery, the
# rectangular-box step, the tree-carried contract functor and the near-sweep
# workgroup -- are in test/gpu/ka_regression_correctness.jl.

@testset "KA regression: checked keys clamp each axis of a rectangular box" begin
    # bounds (8,2,2) at ell=3: unit leaf cells, ell_axes (3,1,1); a body on the
    # y = 2 and z = 2 faces must land in cell 1 of those axes, not cell 2
    x_min = SVector(0.0, 0.0, 0.0); box = SVector(8.0, 2.0, 2.0)
    pos = [3.5 8.0 0.5; 2.0 0.5 2.0; 0.5 2.0 2.0]
    keys = zeros(UInt64, 3)
    KAExt.ka_radix_keys_checked!(keys, zeros(Int32, 1), zeros(Int32, 1), pos, x_min,
        box, 4.0, 3; ell_axes=SVector(3, 1, 1))
    @test [KAExt.ka_decode_morton_key(k, 3) for k in keys] ==
        [(3, 1, 0), (7, 0, 1), (0, 1, 1)]
    @test keys == [FM.morton_key(FM.radix_cell_coord(x_min, 4.0, 3, pos[:, i],
        SVector(3, 1, 1)), 3) for i in 1:3]
end

# an extra source functor meeting only the `_extra_pair_ug` contract: not an
# `AbstractDirectKernel`; `EP=false` also declares `_emits_potential = false`.
# It returns u = 1/r so that row 1 shows whether the potential was written.
struct KAContractKernel{EP} end
FM._emits_potential(::KAContractKernel{false}) = false
@inline function FM._extra_pair_ug(::KAContractKernel, tx, ty, tz, buf, j)
    T = typeof(tx)
    @inbounds begin
        dx = tx - buf[1, j]; dy = ty - buf[2, j]; dz = tz - buf[3, j]
    end
    r2 = dx * dx + dy * dy + dz * dz
    r2 > zero(T) || return zero(T), zero(T), zero(T), zero(T)
    invr = inv(sqrt(r2))
    _, gx, gy, gz = FM._direct_pair_ug(FM.SingularVortex(), dx, dy, dz, r2, invr, buf, j)
    return invr, gx, gy, gz
end
struct KAContractSource{EP}
    inner::PartitionedSmoothedVortex{Float64}
end
FM.source_system_to_buffer!(buffer, i_buffer, s::KAContractSource, i_body) =
    FM.source_system_to_buffer!(buffer, i_buffer, s.inner, i_body)
FM.data_per_body(::KAContractSource) = 8
FM.get_position(s::KAContractSource, i) = FM.get_position(s.inner, i)
FM.strength_dims(::KAContractSource) = 3
FM.get_n_bodies(s::KAContractSource) = FM.get_n_bodies(s.inner)
FM.has_vector_potential(::KAContractSource) = true
FM.body_type(::KAContractSource) = Point{Vortex}
FM.direct_kernel(::KAContractSource{EP}) where EP = KAContractKernel{EP}()

@testset "KA regression: contract-only extra source functors on the device" begin
    Random.seed!(43)
    xt = rand(3, 20) .+ 2
    for EP in (true, false)
        src = KAContractSource{EP}(ka_cpu_smoothed(44, 30, 0.002).inner)
        dev = KAExt.ka_points_from_extra_source(KA.CPU(), xt, src, Float64)
        host = zeros(4, 20)
        FM._host_targets_from_extra_source!(host, KAContractKernel{EP}(), xt, 20,
            FM._radix_extra_source_buffer(Float64, src), Val(false))
        @test dev ≈ host
        @test all(iszero, dev[1, :]) == !EP
    end

end

# `_utick!` timer object that throws at one named stage: the injected failure
# for the occupancy-epoch test below
struct KAThrowAtTick
    name::Symbol
end
Base.get!(t::KAThrowAtTick, name, default) =
    name === t.name ? error("injected failure at $name") : default

@testset "KA regression: a failed refresh leaves no occupancy-epoch snapshot" begin
    sysA = ka_cpu_smoothed(49, 300, 0.002)
    sysB = ka_cpu_smoothed(50, 300, 0.002; scale=0.5, shift=0.25)
    cache = ka_cpu_cache(sysA)          # built (and refreshed) on A's occupancy
    ref = ka_cpu_cache(sysA)
    KAExt.ka_update_radix_state!(ref, (sysB,))
    # fail after the node rebuild of B's occupancy, before its direct pairs,
    # windows and stage groups
    KAExt._KA_UPDATE_TIMERS[] = KAThrowAtTick(:grid_rebuild)
    try
        @test_throws ErrorException KAExt.ka_update_radix_state!(cache, (sysB,))
    finally
        KAExt._KA_UPDATE_TIMERS[] = nothing
    end
    KAExt.ka_update_radix_state!(cache, (sysB,))
    c, r = cache.state.counts, ref.state.counts
    @test (c.n_cells, c.n_nodes, c.n_direct, c.n_routes) ==
        (r.n_cells, r.n_nodes, r.n_direct, r.n_routes)
    nd = c.n_direct; nr = c.n_routes
    @test cache.state.direct_targets[1:nd] == ref.state.direct_targets[1:nd]
    @test cache.state.direct_sources[1:nd] == ref.state.direct_sources[1:nd]
    hc, hr = cache.device_ctx.hierarchical_ctx, ref.device_ctx.hierarchical_ctx
    @test hc.win_valid && hc.win_targets[1:nr] == hr.win_targets[1:nr] &&
        hc.win_sources[1:nr] == hr.win_sources[1:nr] && hc.win_class[1:nr] == hr.win_class[1:nr]
    edges(cc) = [(Array(g.source_idx), Array(g.target_idx)) for g in cc.device_ctx.workspace.m2m_groups]
    @test edges(cache) == edges(ref)
end

@testset "KA regression: nearfield_pass on the direct arm is refused before any work" begin
    sys = ka_cpu_smoothed(51, 200, 0.002)
    cache = ka_cpu_cache(sys)
    sw = FM.DerivativesSwitch(false, true, false, (sys,))
    FM.set_radix_setting!(:RADIX_DIRECT_ARM, true)
    try
        step = cache.step
        @test_throws ArgumentError KAExt.ka_radix_cache_device_step!(cache, (sys,), sw;
            nearfield_pass = c -> nothing)
        @test cache.step == step
    finally
        FM.set_radix_setting!(:RADIX_DIRECT_ARM, false)
    end
end

@testset "KA regression: counting sort needs every buffer to span the key domain" begin
    full = zeros(Int32, 1 << 9); short = zeros(Int32, 1)
    @test KAExt.ka_counting_sort_ready(full, 3, full, full)
    @test !KAExt.ka_counting_sort_ready(full, 3, short, full)
    @test !KAExt.ka_counting_sort_ready(full, 3, full, short)
    @test !KAExt.ka_counting_sort_ready(full, 3, nothing, full)
    # a short prefix falls back to the stable sort instead of running past it
    keys = UInt64.(rand(0:511, 100))
    perm = zeros(Int, 100); sk = similar(keys); ip = zeros(Int, 100)
    KAExt.ka_radix_sort_bodies!(perm, sk, ip, keys; ell=3, histogram=full,
        prefix=short, cursor=full)
    @test perm == sortperm(keys)
end
