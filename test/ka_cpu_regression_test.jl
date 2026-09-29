# Regression tests for the KernelAbstractions extension that need no GPU: each
# runs on the KA CPU backend (or on plain host scalars) and covers one defect
# that was fixed in ext/ka.
#
#   julia --project=test -e 'using Test; include("test/ka_cpu_regression_test.jl")'
using FastMultipole, KernelAbstractions, LinearAlgebra, Random, Test
using FastMultipole.StaticArrays
const KA = KernelAbstractions
const FM = FastMultipole
const KAExt = Base.get_extension(FastMultipole, :FastMultipoleKAExt)

@isdefined(VortexParticles) || include(joinpath(@__DIR__, "vortex.jl"))
@isdefined(Gravitational) || include(joinpath(@__DIR__, "gravitational.jl"))
@isdefined(SmoothedVortex) || include(joinpath(@__DIR__, "interface_test_systems.jl"))

FM.device_backend(::PartitionedSmoothedVortex) = KA.CPU()

function ka_cpu_smoothed(seed, n, sigma)
    Random.seed!(seed)
    base = VortexParticles(rand(3, n), randn(3, n) ./ n, zeros(n);
        potential=zeros(13, n), gradient_stretching=zeros(6, n))
    return PartitionedSmoothedVortex(SmoothedVortex(base, fill(sigma, n)))
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

@testset "KA regression: within-cell subsort runs on the CPU backend" begin
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
    ex = PartitionedSmoothedVortex(ka_cpu_smoothed(24, 40, 0.002).smoothed)
    other = ka_cpu_smoothed(25, 40, 0.002)
    FM.source_revision(::PartitionedSmoothedVortex) = 1
    try
        epoch = ctx.epoch_id[]
        kernel = FM.direct_kernel(ex)
        # an entry left under ex's objectid by a different system (objectid reuse)
        bogus = (; marker = :bogus)
        empty!(ctx.extra_tree_cache)
        ctx.extra_tree_cache[objectid(ex)] = (; system = other, revision = 1, epoch,
            kernel, prepared = bogus)
        ctx.extra_tree_cache[UInt(12345)] = (; system = other, revision = 1,
            epoch = epoch - 1, kernel, prepared = bogus)
        misses = ctx.extra_tree_misses[]
        p = KAExt._ka_extra_tree_prepared!(cache, ex)
        @test p !== bogus
        @test ctx.extra_tree_misses[] == misses + 1
        @test !haskey(ctx.extra_tree_cache, UInt(12345))      # stale epoch evicted
        hits = ctx.extra_tree_hits[]
        @test KAExt._ka_extra_tree_prepared!(cache, ex) === p  # same system: a hit
        @test ctx.extra_tree_hits[] == hits + 1
    finally
        Base.delete_method(which(FM.source_revision, (PartitionedSmoothedVortex,)))
    end
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
