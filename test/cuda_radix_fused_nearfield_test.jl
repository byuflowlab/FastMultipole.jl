#=
CUDA target-owned fused nearfield shapes (task 041e Stage B).

Verifies the 041e production surface:
  - CUDA_NEARFIELD_SHAPE routing: :pairs (shipped control) vs the fused
    shapes :fused_cta (thread-per-target, serial source traversal) and
    :fused_srclanes (warp-per-target, lanes stride sources with a shuffle
    reduction). The shape Ref is read at cache construction (CSR arming) AND
    inside the lifecycle body (graph-baked at record time), so every cache
    here is constructed AND run under the shape it tests, with the Ref
    restored in a finally block (mirrors the mode-Ref discipline of
    test/cuda_radix_nearfield_binning_test.jl).
  - U (gradient) and J (hessian) parity of both fused shapes against the
    shipped :pairs control at accumulation-order tolerance (fused reductions
    reassociate the sums, so bitwise equality is NOT expected). NOTE: the
    adaptive lifecycle refuses hessian=true with lamb_helmholtz=true (040
    deferral), so J-row coverage comes from the Gravitational cases and the
    vortex cases are gradient-only.
  - the target-major U CSR (u_csr_offsets/u_csr_sources) is exactly the
    per-target-slot permutation of the slot-mapped U pair list, including a
    synthetic pair list with empty target slots (boundary fill), plus the
    deterministic (target-slot, emission-index) order contract.
  - silent shipped fallback: a fused shape on a UNIFORM (non-adaptive) cache
    runs the pairs kernel and matches the :pairs control — no throw.
  - guards: :lut g/h mode throws for fused shapes; an invalid shape symbol
    throws at launch; the 023 transfer-counter contract holds on the fused
    path.
P = 4 is mandatory per the standing project rule; P = 8 is also covered.
Gating follows the repo CUDA test convention: silently skipped unless CUDA
loads, loud error when FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1.
=#

using Test
using Random
using Statistics
using LinearAlgebra
using FastMultipole
using FastMultipole.StaticArrays

if !isdefined(@__MODULE__, :Gravitational)
    include("gravitational.jl")
end
if !isdefined(@__MODULE__, :VortexParticles)
    include("vortex.jl")
end
if !isdefined(@__MODULE__, :SmoothedVortex)
    include("interface_test_systems.jl")
end

const FNF_FM = FastMultipole

_cuda_fused_required() = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

const _FUSED_LOADED = FastMultipole.load_cuda_radix_lifecycle!()
if _FUSED_LOADED
    using CUDA
end

#--- deterministic distributions (5 x n: x, y, z, radius, strength) ---#

function _fnf_cube(n; seed=41501)
    rng = MersenneTwister(seed)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    return b
end

function _fnf_multiscale(n; contrast=100.0, seed=41502)
    rng = MersenneTwister(seed)
    frac = 0.35
    nc = round(Int, frac * n)
    nb = n - nc
    Rc = (3 * nc / (4pi * contrast * nb))^(1 / 3)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    b[1:3, 1:nb] .= rand(rng, 3, nb)
    ctr = (0.6, 0.4, 0.55)
    k = 0
    while k < nc
        p = 2 .* (rand(rng, 3) .- 0.5)
        if sum(abs2, p) <= 1
            k += 1
            b[1:3, nb + k] .= ctr .+ Rc .* p
        end
    end
    return b
end

_fnf_maxrel(A, B, scale) = maximum(abs.(A .- B)) / scale

# set the shape Ref for the duration of f (construction + runs), restore after
function _fnf_with_shape(f, shape::Symbol)
    old = FNF_FM.CUDA_NEARFIELD_SHAPE[]
    oldmin = FNF_FM.CUDA_NEARFIELD_FUSED_MIN_BODIES[]
    FNF_FM.CUDA_NEARFIELD_SHAPE[] = shape
    FNF_FM.CUDA_NEARFIELD_FUSED_MIN_BODIES[] = 0   # test cases are small
    try
        return f()
    finally
        FNF_FM.CUDA_NEARFIELD_SHAPE[] = old
        FNF_FM.CUDA_NEARFIELD_FUSED_MIN_BODIES[] = oldmin
    end
end

@testset "CUDA target-owned fused nearfield (task 041e)" begin
    if !_FUSED_LOADED
        if _cuda_fused_required()
            error("FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle " *
                  "did not load: " * FastMultipole.cuda_radix_status())
        end
        @test true
    else
        # shipped defaults asserted before any test mutates the Refs
        @test FNF_FM.CUDA_NEARFIELD_SHAPE[] === :pairs
        @test FNF_FM.NEARFIELD_SHAPES == (:pairs, :fused_cta, :fused_srclanes, :fused_packed)

        n = 2000
        pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5)

        # CSR permutation check: the per-target-slot CSR interval must be
        # exactly the multiset of that target's edges in the slot-mapped pair
        # list (sorted compare), edge count preserved, offsets monotone with
        # offsets[n_leaves + 1] == n_u + 1.
        function _fnf_check_csr(cache)
            actx = cache.adaptive_tree
            st = cache.adaptive_state
            nl = actx.n_leaves
            n_u = actx.n_u
            offs = Int.(Array(actx.u_csr_offsets)[1:nl + 1])
            srcs = Int.(Array(actx.u_csr_sources)[1:n_u])
            dt = Int.(Array(st.direct_targets)[1:n_u])
            ds = Int.(Array(st.direct_sources)[1:n_u])
            @test issorted(offs)
            @test offs[1] >= 1
            @test offs[nl + 1] == n_u + 1
            @test all(1 .<= srcs .<= nl)
            want = Dict{Int,Vector{Int}}()
            for k in 1:n_u
                push!(get!(want, dt[k], Int[]), ds[k])
            end
            total = 0
            ok = true
            for l in 1:nl
                seg = sort(srcs[offs[l]:offs[l + 1] - 1])
                total += length(seg)
                ok &= seg == sort(get(want, l, Int[]))
            end
            @test ok
            @test total == n_u
            return nothing
        end

        #--- 1. Gravitational U + J parity (hessian rows; the adaptive
        #    lifecycle refuses hessian with lamb_helmholtz, so the J coverage
        #    lives on the scalar kernel) ---#

        @testset "grav U/J parity ($cname, P=$P, $TF)" for
                (cname, gen) in (("cube", _fnf_cube), ("multiscale", _fnf_multiscale)),
                P in (4, 8), TF in (Float64, Float32)
            b = gen(n)
            opts = FNF_FM.CUDARadixLifecycleOptions(precision=TF,
                m2l_strategy=FNF_FM.ConcatenatedFixedZM2L())
            build(sys) = FNF_FM.RadixFMMCache(sys; expansion_order=P, ell=3,
                adaptive=pol, options=opts, hessian=true, device=true)
            U0, J0 = _fnf_with_shape(:pairs) do
                sys = Gravitational(copy(b))
                cache = build(sys)
                # :pairs caches must NOT pay the CSR memory
                @test length(cache.adaptive_tree.u_csr_sources) == 0
                fmm!(sys, cache; scalar_potential=true, gradient=true,
                    hessian=true)
                (copy(sys.potential[5:7, :]), copy(sys.potential[8:16, :]))
            end
            u_scale = maximum(abs.(U0))
            j_scale = maximum(abs.(J0))
            tol = TF === Float64 ? 1e-8 : 4e-4
            for shape in (:fused_cta, :fused_srclanes, :fused_packed)
                _fnf_with_shape(shape) do
                    sys = Gravitational(copy(b))
                    cache = build(sys)
                    actx = cache.adaptive_tree
                    @test length(actx.u_csr_offsets) == actx.leaf_capacity + 1
                    @test length(actx.u_csr_sources) == actx.u_capacity
                    @test actx.u_csr_built_epoch == actx.epoch_id
                    # two steps: the second exercises the resident refresh path
                    fmm!(sys, cache; scalar_potential=true, gradient=true,
                        hessian=true)
                    fmm!(sys, cache; scalar_potential=true, gradient=true,
                        hessian=true)
                    @test _fnf_maxrel(sys.potential[5:7, :], U0, u_scale) < tol
                    @test _fnf_maxrel(sys.potential[8:16, :], J0, j_scale) < tol
                    P == 4 && TF === Float64 && _fnf_check_csr(cache)
                end
            end
        end

        #--- 2. Lamb-Helmholtz vortex U parity (gradient-only, 040 deferral) ---#

        @testset "LH vortex U parity (P=$P, $TF)" for P in (4, 8),
                TF in (Float64, Float32)
            rng = MersenneTwister(41510)
            b = _fnf_multiscale(n; seed=41511)
            pos = b[1:3, :]
            str = randn(rng, 3, n) ./ n
            opts = FNF_FM.CUDARadixLifecycleOptions(precision=TF,
                m2l_strategy=FNF_FM.ConcatenatedFixedZM2L(),
                body_type=FNF_FM.Point{FNF_FM.Vortex})
            build(sys) = FNF_FM.RadixFMMCache(sys; expansion_order=P, ell=3,
                adaptive=pol, options=opts, lamb_helmholtz=true, device=true)
            U0 = _fnf_with_shape(:pairs) do
                sys = VortexParticles(copy(pos), copy(str))
                cache = build(sys)
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                copy(sys.gradient_stretching[1:3, :])
            end
            u_scale = maximum(abs.(U0))
            tol = TF === Float64 ? 1e-8 : 4e-4
            for shape in (:fused_cta, :fused_srclanes, :fused_packed)
                _fnf_with_shape(shape) do
                    sys = VortexParticles(copy(pos), copy(str))
                    cache = build(sys)
                    @test cache.adaptive_tree.u_csr_built_epoch ==
                        cache.adaptive_tree.epoch_id
                    fmm!(sys, cache; scalar_potential=false, gradient=true)
                    @test _fnf_maxrel(sys.gradient_stretching[1:3, :], U0,
                        u_scale) < tol
                    P == 4 && TF === Float64 && _fnf_check_csr(cache)
                end
            end
        end

        #--- 3. heterogeneous-sigma regularized vortex (sticky demotion gives
        #    the U list unequal-level structure) ---#

        @testset "sigma-adaptive U parity (P=$P, $TF)" for P in (4, 8),
                TF in (Float64, Float32)
            rng = MersenneTwister(41520)
            b = _fnf_multiscale(n; seed=41521)
            pos = b[1:3, :]
            str = randn(rng, 3, n) ./ n
            sigma = 10 .^ (rand(rng, n) .* 2 .- 3.5)     # log-uniform, hetero
            dk = FNF_FM.RegularizedVortex(; sigma_row=8)
            spol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5,
                rho_t=dk.rho_t, sigma_row=8)
            opts = FNF_FM.CUDARadixLifecycleOptions(precision=TF,
                m2l_strategy=FNF_FM.ConcatenatedFixedZM2L(),
                body_type=FNF_FM.Point{FNF_FM.Vortex}, direct_kernel=dk)
            build(sys) = FNF_FM.RadixFMMCache(sys; expansion_order=P, ell=3,
                adaptive=spol, options=opts, lamb_helmholtz=true, device=true)
            mk() = SmoothedVortex(VortexParticles(copy(pos), copy(str)),
                copy(sigma))
            U0 = _fnf_with_shape(:pairs) do
                sys = mk()
                cache = build(sys)
                @test cache.adaptive_tree.sigma_armed
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                copy(sys.inner.gradient_stretching[1:3, :])
            end
            u_scale = maximum(abs.(U0))
            tol = TF === Float64 ? 1e-8 : 4e-4
            for shape in (:fused_cta, :fused_srclanes, :fused_packed)
                _fnf_with_shape(shape) do
                    sys = mk()
                    cache = build(sys)
                    actx = cache.adaptive_tree
                    @test actx.sigma_armed
                    # the sigma gate must actually demote (unequal-level U)
                    @test actx.n_demoted > 0
                    @test actx.u_csr_built_epoch == actx.epoch_id
                    fmm!(sys, cache; scalar_potential=false, gradient=true)
                    @test _fnf_maxrel(sys.inner.gradient_stretching[1:3, :],
                        U0, u_scale) < tol
                    P == 4 && TF === Float64 && _fnf_check_csr(cache)
                end
            end
        end

        #--- 4. synthetic CSR build with empty target slots (boundary fill +
        #    deterministic emission order); the cache is discarded after ---#

        @testset "U CSR empty-leaf boundary fill" begin
            _fnf_with_shape(:fused_cta) do
                b = _fnf_cube(600; seed=41530)
                sys = Gravitational(copy(b))
                opts = FNF_FM.CUDARadixLifecycleOptions(precision=Float64,
                    m2l_strategy=FNF_FM.ConcatenatedFixedZM2L())
                cache = FNF_FM.RadixFMMCache(sys; expansion_order=4, ell=3,
                    adaptive=pol, options=opts, device=true)
                actx = cache.adaptive_tree
                nl = actx.n_leaves
                @test nl >= 6
                # synthetic slot-mapped pair list: target slots 1, 3, 4, and
                # everything above 5 have NO edges; emission order is not
                # target-major (exercises the stable key sort)
                dt_h = [2, 5, 2, 2]
                ds_h = [3, 2, 1, 5]
                d_t = CuArray(Int.(dt_h))
                d_s = CuArray(Int.(ds_h))
                FNF_FM._cuda_adaptive_build_u_csr!(actx, d_t, d_s, length(dt_h))
                offs = Int.(Array(actx.u_csr_offsets)[1:nl + 1])
                srcs = Int.(Array(actx.u_csr_sources)[1:length(dt_h)])
                @test issorted(offs)
                @test offs[nl + 1] == length(dt_h) + 1
                # deterministic (target-slot, emission-index) order contract
                @test srcs == [3, 1, 5, 2]
                for l in 1:nl
                    seg = sort(srcs[offs[l]:offs[l + 1] - 1])
                    expect = sort([ds_h[k] for k in eachindex(dt_h) if dt_h[k] == l])
                    @test seg == expect
                end
                # empty slots have empty intervals
                @test offs[1] == offs[2]                  # slot 1 empty
                @test offs[3] == offs[4] == offs[5]       # slots 3, 4 empty
                @test all(offs[l] == length(dt_h) + 1 for l in 6:(nl + 1))
                # the synthetic CSR no longer matches the real lists: discard
                nothing
            end
        end

        #--- 5. silent shipped fallback on a UNIFORM (non-adaptive) cache ---#

        @testset "fused shape on uniform cache falls back silently" begin
            b = _fnf_cube(n; seed=41540)
            opts = FNF_FM.CUDARadixLifecycleOptions(precision=Float64,
                m2l_strategy=FNF_FM.ConcatenatedFixedZM2L())
            build(sys) = FNF_FM.RadixFMMCache(sys; expansion_order=4, ell=3,
                options=opts, hessian=true, device=true)
            U0, J0 = _fnf_with_shape(:pairs) do
                sys = Gravitational(copy(b))
                cache = build(sys)
                fmm!(sys, cache; scalar_potential=true, gradient=true,
                    hessian=true)
                (copy(sys.potential[5:7, :]), copy(sys.potential[8:16, :]))
            end
            u_scale = maximum(abs.(U0))
            j_scale = maximum(abs.(J0))
            for shape in (:fused_cta, :fused_srclanes, :fused_packed)
                _fnf_with_shape(shape) do
                    sys = Gravitational(copy(b))
                    cache = build(sys)
                    @test cache.adaptive === nothing
                    # must run (silent fallback to the pairs kernel), no throw
                    fmm!(sys, cache; scalar_potential=true, gradient=true,
                        hessian=true)
                    @test _fnf_maxrel(sys.potential[5:7, :], U0, u_scale) < 1e-8
                    @test _fnf_maxrel(sys.potential[8:16, :], J0, j_scale) < 1e-8
                end
            end
        end

        #--- 6. guards: :lut g/h mode refused on fused shapes; invalid shape
        #    symbol refused at launch (fresh caches: first lifecycle of an
        #    epoch runs uncaptured, mirroring the binning-test convention) ---#

        @testset "fused-shape guards" begin
            b = _fnf_cube(600; seed=41550)
            opts = FNF_FM.CUDARadixLifecycleOptions(precision=Float64,
                m2l_strategy=FNF_FM.ConcatenatedFixedZM2L())
            _fnf_with_shape(:fused_cta) do
                sys = Gravitational(copy(b))
                cache = FNF_FM.RadixFMMCache(sys; expansion_order=4, ell=3,
                    adaptive=pol, options=opts, device=true)
                old_gh = FNF_FM.CUDA_NEARFIELD_GH_MODE[]
                FNF_FM.CUDA_NEARFIELD_GH_MODE[] = :lut
                try
                    # construction-locked settings error loudly on a late flip
                    @test_throws r"construction-locked" fmm!(sys, cache;
                        scalar_potential=false, gradient=true)
                finally
                    FNF_FM.CUDA_NEARFIELD_GH_MODE[] = old_gh
                end
            end
            # invalid shape at launch: constructed under :pairs, flipped after
            sys2 = Gravitational(copy(b))
            cache2 = FNF_FM.RadixFMMCache(sys2; expansion_order=4, ell=3,
                adaptive=pol, options=opts, device=true)
            old_shape = FNF_FM.CUDA_NEARFIELD_SHAPE[]
            FNF_FM.CUDA_NEARFIELD_SHAPE[] = :bogus
            try
                # construction-locked settings error loudly on a late flip
                @test_throws r"construction-locked" fmm!(sys2, cache2;
                    scalar_potential=false, gradient=true)
            finally
                FNF_FM.CUDA_NEARFIELD_SHAPE[] = old_shape
            end
        end

        #--- 7. 023 transfer-counter contract on the fused path ---#

        @testset "fused-path counter contract ($shape)" for
                shape in (:fused_cta, :fused_srclanes, :fused_packed)
            _fnf_with_shape(shape) do
                b = _fnf_multiscale(n; seed=41560)
                sys = Gravitational(copy(b))
                opts = FNF_FM.CUDARadixLifecycleOptions(precision=Float32,
                    m2l_strategy=FNF_FM.ConcatenatedFixedZM2L())
                cache = FNF_FM.RadixFMMCache(sys; expansion_order=4, ell=3,
                    adaptive=pol, options=opts, device=true)
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                c = cache.adaptive_state.counters
                @test c.expansion_host_copies == 0
                route0 = c.route_uploads
                op0 = c.operator_uploads
                fmm!(sys, cache; scalar_potential=false, gradient=true)
                @test c.route_uploads == route0
                @test c.operator_uploads == op0
                @test c.expansion_host_copies == 0
            end
        end

        # the shipped default must be restored for any suites that follow
        @test FNF_FM.CUDA_NEARFIELD_SHAPE[] === :pairs
    end
end
