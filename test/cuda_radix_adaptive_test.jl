#=
CUDA device-resident adaptive octree lifecycle (task 041).

Verifies the device mirror of the 039/040 host adaptive machinery:
  - structural parity vs the host 039 tree + DTR lists (EXACT node table,
    level offsets, balance-split count, U/W/X pair sets, V route multiset and
    class histogram), balanced and unbalanced, q in (3, 12), K_max in (8, 32),
    including a sigma-gated (sticky demotion) variant;
  - lifecycle parity vs the host 040 adaptive lifecycle (velocity fields agree
    to accumulation-order tolerance) and the phase accuracy gate (velocity
    rel RMS <= 1e-3 vs direct) on cube / filament / multiscale, P in (4, 8),
    Float64 and Float32, plus dense-CUDA and precomputed-y strategy coverage;
  - Lamb-Helmholtz vortex accuracy (device M2T/S2L phi+chi channels);
  - the 023 counter contract (route/operator uploads constant after
    construction, expansion_host_copies == 0), zero device allocation of the
    warmed adaptive lifecycle body, and occupancy-epoch semantics;
  - task-041 construction guards (split veto unsupported on device, TwoPass +
    adaptive, ungated regularized kernels);
  - a uniform-device non-regression smoke (full non-regression = the existing
    CUDA suites, run separately).
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

const ACU_FM = FastMultipole

_cuda_adaptive_required() = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

#--- distributions (5 x n: x, y, z, radius, strength), task-040 generators ---#

function _acu_cube(n; seed=41101)
    rng = MersenneTwister(seed)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    return b
end

function _acu_multiscale(n; contrast=100.0, seed=41102)
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

function _acu_filament(n; seed=41103)
    rng = MersenneTwister(seed)
    nf = round(Int, 0.6n)
    b = rand(rng, 5, n)
    b[4, :] .*= 1e-3
    b[5, :] ./= n
    core = 0.004
    for p in 1:nf
        t = 4pi * (p - 1) / nf
        c = (0.5 + 0.35cos(t), 0.5 + 0.35sin(t), 0.15 + 0.7t / (4pi))
        b[1:3, p] .= c .+ core .* randn(rng, 3)
    end
    b[1:3, nf+1:end] .= rand(rng, 3, n - nf)
    return b
end

_acu_rel_rms(a, b) = sqrt(mean(abs2, a .- b)) / sqrt(mean(abs2, b))

@testset "CUDA adaptive octree lifecycle (task 041)" begin
    loaded = FastMultipole.load_cuda_radix_lifecycle!()
    if !loaded
        if _cuda_adaptive_required()
            error("FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle " *
                  "did not load: " * FastMultipole.cuda_radix_status())
        end
    else
        @eval using CUDA

        #--- a. structural parity vs the host 039 tree + lists ---#

        @testset "device structural parity vs host 039 ($name, K=$K, q=$q, bal=$bal)" for
                (name, gen) in (("uniform", _acu_cube), ("multiscale", _acu_multiscale),
                                ("filament", _acu_filament)),
                K in (8, 32), q in (3, 12), bal in (true, false)
            n = 2000
            b = gen(n; seed=41110)
            pol = AdaptiveTreePolicy(K_max=K, ell_max=7, near_radius2=q, balance=bal)
            sys_d = Gravitational(copy(b))
            dcache = ACU_FM.RadixFMMCache(sys_d; expansion_order=4, ell=3,
                adaptive=pol, device=true)
            actx = dcache.adaptive_tree
            # host reference tree + lists on the SAME root cube
            sys_h = Gravitational(copy(b))
            htree = ACU_FM.AdaptiveRadixTree((sys_h,); policy=pol,
                root=(dcache.x_min, dcache.h0))
            hlists = ACU_FM.AdaptiveInteractionLists(htree)
            ACU_FM.build_adaptive_interaction_lists!(hlists, htree)
            # node table
            nn = actx.n_nodes
            @test nn == htree.n_nodes
            @test actx.n_leaves == htree.n_leaves
            @test actx.n_balance_splits == htree.n_balance_splits
            @test actx.level_offsets == htree.level_offsets
            g = actx.grid
            @test Array(g.node_keys)[1:nn] == htree.node_keys[1:nn]
            @test Int.(Array(g.node_levels)[1:nn]) == Int.(htree.node_levels[1:nn])
            @test Int.(Array(g.parent_index)[1:nn]) == Int.(htree.parent_index[1:nn])
            @test Int.(Array(g.child_ranges)[:, 1:nn]) == Int.(htree.child_ranges[:, 1:nn])
            @test Int.(Array(actx.node_lo)[1:nn]) == Int.(htree.node_lo[1:nn])
            @test Int.(Array(actx.node_hi)[1:nn]) == Int.(htree.node_hi[1:nn])
            # sorted body permutation partition (same key sort; equal-key order
            # may differ between sort backends, so compare keys, not perm)
            # U/W/X pair sets
            @test actx.n_u == hlists.n_u
            @test actx.n_w == hlists.n_w
            @test actx.n_x == hlists.n_x
            du = Set(zip(Int.(Array(actx.u_targets)[1:actx.n_u]),
                         Int.(Array(actx.u_sources)[1:actx.n_u])))
            hu = Set(zip(hlists.u_targets[1:hlists.n_u], hlists.u_sources[1:hlists.n_u]))
            @test du == hu
            dw = Set(zip(Int.(Array(actx.w_targets)[1:actx.n_w]),
                         Int.(Array(actx.w_sources)[1:actx.n_w])))
            hw = Set(zip(hlists.w_targets[1:hlists.n_w], hlists.w_sources[1:hlists.n_w]))
            @test dw == hw
            dx = Set(zip(Int.(Array(actx.x_targets)[1:actx.n_x]),
                         Int.(Array(actx.x_sources)[1:actx.n_x])))
            hx = Set(zip(hlists.x_targets[1:hlists.n_x], hlists.x_sources[1:hlists.n_x]))
            @test dx == hx
            # V route stream: multiset + class partition
            nr = actx.n_routes
            @test nr == hlists.n_routes
            dvt = Int.(Array(actx.route_targets)[1:nr])
            dvs = Int.(Array(actx.route_sources)[1:nr])
            dvc = Int.(Array(actx.route_class)[1:nr])
            hv = Set(zip(hlists.route_targets[1:nr], hlists.route_sources[1:nr],
                Int.(hlists.route_class[1:nr])))
            @test Set(zip(dvt, dvs, dvc)) == hv
            @test length(unique(zip(dvt, dvs))) == nr    # no duplicate pairs
            @test actx.class_starts == hlists.class_starts
            # every device route sits inside its class's CSR range
            ok_csr = true
            for r in 1:nr
                c = dvc[r]
                ok_csr &= actx.class_starts[c] <= r < actx.class_starts[c + 1]
            end
            @test ok_csr
        end

        #--- a2. sigma-gated structural parity (sticky demotion) ---#

        @testset "device structural parity: sigma gate (q=$q)" for q in (3, 12)
            n = 2000
            b = _acu_multiscale(n; seed=41120)
            rng = MersenneTwister(41121)
            b[4, :] .= 10 .^ (rand(rng, n) .* 2 .- 3.5)   # heterogeneous sigma
            pol = AdaptiveTreePolicy(K_max=16, ell_max=7, near_radius2=q,
                rho_t=4.789, sigma_row=4)
            sys_d = Gravitational(copy(b))
            dcache = ACU_FM.RadixFMMCache(sys_d; expansion_order=4, ell=3,
                adaptive=pol, device=true)
            actx = dcache.adaptive_tree
            @test actx.sigma_armed
            @test actx.n_demoted > 0
            sys_h = Gravitational(copy(b))
            htree = ACU_FM.AdaptiveRadixTree((sys_h,); policy=pol,
                root=(dcache.x_min, dcache.h0), sigma=b[4, :])
            hlists = ACU_FM.AdaptiveInteractionLists(htree)
            ACU_FM.build_adaptive_interaction_lists!(hlists, htree)
            @test actx.n_nodes == htree.n_nodes
            @test actx.n_u == hlists.n_u
            @test actx.n_w == hlists.n_w
            @test actx.n_x == hlists.n_x
            @test actx.n_routes == hlists.n_routes
            @test actx.n_demoted == hlists.n_demoted
            du = Set(zip(Int.(Array(actx.u_targets)[1:actx.n_u]),
                         Int.(Array(actx.u_sources)[1:actx.n_u])))
            hu = Set(zip(hlists.u_targets[1:hlists.n_u], hlists.u_sources[1:hlists.n_u]))
            @test du == hu
            @test actx.class_starts == hlists.class_starts
        end

        #--- b. lifecycle parity vs the host adaptive lifecycle (040) ---#

        @testset "device lifecycle parity ($name, P=$P, $TF)" for
                (name, gen) in (("cube", _acu_cube), ("filament", _acu_filament),
                                ("multiscale", _acu_multiscale)),
                P in (4, 8), TF in (Float64, Float32)
            n = 2000
            b = gen(n; seed=41130)
            ref = Gravitational(copy(b))
            FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
            gref = ref.potential[5:7, :]
            pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5)
            opts = ACU_FM.CUDARadixLifecycleOptions(precision=TF,
                m2l_strategy=ACU_FM.ConcatenatedFixedZM2L())
            sys_h = Gravitational(copy(b))
            hcache = ACU_FM.RadixFMMCache(sys_h; expansion_order=P, ell=3,
                adaptive=pol, options=opts)
            fmm!(sys_h, hcache; scalar_potential=true, gradient=true)
            sys_d = Gravitational(copy(b))
            dcache = ACU_FM.RadixFMMCache(sys_d; expansion_order=P, ell=3,
                adaptive=pol, options=opts, device=true)
            if name != "cube"
                @test dcache.adaptive_tree.n_w > 0
                @test dcache.adaptive_tree.n_x > 0
            end
            fmm!(sys_d, dcache; scalar_potential=true, gradient=true)
            # host-device agreement (accumulation-order tolerance)
            tol_hd = TF === Float64 ? 1e-10 : 5e-4
            @test _acu_rel_rms(sys_d.potential[5:7, :], sys_h.potential[5:7, :]) <= tol_hd
            # phase accuracy gate vs direct
            @test _acu_rel_rms(sys_d.potential[5:7, :], gref) <= 1e-3
            @test maximum(abs.(sys_d.potential[1, :] .- ref.potential[1, :])) /
                maximum(abs.(ref.potential[1, :])) <= 1e-3
        end

        #--- b2. strategy coverage: dense CUDA + precomputed-y ---#

        @testset "device lifecycle strategies ($sname)" for (sname, mkopts) in (
                ("dense", () -> ACU_FM.CUDARadixLifecycleOptions(
                    m2l_strategy=ACU_FM.DenseTranslationM2L(apply_chunk=64,
                        build_chunk=8))),
                ("precomputed_y", () -> ACU_FM.CUDARadixLifecycleOptions(
                    operator=ACU_FM.FactoredRotationM2L(),
                    m2l_strategy=ACU_FM.PrecomputedFactoredYM2L())),
            )
            n = 2000
            b = _acu_multiscale(n; seed=41140)
            ref = Gravitational(copy(b))
            FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
            gref = ref.potential[5:7, :]
            pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5)
            sys_d = Gravitational(copy(b))
            dcache = ACU_FM.RadixFMMCache(sys_d; expansion_order=4, ell=3,
                adaptive=pol, options=mkopts(), device=true)
            fmm!(sys_d, dcache; scalar_potential=true, gradient=true)
            @test _acu_rel_rms(sys_d.potential[5:7, :], gref) <= 1e-3
        end

        #--- c. Lamb-Helmholtz vortex accuracy (device M2T/S2L channels) ---#

        @testset "device LH vortex accuracy ($name, P=$P)" for
                (name, seed) in (("cube", 41151), ("multiscale", 41152)), P in (4, 8)
            n = 2000
            rng = MersenneTwister(41150)
            str = randn(rng, 3, n) ./ n
            b = name == "cube" ? _acu_cube(n; seed=seed) : _acu_multiscale(n; seed=seed)
            pos = b[1:3, :]
            ref = VortexParticles(copy(pos), copy(str))
            FastMultipole.direct!(ref; scalar_potential=false, gradient=true)
            gref = ref.gradient_stretching[1:3, :]
            pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5)
            opts = ACU_FM.CUDARadixLifecycleOptions(precision=Float64,
                m2l_strategy=ACU_FM.ConcatenatedFixedZM2L(),
                body_type=ACU_FM.Point{ACU_FM.Vortex})
            sys_h = VortexParticles(copy(pos), copy(str))
            hcache = ACU_FM.RadixFMMCache(sys_h; expansion_order=P, ell=3,
                adaptive=pol, options=opts, lamb_helmholtz=true)
            fmm!(sys_h, hcache; scalar_potential=false, gradient=true)
            sys_d = VortexParticles(copy(pos), copy(str))
            dcache = ACU_FM.RadixFMMCache(sys_d; expansion_order=P, ell=3,
                adaptive=pol, options=opts, lamb_helmholtz=true, device=true)
            name == "multiscale" && @test dcache.adaptive_tree.n_w > 0
            fmm!(sys_d, dcache; scalar_potential=false, gradient=true)
            @test _acu_rel_rms(sys_d.gradient_stretching[1:3, :],
                sys_h.gradient_stretching[1:3, :]) <= 1e-10
            @test _acu_rel_rms(sys_d.gradient_stretching[1:3, :], gref) <= 1e-3
        end

        #--- d. counter + zero-allocation + epoch contract ---#

        @testset "device counter/zero-alloc/epoch contract" begin
            n = 1500
            b = _acu_multiscale(n; seed=41160)
            sys = Gravitational(copy(b))
            pol = AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5)
            cache = ACU_FM.RadixFMMCache(sys; expansion_order=4, ell=3,
                adaptive=pol, device=true)
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            state = cache.adaptive_state
            actx = cache.adaptive_tree
            c = state.counters
            base_route = c.route_uploads
            base_operator = c.operator_uploads
            @test c.expansion_host_copies == 0
            # epoch fast path: unmoved bodies leave the epoch untouched
            e1 = actx.epoch_id
            ACU_FM.update_cuda_radix_state!(cache, (sys,))
            @test actx.epoch_id == e1
            # stepping with moved bodies: counters stay flat, arrays identical
            out_id = objectid(state.output)
            mp_id = objectid(state.multipoles.phi)
            rng = MersenneTwister(41161)
            for step in 1:3
                for i in eachindex(sys.bodies)
                    bd = sys.bodies[i]
                    pos = clamp.(bd.position .+ 0.05 .* (rand(rng, SVector{3,Float64}) .- 0.5),
                        0.001, 0.999)
                    sys.bodies[i] = Body(pos, bd.radius, bd.strength)
                end
                fmm!(sys, cache; scalar_potential=true, gradient=true)
                @test c.route_uploads == base_route
                @test c.operator_uploads == base_operator
                @test c.expansion_host_copies == 0
            end
            @test objectid(cache.adaptive_state.output) == out_id
            @test objectid(cache.adaptive_state.multipoles.phi) == mp_id
            # a big move changes the occupancy epoch
            e2 = actx.epoch_id
            @test e2 > e1     # the 0.05 jitter moved bodies across leaves
            # warmed lifecycle body: zero device allocation (runtime @eval:
            # CUDA.@allocated is a macro and CUDA only loads inside this branch)
            ACU_FM.run_cuda_adaptive_radix_lifecycle!(cache.adaptive_state,
                cache.adaptive_tree)
            dev_alloc = @eval CUDA.@allocated ACU_FM.run_cuda_adaptive_radix_lifecycle!(
                $(cache).adaptive_state, $(cache).adaptive_tree)
            @test dev_alloc == 0
        end

        #--- e. construction guards ---#

        @testset "device adaptive guards" begin
            n = 300
            b = _acu_cube(n; seed=41170)
            sys = Gravitational(copy(b))
            # split veto unsupported on device (recorded 041 limitation)
            @test_throws ArgumentError ACU_FM.RadixFMMCache(sys;
                expansion_order=4, ell=3, device=true,
                adaptive=AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5,
                    split_veto=true, rho_t=4.789, sigma_row=4))
            # TwoPassVortex + adaptive: refused (shared host guard)
            base = generate_vortex(41171, n)
            sigma = fill(0.004, n)
            opts_tp = ACU_FM.CUDARadixLifecycleOptions(precision=Float64,
                m2l_strategy=ACU_FM.ConcatenatedFixedZM2L(),
                body_type=ACU_FM.Point{ACU_FM.Vortex},
                direct_kernel=ACU_FM.TwoPassVortex(; sigma_row=8))
            @test_throws ArgumentError ACU_FM.RadixFMMCache(
                SmoothedVortex(base, sigma); expansion_order=4, ell=3,
                options=opts_tp, lamb_helmholtz=true, device=true,
                adaptive=AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5,
                    rho_t=4.789, sigma_row=8))
            # regularized kernel without the armed per-cell gate: refused
            @test_throws ArgumentError ACU_FM.RadixFMMCache(
                SmoothedVortex(base, sigma); expansion_order=4, ell=3,
                device=true,
                adaptive=AdaptiveTreePolicy(K_max=16, ell_max=6, near_radius2=5))
        end

        #--- f. uniform-device non-regression smoke ---#

        @testset "uniform device non-regression smoke" begin
            n = 2000
            b = _acu_cube(n; seed=41180)
            ref = Gravitational(copy(b))
            FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
            sys = Gravitational(copy(b))
            cache = ACU_FM.RadixFMMCache(sys; expansion_order=4, ell=3,
                device=true)
            @test cache.adaptive === nothing
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            @test _acu_rel_rms(sys.potential[5:7, :], ref.potential[5:7, :]) <= 1e-3
        end
    end
end
