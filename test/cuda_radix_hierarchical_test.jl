using FastMultipole
using FastMultipole.StaticArrays
using Random
using Test

if !isdefined(@__MODULE__, :generate_gravitational)
    include("gravitational.jl")
end

const HCU_FM = FastMultipole

_cuda_hier_required() = get(ENV, "FASTMULTIPOLE_REQUIRE_CUDA_TESTS", "0") == "1"

# Same accuracy contract the host suite uses: an epsilon whose analytic classifier
# rejects exactly the requested rigid near set at this (h0, ell).
function _hcu_epsilon(P, q, h0, ell, ::Type{TF}, LH) where TF
    q == 3 && return TF(1e12)
    probe = ConstantPStencilConfig(P, one(TF); lamb_helmholtz=LH)
    upper = constant_p_stencil_bound(TF(h0), ell, probe, SVector(2, 2, 2))
    lower = constant_p_stencil_bound(TF(h0), ell, probe, SVector(3, 2, 0))
    return (upper + lower) / 2
end

function _hcu_policy(P, q, h0, ell, ::Type{TF}, LH; window_classes=8) where TF
    return HierarchicalRigidStencil(
        ConstantPStencilConfig(P, _hcu_epsilon(P, q, h0, ell, TF, LH);
            lamb_helmholtz=LH);
        near_radius2=q, window_classes)
end

# Exactly-once near/far coverage over leaf pairs, from device-generated lists.
function _hcu_coverage(dcache, hcache)
    ds = dcache.state
    hs = hcache.state
    hctx = hs.interaction_list
    grid_levels = Array(ds.grid.node_levels)
    node_coords = Array(ds.grid.node_coords)
    leaf_to_node = Array(ds.grid.leaf_to_node)
    C = ds.counts.n_cells
    hits = zeros(Int, C, C)
    dt = Array(ds.direct_targets)
    dsrc = Array(ds.direct_sources)
    for i in 1:ds.counts.n_direct
        hits[dt[i], dsrc[i]] += 1
    end
    leaf_coords = [SVector{3,Int}(node_coords[:, leaf_to_node[c]]) for c in 1:C]
    noffsets = length(hctx.tables.push_offsets)
    K = hctx.window_classes
    for level in 2:ds.grid.ell, first in 1:K:noffsets
        last = min(first + K - 1, noffsets)
        n = HCU_FM.cuda_hierarchical_route_window!(ds, level, first, last)
        n == 0 && continue
        shift = ds.grid.ell - level
        sources = Array(view(ds.route_sources, 1:n))
        targets = Array(view(ds.route_targets, 1:n))
        for r in 1:n
            sc = SVector{3,Int}(node_coords[:, sources[r]])
            tc = SVector{3,Int}(node_coords[:, targets[r]])
            @assert grid_levels[sources[r]] == level
            @assert grid_levels[targets[r]] == level
            source_leaves = findall(c -> (c .>> shift) == sc, leaf_coords)
            target_leaves = findall(c -> (c .>> shift) == tc, leaf_coords)
            for t in target_leaves, s in source_leaves
                hits[t, s] += 1
            end
        end
    end
    return hits
end

function _hcu_random_physical!(buf, rng)
    buf.phi .= randn(rng, eltype(buf.phi), size(buf.phi))
    for n in 0:buf.basis_info.orders.P_phi
        buf.phi[HCU_FM.flat_basis_index(n, 0, 2), :] .= 0
    end
    buf.chi .= randn(rng, eltype(buf.chi), size(buf.chi))
    for n in 0:buf.basis_info.orders.P_active
        buf.chi[HCU_FM.flat_basis_index(n, 0, 2), :] .= 0
    end
    return buf
end

const _HCU_STRATEGIES = (
    (:concat, () -> CUDARadixLifecycleOptions(; m2l_strategy=ConcatenatedFixedZM2L())),
    (:factored, () -> CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
        m2l_strategy=ConcatenatedFixedZM2L())),
    (:precomputed_y, () -> CUDARadixLifecycleOptions(; operator=FactoredRotationM2L(),
        m2l_strategy=PrecomputedFactoredYM2L())),
    (:dense, () -> CUDARadixLifecycleOptions(;
        m2l_strategy=DenseTranslationM2L(apply_chunk=64, build_chunk=8))),
)

@testset "CUDA hierarchical rigid M2L (task 027)" begin
    loaded = FastMultipole.load_cuda_radix_lifecycle!()
    if !loaded
        if _cuda_hier_required()
            error("FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 but CUDA radix lifecycle did " *
                  "not load: " * FastMultipole.cuda_radix_status())
        end
        # CUDA-absent hosts must still fail loudly on the device gate
        sys = generate_gravitational(1, 50)
        @test_throws Exception RadixFMMCache(sys; expansion_order=4, ell=3,
            device=true, policy=HierarchicalRigidStencil(4, 1e12; near_radius2=3))
    else
        @eval using CUDA
        P = 4
        bounds = (SVector(-0.1, -0.1, -0.1), 1.2)

        #--- occupancy, route, and direct-pair parity vs the 026 host oracle ---#

        for q in (12, 3), ell in (3, 4, 5)
            n = ell == 5 ? 4000 : 600
            sys = generate_gravitational(27000 + ell, n)
            ref = generate_gravitational(27000 + ell, n)
            policy = _hcu_policy(P, q, 0.6, ell, Float64, false; window_classes=8)
            hcache = RadixFMMCache(sys; expansion_order=P, ell, bounds, policy)
            dcache = RadixFMMCache(ref; expansion_order=P, ell, bounds, policy,
                device=true)
            hs, ds = hcache.state, dcache.state
            hctx, dctx = hs.interaction_list, ds.interaction_list
            @test dctx isa HCU_FM.DeviceHierarchicalM2LContext
            @test ds.counts.n_cells == hs.counts.n_cells
            @test ds.counts.n_nodes == hs.counts.n_nodes

            # per-level occupancy scatter
            @test Array(dctx.node_at) == hctx.occupancy.node_at
            @test dctx.nodes_per_level ==
                [hcache.level_offsets[L + 2] - hcache.level_offsets[L + 1]
                 for L in 0:ell]

            # direct pairs: leaf-only near set, elementwise host order
            @test ds.counts.n_direct == hs.counts.n_direct
            nd = hs.counts.n_direct
            @test Array(ds.direct_targets)[1:nd] == hs.direct_targets[1:nd]
            @test Array(ds.direct_sources)[1:nd] == hs.direct_sources[1:nd]

            # route windows: identical multiset AND elementwise emission order
            noffsets = length(hctx.tables.push_offsets)
            nonleaf_routes = 0
            leaf_routes = 0
            # ell = 5 sweeps ~4 x 218 windows per radius; sample them there and
            # compare every window exhaustively at the smaller depths
            stride = ell == 5 ? 8 : 1
            windows = collect(1:8:noffsets)
            for level in 2:ell, first in windows[1:stride:end]
                last = min(first + 7, noffsets)
                nh = HCU_FM.build_hierarchical_routes_window!(
                    hs.route_levels, hs.route_offsets, hs.route_targets,
                    hs.route_sources, hs.scratch.m2l_concat.route_class,
                    hctx, hs.grid, level, first, last)
                nd_ = HCU_FM.cuda_hierarchical_route_window!(ds, level, first, last)
                @test nd_ == nh
                nh == 0 && continue
                @test Array(view(ds.route_levels, 1:nh)) == hs.route_levels[1:nh]
                @test Array(view(ds.route_offsets, :, 1:nh)) ==
                    hs.route_offsets[:, 1:nh]
                @test Array(view(ds.route_sources, 1:nh)) == hs.route_sources[1:nh]
                @test Array(view(ds.route_targets, 1:nh)) == hs.route_targets[1:nh]
                @test Array(view(ds.scratch.m2l_concat.route_class, 1:nh)) ==
                    hs.scratch.m2l_concat.route_class[1:nh]
                level == ell ? (leaf_routes += nh) : (nonleaf_routes += nh)
            end
            @test leaf_routes > 0
            @test nonleaf_routes > 0

            # combined near/far coverage of every leaf pair is exactly once
            ell <= 4 && @test all(_hcu_coverage(dcache, hcache) .== 1)

            # empty window (a level with no occupied nodes cannot occur here, but a
            # partial final window must be valid)
            tail = (noffsets ÷ 8) * 8 + 1
            if tail <= noffsets
                @test HCU_FM.cuda_hierarchical_route_window!(ds, ell, tail, noffsets) >= 0
            end
        end

        #--- windowing invariance: K = 1, 8, and one whole-level window ---#

        let ell = 3
            outs = Matrix{Float64}[]
            for K in (1, 8, 1740)
                sys = generate_gravitational(27100, 800)
                cache = RadixFMMCache(sys; expansion_order=P, ell, bounds,
                    device=true,
                    policy=_hcu_policy(P, 12, 0.6, ell, Float64, false;
                        window_classes=K))
                fmm!(sys, cache; scalar_potential=true, gradient=true)
                push!(outs, copy(sys.potential))
            end
            for out in outs[2:end]
                @test out[1, :] ≈ outs[1][1, :] atol=2e-12 rtol=2e-12
                @test out[5:7, :] ≈ outs[1][5:7, :] atol=2e-11 rtol=2e-12
            end
        end

        #--- device vs host-026 lifecycle parity across the strategy matrix ---#

        for q in (12, 3), TF in (Float64, Float32), LH in (false, true),
                ell in (3, 4)
            (TF === Float32 && ell == 4) && continue
            base = Matrix{TF}[]
            for (name, mkopts) in _HCU_STRATEGIES
                hsys = generate_gravitational(27200, 400)
                dsys = generate_gravitational(27200, 400)
                policy = _hcu_policy(P, q, 0.6, ell, TF, LH; window_classes=8)
                opts = mkopts()
                opts = CUDARadixLifecycleOptions(; precision=TF,
                    operator=opts.operator, m2m_strategy=opts.m2m_strategy,
                    m2l_strategy=opts.m2l_strategy)
                hcache = RadixFMMCache(hsys; expansion_order=P, ell, bounds,
                    policy, options=opts, lamb_helmholtz=LH)
                dcache = RadixFMMCache(dsys; expansion_order=P, ell, bounds,
                    policy, options=opts, device=true, lamb_helmholtz=LH)
                fmm!(hsys, hcache; scalar_potential=!LH, gradient=true)
                fmm!(dsys, dcache; scalar_potential=!LH, gradient=true)
                tol = TF === Float32 ? 2e-3 : 1e-9
                @test maximum(abs.(dsys.potential[5:7, :] .-
                    hsys.potential[5:7, :])) < tol
                push!(base, copy(dsys.potential))
                @test dcache.state.counts.n_routes ==
                    hcache.state.interaction_list.total_routes
            end
            # every strategy agrees with the concatenated reference on device
            for out in base[2:end]
                atol = TF === Float32 ? 1e-3 : 3e-11
                @test out[5:7, :] ≈ base[1][5:7, :] atol=atol rtol=1e-4
            end
        end

        #--- seeded nonzero chi: exercises the asymmetric dense LH level diagonals ---#

        let ell = 3, q = 12
            rng = MersenneTwister(0x27c)
            seed = nothing
            outs = Tuple{Matrix{Float64},Matrix{Float64}}[]
            for (name, mkopts) in _HCU_STRATEGIES
                sys = generate_gravitational(27300, 300)
                cache = RadixFMMCache(sys; expansion_order=P, ell, bounds,
                    lamb_helmholtz=true, options=mkopts(), device=true,
                    policy=_hcu_policy(P, q, 0.6, ell, Float64, true;
                        window_classes=8))
                mult = cache.state.multipoles
                if seed === nothing
                    host_seed = HCU_FM.FlatCoefficientBuffer(Float64,
                        mult.basis_info, size(mult.phi, 2))
                    _hcu_random_physical!(host_seed, rng)
                    seed = (copy(host_seed.phi), copy(host_seed.chi))
                end
                copyto!(mult.phi, seed[1])
                copyto!(mult.chi, seed[2])
                HCU_FM._launch_cuda_resident_m2l!(cache.state)
                push!(outs, (Array(cache.state.locals.phi),
                    Array(cache.state.locals.chi)))
            end
            for out in outs[2:end]
                @test out[1] ≈ outs[1][1] rtol=1e-7 atol=1e-9
                @test out[2] ≈ outs[1][2] rtol=1e-7 atol=1e-9
            end
        end

        #--- hierarchical dense storage: one operator per union offset ---#

        for ell in (3, 4)
            sys = generate_gravitational(27400, 300)
            cache = RadixFMMCache(sys; expansion_order=P, ell, bounds, device=true,
                options=CUDARadixLifecycleOptions(;
                    m2l_strategy=DenseTranslationM2L(apply_chunk=64, build_chunk=8)),
                policy=_hcu_policy(P, 12, 0.6, ell, Float64, false; window_classes=8))
            hctx = cache.state.interaction_list
            plan = cache.state.scratch.m2l_concat
            noffsets = length(hctx.tables.push_offsets)
            # independent of ell: the (level, offset) duplication is replaced by the
            # task-025 level scaling
            @test size(plan.operators, 3) == noffsets
            @test plan.nclasses == noffsets
            @test size(hctx.source_scale) == (plan.ndof, ell - 1)
            @test size(hctx.target_scale) == (plan.ndof, ell - 1)
            # only the dense strategy carries level scales
            csys = generate_gravitational(27400, 300)
            ccache = RadixFMMCache(csys; expansion_order=P, ell, bounds, device=true,
                policy=_hcu_policy(P, 12, 0.6, ell, Float64, false; window_classes=8))
            @test isempty(ccache.state.interaction_list.source_scale)
            @test isempty(ccache.state.interaction_list.target_scale)
        end

        #--- unfused reference drivers must match the production drivers ---#

        let ell = 3
            for (name, mkopts, flag) in (
                    (:dense, _HCU_STRATEGIES[4][2], HCU_FM.DENSE_CUDA_FUSED),
                    (:precomputed_y, _HCU_STRATEGIES[3][2],
                        HCU_FM.PRECOMPUTED_CUDA_WHOLE_PASS))
                sys_a = generate_gravitational(27500, 400)
                sys_b = generate_gravitational(27500, 400)
                policy = _hcu_policy(P, 12, 0.6, ell, Float64, false; window_classes=8)
                ca = RadixFMMCache(sys_a; expansion_order=P, ell, bounds, device=true,
                    options=mkopts(), policy)
                fmm!(sys_a, ca; scalar_potential=true, gradient=true)
                old = flag[]
                try
                    flag[] = !old
                    cb = RadixFMMCache(sys_b; expansion_order=P, ell, bounds,
                        device=true, options=mkopts(), policy)
                    fmm!(sys_b, cb; scalar_potential=true, gradient=true)
                finally
                    flag[] = old
                end
                @test sys_b.potential[1, :] ≈ sys_a.potential[1, :] atol=2e-11 rtol=1e-10
                @test sys_b.potential[5:7, :] ≈ sys_a.potential[5:7, :] atol=2e-10 rtol=1e-10
            end
        end

        #--- resident contract: counters, identity, steady-state allocation ---#

        let ell = 3
            sys = generate_gravitational(27600, 1200)
            cache = RadixFMMCache(sys; expansion_order=P, ell, bounds, device=true,
                policy=_hcu_policy(P, 12, 0.6, ell, Float64, false; window_classes=8))
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            counters = cache.state.counters
            base_route = counters.route_uploads
            base_operator = counters.operator_uploads
            base_metadata = counters.metadata_downloads
            hctx = cache.state.interaction_list
            ids = (objectid(hctx.node_at), objectid(hctx.route_flags),
                objectid(hctx.route_prefix), objectid(hctx.d_push_offsets),
                objectid(hctx.d_class_of), objectid(hctx.d_near_offsets),
                objectid(hctx.source_scale), objectid(cache.state.route_sources),
                objectid(cache.state.route_targets),
                objectid(cache.state.direct_targets),
                objectid(cache.state.multipoles.phi),
                objectid(cache.state.locals.phi),
                objectid(cache.state.scratch.m2l_concat.route_class))
            rng = MersenneTwister(27600)
            for step in 1:2
                for i in eachindex(sys.bodies)
                    b = sys.bodies[i]
                    pos = clamp.(b.position .+
                        0.02 .* (rand(rng, SVector{3,Float64}) .- 0.5), -0.05, 1.05)
                    sys.bodies[i] = Body(pos, b.radius, b.strength)
                end
                fmm!(sys, cache; scalar_potential=true, gradient=true)
            end
            @test counters.route_uploads == base_route
            @test counters.operator_uploads == base_operator
            @test counters.expansion_host_copies == 0
            @test counters.metadata_downloads == base_metadata + 6
            @test ids == (objectid(hctx.node_at), objectid(hctx.route_flags),
                objectid(hctx.route_prefix), objectid(hctx.d_push_offsets),
                objectid(hctx.d_class_of), objectid(hctx.d_near_offsets),
                objectid(hctx.source_scale), objectid(cache.state.route_sources),
                objectid(cache.state.route_targets),
                objectid(cache.state.direct_targets),
                objectid(cache.state.multipoles.phi),
                objectid(cache.state.locals.phi),
                objectid(cache.state.scratch.m2l_concat.route_class))
            # Warmed steady-state device allocation (runtime @eval: CUDA.@allocated
            # cannot be macroexpanded when this file is loaded without CUDA).
            # Unlike the flat path, the hierarchical M2L stage owns the per-window
            # flag/scan/compact, so it inherits CUDA.jl's pool-served `accumulate!`
            # scan scratch — the same already-documented behavior the flat path
            # shows in its update stage. The contract that must hold is that the
            # amount is *stable* across repeated warmed calls: nothing grows, and
            # no persistent array is reallocated.
            @eval CUDA.@allocated fmm!($sys, $cache; scalar_potential=true,
                gradient=true)
            step_a = @eval CUDA.@allocated fmm!($sys, $cache; scalar_potential=true,
                gradient=true)
            step_b = @eval CUDA.@allocated fmm!($sys, $cache; scalar_potential=true,
                gradient=true)
            @test step_b == step_a
            @eval CUDA.@allocated HCU_FM._launch_cuda_resident_m2l!($(cache).state)
            m2l_a = @eval CUDA.@allocated HCU_FM._launch_cuda_resident_m2l!(
                $(cache).state)
            m2l_b = @eval CUDA.@allocated HCU_FM._launch_cuda_resident_m2l!(
                $(cache).state)
            @test m2l_b == m2l_a

            # telemetry
            @test hctx.total_routes == cache.state.counts.n_routes
            @test sum(hctx.routes_per_level) == hctx.total_routes
            @test count(>(0), hctx.routes_per_level) > 1
            hctx.profile_stages = true
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            @test hctx.update_stage_ns[1] > 0
            @test hctx.update_stage_ns[2] > 0
            @test hctx.update_stage_ns[3] > 0
            @test hctx.update_stage_ns[4] > 0
            @test all(>(0), hctx.m2l_level_ns[3:(ell + 1)])
            hctx.profile_stages = false
        end

        #--- the task-027 production default, on device, with no policy passed ---#
        # Every other block here passes an explicit policy, and the flat device
        # regression suite is pinned to flat, so without this the shipped default
        # would never actually execute on the GPU.
        let ell = 4
            sys = generate_gravitational(27650, 3000)
            ref = generate_gravitational(27650, 3000)
            cache = RadixFMMCache(sys; expansion_order=P, ell, bounds, device=true)
            @test cache.policy isa HierarchicalRigidStencil
            @test cache.policy.near_radius2 == 12
            @test cache.policy.window_classes == HCU_FM.RADIX_DEVICE_WINDOW_CLASSES
            @test cache.state.interaction_list isa HCU_FM.DeviceHierarchicalM2LContext
            fmm!(sys, cache; scalar_potential=true, gradient=true)
            HCU_FM.direct!(ref; scalar_potential=true, gradient=true)
            # the derived tolerance reproduces the rigid near set exactly, so the
            # accuracy is the theta=0.5 stencil's, not the flat classifier's
            @test maximum(abs.(sys.potential[1, :] .- ref.potential[1, :])) < 1e-5
            @test maximum(abs.(sys.potential[5:7, :] .- ref.potential[5:7, :])) < 1e-3
            # host default agrees with the device default at matched policy
            hsys = generate_gravitational(27650, 3000)
            hcache = RadixFMMCache(hsys; expansion_order=P, ell, bounds,
                window_classes=HCU_FM.RADIX_DEVICE_WINDOW_CLASSES)
            @test hcache.policy isa HierarchicalRigidStencil
            fmm!(hsys, hcache; scalar_potential=true, gradient=true)
            @test maximum(abs.(sys.potential[5:7, :] .- hsys.potential[5:7, :])) < 1e-9
        end

        #--- flat device path is unchanged ---#

        let ell = 3
            sys = generate_gravitational(27700, 500)
            flat = RadixFMMCache(sys; expansion_order=P, ell, bounds, device=true,
                stencil_epsilon=1e-4)
            @test flat.policy isa ConstantPAnalyticStencil
            @test flat.state.interaction_list === nothing
            @test flat.state.counts.n_routes > 0
        end

        #--- construction gates ---#

        # incompatible epsilon: the analytic classifier must reproduce the rigid
        # near set exactly
        bad = generate_gravitational(27800, 100)
        @test_throws ArgumentError RadixFMMCache(bad; expansion_order=P, ell=3,
            bounds, device=true, policy=HierarchicalRigidStencil(P, 1e-4))
        # the device path has no Morton fallback for the per-level lookup
        @test_throws ArgumentError RadixFMMCache(bad; expansion_order=P, ell=3,
            bounds, device=true,
            policy=HierarchicalRigidStencil(
                ConstantPStencilConfig(P, _hcu_epsilon(P, 12, 0.6, 3, Float64, false));
                dense_occupancy_max_bytes=0))
    end
end
