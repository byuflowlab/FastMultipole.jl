using FastMultipole
using FastMultipole.StaticArrays
using Random
using Test

if !isdefined(@__MODULE__, :generate_gravitational)
    include("gravitational.jl")
end

const TRIM_FM = FastMultipole

# One body at the center of each chosen leaf cell of a unit-leaf-width grid
# anchored at the origin, so occupancy is exact and per-cell.
function _trim_system(coords::Vector{SVector{3,Int}})
    n = length(coords)
    bodies = zeros(8, n)
    for (i, c) in enumerate(coords)
        bodies[1:3, i] .= Float64.(c) .+ 0.5
        bodies[4, i] = 1e-3
        bodies[5, i] = 1.0 / n
    end
    return Gravitational(bodies)
end

function _trim_sparse_coords(rng, ell_axes::SVector{3,Int}, frac)
    N = (1 << ell_axes[1], 1 << ell_axes[2], 1 << ell_axes[3])
    coords = SVector{3,Int}[]
    for z in 0:(N[3] - 1), y in 0:(N[2] - 1), x in 0:(N[1] - 1)
        rand(rng) < frac && push!(coords, SVector(x, y, z))
    end
    # always occupy the two extreme corners so the flat-top level is exercised
    push!(coords, SVector(0, 0, 0))
    push!(coords, SVector(N[1] - 1, N[2] - 1, N[3] - 1))
    return sort!(unique(coords); by=c -> (c[3], c[2], c[1]))
end

# Exhaustive exact-once coverage audit (task 037 stage 3, the enforcement of
# the design record's theory verdict): count how many times each ordered
# occupied-leaf pair is covered by direct ∪ (M2L at some active level) and
# return the nc x nc hit matrix — every entry must be exactly 1.
function _trim_coverage_hits(cache)
    state = cache.state
    grid = state.grid
    ctx = state.interaction_list
    ell = grid.ell
    nc = state.counts.n_cells
    hits = zeros(Int, nc, nc)
    for i in 1:state.counts.n_direct
        hits[state.direct_targets[i], state.direct_sources[i]] += 1
    end
    leaf_base = cache.level_offsets[ell + 1]
    coords = [SVector{3,Int}(grid.node_coords[:, leaf_base + c]) for c in 1:nc]
    # leaf descendants of every (level, node coordinate)
    desc = Dict{Tuple{Int,SVector{3,Int}},Vector{Int}}()
    for (ci, c) in enumerate(coords), L in 0:ell
        push!(get!(() -> Int[], desc, (L, c .>> (ell - L))), ci)
    end
    noffsets = length(ctx.tables.push_offsets)
    for L in ctx.first_m2l_level:ell,
            first_offset in 1:ctx.window_classes:noffsets
        last_offset = min(first_offset + ctx.window_classes - 1, noffsets)
        nr = TRIM_FM.build_hierarchical_routes_window!(state.route_levels,
            state.route_offsets, state.route_targets, state.route_sources,
            nothing, ctx, grid, L, first_offset, last_offset)
        for r in 1:nr
            sc = SVector{3,Int}(grid.node_coords[:, state.route_sources[r]])
            tc = SVector{3,Int}(grid.node_coords[:, state.route_targets[r]])
            for t in desc[(L, tc)], s in desc[(L, sc)]
                hits[t, s] += 1
            end
        end
    end
    return hits
end

@testset "radix active-level trimming (task 037 stage 3)" begin

    #--- (a) root-level resolution and the flat-top cap guard ---#

    @test TRIM_FM._radix_root_level(SVector(4, 4, 4), 4, 5) == (1, 1)
    @test TRIM_FM._radix_root_level(SVector(3, 3, 3), 3, 5) == (1, 1)
    @test TRIM_FM._radix_root_level(SVector(4, 2, 2), 4, 5) == (2, 1)
    @test TRIM_FM._radix_root_level(SVector(3, 3, 1), 3, 5) == (2, 1)
    @test TRIM_FM._radix_root_level(SVector(4, 4, 2), 4, 5) == (2, 1)
    @test TRIM_FM._radix_root_level(SVector(6, 2, 2), 6, 5) == (4, 1)
    # cap guard: (8,8,2) at R=6 has 127^2 - near ≈ 16k flat-top classes,
    # over the 4096 cap; R lowers until the count fits
    R, L_allnear = TRIM_FM._radix_root_level(SVector(8, 8, 2), 8, 5)
    @test L_allnear == 1
    @test R < 6
    @test TRIM_FM._radix_flat_top_count(SVector(8, 8, 2), 8, R, 5) <=
        TRIM_FM.RADIX_FLAT_TOP_CLASS_CAP
    @test TRIM_FM._radix_flat_top_count(SVector(8, 8, 2), 8, R + 1, 5) >
        TRIM_FM.RADIX_FLAT_TOP_CLASS_CAP
    # root counts saturate axes and reproduce cubic counts on equal depths
    @test TRIM_FM._radix_root_counts(SVector(4, 2, 2), 4, 2) == SVector(4, 1, 1)
    @test TRIM_FM._radix_root_counts(SVector(3, 3, 1), 3, 2) == SVector(4, 4, 1)
    @test TRIM_FM._radix_root_counts(SVector(4, 4, 4), 4, 2) == SVector(4, 4, 4)

    #--- (b) schedule anchoring over the active levels ---#

    let eps0 = rigid_stencil_epsilon(4, 32.0, 6, 5)
        pol(qs) = HierarchicalRigidStencil(ConstantPStencilConfig(4, eps0);
            near_radius2=5, level_radii2=qs)
        axes622 = SVector(6, 2, 2)
        # active levels 4:6 (flat-top at R=4): both anchorings accepted
        _, _, qs_active, R6, f6 = TRIM_FM._hierarchical_scheduled_tables(
            pol((6, 5, 5)), 6, axes622)
        @test (R6, f6) == (4, 4)
        @test qs_active == [6, 5, 5]
        _, _, qs_legacy, _, _ = TRIM_FM._hierarchical_scheduled_tables(
            pol((6, 6, 6, 5, 5)), 6, axes622)
        @test qs_legacy == [6, 5, 5]        # sliced to levels 4:6
        @test_throws ArgumentError TRIM_FM._hierarchical_scheduled_tables(
            pol((6, 5, 5, 5)), 6, axes622)  # neither anchoring
        # cubic: both anchorings coincide (identity slice)
        _, _, qs_cubic, Rc, fc = TRIM_FM._hierarchical_scheduled_tables(
            pol((6, 5, 5, 5, 5)), 6, SVector(6, 6, 6))
        @test (Rc, fc) == (1, 2)
        @test qs_cubic == [6, 5, 5, 5, 5]
    end

    #--- (c) flat-top table shape ---#

    let ft = TRIM_FM._rigid_flat_top_tables(6, SVector(4, 1, 1))
        @test ft.push_offsets == [SVector(-3, 0, 0), SVector(3, 0, 0)]
        # all 8 phases admit every flat-top offset
        @test all(!iszero, ft.class_of)
    end

    #--- (d) exhaustive exact-once coverage + accuracy on small grids ---#

    rng = MersenneTwister(0x037)
    cases = (
        (SVector(4, 2, 2), 4, (16.0, 4.0, 4.0), 0.18),
        (SVector(3, 3, 1), 3, (8.0, 8.0, 2.0), 0.30),
        (SVector(4, 4, 2), 4, (16.0, 16.0, 4.0), 0.06),
        (SVector(3, 3, 3), 3, (8.0, 8.0, 8.0), 0.25),   # cubic degeneration
    )
    origin = SVector(0.0, 0.0, 0.0)
    for (ell_axes, ell, extent, frac) in cases,
            schedule in (nothing, ())           # default schedule and uniform q
        coords = _trim_sparse_coords(rng, ell_axes, frac)
        sys = _trim_system(coords)
        kwargs = schedule === nothing ? (;) : (; near_radius2=5)
        cache = RadixFMMCache(sys; expansion_order=3, ell,
            bounds=(origin, extent), kwargs...)
        @test cache.ell_axes == ell_axes
        ctx = cache.state.interaction_list
        cubic = ell_axes == SVector(ell, ell, ell)
        @test cache.root_level == (cubic ? 1 : 2)
        @test ctx.first_m2l_level == 2
        # trimmed levels carry no nodes and no routes
        @test all(cache.level_offsets[1:cache.root_level + 1] .== 0)
        @test cache.level_offsets[cache.root_level + 2] > 0
        fmm!(sys, cache; scalar_potential=true, gradient=true)
        @test all(ctx.routes_per_level[1:ctx.first_m2l_level] .== 0)
        @test sum(ctx.routes_per_level) == cache.state.counts.n_routes
        hits = _trim_coverage_hits(cache)
        @test all(hits .== 1)
        # accuracy against direct summation (P=4 truncation gates, task 023)
        ref = _trim_system(coords)
        FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
        @test maximum(abs.(sys.potential[1, :] .- ref.potential[1, :])) < 2e-3
        @test maximum(abs.(sys.potential[5:7, :] .- ref.potential[5:7, :])) < 2e-1
    end

    #--- (e) trimmed hierarchical vs flat analytic oracle ---#

    for (ell_axes, ell, extent, frac) in cases[1:3]
        coords = _trim_sparse_coords(rng, ell_axes, frac)
        h0 = maximum(extent) / 2
        eps0 = rigid_stencil_epsilon(4, h0, ell, 5)
        a = _trim_system(coords)
        b = _trim_system(coords)
        ch = RadixFMMCache(a; expansion_order=3, ell, bounds=(origin, extent))
        cf = RadixFMMCache(b; expansion_order=3, ell, bounds=(origin, extent),
            stencil_epsilon=eps0)
        @test cf.root_level == 0            # flat policy stays untrimmed
        fmm!(a, ch; scalar_potential=true, gradient=true)
        fmm!(b, cf; scalar_potential=true, gradient=true)
        # both engines truncate at the same P; the gap is bounded by the sum
        # of the two truncation errors
        @test maximum(abs.(a.potential[1, :] .- b.potential[1, :])) < 4e-3
        @test maximum(abs.(a.potential[5:7, :] .- b.potential[5:7, :])) < 4e-1
    end

    #--- (e2) cross-strategy parity on a trimmed rectangular cache ---#
    # concat, precomputed-y, and dense all consume the re-anchored
    # (L - first_m2l_level) * noffsets class numbering; they must agree to
    # roundoff on identical bodies (Float64, P=4 literature order).

    let (ell_axes, ell, extent, frac) = cases[1]
        coords = _trim_sparse_coords(rng, ell_axes, frac)
        outs = Matrix{Float64}[]
        for strategy in (FastMultipole.ConcatenatedFixedZM2L(),
                FastMultipole.PrecomputedFactoredYM2L(),
                FastMultipole.DenseTranslationM2L())
            operator = strategy isa FastMultipole.PrecomputedFactoredYM2L ?
                FastMultipole.FactoredRotationM2L() :
                FastMultipole.MaterializedYRotationM2L()
            opts = CUDARadixLifecycleOptions(; precision=Float64,
                m2l_strategy=strategy, operator)
            s = _trim_system(coords)
            c = RadixFMMCache(s; expansion_order=3, ell,
                bounds=(origin, extent), options=opts)
            @test c.root_level == 2
            fmm!(s, c; scalar_potential=true, gradient=true)
            push!(outs, copy(s.potential))
        end
        for k in 2:length(outs)
            @test maximum(abs.(outs[k] .- outs[1])) < 1e-11
        end
    end

    #--- (f) multi-root tree edges and zero-allocation refresh ---#

    let (ell_axes, ell, extent, frac) = cases[1]
        coords = _trim_sparse_coords(rng, ell_axes, frac)
        sys = _trim_system(coords)
        cache = RadixFMMCache(sys; expansion_order=3, ell,
            bounds=(origin, extent))
        state = cache.state
        n_root = cache.level_offsets[cache.root_level + 2]
        @test n_root >= 1
        @test length(state.m2m_parent_routes) ==
            state.counts.n_nodes - n_root
        @test all(state.grid.parent_index[1:n_root] .== 0)
        @test all(state.grid.parent_index[(n_root + 1):state.counts.n_nodes] .> 0)
        # stage groups cover levels root_level:ell only
        ws = state.scratch
        @test length(ws.m2m_groups) == ell - cache.root_level
        @test length(ws.l2l_groups) == ell - cache.root_level
        # bounded-allocation recurring refresh (the strict zero-allocation
        # contract is device-side; the host suites gate the warmed step at the
        # same bound — see radix_fmm_timestepping_test.jl)
        TRIM_FM.update_radix_state!(cache, (sys,))
        TRIM_FM.update_radix_state!(cache, (sys,))
        @test (@allocated TRIM_FM.update_radix_state!(cache, (sys,))) <= 64 * 1024
    end

    #--- (g) zero-M2L degenerate geometry runs pure direct (task 052c) ---#
    # On a (1,1,2) grid with q = 12 every root offset lies inside the near
    # ball (max |o|^2 = 1 + 1 + 9 = 11), so the hierarchy has zero M2L levels
    # (first_m2l_level == ell + 1). The cache must degenerate to direct-only
    # evaluation — the efficient answer for a field this small — not throw.

    let ell_axes = SVector(1, 1, 2), ell = 2, extent = (2.0, 2.0, 4.0)
        @test TRIM_FM._radix_root_level(ell_axes, ell, 12) == (2, 2)
        coords = _trim_sparse_coords(rng, ell_axes, 0.9)
        sys = _trim_system(coords)
        cache = RadixFMMCache(sys; expansion_order=3, ell,
            bounds=(origin, extent), near_radius2=12)
        ctx = cache.state.interaction_list
        @test ctx.first_m2l_level == ell + 1
        @test isempty(cache.accepted_offsets)
        @test isempty(ctx.tables.push_offsets)
        fmm!(sys, cache; scalar_potential=true, gradient=true)
        @test cache.state.counts.n_routes == 0
        # direct covers every ordered leaf pair exactly once — the exact-once
        # audit degenerates to the direct list alone
        hits = _trim_coverage_hits(cache)
        @test all(hits .== 1)
        # with no expansions anywhere the answer is the direct sum: the gap to
        # direct! sits at the kernel softening-convention level (~5e-9 with the
        # 1e-3 body radius here), orders below the ~2e-3 truncation-level
        # tolerances of section (d)
        ref = _trim_system(coords)
        FastMultipole.direct!(ref; scalar_potential=true, gradient=true)
        @test maximum(abs.(sys.potential[1, :] .- ref.potential[1, :])) < 1e-7
        @test maximum(abs.(sys.potential[5:7, :] .- ref.potential[5:7, :])) < 1e-6
        # an explicit level schedule cannot anchor to zero active levels
        @test_throws ArgumentError RadixFMMCache(_trim_system(coords);
            expansion_order=3, ell, bounds=(origin, extent), near_radius2=12,
            level_radii2=(12,))
    end
end
