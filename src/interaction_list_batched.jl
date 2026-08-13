const RADIX_PRODUCTION_NORMALIZATION = 1 / (4 * π)

@inline _stencil_production_factor(::ConstantPStencilConfig{<:Any,<:Any,:analytic}) = 1
@inline _stencil_production_factor(::ConstantPStencilConfig{<:Any,<:Any,:production}) = RADIX_PRODUCTION_NORMALIZATION

@inline _stencil_lamb_helmholtz(::ConstantPStencilConfig{<:Any,LH,<:Any}) where {LH} = LH

function constant_p_stencil_bound(P::Integer, offset::SVector{3,<:Integer}, source_strength, cell_half_width)
    TF = promote_type(typeof(source_strength), typeof(cell_half_width), Float64)
    dnorm = norm(SVector{3,TF}(offset[1], offset[2], offset[3]))
    c = 2 * dnorm / sqrt(TF(3))
    c > TF(2) || return TF(Inf)
    rho = TF(cell_half_width) * sqrt(TF(3))
    bound = 2 * TF(source_strength) / (rho * (c - TF(2))) * (1 / (c - TF(1)))^(Int(P) + 1)
    return isfinite(bound) ? bound : TF(Inf)
end

function constant_p_stencil_bound(grid::RadixGrid, config::ConstantPStencilConfig,
        offset::SVector{3,<:Integer})
    B_phi = constant_p_stencil_bound(
        config.P_phi, offset, config.source_strength, radix_cell_half_width(grid),
    )
    if _stencil_lamb_helmholtz(config)
        B_chi = constant_p_stencil_bound(
            config.P_phi + 1, offset, config.chi_strength, radix_cell_half_width(grid),
        )
        R = norm(radix_displacement(grid, offset))
        return _stencil_production_factor(config) * (B_phi + (1 + 2R) * B_chi)
    else
        return _stencil_production_factor(config) * B_phi
    end
end

@inline constant_p_stencil_accepts(grid::RadixGrid, config::ConstantPStencilConfig,
        offset::SVector{3,<:Integer}) =
    constant_p_stencil_bound(grid, config, offset) <= config.epsilon

function accepted_radix_stencil(grid::RadixGrid, config::ConstantPStencilConfig)
    G = radix_resolution(grid)
    offsets = SVector{3,Int}[]
    for k in -(G - 1):(G - 1), j in -(G - 1):(G - 1), i in -(G - 1):(G - 1)
        offset = SVector{3,Int}(i, j, k)
        constant_p_stencil_accepts(grid, config, offset) && push!(offsets, offset)
    end
    return offsets
end

"""
    radix_implicit_stencil(grid, config)

Build the [`RadixImplicitStencil`](@ref) for a constant-`P` policy: one pass over the
bounded `(2G-1)^3` offset box classifies every offset as accepted (M2L) or rejected
(direct near/self complement), and one O(cells) pass fills the dense coord -> cell
occupancy map. Together these determine pair membership procedurally — membership
for any offset class is `cell_at[coord(target) - offset] != 0` — without a cells^2
complement scan. Callers that request a [`RadixInteractionList`](@ref) still
materialize the matching target/source routes and direct pairs.
"""
function radix_implicit_stencil(grid::RadixGrid, config::ConstantPStencilConfig)
    G = radix_resolution(grid)
    accepted, rejected = classify_radix_stencil_offsets(grid, config)
    cell_at = zeros(Int32, G, G, G)
    refresh_cell_at!(cell_at, grid.cell_keys, length(grid.cell_keys), grid.ell)
    return RadixImplicitStencil(accepted, rejected, cell_at)
end

"""
    classify_radix_stencil_offsets(grid, config)

Classify every offset of the bounded `(2G-1)^3` box as accepted (M2L) or rejected
(direct near/self complement) for a constant-`P` policy. The classification depends
only on the grid geometry (`h0`, `ell`) and the config — never on occupancy — so a
fixed-box cache can compute it once at construction.
"""
classify_radix_stencil_offsets(grid::RadixGrid, config::ConstantPStencilConfig) =
    classify_radix_stencil_offsets(grid.h0, grid.ell, config)

function classify_radix_stencil_offsets(h0::Real, ell::Integer,
        config::ConstantPStencilConfig)
    G = 1 << Int(ell)
    accepted = SVector{3,Int}[]
    rejected = SVector{3,Int}[]
    for k in -(G - 1):(G - 1), j in -(G - 1):(G - 1), i in -(G - 1):(G - 1)
        offset = SVector{3,Int}(i, j, k)
        bound = constant_p_stencil_bound(h0, ell, config, offset)
        bound <= config.epsilon ? push!(accepted, offset) : push!(rejected, offset)
    end
    sort!(accepted; by=o -> (o[3], o[2], o[1]))
    return accepted, rejected
end

@inline _rigid_offset_order(o) = (o[3], o[2], o[1])
@inline function _rigid_orbit_key(o)
    a = abs(Int(o[1]))
    b = abs(Int(o[2]))
    c = abs(Int(o[3]))
    hi = max(a, b, c)
    lo = min(a, b, c)
    return (hi, a + b + c - hi - lo, lo)
end
@inline _rigid_near(o, radius2::Int) =
    o[1] * o[1] + o[2] * o[2] + o[3] * o[3] <= radius2
@inline _rigid_phase_index(c1::Integer, c2::Integer, c3::Integer) =
    1 + (Int(c1) & 1) + 2 * (Int(c2) & 1) + 4 * (Int(c3) & 1)

"""
    rigid_stencil_epsilon(P_phi, h0, ell, near_radius2; lamb_helmholtz=false, TF=Float64)

Tolerance whose analytic constant-`P` classifier rejects **exactly** the rigid near
set `{o : |o|^2 <= near_radius2}` at this box and depth — the accuracy contract
[`HierarchicalRigidStencil`](@ref) verifies at construction.

The bound depends on the offset only through `|o|`, and it is monotonically
decreasing, so any tolerance strictly between the bound at the farthest near offset
and the bound at the nearest far offset separates the two sets exactly. For
`near_radius2 = 3` the bound is intrinsically infinite inside the near set (the
classic FMM case, `c <= 2` in the Gumerov bound), so any tolerance at or above the
nearest far offset's bound works.
"""
function rigid_stencil_epsilon(P_phi::Integer, h0::Real, ell::Integer,
        near_radius2::Integer; lamb_helmholtz::Bool=false, TF::Type=Float64)
    q = _validate_rigid_near_radius2(near_radius2, "rigid stencil")
    probe = ConstantPStencilConfig(P_phi, one(TF); lamb_helmholtz)
    # Enumerate representatives of the farthest occupied near shell and the
    # nearest nonempty far shell.  Not every integer is a sum of three squares
    # (q=7 is redundant with q=6), hence this cannot safely assume q + 1.
    extent = isqrt(q)
    near_shell = SVector{3,Int}[]
    for z in -extent:extent, y in -extent:extent, x in -extent:extent
        x*x + y*y + z*z == q && push!(near_shell, SVector(x, y, z))
    end
    isempty(near_shell) && error("supported rigid radius q=$q has no lattice shell")
    far_q = q + 1
    far_shell = SVector{3,Int}[]
    while isempty(far_shell)
        far_extent = isqrt(far_q)
        for z in -far_extent:far_extent, y in -far_extent:far_extent,
                x in -far_extent:far_extent
            x*x + y*y + z*z == far_q && push!(far_shell, SVector(x, y, z))
        end
        isempty(far_shell) && (far_q += 1)
    end
    near_far = (first(near_shell), first(far_shell))
    upper = constant_p_stencil_bound(TF(h0), ell, probe, near_far[1])
    lower = constant_p_stencil_bound(TF(h0), ell, probe, near_far[2])
    isfinite(lower) || throw(ArgumentError(
        "no finite constant-P tolerance separates the rigid near set at " *
        "P_phi=$P_phi, ell=$ell, near_radius2=$near_radius2"))
    return isfinite(upper) ? (upper + lower) / 2 : 2 * lower
end

"""
    RigidHierarchicalTables(near_radius2)

Enumerate the level-invariant source-major V-list from task 025.  This is
construction-time work and contains no occupancy-dependent state.
"""
function RigidHierarchicalTables(near_radius2::Integer)
    q = _validate_rigid_near_radius2(near_radius2, "rigid hierarchical")
    near_extent = isqrt(q)
    near_offsets = SVector{3,Int}[]
    for z in -near_extent:near_extent, y in -near_extent:near_extent,
            x in -near_extent:near_extent
        o = SVector{3,Int}(x, y, z)
        _rigid_near(o, q) && push!(near_offsets, o)
    end
    sort!(near_offsets; by=_rigid_offset_order)

    # If a parent offset is near, each child coordinate is bounded by twice
    # the parent extent plus its phase bit.
    extent = 2isqrt(q) + 1
    by_phase = [SVector{3,Int}[] for _ in 1:8]
    union_offsets = Set{SVector{3,Int}}()
    for phase in 0:7
        ux = phase & 1
        uy = (phase >> 1) & 1
        uz = (phase >> 2) & 1
        phase_offsets = by_phase[phase + 1]
        for z in -extent:extent, y in -extent:extent, x in -extent:extent
            o = SVector{3,Int}(x, y, z)
            _rigid_near(o, q) && continue
            parent = SVector{3,Int}(
                fld(ux + x, 2), fld(uy + y, 2), fld(uz + z, 2))
            _rigid_near(parent, q) || continue
            push!(phase_offsets, o)
            push!(union_offsets, o)
        end
        sort!(phase_offsets; by=_rigid_offset_order)
    end
    push_offsets = sort!(collect(union_offsets); by=_rigid_offset_order)
    offset_id = Dict(o => Int32(i) for (i, o) in enumerate(push_offsets))
    phase_index = Int32[]
    starts = Vector{Int}(undef, 9)
    class_of = zeros(Int32, 8, length(push_offsets))
    for phase in 1:8
        starts[phase] = length(phase_index) + 1
        for o in by_phase[phase]
            k = offset_id[o]
            push!(phase_index, k)
            class_of[phase, k] = k
        end
    end
    starts[9] = length(phase_index) + 1
    return RigidHierarchicalTables(near_offsets, push_offsets, Tuple(starts),
        phase_index, class_of)
end

# Scheduled transition table: children outside q_child are emitted when their
# parent lies inside q_parent. For q_parent == q_child this is exactly the
# production fixed-radius V-list. Keeping this constructor internal avoids a new
# public geometry surface while making the exact-once transition explicit.
function _rigid_transition_tables(q_parent::Integer, q_child::Integer)
    qp = _validate_rigid_near_radius2(q_parent, "rigid parent transition")
    qc = _validate_rigid_near_radius2(q_child, "rigid child transition")
    qc <= qp || throw(ArgumentError(
        "rigid transition requires q_child <= q_parent; got $qc > $qp"))
    child_base = RigidHierarchicalTables(qc)
    extent = 2isqrt(qp) + 1
    by_phase = [SVector{3,Int}[] for _ in 1:8]
    union_offsets = Set{SVector{3,Int}}()
    for phase in 0:7
        ux = phase & 1
        uy = (phase >> 1) & 1
        uz = (phase >> 2) & 1
        phase_offsets = by_phase[phase + 1]
        for z in -extent:extent, y in -extent:extent, x in -extent:extent
            o = SVector{3,Int}(x, y, z)
            _rigid_near(o, qc) && continue
            parent = SVector{3,Int}(
                fld(ux + x, 2), fld(uy + y, 2), fld(uz + z, 2))
            _rigid_near(parent, qp) || continue
            push!(phase_offsets, o)
            push!(union_offsets, o)
        end
        sort!(phase_offsets; by=_rigid_offset_order)
    end
    push_offsets = sort!(collect(union_offsets); by=_rigid_offset_order)
    offset_id = Dict(o => Int32(i) for (i, o) in enumerate(push_offsets))
    phase_index = Int32[]
    starts = Vector{Int}(undef, 9)
    class_of = zeros(Int32, 8, length(push_offsets))
    for phase in 1:8
        starts[phase] = length(phase_index) + 1
        for o in by_phase[phase]
            k = offset_id[o]
            push!(phase_index, k)
            class_of[phase, k] = k
        end
    end
    starts[9] = length(phase_index) + 1
    return RigidHierarchicalTables(child_base.near_offsets, push_offsets,
        Tuple(starts), phase_index, class_of)
end

"""
Build the shared offset union and per-level phase masks for an internal radius
schedule. The leaf table supplies the direct list; every M2L level selects a
complete rigid (and therefore complete cubic-symmetry-orbit) table. Uniform
policies take this same path, which keeps scheduled and production geometry
directly comparable.
"""
function _hierarchical_scheduled_tables(policy::HierarchicalRigidStencil, ell::Int)
    ell >= 2 || throw(ArgumentError(
        "HierarchicalRigidStencil requires ell >= 2 (the first M2L level is 2)"))
    nlevels = max(ell - 1, 0)
    qs = isempty(policy.level_radii2) ? fill(policy.near_radius2, nlevels) :
        collect(policy.level_radii2)
    length(qs) == nlevels || throw(ArgumentError(
        "hierarchical level schedule has $(length(qs)) entries, but ell=$ell " *
        "requires $nlevels entries for levels 2:$ell"))
    all(qs[i + 1] <= qs[i] for i in 1:length(qs)-1) || throw(ArgumentError(
        "hierarchical level schedule must be non-increasing with depth; got $(Tuple(qs))"))
    isempty(qs) || last(qs) == policy.near_radius2 || throw(ArgumentError(
        "hierarchical schedule leaf radius must equal near_radius2"))

    level_tables = [_rigid_transition_tables(j == 1 ? qs[j] : qs[j - 1], qs[j])
                    for j in eachindex(qs)]
    leaf = RigidHierarchicalTables(last(qs))
    push_offsets = sort!(collect(union((Set(t.push_offsets) for t in level_tables)...));
        by=_rigid_offset_order)
    offset_id = Dict(o => k for (k, o) in enumerate(push_offsets))
    level_class_of = zeros(Int32, 8, length(push_offsets), ell + 1)
    for (j, table) in enumerate(level_tables)
        L = j + 1
        for (oldk, o) in enumerate(table.push_offsets)
            k = offset_id[o]
            # Only membership is load-bearing; carrying the shared-union id
            # rather than the per-level one aids host-side diagnostics.
            @views level_class_of[:, k, L + 1] .=
                ifelse.(table.class_of[:, oldk] .== 0, Int32(0), Int32(k))
        end
    end
    # Legacy fields retain leaf semantics. Route construction uses the explicit
    # level masks above; direct construction uses leaf.near_offsets.
    leaf_class = zeros(Int32, 8, length(push_offsets))
    leaf_map = Dict(o => k for (k, o) in enumerate(leaf.push_offsets))
    for (k, o) in enumerate(push_offsets)
        oldk = get(leaf_map, o, 0)
        oldk == 0 || (@views leaf_class[:, k] .=
            ifelse.(leaf.class_of[:, oldk] .== 0, Int32(0), Int32(k)))
    end
    tables = RigidHierarchicalTables(leaf.near_offsets, push_offsets,
        leaf.phase_starts, leaf.phase_index, leaf_class)
    return tables, level_class_of, qs
end

function _verify_hierarchical_classifier!(h0, ell::Int,
        policy::HierarchicalRigidStencil, tables::RigidHierarchicalTables)
    ell >= 2 || throw(ArgumentError(
        "HierarchicalRigidStencil requires ell >= 2 (the first M2L level is 2)"))
    # The task-025 accepted/rejected boundary lies strictly inside the rigid
    # push-union cube.  Evaluate the production analytic
    # classifier on that complete cube without materializing the full
    # `(2^(ell+1)-1)^3` flat route-class domain.
    extent = 2isqrt(policy.near_radius2) + 1
    actual = Set{SVector{3,Int}}()
    for z in -extent:extent, y in -extent:extent, x in -extent:extent
        o = SVector{3,Int}(x, y, z)
        constant_p_stencil_bound(h0, ell, policy.config, o) <= policy.config.epsilon ||
            push!(actual, o)
    end
    expected = Set(tables.near_offsets)
    actual == expected && return nothing
    missing = length(setdiff(expected, actual))
    extra = length(setdiff(actual, expected))
    throw(ArgumentError(
        "HierarchicalRigidStencil accuracy gate failed at ell=$ell: the analytic " *
        "classifier at epsilon=$(policy.config.epsilon) rejects " *
        "$(length(actual)) offsets, but near_radius2=" *
        "$(policy.near_radius2) requires exactly $(length(expected)) offsets " *
        "(missing=$missing, extra=$extra). If you constructed this policy with " *
        "an explicit tolerance, choose one compatible with " *
        "rigid_stencil_epsilon(P, h0, ell, near_radius2) (it must scale with " *
        "2^ell as derived in task 025). If the tolerance was machine-derived, " *
        "the analytic bound at this (P, ell, box) cannot realize this rigid " *
        "near set; try the other near_radius2 or pass " *
        "policy=ConstantPAnalyticStencil(...) for the flat path."))
end

function _hierarchical_class_metadata(tables::RigidHierarchicalTables, ell::Int)
    noffsets = length(tables.push_offsets)
    nclasses = max(ell - 1, 0) * noffsets
    class_level = Vector{Int32}(undef, nclasses)
    class_offset = Matrix{Int32}(undef, 3, nclasses)
    effective_offsets = Vector{SVector{3,Int}}(undef, nclasses)
    c = 0
    @inbounds for level in 2:ell
        scale = 1 << (ell - level)
        for o in tables.push_offsets
            c += 1
            class_level[c] = Int32(level)
            class_offset[1, c] = Int32(o[1])
            class_offset[2, c] = Int32(o[2])
            class_offset[3, c] = Int32(o[3])
            # Existing resident plan builders take integer offsets at leaf
            # reference width.  This exact binary rescaling gives level-true
            # radii while preserving direction.
            effective_offsets[c] = scale * o
        end
    end
    return class_level, class_offset, effective_offsets
end

@inline function _hierarchical_node_lookup(occupancy::RadixLevelOccupancy,
        grid::DeviceRadixGrid, level_offsets, level::Int, coord::SVector{3,Int})
    _radix_coord_inbounds_at_level(coord, level) || return 0
    if !isempty(occupancy.node_at)
        G = 1 << level
        linear = coord[1] + G * (coord[2] + G * coord[3])
        return Int(occupancy.node_at[occupancy.level_base[level + 1] + linear + 1])
    end
    key = morton_key(coord, level)
    lo = level_offsets[level + 1] + 1
    hi = level_offsets[level + 2]
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        mk = grid.node_keys[mid]
        mk < key ? (lo = mid + 1) : (hi = mid - 1)
    end
    return lo <= level_offsets[level + 2] && grid.node_keys[lo] == key ? lo : 0
end

"""
Generate one `(level, consecutive-offset-classes)` route window.  Routes are
source-major inside each class and endpoints are flat node indices.
"""
@inline function build_hierarchical_routes_window!(route_levels, route_offsets,
        route_targets, route_sources, route_class, ctx::HostHierarchicalM2LContext,
        grid::DeviceRadixGrid, level::Integer, first_offset::Integer,
        last_offset::Integer)
    L = Int(level)
    noffsets = length(ctx.tables.push_offsets)
    1 <= first_offset <= last_offset <= noffsets ||
        throw(ArgumentError("invalid hierarchical offset window $first_offset:$last_offset"))
    first_source = ctx.level_offsets[L + 1] + 1
    last_source = ctx.level_offsets[L + 2]
    n_routes = 0
    @inbounds for k in Int(first_offset):Int(last_offset)
        o = ctx.tables.push_offsets[k]
        global_class = (L - 2) * noffsets + k
        for source in first_source:last_source
            phase = _rigid_phase_index(grid.node_coords[1, source],
                grid.node_coords[2, source], grid.node_coords[3, source])
            ctx.level_class_of[phase, k, L + 1] == 0 && continue
            source_coord = SVector{3,Int}(grid.node_coords[1, source],
                grid.node_coords[2, source], grid.node_coords[3, source])
            target = _hierarchical_node_lookup(ctx.occupancy, grid,
                ctx.level_offsets, L, source_coord + o)
            target == 0 && continue
            n_routes += 1
            n_routes <= length(route_sources) || throw(AssertionError(
                "hierarchical route window exceeded capacity $(length(route_sources)); " *
                "increase window storage or reduce window_classes"))
            route_levels[n_routes] = L
            route_offsets[1, n_routes] = o[1]
            route_offsets[2, n_routes] = o[2]
            route_offsets[3, n_routes] = o[3]
            route_targets[n_routes] = target
            route_sources[n_routes] = source
            route_class === nothing || (route_class[n_routes] = Int32(global_class))
        end
    end
    return n_routes
end

function build_hierarchical_direct_pairs!(direct_targets, direct_sources,
        ctx::HostHierarchicalM2LContext, grid::DeviceRadixGrid, n_cells::Integer;
        nearfield::Bool=true, self_induced::Bool=true)
    (nearfield || self_induced) || return 0
    L = grid.ell
    leaf_base = ctx.level_offsets[L + 1]
    n_direct = 0
    @inbounds for target_cell in 1:Int(n_cells)
        target_node = leaf_base + target_cell
        target_coord = SVector{3,Int}(grid.node_coords[1, target_node],
            grid.node_coords[2, target_node], grid.node_coords[3, target_node])
        for o in ctx.tables.near_offsets
            isself = iszero(o[1]) && iszero(o[2]) && iszero(o[3])
            (isself ? self_induced : nearfield) || continue
            source_node = _hierarchical_node_lookup(ctx.occupancy, grid,
                ctx.level_offsets, L, target_coord - o)
            source_node == 0 && continue
            n_direct += 1
            n_direct <= length(direct_targets) || throw(AssertionError(
                "hierarchical direct-pair buffer exceeded capacity"))
            direct_targets[n_direct] = target_cell
            direct_sources[n_direct] = source_node - leaf_base
        end
    end
    return n_direct
end

# Grid-free constant-P bound from the fixed Morton domain (task 023): identical
# arithmetic to the RadixGrid method, with cell_half_width = h0 / G and
# displacement = offset * (2 h0 / G).
function constant_p_stencil_bound(h0::Real, ell::Integer, config::ConstantPStencilConfig,
        offset::SVector{3,<:Integer})
    G = 1 << Int(ell)
    cell_half_width = h0 / G
    B_phi = constant_p_stencil_bound(
        config.P_phi, offset, config.source_strength, cell_half_width,
    )
    if _stencil_lamb_helmholtz(config)
        B_chi = constant_p_stencil_bound(
            config.P_phi + 1, offset, config.chi_strength, cell_half_width,
        )
        Δ = (2 * h0) / G
        TF = typeof(Δ)
        R = norm(Δ * SVector{3,TF}(offset[1], offset[2], offset[3]))
        return _stencil_production_factor(config) * (B_phi + (1 + 2R) * B_chi)
    else
        return _stencil_production_factor(config) * B_phi
    end
end

"""
    refresh_cell_at!(cell_at, cell_keys, n_cells, ell)

Refill the dense coord -> occupied-cell map in place: zero it, then scatter the
first `n_cells` Morton keys. Factored out of [`radix_implicit_stencil`](@ref) so
the recurring update path (task 023) can refresh occupancy without reallocating.
"""
function refresh_cell_at!(cell_at::AbstractArray{Int32,3}, cell_keys, n_cells::Integer,
        ell::Integer)
    fill!(cell_at, Int32(0))
    @inbounds for cell in 1:n_cells
        coord = morton_decode(cell_keys[cell], ell)
        cell_at[coord[1] + 1, coord[2] + 1, coord[3] + 1] = Int32(cell)
    end
    return cell_at
end

@inline function _radix_cell_at(cell_at::AbstractArray{Int32,3}, coord::SVector{3,<:Integer})
    (0 <= coord[1] < size(cell_at, 1) && 0 <= coord[2] < size(cell_at, 2) &&
        0 <= coord[3] < size(cell_at, 3)) || return 0
    return Int(@inbounds cell_at[coord[1] + 1, coord[2] + 1, coord[3] + 1])
end

@inline _radix_cell_at(stencil::RadixImplicitStencil, coord::SVector{3,<:Integer}) =
    _radix_cell_at(stencil.cell_at, coord)

@inline _radix_cell_coords(grid::RadixGrid) =
    SVector{3,Int}[radix_cell_coord(grid, cell) for cell in eachindex(grid.cell_keys)]

@inline _chebyshev_norm(d::SVector{3,<:Integer}) =
    max(abs(Int(d[1])), abs(Int(d[2])), abs(Int(d[3])))

@inline _radix_is_leaf_direct(::ParentNeighborM2L, leaf_offset::SVector{3,<:Integer}) =
    _chebyshev_norm(leaf_offset) <= 1

@inline _radix_is_m2l(::ParentNeighborM2L, child_offset::SVector{3,<:Integer},
        parent_offset::SVector{3,<:Integer}) =
    _chebyshev_norm(parent_offset) <= 1 && _chebyshev_norm(child_offset) > 1

const _RADIX_CHILD_PHASES = ntuple(i -> SVector{3,Int}(
    (i - 1) & 0x1,
    ((i - 1) >> 1) & 0x1,
    ((i - 1) >> 2) & 0x1,
), 8)

const _RADIX_DIRECT_OFFSETS = let offsets = SVector{3,Int}[]
    for k in -1:1, j in -1:1, i in -1:1
        push!(offsets, SVector{3,Int}(i, j, k))
    end
    Tuple(offsets)
end

function _radix_parent_neighbor_m2l_candidates(target_phase::SVector{3,<:Integer})
    target_child_coord = SVector{3,Int}(target_phase[1], target_phase[2], target_phase[3])
    candidates = SVector{3,Int}[]
    for k_parent in -1:1, j_parent in -1:1, i_parent in -1:1
        parent_offset = SVector{3,Int}(i_parent, j_parent, k_parent)
        for source_phase in _RADIX_CHILD_PHASES
            source_child_coord = 2 * (-parent_offset) + source_phase
            child_offset = target_child_coord - source_child_coord
            if _radix_is_m2l(ParentNeighborM2L(), child_offset, parent_offset)
                push!(candidates, child_offset)
            end
        end
    end
    sort!(candidates; by=o -> (o[3], o[2], o[1]))
    return Tuple(candidates)
end

const _RADIX_PARENT_NEIGHBOR_M2L_CANDIDATES = ntuple(
    i -> _radix_parent_neighbor_m2l_candidates(_RADIX_CHILD_PHASES[i]), 8,
)

@inline _radix_phase_index(phase::SVector{3,<:Integer}) =
    Int(phase[1] + 2 * phase[2] + 4 * phase[3] + 1)

@inline _radix_leaf_phase(coord::SVector{3,<:Integer}) =
    SVector{3,Int}(coord[1] & 0x1, coord[2] & 0x1, coord[3] & 0x1)

@inline _radix_m2l_candidates(::ParentNeighborM2L, target_phase::SVector{3,<:Integer}) =
    _RADIX_PARENT_NEIGHBOR_M2L_CANDIDATES[_radix_phase_index(target_phase)]

@inline _radix_direct_offsets(::ParentNeighborM2L) = _RADIX_DIRECT_OFFSETS

@inline _radix_level_coord(leaf_coord::SVector{3,<:Integer}, leaf_level::Integer, level::Integer) =
    SVector{3,Int}(
        Int(leaf_coord[1]) >> (Int(leaf_level) - Int(level)),
        Int(leaf_coord[2]) >> (Int(leaf_level) - Int(level)),
        Int(leaf_coord[3]) >> (Int(leaf_level) - Int(level)),
    )

function _radix_ancestor_leaf_map(grid::RadixGrid)
    map = Dict{Tuple{Int,SVector{3,Int}},Vector{Int}}()
    for cell in eachindex(grid.cell_keys)
        leaf_coord = radix_cell_coord(grid, cell)
        for level in 0:grid.ell
            coord = _radix_level_coord(leaf_coord, grid.ell, level)
            push!(get!(() -> Int[], map, (level, coord)), cell)
        end
    end
    return map
end

@inline _radix_coord_inbounds_at_level(coord::SVector{3,<:Integer}, level::Integer) = begin
    G = 1 << Int(level)
    0 <= coord[1] < G && 0 <= coord[2] < G && 0 <= coord[3] < G
end

foreach_radix_m2l_pair(f, strategy::RadixTraversalStrategy, grid::RadixGrid;
        farfield::Bool=true) =
    foreach_radix_m2l_pair(f, strategy, ParentNeighborM2L(), grid; farfield)

foreach_radix_direct_pair(f, strategy::RadixTraversalStrategy, grid::RadixGrid;
        nearfield::Bool=true, self_induced::Bool=true) =
    foreach_radix_direct_pair(f, strategy, ParentNeighborM2L(), grid; nearfield, self_induced)

build_radix_interaction_list(strategy::RadixTraversalStrategy, grid::RadixGrid;
        farfield::Bool=true, nearfield::Bool=true, self_induced::Bool=true) =
    build_radix_interaction_list(
        strategy, ParentNeighborM2L(), grid; farfield, nearfield, self_induced,
    )

function foreach_radix_m2l_pair(f, strategy::RadixTraversalStrategy,
        policy::RadixSeparationPolicy, grid::RadixGrid; farfield::Bool=true)
    farfield || return nothing
    return foreach_radix_m2l_route(strategy, policy, grid; farfield) do level, offset, target_cell, source_cell
        f(offset, target_cell, source_cell)
    end
end

function foreach_radix_m2l_route(f, strategy::RadixTraversalStrategy,
        policy::RadixSeparationPolicy, grid::RadixGrid; farfield::Bool=true)
    farfield || return nothing
    return _foreach_radix_m2l_route(f, strategy, policy, grid)
end

foreach_radix_m2l_route(f, strategy::RadixTraversalStrategy, grid::RadixGrid;
        farfield::Bool=true) =
    foreach_radix_m2l_route(f, strategy, ParentNeighborM2L(), grid; farfield)

function _foreach_radix_m2l_route(f, ::RadixTraversalStrategy,
        policy::ParentNeighborM2L, grid::RadixGrid)
    ancestor_leaf_cells = _radix_ancestor_leaf_map(grid)
    for target_cell in eachindex(grid.cell_keys)
        target_leaf_coord = radix_cell_coord(grid, target_cell)
        for level in 1:grid.ell
            target_coord = _radix_level_coord(target_leaf_coord, grid.ell, level)
            target_phase = _radix_leaf_phase(target_coord)
            for offset in _radix_m2l_candidates(policy, target_phase)
                source_coord = target_coord - offset
                _radix_coord_inbounds_at_level(source_coord, level) || continue
                source_cells = get(ancestor_leaf_cells, (level, source_coord), nothing)
                source_cells === nothing && continue
                for source_cell in source_cells
                    f(level, offset, target_cell, source_cell)
                end
            end
        end
    end
    return nothing
end

# Constant-P M2L routes, classified by implicit lookup and emitted class-major (offset outer): for each
# fixed accepted offset the member pairs are exactly the occupied (target, target-offset)
# coordinate pairs, resolved by O(1) occupancy lookups.
function _foreach_radix_m2l_route(f, ::RadixTraversalStrategy,
        policy::ConstantPAnalyticStencil, grid::RadixGrid)
    stencil = radix_implicit_stencil(grid, policy.config)
    return _foreach_radix_m2l_route_implicit(f, stencil, _radix_cell_coords(grid), grid.ell)
end

function _foreach_radix_m2l_route_implicit(f, stencil::RadixImplicitStencil,
        coords::Vector{SVector{3,Int}}, level::Integer)
    for offset in stencil.accepted_offsets
        for target_cell in eachindex(coords)
            source_cell = _radix_cell_at(stencil, coords[target_cell] - offset)
            source_cell == 0 && continue
            f(level, offset, target_cell, source_cell)
        end
    end
    return nothing
end

function foreach_radix_direct_pair(f, ::RadixTraversalStrategy, policy::ParentNeighborM2L,
        grid::RadixGrid; nearfield::Bool=true, self_induced::Bool=true)
    (nearfield || self_induced) || return nothing
    for target_cell in eachindex(grid.cell_keys)
        target_coord = radix_cell_coord(grid, target_cell)
        for offset in _radix_direct_offsets(policy)
            source_cell = radix_cell_index(grid, target_coord - offset)
            source_cell == 0 && continue
            if offset == SVector{3,Int}(0, 0, 0)
                self_induced && f(offset, target_cell, source_cell)
            else
                nearfield && f(offset, target_cell, source_cell)
            end
        end
    end
    return nothing
end

# Constant-P direct near/self complement, classified by implicit lookup: every offset between
# occupied cells lies in the bounded stencil box, so the complement of the accepted set
# is exactly the (small) rejected-offset set. O(cells * |rejected|) occupancy lookups
# replace the former O(cells^2) full pair scan.
function foreach_radix_direct_pair(f, ::RadixTraversalStrategy, policy::ConstantPAnalyticStencil,
        grid::RadixGrid; nearfield::Bool=true, self_induced::Bool=true)
    (nearfield || self_induced) || return nothing
    stencil = radix_implicit_stencil(grid, policy.config)
    return _foreach_radix_direct_pair_implicit(
        f, stencil, _radix_cell_coords(grid); nearfield, self_induced,
    )
end

function _foreach_radix_direct_pair_implicit(f, stencil::RadixImplicitStencil,
        coords::Vector{SVector{3,Int}}; nearfield::Bool=true, self_induced::Bool=true)
    (nearfield || self_induced) || return nothing
    for target_cell in eachindex(coords)
        target_coord = coords[target_cell]
        for offset in stencil.rejected_offsets
            source_cell = _radix_cell_at(stencil, target_coord - offset)
            source_cell == 0 && continue
            if offset == SVector{3,Int}(0, 0, 0)
                self_induced && f(offset, target_cell, source_cell)
            else
                nearfield && f(offset, target_cell, source_cell)
            end
        end
    end
    return nothing
end

# Constant-P list assembly from the implicit stencil classifier: one stencil build serves
# both passes, while each accepted pair is still materialized into an offset batch — already
# in the canonical (level, z, y, x) batch order, so no Dict and no final sort — and the
# direct complement comes from the bounded rejected-offset set.
function build_radix_interaction_list(::RadixTraversalStrategy,
        policy::ConstantPAnalyticStencil, grid::RadixGrid; farfield::Bool=true,
        nearfield::Bool=true, self_induced::Bool=true)
    stencil = radix_implicit_stencil(grid, policy.config)
    coords = _radix_cell_coords(grid)
    ncells = length(coords)
    batches = RadixM2LBatch{Int}[]
    if farfield
        sizehint!(batches, length(stencil.accepted_offsets))
        for offset in stencil.accepted_offsets
            # count first so each batch retains exactly its member count (the former
            # ncells sizehint over-retained ~2 GB at n=1e5/ell=4; task 023)
            npairs = 0
            @inbounds for target_cell in 1:ncells
                _radix_cell_at(stencil, coords[target_cell] - offset) == 0 || (npairs += 1)
            end
            npairs == 0 && continue
            targets = Vector{Int}(undef, npairs)
            sources = Vector{Int}(undef, npairs)
            i = 0
            @inbounds for target_cell in 1:ncells
                source_cell = _radix_cell_at(stencil, coords[target_cell] - offset)
                source_cell == 0 && continue
                i += 1
                targets[i] = target_cell
                sources[i] = source_cell
            end
            push!(batches, RadixM2LBatch(grid.ell, offset, targets, sources))
        end
    end
    ndirect = 0
    _foreach_radix_direct_pair_implicit(stencil, coords; nearfield, self_induced) do offset, target_cell, source_cell
        ndirect += 1
    end
    direct_pairs = Vector{SVector{2,Int}}(undef, ndirect)
    i_direct = 0
    _foreach_radix_direct_pair_implicit(stencil, coords; nearfield, self_induced) do offset, target_cell, source_cell
        i_direct += 1
        direct_pairs[i_direct] = SVector{2,Int}(target_cell, source_cell)
    end
    return RadixInteractionList{Int}(batches, direct_pairs)
end

"""
    build_radix_routes!(route_levels, route_offsets, route_targets, route_sources,
        route_class, direct_targets, direct_sources, accepted_offsets,
        rejected_offsets, cell_at, coords, leaf_to_node, ell, n_cells;
        farfield=true, nearfield=true, self_induced=true) -> (n_routes, n_direct)

In-place constant-`P` route generation (task 023): writes the flattened M2L routes
(offset-class-major, matching `build_radix_interaction_list` +
`_flatten_radix_routes_host` elementwise) and the direct near/self pairs
(target-major, matching `_foreach_radix_direct_pair_implicit` order) directly into
preallocated flat arrays, so recurring time steps materialize no batch vectors.
Route targets/sources are node indices (via `leaf_to_node`); direct pairs are leaf
cell indices. `route_class[i]` receives the 1-based index of route `i`'s offset in
`accepted_offsets` (pass `nothing` to skip). `coords[1:n_cells]` are the decoded
cell coordinates. Returns the valid prefix lengths.
"""
function build_radix_routes!(route_levels, route_offsets, route_targets, route_sources,
        route_class, direct_targets, direct_sources,
        accepted_offsets, rejected_offsets, cell_at::AbstractArray{Int32,3},
        coords, leaf_to_node, ell::Integer, n_cells::Integer;
        farfield::Bool=true, nearfield::Bool=true, self_induced::Bool=true)
    n_routes = 0
    if farfield
        @inbounds for (k, offset) in enumerate(accepted_offsets)
            for target_cell in 1:n_cells
                source_cell = _radix_cell_at(cell_at, coords[target_cell] - offset)
                source_cell == 0 && continue
                n_routes += 1
                route_levels[n_routes] = ell
                route_offsets[1, n_routes] = offset[1]
                route_offsets[2, n_routes] = offset[2]
                route_offsets[3, n_routes] = offset[3]
                route_targets[n_routes] = leaf_to_node[target_cell]
                route_sources[n_routes] = leaf_to_node[source_cell]
                route_class === nothing || (route_class[n_routes] = Int32(k))
            end
        end
    end
    n_direct = 0
    if nearfield || self_induced
        @inbounds for target_cell in 1:n_cells
            target_coord = coords[target_cell]
            for offset in rejected_offsets
                source_cell = _radix_cell_at(cell_at, target_coord - offset)
                source_cell == 0 && continue
                is_self = offset[1] == 0 && offset[2] == 0 && offset[3] == 0
                (is_self ? self_induced : nearfield) || continue
                n_direct += 1
                direct_targets[n_direct] = target_cell
                direct_sources[n_direct] = source_cell
            end
        end
    end
    return n_routes, n_direct
end

function build_radix_interaction_list(strategy::RadixTraversalStrategy,
        policy::RadixSeparationPolicy, grid::RadixGrid; farfield::Bool=true,
        nearfield::Bool=true, self_induced::Bool=true)
    batches_by_route = Dict{Tuple{Int,SVector{3,Int}},RadixM2LBatch{Int}}()
    if farfield
        foreach_radix_m2l_route(strategy, policy, grid; farfield) do level, offset, target_cell, source_cell
            route = (level, offset)
            batch = get!(() -> RadixM2LBatch(level, offset, Int[], Int[]), batches_by_route, route)
            push!(batch.targets, target_cell)
            push!(batch.sources, source_cell)
        end
    end
    direct_pairs = SVector{2,Int}[]
    foreach_radix_direct_pair(strategy, policy, grid; nearfield, self_induced) do offset, target_cell, source_cell
        push!(direct_pairs, SVector{2,Int}(target_cell, source_cell))
    end
    batches = collect(values(batches_by_route))
    sort!(batches; by=batch -> (batch.level, batch.offset[3], batch.offset[2], batch.offset[1]))
    return RadixInteractionList{Int}(batches, direct_pairs)
end
