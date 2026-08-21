#------- matrix storage -------#

function Matrices(sizes::Vector{Tuple{Int,Int}}, TF=Float64)
    # preallocate matrix storage
    n_matrix = sum(m * n for (m, n) in sizes)
    n_rhs = sum(m for (m,_) in sizes)
    data = Vector{TF}(undef, n_matrix)
    rhs = Vector{TF}(undef, n_rhs)

    # offsets
    matrix_offsets = Vector{Int}(undef, length(sizes))
    rhs_offsets = Vector{Int}(undef, length(sizes))
    matrix_offset = 1
    rhs_offset = 1
    for i in 1:length(sizes)
        matrix_offsets[i] = matrix_offset
        rhs_offsets[i] = rhs_offset

        m, n = sizes[i]
        matrix_offset += m * n
        rhs_offset += m
    end

    return Matrices{TF}(data, rhs, sizes, matrix_offsets, rhs_offsets)
end

function EmptyMatrices(TF=Float64)
    # preallocate empty matrix storage
    return Matrices{TF}(zeros(TF, 0), zeros(TF, 0), Tuple{Int,Int}[], Int[], Int[])
end

@inline function get_matrix_range(ms::Matrices, k::Int, m, n)
    # get the range of values corresponding to the k-th matrix
    m, n = ms.sizes[k]
    matrix_offset = ms.matrix_offsets[k]
    return matrix_offset:matrix_offset + m*n - 1
end

@inline function get_rhs_range(ms::Matrices, k::Int, m)
    # get the range of values corresponding to the k-th rhs vector
    rhs_offset = ms.rhs_offsets[k]
    return rhs_offset:rhs_offset + m - 1
end

function get_matrix_vector(ms::Matrices, k::Int)
    m, n = ms.sizes[k]
    vrange = get_rhs_range(ms, k, m)
    mrange = get_matrix_range(ms, k, m, n)
    mat = @view ms.data[mrange]
    return reshape(mat, m, n), view(ms.rhs, vrange)
end

function build_leaf_lu_cache(self_matrices::Matrices{TF}) where TF
    start_time = time_ns()
    data = copy(self_matrices.data)
    factors = map(eachindex(self_matrices.sizes)) do k
        m, n = self_matrices.sizes[k]
        m == n || throw(DimensionMismatch(
            "FastGaussSeidel self-influence block $k must be square, got $(m)×$(n)"))
        matrix_range = get_matrix_range(self_matrices, k, m, n)
        factor_matrix = reshape(view(data, matrix_range), m, n)
        lu!(factor_matrix; check=true)
    end
    build_time = (time_ns() - start_time) * 1e-9
    bytes = sizeof(data) + sum(sizeof(F.ipiv) for F in factors)
    return LeafLUCache{TF,eltype(factors)}(data, factors, build_time, bytes)
end

@inline function solve_leaf!(leaf_strengths, self_matrices::Matrices,
                             ::Nothing, i_leaf::Int)
    mat, rhs = get_matrix_vector(self_matrices, i_leaf)
    leaf_strengths .= mat \ rhs
    return leaf_strengths
end

@inline function solve_leaf!(leaf_strengths, self_matrices::Matrices,
                             cache::LeafLUCache, i_leaf::Int)
    _, rhs = get_matrix_vector(self_matrices, i_leaf)
    ldiv!(leaf_strengths, cache.factorizations[i_leaf], rhs)
    return leaf_strengths
end

function set_unit_strength!(source_buffers::AbstractVector{<:Matrix}, source_systems::Tuple)
    for i_source_system in eachindex(source_systems)
        source_system = source_systems[i_source_system]
        source_buffer = source_buffers[i_source_system]

        set_unit_strength!(source_buffer, source_system)
    end
end

function set_unit_strength!(source_buffer::AbstractMatrix{TF}, source_system) where TF
    # set the source strength to unit
    n_bodies = get_n_bodies(source_system)
    unit_strength = one(TF)
    dim = strength_dims(source_system)
    for j in 1:n_bodies
        value_to_strength!(source_buffer, source_system, j, one(TF))
    end
end

function save_strengths(source_buffers::AbstractVector{<:Matrix}, source_systems::Tuple)
    # save the strengths of the source systems
    TF = promote_type(eltype.(source_buffers)...)
    old_strengths = Tuple(Matrix{TF}(undef, strength_dims(source_systems[k]), get_n_bodies(source_systems[k])) for k in eachindex(source_systems))
    for k in eachindex(source_systems)
        source_system = source_systems[k]
        source_buffer = source_buffers[k]
        old_strengths[k] .= view(source_buffer, 5:5 + strength_dims(source_system) - 1, :)
    end

    return old_strengths
end

function restore_strengths!(source_buffers::AbstractVector{<:Matrix}, source_systems::Tuple, old_strengths)
    # restore the strengths of the source systems
    for k in eachindex(source_systems)
        source_system = source_systems[k]
        source_buffer = source_buffers[k]
        source_buffer[5:5 + strength_dims(source_system) - 1, :] .= old_strengths[k]
    end
end

function nonself_influence_matrices(target_buffers::AbstractVector{<:Matrix}, source_buffers::AbstractVector{<:Matrix}, source_systems::Tuple, target_tree::Tree{TF,<:Any}, source_tree::Tree, direct_list, derivatives_switches) where TF

    #--- sort by source ---#

    source_branches = source_tree.branches
    target_branches = target_tree.branches
    sorted_list = sort_by_source(direct_list, source_branches)

    #--- pre-allocate influence matrices ---#

    if length(sorted_list) > 0

        # preallocate sizes
        sizes = Vector{Tuple{Int,Int}}(undef, length(source_tree.leaf_index))

        # number of non-empty matrices
        this_source = 0
        n_matrices = 0
        for (i_target,j_source) in sorted_list
            if j_source != this_source
                n_matrices += 1
                this_source = j_source
            end
        end

        # generate matrix map
        matrix_map = Vector{Int}(undef, n_matrices)
        i_leaf = 1
        i_matrix = 1
        this_source = 0
        for (_, j_source) in sorted_list

            # found a non-empty matrix
            if j_source != this_source
                # upper bound of number of leaves we've skipped
                n_missing_ub = j_source - source_tree.leaf_index[i_leaf]

                # increment i_leaf until we line up with j_source
                for i_plus in 1:n_missing_ub
                    # i_leaf is empty
                    sizes[i_leaf] = (0,0)

                    # move on to the next leaf
                    i_leaf += 1

                    # check if we've reached j_source
                    if source_tree.leaf_index[i_leaf] == j_source
                        break
                    end
                end

                # update matrix_map
                matrix_map[i_matrix] = i_leaf

                # recurse
                i_leaf += 1
                i_matrix += 1
                this_source = j_source
            end
        end

        # fill in the rest with zeros
        for i in i_leaf:length(source_tree.leaf_index)
            sizes[i] = (0,0)
        end

        # populate sizes
        this_leaf = source_tree.leaf_index[1]
        this_source = sorted_list[1][2]
        i_matrix = 1
        n = 0
        for (i_target, j_source) in sorted_list

            # just finished a matrix
            if j_source != this_source

                # save this size
                m = get_n_bodies(source_branches[this_source].bodies_index)
                sizes[matrix_map[i_matrix]] = (n,m)

                # increment indices for the next
                i_matrix += 1
                this_source = j_source
                n = 0
            end

            # accumulate n
            n += get_n_bodies(target_branches[i_target].bodies_index)
        end

        # add the final matrix
        this_source = sorted_list[end][2]
        m = get_n_bodies(source_branches[this_source].bodies_index)
        sizes[matrix_map[i_matrix]] = (n,m)

        # construct influence matrices
        matrices = Matrices(sizes, TF)

        #--- populate influence matrices ---#

        # store strengths for later
        old_strengths = save_strengths(source_buffers, source_systems)

        # set sources to unit strength
        set_unit_strength!(source_buffers, source_systems)

        # loop over direct list
        this_source = 0
        i_matrix = 0
        i_target_start = 1
        i_source_start = 1
        matrix, influence = get_matrix_vector(matrices, 1)

        for (i_target,j_source) in sorted_list

            # start on the next matrix
            if j_source != this_source
                i_matrix += 1
                this_source = j_source
                i_source_start = 1
                i_target_start = 1
                matrix, influence = get_matrix_vector(matrices, matrix_map[i_matrix])
            end

            # loop over source systems
            this_i_source_start = i_source_start

            for i_source_system in eachindex(source_systems)

                # get view of matrix corresponding to this source system
                source_index = source_branches[j_source].bodies_index[i_source_system]
                n_sources = length(source_index)

                # unpack source buffer
                source_system = source_systems[i_source_system]
                source_buffer = source_buffers[i_source_system]
                # this_source_buffer = view(source_buffer, :, source_index)

                # loop over target systems
                for i_target_system in eachindex(target_buffers)

                    # get view of matrix corresponding to this source and target system
                    target_index = target_branches[i_target].bodies_index[i_target_system]
                    n_targets = length(target_index)
                    this_matrix = view(matrix, i_target_start:i_target_start + n_targets - 1, this_i_source_start:this_i_source_start + n_sources - 1)
                    this_influence = view(influence, i_target_start:i_target_start + n_targets - 1)

                    # unpack target buffer
                    target_buffer = target_buffers[i_target_system]
                    this_target_buffer = view(target_buffer, :, target_index)
                    target_source_buffer = source_buffers[i_target_system]
                    this_source_buffer = view(target_source_buffer, :, target_index)
                    derivatives_switch = derivatives_switches[i_target_system]

                    # loop over source bodies
                    for (isb,i_source_body) in enumerate(source_index)

                        # reset targets
                        reset!(target_buffer, target_index)

                        # direct influence on these targets
                        direct!(target_buffer, target_index, derivatives_switch, source_system, source_buffer, i_source_body:i_source_body)

                        # compute influences
                        influence!(this_influence, this_target_buffer, derivatives_switch, source_system, this_source_buffer)

                        # update matrix
                        this_matrix[:,isb] .= this_influence

                    end

                    # update target starting index
                    i_target_start += n_targets
                end

                # update source starting index
                this_i_source_start += length(source_index)
            end
        end

        # restore old strengths
        restore_strengths!(source_buffers, source_systems, old_strengths)

        # zero rhs
        matrices.rhs .= zero(TF)

    else
        matrices = EmptyMatrices(TF)
    end

    return matrices, sorted_list
end

"""
    index_by_source(sorted_list::Vector{SVector{2,Int}}, leaf_index::Vector{Int})

Constructs an index map that maps each leaf to the range of indices in the sorted list where it acts as a source. It is possible for some leaves to never act as sources, in which case the range will be empty.
"""
function index_by_source(sorted_list::Vector{SVector{2,Int32}}, leaf_index::Vector{Int})

    # preallocate index map
    # index_map = Vector{UnitRange{Int}}(undef, length(leaf_index))
    index_map = fill(1:0, length(leaf_index)) # initialize with empty ranges

    # check if sorted_list is empty
    if length(sorted_list) > 0

        # loop over direct list
        this_source = 0
        i_start = 1
        i_leaf = 1
        this_source = sorted_list[1][2] # start with the first source
        for (i_list, (i_target, j_source)) in enumerate(sorted_list)

            # finished a non-empty matrix
            if j_source != this_source

                # upper bound of number of leaves we've skipped
                n_missing_ub = this_source - leaf_index[i_leaf]

                # increment i_leaf until we line up with j_source
                for i_plus in 1:n_missing_ub

                    # no interactions for this leaf
                    # index_map[i_leaf] = i_start:i_start - 1

                    # move on to the next leaf
                    i_leaf += 1

                    # check if we've reached j_source
                    if leaf_index[i_leaf] == this_source
                        break
                    end
                end

                # store leaf index range
                index_map[i_leaf] = i_start:i_list-1

                # recurse
                i_leaf += 1
                this_source = j_source
                i_start = i_list
            end
        end

        #--- last index ---#

        # upper bound of number of leaves we've skipped
        n_missing_ub = this_source - leaf_index[i_leaf]

        # increment i_leaf until we line up with j_source
        for i_plus in 1:n_missing_ub

            # no interactions for this leaf
            # index_map[i_leaf] = i_start:i_start - 1

            # move on to the next leaf
            i_leaf += 1

            # check if we've reached j_source
            if leaf_index[i_leaf] == this_source
                break
            end
        end

        # store leaf index range
        index_map[i_leaf] = i_start:length(sorted_list)
        i_start = length(sorted_list) + 1

        #--- all remaining leaves do not act as sources ---#

        # for i in i_leaf + 1:length(leaf_index)
        #     index_map[i] = i_start:i_start - 1
        # end
    end

    return index_map
end

"""
    self_influence_matrices(target_buffers, source_buffers, source_systems, target_tree, source_tree, derivatives_switches)

Constructs influence matrices for all leaves of the tree. (Assumes source tree and target trees are identical.)
"""
function self_influence_matrices(target_buffers, source_buffers, source_systems, target_tree::Tree{TF,<:Any}, source_tree, derivatives_switches) where TF

    #--- pre-allocate influence matrices ---#

    # get sizes
    sizes = Vector{Tuple{Int,Int}}(undef,length(source_tree.leaf_index))
    for (i, i_branch) in enumerate(source_tree.leaf_index)
        n_sources = get_n_bodies(source_tree.branches[i_branch].bodies_index)
        sizes[i] = (n_sources, n_sources)
    end

    # construct influence matrices
    matrices = Matrices(sizes, TF)

    #--- populate influence matrices ---#

    # store strengths for later
    old_strengths = save_strengths(source_buffers, source_systems)

    # set sources to unit strength
    set_unit_strength!(source_buffers, source_systems)

    # populate influence matrices
    for (i_matrix, i_leaf) in enumerate(source_tree.leaf_index)
        # get branch
        branch = source_tree.branches[i_leaf]

        # get view of matrix corresponding to this branch
        matrix, influence = get_matrix_vector(matrices, i_matrix)

        # loop over source systems
        i_source_start = 0
        for i_source_system in eachindex(source_systems)

            # unpack source system and buffer
            source_system = source_systems[i_source_system]
            source_buffer = source_buffers[i_source_system]
            source_bodies_index = branch.bodies_index[i_source_system]

            # loop over target systems
            i_target_start = 1
            for i_target_system in eachindex(target_buffers)

                # unpack target buffer
                target_buffer = target_buffers[i_target_system]
                derivatives_switch = derivatives_switches[i_target_system]
                target_bodies_index = branch.bodies_index[i_target_system]

                # view of influence corresponding to these targets
                n_targets = length(target_bodies_index)
                this_influence = view(influence, i_target_start:i_target_start + n_targets - 1)

                # loop over bodies in this branch
                targets_view = view(target_buffer, :, target_bodies_index)
                sources_view = view(source_buffer, :, source_bodies_index)
                this_matrix_block = view(matrix, i_target_start:i_target_start + n_targets - 1, i_source_start + 1:i_source_start + length(source_bodies_index))
                for (isb, i_source_body) in enumerate(source_bodies_index)

                    # reset targets
                    reset!(target_buffer, target_bodies_index)

                    # direct influence on these targets
                    direct!(target_buffer, target_bodies_index, derivatives_switch, source_system, source_buffer, i_source_body:i_source_body)

                    # compute influences
                    influence!(this_influence, targets_view, derivatives_switch, source_system, sources_view)

                    # update matrix
                    this_matrix_block[:, isb] .= this_influence
                end

                # update target starting index
                i_target_start += length(branch.bodies_index[i_target_system])
            end

            # update source starting index
            i_source_start += length(source_bodies_index)
        end
    end

    # restore old strengths
    restore_strengths!(source_buffers, source_systems, old_strengths)

    # zero rhs
    matrices.rhs .= zero(TF)

    return matrices
end

"""
    map_by_leaf(source_tree::Tree)

Constructs a mapping from each leaf to the range of indices in the strengths vector that correspond to the bodies in that leaf.
"""
function map_by_leaf(source_tree::Tree)
    strengths_by_leaf = Vector{UnitRange{Int}}(undef, length(source_tree.leaf_index))
    i_start = 1
    for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)

        # get bodies index
        bodies_index = source_tree.branches[i_branch].bodies_index

        # get number of bodies in this branch
        n_bodies = get_n_bodies(bodies_index)

        # store strength index
        strengths_by_leaf[i_leaf] = i_start:i_start + n_bodies - 1
        i_start += n_bodies

    end
    return strengths_by_leaf
end

"""
    map_by_branch(target_tree::Tree)

Constructs a mapping from each branch to the range of indices in the targets vector that correspond to the bodies in that branch.

This relies on the fact that `tree.leaf_index` is sorted in order of increasing branch index.

"""
function map_by_branch(target_tree::Tree)
    targets_by_branch = fill(1:0, length(target_tree.branches)) # initialize with empty ranges
    i_start = 1
    for i_branch in target_tree.leaf_index

        # get bodies index
        bodies_index = target_tree.branches[i_branch].bodies_index

        # get number of bodies in this branch
        n_bodies = get_n_bodies(bodies_index)

        # store target index
        targets_by_branch[i_branch] = i_start:i_start + n_bodies - 1
        i_start += n_bodies
    end
    return targets_by_branch
end

function add_self_interactions(direct_list::Vector{SVector{2,Int32}}, source_tree::Tree)
    # prepare container
    full_direct_list = copy(direct_list)
    i_start = length(full_direct_list)
    resize!(full_direct_list, length(full_direct_list) + length(source_tree.leaf_index))

    # add leaf-on-self interactions
    for (i_leaf,i_branch) in enumerate(source_tree.leaf_index)
        full_direct_list[i_start + i_leaf] = SVector{2,Int32}(i_branch, i_branch)
    end

    # sort by target
    full_direct_list = sort_by_target(full_direct_list, source_tree.branches)

    return full_direct_list
end

"""
    assert_shared_topology(target_tree, source_tree)

Errors if the two trees do not share identical octree topology (branch count and
per-branch `bodies_index`/`branch_index`). `FastGaussSeidel` requires this invariant
because its interaction lists and influence matrices use source- and target-tree branch
indices interchangeably; a silent divergence would mis-associate blocks rather than crash.
"""
function assert_shared_topology(target_tree::Tree, source_tree::Tree)
    length(target_tree.branches) == length(source_tree.branches) || error(
        "FastGaussSeidel requires structurally identical source/target trees, but got " *
        "$(length(target_tree.branches)) target vs $(length(source_tree.branches)) source branches; " *
        "the target tree should replay the source tree's topology — please file an issue.")
    for (i_branch, (tb, sb)) in enumerate(zip(target_tree.branches, source_tree.branches))
        (tb.bodies_index == sb.bodies_index && tb.branch_index == sb.branch_index) || error(
            "FastGaussSeidel requires structurally identical source/target trees, but branch " *
            "$i_branch differs: target (bodies_index=$(tb.bodies_index), branch_index=$(tb.branch_index)) " *
            "vs source (bodies_index=$(sb.bodies_index), branch_index=$(sb.branch_index)); " *
            "the target tree should replay the source tree's topology — please file an issue.")
    end
end

FastGaussSeidel(system; optargs...) = FastGaussSeidel((system,); optargs...)

FastGaussSeidel(systems::Tuple; optargs...) = FastGaussSeidel(systems, systems; optargs...)

FastGaussSeidel(target_systems, source_systems; optargs...) = FastGaussSeidel((target_systems,), (source_systems,); optargs...)

function FastGaussSeidel(target_systems::Tuple, source_systems::Tuple;
    expansion_order=4, multipole_acceptance=0.5, leaf_size=30,
    interaction_list_method=Barba(), shrink=true, recenter=false,
    derivatives_switches=DerivativesSwitch(true, true, false, target_systems),
    extra_farfield=false, cache_leaf_lu::Bool=true,
    sweep_order::Symbol=:lexicographic
)
    sweep_order in (:lexicographic, :colored) || throw(ArgumentError(
        "sweep_order must be :lexicographic or :colored (got $(repr(sweep_order)))"))

    #--- identical source and target trees ---#

    @assert target_systems === source_systems "different sources and targets are not yet supported for FastGaussSeidel"
    @assert interaction_list_method == Barba() "only the Barba() interaction_list_method is currently supported for FastGaussSeidel"

    #--- generate octree ---#

    # promote leaf_size to vector
    leaf_size = to_vector(leaf_size, length(source_systems))

    # create trees; the target tree REPLAYS the source tree's topology (identical branch
    # structure and body order) with a target-role shrink pass, since the block
    # bookkeeping below indexes both trees interchangeably and independently built trees
    # can diverge (source subdivision stops early at large body radii)
    TF = promote_type(numtype.(target_systems)...)
    switches = DerivativesSwitch(true, true, true, source_systems)
    source_tree = Tree(source_systems, false, switches; expansion_order, leaf_size, shrink, recenter, interaction_list_method)
    switches = DerivativesSwitch(true, true, true, target_systems)
    target_tree = Tree(source_tree, target_systems, switches; shrink, recenter)
    assert_shared_topology(target_tree, source_tree)

    #--- ensure no leaves have fewer than 2 bodies ---#

    # n_bodies_min = 1000000
    # for i_branch in source_tree.leaf_index
    #     n_bodies = get_n_bodies(source_tree.branches[i_branch].bodies_index)
    #     n_bodies_min = min(n_bodies_min, n_bodies)
    # end
    # @assert n_bodies_min >= 2 "The smallest leaf has only $n_bodies_min bodies, but at least 2 are required for FastGaussSeidel.\nConsider increasing the leaf_size or using a different interaction_list_method."

    #--- build interaction lists ---#

    farfield, nearfield, self_induced = true, true, false # self-induced interactions accounted for in self-influence matrices
    m2l_list, direct_list = build_interaction_lists(target_tree.branches, source_tree.branches, leaf_size, multipole_acceptance, farfield, nearfield, self_induced, interaction_list_method)

    # Canonical owner-major order is required before any threaded M2L
    # assignments or matrix/index maps are constructed.  `assign_m2l!`
    # partitions only at contiguous target boundaries; the parallel list
    # builder's assignment-order concatenation does not guarantee that all
    # interactions for one target are contiguous, which otherwise permits
    # concurrent `+=` into the same target expansion.
    # The counting sorts are stable and O(list + branches): source first,
    # then target, yields lexicographic (target, source) order.
    m2l_list = sort_by_target(sort_by_source(m2l_list, source_tree.branches),
                              target_tree.branches)
    direct_list = sort_by_target(sort_by_source(direct_list, source_tree.branches),
                                 target_tree.branches)

    #--- build non-self influence matrices ---#

    nonself_matrices, sorted_list = nonself_influence_matrices(target_tree.buffers, source_tree.buffers, source_systems, target_tree, source_tree, direct_list, derivatives_switches)
    old_influence_storage = similar(nonself_matrices.rhs)

    #--- full direct list includes leaf-on-self interactions ---#

    full_direct_list = add_self_interactions(direct_list, source_tree)

    #--- index by source ---#

    index_map = index_by_source(sorted_list, source_tree.leaf_index)

    #--- build self-influence matrices ---#

    self_matrices = self_influence_matrices(target_tree.buffers, source_tree.buffers, source_systems, target_tree, source_tree, derivatives_switches)
    leaf_lu_cache = cache_leaf_lu ? build_leaf_lu_cache(self_matrices) : nothing

    #--- source strength vector ---#

    strengths = zeros(TF, get_n_bodies(source_systems))
    strengths_by_leaf = map_by_leaf(source_tree)

    #--- mapping to targets by branch ---#

    targets_by_branch = map_by_branch(target_tree)

    #--- external right-hand side vector ---#

    extra_right_hand_side = Vector{TF}(undef, get_n_bodies(target_systems))

    #--- influences per system ---#

    influences_per_system = Vector{Vector{TF}}(undef, length(target_systems))
    for i_target_system in eachindex(target_systems)
        # get number of bodies in this target system
        n_bodies = get_n_bodies(target_systems[i_target_system])

        # create vector for influences
        influences_per_system[i_target_system] = zeros(TF, n_bodies)
    end

    #--- residual vector ---#

    # get max number of strengths in a leaf
    n_max = 0
    for i_leaf in eachindex(self_matrices.sizes)
        n_sources = self_matrices.sizes[i_leaf][1]
        n_max = max(n_max, n_sources)
    end

    # construct residual vector
    residual_vector = Vector{TF}(undef, n_max)

    #--- colored-sweep structures (opt-in; empty when lexicographic) ---#

    if sweep_order === :colored
        leaf_colors, leaves_by_color = color_leaves(source_tree, sorted_list,
                                                   index_map, targets_by_branch)
    else
        leaf_colors = Int[]
        leaves_by_color = Vector{Int}[]
    end

    return FastGaussSeidel{TF,length(source_systems),typeof(interaction_list_method),typeof(leaf_lu_cache)}(
        self_matrices,
        leaf_lu_cache,
        cache_leaf_lu,
        nonself_matrices,
        index_map,
        m2l_list,
        sorted_list,
        full_direct_list,
        interaction_list_method,
        multipole_acceptance,
        strengths,
        strengths_by_leaf,
        targets_by_branch,
        source_tree,
        target_tree,
        old_influence_storage,
        extra_right_hand_side,
        influences_per_system,
        residual_vector,
        extra_farfield,
        sweep_order,
        leaf_colors,
        leaves_by_color,
        Ref(false),
    )
end

"""
    transform_solver!(solver::FastGaussSeidel, target_systems::Tuple, R, t)

Update a `FastGaussSeidel` solver for a RIGID motion `x -> R*x + t` of its
(co-moving) systems, so the solver can be reused across timesteps instead of
being rebuilt: both trees are transformed with [`transform_tree!`](@ref)
(interaction lists are exactly invariant) and the target buffer
positions/metadata — written only at construction; `solve!` refreshes source
buffers and target INFLUENCE rows per call, never target positions — are
refreshed from the moved `target_systems`. Call AFTER the systems have moved.

Without this, `solve!` on a moved system silently forms far-field expansions
about construction-time branch centers: bodies leave their branches as
rotation accumulates and the multipole acceptance criterion that justified
the interaction lists no longer holds — error grows with rotation angle.

The dense self/nonself influence matrices stay untouched: their
scalar-potential rows depend only on relative geometry, which rigid motion
preserves exactly. Their GRADIENT rows are direction-carrying and do NOT
rotate with the body, so once a solver has been transformed, `solve!` with
`gradient=true` refuses loudly (scalar-potential solves remain exact).

Reusable dense blocks also require every kernel-consumed auxiliary geometry,
active kernel offset, and source-buffer radius to remain invariant under the
same rigid motion. Rebuild the solver after any non-rigid change to those
quantities.
"""
function transform_solver!(solver::FastGaussSeidel, target_systems::Tuple, R, t)
    get_n_bodies(target_systems) == length(solver.strengths) ||
        throw(ArgumentError("transform_solver!: target body count " *
            "$(get_n_bodies(target_systems)) does not match the solver's " *
            "$(length(solver.strengths)) — rigid motion cannot change body counts"))
    transform_tree!(solver.source_tree, R, t)
    solver.target_tree === solver.source_tree ||
        transform_tree!(solver.target_tree, R, t)
    switches = DerivativesSwitch(true, true, true, target_systems)
    target_to_buffer!(solver.target_tree.buffers, target_systems,
        solver.target_tree.sort_index_list, switches)
    solver.transformed[] = true
    return solver
end

function reset!(v::Vector{TF}) where TF<:Number
    # reset vector to zero
    v .= zero(TF)
end

function reset!(influences_per_system::Vector{<:AbstractVector})
    for i in eachindex(influences_per_system)
        reset!(influences_per_system[i])
    end
end

function update_by_leaf!(strengths::Vector, strengths_by_leaf::Vector{UnitRange{Int}}, source_systems::Tuple, source_buffers::AbstractVector{<:Matrix}, source_tree::Tree)
    # update strengths by leaf
    for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)
        # get bodies index
        bodies_index = source_tree.branches[i_branch].bodies_index

        if get_n_bodies(bodies_index) > 0
            # get strengths for this leaf
            strength_index = strengths_by_leaf[i_leaf]
            these_strengths = view(strengths, strength_index)

            # loop over source systems
            i_strength = 1
            for i_source_system in eachindex(source_systems)
                # unpack source system and buffer
                source_system = source_systems[i_source_system]
                source_buffer = source_buffers[i_source_system]

                # update strengths
                for i_body in bodies_index[i_source_system]
                    # set strength value
                    these_strengths[i_strength] = strength_to_value(source_buffer, source_system, i_body)

                    # increment index
                    i_strength += 1
                end
            end
            @assert i_strength - 1 == get_n_bodies(bodies_index) "number of strengths does not match number of bodies in leaf $(i_leaf)"
        end
    end
end

function update_by_leaf!(source_buffers::AbstractVector{<:Matrix}, source_systems::Tuple, strengths::Vector, strengths_by_leaf::Vector{UnitRange{Int}}, source_tree::Tree, rlx=1.0)
    # update strengths by leaf
    for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)
        # get bodies index
        bodies_index = source_tree.branches[i_branch].bodies_index

        if get_n_bodies(bodies_index) > 0
            # get strengths for this leaf
            strength_index = strengths_by_leaf[i_leaf]
            these_strengths = view(strengths, strength_index)

            # loop over source systems
            i_strength = 1
            for i_source_system in eachindex(source_systems)
                # unpack source system and buffer
                source_system = source_systems[i_source_system]
                source_buffer = source_buffers[i_source_system]

                # update strengths
                for i_body in bodies_index[i_source_system]
                    # set strength value
                    value_to_strength!(source_buffer, source_system, i_body, these_strengths[i_strength], rlx)

                    # increment index
                    i_strength += 1
                end
            end
            @assert i_strength - 1 == get_n_bodies(bodies_index) "number of strengths does not match number of bodies in leaf $(i_leaf)"
        end
    end
end

"""
    update_nonself_influence!(right_hand_side, nonself_matrices::Matrices, old_influence_storage::Vector, source_tree::Tree, target_tree::Tree, strengths_by_leaf::Vector{UnitRange{Int}}, index_map::Vector{UnitRange{Int}}, direct_list::Vector{SVector{2,Int32}}, targets_by_branch::Vector{UnitRange{Int}})

Updates the right-hand side vector based on the non-self influence matrices and the current strengths of the source systems. Does this by removing the old influence and adding the new.

"""
function update_nonself_influence!(right_hand_side, strengths::Vector, nonself_matrices::Matrices, old_influence_storage::Vector, source_tree::Tree, target_tree::Tree, strengths_by_leaf::Vector{UnitRange{Int}}, index_map::Vector{UnitRange{Int}}, direct_list::Vector{SVector{2,Int32}}, targets_by_branch::Vector{UnitRange{Int}})

    # check if there are any direct interactions
    if length(direct_list) > 0

        # loop over source leaves
        for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)
            update_nonself_influence!(right_hand_side, strengths, nonself_matrices, old_influence_storage, i_leaf, source_tree, target_tree, strengths_by_leaf, index_map, direct_list, targets_by_branch)
        end
    end
end

function update_nonself_influence!(right_hand_side, strengths::Vector, nonself_matrices::Matrices, old_influence_storage, i_leaf::Int, source_tree::Tree, target_tree::Tree, strengths_by_leaf::Vector{UnitRange{Int}}, index_map::Vector{UnitRange{Int}}, direct_list::Vector{SVector{2,Int32}}, targets_by_branch::Vector{UnitRange{Int}})
    compute_nonself_products!(strengths, nonself_matrices, old_influence_storage, i_leaf, strengths_by_leaf)
    scatter_nonself_influence!(right_hand_side, nonself_matrices, old_influence_storage, i_leaf, target_tree, index_map, direct_list, targets_by_branch)
    return nothing
end

# Thread-safe half of update_nonself_influence!: every write lands in leaf
# `i_leaf`'s own disjoint blocks of `nonself_matrices.rhs` and
# `old_influence_storage`, so distinct leaves may run concurrently.
function compute_nonself_products!(strengths::Vector, nonself_matrices::Matrices, old_influence_storage, i_leaf::Int, strengths_by_leaf::Vector{UnitRange{Int}})

    # unpack influence matrix and right-hand side
    mat, target_influence = get_matrix_vector(nonself_matrices, i_leaf)
    m = size(mat, 1)
    rhs_offset = nonself_matrices.rhs_offsets[i_leaf]
    old_influence = view(old_influence_storage, rhs_offset:rhs_offset + m - 1)

    if length(target_influence) > 0

        # unpack strengths
        leaf_strengths = view(strengths, strengths_by_leaf[i_leaf])

        # copy the old influence for later
        old_influence .= target_influence

        # compute the influence
        mul!(target_influence, mat, leaf_strengths)
    end
    return nothing
end

# Serial half of update_nonself_influence!: applies leaf `i_leaf`'s
# already-computed old/new products to the shared `right_hand_side` — target
# rows overlap between leaves, so calls must not run concurrently.
function scatter_nonself_influence!(right_hand_side, nonself_matrices::Matrices, old_influence_storage, i_leaf::Int, target_tree::Tree, index_map::Vector{UnitRange{Int}}, direct_list::Vector{SVector{2,Int32}}, targets_by_branch::Vector{UnitRange{Int}})

    _, target_influence = get_matrix_vector(nonself_matrices, i_leaf)
    m = length(target_influence)
    rhs_offset = nonself_matrices.rhs_offsets[i_leaf]
    old_influence = view(old_influence_storage, rhs_offset:rhs_offset + m - 1)

    if length(target_influence) > 0

        # determine which target branches this leaf influences
        direct_list_indices = index_map[i_leaf]
        i_influence_start = 1
        for index in direct_list_indices

            # determine which target leaf
            i_target, _ = direct_list[index]

            # how many targets
            n_targets = get_n_bodies(target_tree.branches[i_target].bodies_index)

            # influence on this target leaf
            this_influence = view(target_influence, i_influence_start:i_influence_start + n_targets - 1)
            this_old_influence = view(old_influence, i_influence_start:i_influence_start + n_targets - 1)
            i_influence_start += n_targets

            # get influence at the appropriate target
            this_rhs = view(right_hand_side, targets_by_branch[i_target])

            # remove old influence from right-hand side
            this_rhs .+= this_old_influence

            # add influence to the right-hand side
            this_rhs .-= this_influence
        end
    end
    return nothing
end

"""
    color_leaves(source_tree, sorted_list, index_map, targets_by_branch)

Greedy graph coloring of the source leaves by direct-interaction conflict, for
`sweep_order=:colored` (deterministic: pure function of the sorted lists,
independent of thread count). Two leaves conflict when one's nonself update
WRITES right-hand-side rows that overlap the other's OWN rows (its leaf-solve
READ set) — including overlap through non-leaf target branches, whose row
ranges span several leaves' rows. Within one color no leaf reads rows another
writes, so deferring all of a color's scatters to the color boundary
reproduces sequential Gauss-Seidel in color-major leaf order EXACTLY (the
serial ascending-leaf scatter preserves the sequential floating-point
accumulation order on shared target rows).

Returns `(leaf_colors, leaves_by_color)`, colors 1-based, leaves ascending
within each color.
"""
function color_leaves(source_tree::Tree, sorted_list::Vector{SVector{2,Int32}},
                      index_map::Vector{UnitRange{Int}},
                      targets_by_branch::Vector{UnitRange{Int}})

    n_leaves = length(source_tree.leaf_index)

    # each leaf's own rhs rows (ascending, disjoint: leaves partition bodies
    # in tree order)
    leaf_ranges = [targets_by_branch[i_branch] for i_branch in source_tree.leaf_index]
    leaf_starts = [first(r) for r in leaf_ranges]

    # leaves whose own rows overlap a target-branch row range. Leaf ranges
    # TILE the target rows contiguously in ascending order (each body lives in
    # exactly one leaf), so two binary searches bracket the overlap exactly.
    function overlapping_leaves(rows::UnitRange{Int})
        isempty(rows) && return 1:0
        lo = max(searchsortedlast(leaf_starts, first(rows)), 1)
        hi = max(searchsortedlast(leaf_starts, last(rows)), 1)
        return lo:hi
    end

    # adjacency: L writes targets_by_branch[i_target] for each of its direct
    # entries -> conflict with every leaf overlapping those rows (symmetrized)
    adjacency = [Set{Int}() for _ in 1:n_leaves]
    for i_leaf in 1:n_leaves
        for index in index_map[i_leaf]
            i_target, _ = sorted_list[index]
            for k_leaf in overlapping_leaves(targets_by_branch[i_target])
                k_leaf == i_leaf && continue
                push!(adjacency[i_leaf], k_leaf)
                push!(adjacency[k_leaf], i_leaf)
            end
        end
    end

    # greedy coloring, ascending leaf order, smallest admissible color
    leaf_colors = zeros(Int, n_leaves)
    used = Int[]
    for i_leaf in 1:n_leaves
        empty!(used)
        for k in adjacency[i_leaf]
            leaf_colors[k] > 0 && push!(used, leaf_colors[k])
        end
        c = 1
        while c in used
            c += 1
        end
        leaf_colors[i_leaf] = c
    end

    n_colors = n_leaves == 0 ? 0 : maximum(leaf_colors)
    leaves_by_color = [Int[] for _ in 1:n_colors]
    for i_leaf in 1:n_leaves   # ascending within color by construction
        push!(leaves_by_color[leaf_colors[i_leaf]], i_leaf)
    end

    return leaf_colors, leaves_by_color
end

"""
One Gauss-Seidel sweep over all source leaves.

- `sweep_order == :lexicographic` (default): the historical serial loop —
  solve leaf, immediately scatter its nonself update — bit-identical to the
  pre-refactor code. NOTE the historical `reverse_pass` quirk is preserved:
  the legacy loop iterated `enumerate(reverse(leaf_index))` but used only the
  enumeration counter `i_leaf`, so the "reverse" sweep visited leaves in
  FORWARD order; both sweep orders keep that behavior (flagged for review —
  changing it would alter the reverse_pass iteration).
- `sweep_order == :colored`: within each color, leaf solves and nonself
  PRODUCTS run in parallel (all writes per leaf are disjoint); the RHS
  scatter then runs serially in ascending leaf order at the color boundary.
  Coloring guarantees no same-color leaf reads rows another writes, so this
  reproduces sequential GS in color-major leaf order exactly (see
  `color_leaves`), deterministically at any thread count.
"""
function gs_sweep!(strengths, self_matrices, leaf_lu_cache, right_hand_side,
                   nonself_matrices, old_influence_storage, source_tree,
                   target_tree, strengths_by_leaf, index_map, direct_list,
                   targets_by_branch, solver, reverse_sweep::Bool)

    if solver.sweep_order === :colored

        for leaves in solver.leaves_by_color
            # parallel: per-leaf-disjoint writes only (strengths block +
            # nonself product/old-influence blocks)
            Threads.@threads for i_leaf in leaves
                leaf_strengths = view(strengths, strengths_by_leaf[i_leaf])
                solve_leaf!(leaf_strengths, self_matrices, leaf_lu_cache, i_leaf)
                length(direct_list) > 0 && compute_nonself_products!(strengths,
                    nonself_matrices, old_influence_storage, i_leaf,
                    strengths_by_leaf)
            end
            # serial scatter, fixed ascending order: preserves the sequential
            # accumulation order on shared target rows (determinism + the
            # color-major-GS equivalence)
            if length(direct_list) > 0
                for i_leaf in leaves
                    scatter_nonself_influence!(right_hand_side,
                        nonself_matrices, old_influence_storage, i_leaf,
                        target_tree, index_map, direct_list, targets_by_branch)
                end
            end
        end

    else # :lexicographic — the historical serial loop, bit-identical

        for (i_leaf, i_branch) in enumerate(source_tree.leaf_index)
            leaf_strengths = view(strengths, strengths_by_leaf[i_leaf])
            solve_leaf!(leaf_strengths, self_matrices, leaf_lu_cache, i_leaf)
            length(direct_list) > 0 && update_nonself_influence!(
                right_hand_side, strengths, nonself_matrices,
                old_influence_storage, i_leaf, source_tree, target_tree,
                strengths_by_leaf, index_map, direct_list, targets_by_branch)
        end

    end

    return nothing
end

solve!(system, solver::FastGaussSeidel; optargs...) = solve!((system,), solver; optargs...)

solve!(systems::Tuple, solver::FastGaussSeidel; optargs...) = solve!(systems, systems, solver; optargs...)

solve!(target_systems, source_systems, solver::FastGaussSeidel; optargs...) = solve!((target_systems,), (source_systems,), solver; optargs...)

function solve!(target_systems::Tuple, source_systems::Tuple, solver::FastGaussSeidel{TF,N};
    scalar_potential=false, gradient=true, hessian=false,
    max_iterations=10, inner_iterations=1, tolerance=1e-3,
    rlx=1.0, reverse_pass=false, verbose=true, final_update=true,
    callback=nothing
) where {TF,N}

    #--- refuse direction-carrying outputs on a transformed solver ---#

    # the dense influence matrices embed build-time gradient rows, which do
    # not rotate with the body (see transform_solver!)
    if solver.transformed[] &&
            (any(to_vector(gradient, N)) || any(to_vector(hessian, N)))
        throw(ArgumentError("this FastGaussSeidel solver has been rigidly " *
            "transformed (transform_solver!): its dense influence matrices " *
            "embed build-time gradient/hessian rows, which do not rotate " *
            "with the body — only scalar_potential solves are valid. " *
            "Rebuild the solver for gradient solves under rigid motion."))
    end

    #--- derivatives switches ---#

    derivatives_switches = DerivativesSwitch(scalar_potential, gradient, hessian, target_systems)

    #--- unpack containers ---#

    source_tree = solver.source_tree
    target_tree = solver.target_tree
    source_buffers = source_tree.buffers
    target_buffers = target_tree.buffers
    self_matrices = solver.self_matrices
    leaf_lu_cache = solver.leaf_lu_cache
    nonself_matrices = solver.nonself_matrices
    index_map = solver.index_map
    m2l_list = solver.m2l_list
    direct_list = solver.direct_list
    full_direct_list = solver.full_direct_list
    interaction_list_method = solver.interaction_list_method
    multipole_acceptance = solver.multipole_acceptance
    strengths = solver.strengths
    strengths_by_leaf = solver.strengths_by_leaf
    targets_by_branch = solver.targets_by_branch
    influences_per_system = solver.influences_per_system
    old_influence_storage = solver.old_influence_storage
    right_hand_side = self_matrices.rhs
    extra_right_hand_side = solver.extra_right_hand_side
    residual_vector = solver.residual_vector

    #--- external right-hand side based on current influence ---#

    # reset and update buffers
    target_influence_to_buffer!(target_buffers, target_systems, derivatives_switches, target_tree.sort_index_list)

    # run influence function on buffers
    reset!(extra_right_hand_side)
    influence!(extra_right_hand_side, influences_per_system, target_buffers, source_systems, source_buffers, source_tree, derivatives_switches)

    # set the right-hand side to the external influence
    # NOTE: this only happens once as it is not reset in the iterations
    right_hand_side .= extra_right_hand_side

    #--- update strengths ---#

    system_to_buffer!(source_buffers, source_systems, source_tree.sort_index_list)
    update_by_leaf!(strengths, strengths_by_leaf, source_systems, source_buffers, source_tree)

    #--- update strengths in buffers to match ---#

    update_by_leaf!(source_buffers, source_systems, strengths, strengths_by_leaf, source_tree)

    #--- non-self influence ---#

    # add nonself influence to the right-hand side
    nonself_matrices.rhs .= zero(TF) # reset rhs
    update_nonself_influence!(right_hand_side, strengths, nonself_matrices, old_influence_storage, source_tree, target_tree, strengths_by_leaf, index_map, direct_list, targets_by_branch)

    #--- fast gauss seidel iterations ---#

    # prepare inputs
    empty_direct_list = Vector{Tuple{Int,Int}}(undef, 0)
    mse = one(TF) * 100000
    # mse_best = mse

    Δ = zero(TF) # track change in strengths
    strengths_old = similar(strengths)

    # begin iterations
    for iteration in 1:max_iterations

        #--- farfield influence ---#

        # fmm call
        reset!(target_buffers)

        fmm!(target_systems, target_tree, source_systems, source_tree, source_tree.leaf_size, m2l_list, empty_direct_list, derivatives_switches, interaction_list_method;
            source_tree.expansion_order, error_tolerance=nothing,
            upward_pass=true, horizontal_pass=true, downward_pass=true,
            # horizontal_pass_verbose::Bool=false,
            reset_target_tree=true, reset_source_tree=true,
            # nearfield_device::Bool=false,
            tune=false, update_target_systems=false, multipole_acceptance,
            # t_source_tree=0.0, t_target_tree=0.0, t_lists=0.0,
            # silence_warnings=false,
            extra_farfield=solver.extra_farfield,
        )

        # move farfield influence to the right-hand side
        reset!(extra_right_hand_side)
        influence!(extra_right_hand_side, influences_per_system, target_buffers, source_systems, source_buffers, source_tree, derivatives_switches)
        right_hand_side .+= extra_right_hand_side

        #--- check residual ---#

        # note that `right_hand_side` now contains external, nonself, and farfield influence
        mse = residual!(residual_vector, self_matrices, strengths, strengths_by_leaf)

        # convergence-history hook: called once per outer iteration with the
        # exact residual the loop's tolerance check uses (max-abs, despite the
        # MSE label); default nothing → zero behavior change
        callback === nothing || callback(iteration, mse)

        verbose && println("Iteration $(iteration):\tMSE = $(mse),\tdelta = $(Δ)")

        # if mse > mse_best * 10 # stop if mse begins increasing
        #     @warn "FastGaussSeidel stopped early at iteration $(iteration) with MSE = $(mse) (previous was $(mse_best))"
        #     break
        # end
        # mse_best = min(mse_best, mse)

        if mse <= tolerance
            if verbose
                @info "FastGaussSeidel converged after $(iteration-1) iterations with MSE = $(mse); delta = $(Δ)"
            end
            break
        end

        #--- nearfield influence and solve ---#

        strengths_old .= strengths

        for i_inner in 1:inner_iterations

            gs_sweep!(strengths, self_matrices, leaf_lu_cache, right_hand_side,
                nonself_matrices, old_influence_storage, source_tree,
                target_tree, strengths_by_leaf, index_map, direct_list,
                targets_by_branch, solver, false)

            if reverse_pass
                #--- reverse pass ---#
                gs_sweep!(strengths, self_matrices, leaf_lu_cache,
                    right_hand_side, nonself_matrices, old_influence_storage,
                    source_tree, target_tree, strengths_by_leaf, index_map,
                    direct_list, targets_by_branch, solver, true)
            end

        end

        # get delta
        strengths_old .-= strengths
        strengths_old .*= strengths_old
        Δ = sqrt(maximum(strengths_old))

        #--- update strengths in buffers ---#

        update_by_leaf!(source_buffers, source_systems, strengths, strengths_by_leaf, source_tree, rlx)

        #--- restore right hand side to exclude farfield influence ---#

        right_hand_side .-= extra_right_hand_side

    end

    #--- check if we converged ---#
    if mse > tolerance
        if verbose
            @warn "FastGaussSeidel did not converge after $(max_iterations) iterations with MSE = $(mse)"
        end
    end

    #--- final update of systems ---#

    # use new strengths to get the full influence (farfield was already computed)
    if final_update
        fmm!(target_systems, target_tree, source_systems, source_tree, source_tree.leaf_size, m2l_list, full_direct_list, derivatives_switches, interaction_list_method;
                expansion_order=source_tree.expansion_order, error_tolerance=nothing,
                upward_pass=false, horizontal_pass=false, downward_pass=false, # just nearfield influence
                # horizontal_pass_verbose::Bool=false,
                reset_target_tree=false, reset_source_tree=false, # false now
                # nearfield_device::Bool=false,
                tune=false, update_target_systems=true, multipole_acceptance, # update_target_systems is `true` now
                # t_source_tree=0.0, t_target_tree=0.0, t_lists=0.0,
                # silence_warnings=false,
                extra_farfield=solver.extra_farfield
            )
    end

    # update source system strengths
    buffer_to_system_strength!(source_systems, source_tree)

end

"""
    influence!(sorted_influences, influences_per_system, target_buffers, source_systems, source_buffers, source_tree, derivatives_switches)

Evaluate the influence as pertains to the boundary element influence matrix and subtracts it from `sorted_influences` (which would act like the RHS of a linear system). Based on the current state of the `target_buffers` and `source_buffers`. Note that `source_systems` is provided solely for dispatch. Note also that `influences_per_system` is overwritten each time.

* `sorted_influences::Vector{Float64}`: single vector containing the influence for every body in the target buffers, sorted by source branch in the direct interaction list
* `influences_per_system::Vector{Vector{Float64}}`: vector of vectors containing the influence for each target system, sorted the same way as the buffers
* `target_buffers::Vector{Matrix{Float64}}`: target buffers used to compute the influence
* `source_systems::Tuple`: system objects used for dispatch
* `source_buffers::Vector{Matrix{Float64}}`: source buffers used to compute the influence

"""
function influence!(sorted_influences::Vector{TF}, influences_per_system::Vector{Vector{TF}}, target_buffers::AbstractVector{<:Matrix}, source_systems::Tuple, source_buffers::AbstractVector{<:Matrix}, source_tree::Tree, derivatives_switches::Tuple) where TF
    @assert length(target_buffers) == length(source_buffers) == length(source_systems)

    #--- evaluate influences ---#

    for i_system in eachindex(target_buffers)
        # unpack containers
        influence = influences_per_system[i_system]
        target_buffer = target_buffers[i_system]
        source_buffer = source_buffers[i_system]
        source_system = source_systems[i_system]
        derivatives_switch = derivatives_switches[i_system]

        # evaluate influence
        influence!(influence, target_buffer, derivatives_switch, source_system, source_buffer)
    end

    #--- sort by source ---#

    i_influence = 1
    for i_leaf in source_tree.leaf_index
        # unpack bodies index
        bodies_index = source_tree.branches[i_leaf].bodies_index

        for i_system in eachindex(source_systems)
            # unpack containers
            index = bodies_index[i_system]
            influences = influences_per_system[i_system]

            # update influences
            sorted_influences[i_influence:i_influence + length(index) - 1] .-= view(influences, index)
            i_influence += length(index)
        end
    end
end

#------- Block Jacobi Preconditioner -------#

"""
    JacobiPreconditioner{TF}

Block Jacobi preconditioner that partitions bodies into uniform grid cells and
stores per-cell LU-factorized influence matrices.

`cell_body_indices[k]` contains the original (global) body indices for cell `k`,
matching the row/column ordering of `lu_factorizations[k]`.
"""
struct JacobiPreconditioner{TF,LF<:LU}
    lu_factorizations::Vector{LF}
    cell_body_indices::Vector{Vector{Int}}  # original body indices per cell
    cell_nonsingular::Vector{Bool}          # whether each cell's LU is nonsingular
    n_bodies::Int
end

"""
    JacobiPreconditioner(target_systems, source_systems; cell_size, derivatives_switches)

Build a block Jacobi preconditioner by partitioning bodies into uniform grid cells
and assembling/factorizing the self-influence matrix for each cell.

Bodies are hashed into a uniform grid (cell list) following the pattern of FLOWVPM's
`merge_particles!`. For each non-empty cell, a dense influence matrix is assembled
column-by-column using `direct!` with unit strengths (matching `self_influence_matrices`),
then LU-factorized.
"""
JacobiPreconditioner(system; optargs...) = JacobiPreconditioner((system,); optargs...)
JacobiPreconditioner(systems::Tuple; optargs...) = JacobiPreconditioner(systems, systems; optargs...)

function JacobiPreconditioner(target_systems::Tuple, source_systems::Tuple;
    cell_size::Real,
    derivatives_switches=DerivativesSwitch(true, true, false, target_systems),
)
    @assert target_systems === source_systems "different sources and targets are not yet supported for JacobiPreconditioner"

    TF = get_type(target_systems, source_systems)
    n_bodies_total = get_n_bodies(source_systems)

    #--- build uniform cell list ---#

    # collect positions (global indexing across all systems)
    positions = Vector{SVector{3,TF}}(undef, n_bodies_total)
    body_offset = 0
    for system in source_systems
        nb = get_n_bodies(system)
        for i in 1:nb
            positions[body_offset + i] = get_position(system, i)
        end
        body_offset += nb
    end

    # compute bounding box
    xmin = ymin = zmin = typemax(TF)
    xmax = ymax = zmax = typemin(TF)
    for pos in positions
        xmin = min(xmin, pos[1]); xmax = max(xmax, pos[1])
        ymin = min(ymin, pos[2]); ymax = max(ymax, pos[2])
        zmin = min(zmin, pos[3]); zmax = max(zmax, pos[3])
    end

    # grid dimensions
    Nx = max(1, floor(Int, (xmax - xmin) / cell_size) + 1)
    Ny = max(1, floor(Int, (ymax - ymin) / cell_size) + 1)
    Nz = max(1, floor(Int, (zmax - zmin) / cell_size) + 1)
    n_cells = Nx * Ny * Nz

    # hash bodies into cells (counting sort)
    cell_keys = Vector{Int}(undef, n_bodies_total)
    cell_offsets = zeros(Int, n_cells + 1)

    for i in 1:n_bodies_total
        pos = positions[i]
        ix = clamp(floor(Int, (pos[1] - xmin) / cell_size), 0, Nx - 1)
        iy = clamp(floor(Int, (pos[2] - ymin) / cell_size), 0, Ny - 1)
        iz = clamp(floor(Int, (pos[3] - zmin) / cell_size), 0, Nz - 1)
        key = ix + iy * Nx + iz * Nx * Ny
        cell_keys[i] = key
        cell_offsets[key + 2] += 1
    end

    # prefix sum
    for i in 2:(n_cells + 1)
        cell_offsets[i] += cell_offsets[i - 1]
    end

    # place bodies in sorted order (sorted_indices[j] = original body index)
    sorted_indices = Vector{Int}(undef, n_bodies_total)
    counts = copy(cell_offsets)
    for i in 1:n_bodies_total
        key = cell_keys[i] + 1
        counts[key] += 1
        sorted_indices[counts[key]] = i
    end

    # collect non-empty cells as vectors of original body indices
    cell_body_indices = Vector{Vector{Int}}()
    for key in 0:(n_cells - 1)
        range_start = cell_offsets[key + 1] + 1
        range_stop = cell_offsets[key + 2]
        range_start > range_stop && continue
        push!(cell_body_indices, sorted_indices[range_start:range_stop])
    end

    #--- allocate buffers ---#

    # use full-size target buffers (hessian=true) so reset! can zero rows 4:16
    full_switches = DerivativesSwitch(true, true, true, target_systems)
    target_buffers = allocate_buffers(target_systems, true, TF, full_switches)
    source_buffers = allocate_buffers(source_systems, false, TF, DerivativesSwitch(false, false, false, source_systems))

    # load systems into buffers (unsorted — identity permutation)
    target_to_buffer!(target_buffers, target_systems, SVector{length(target_systems)}([1:get_n_bodies(system) for system in target_systems]), full_switches)
    system_to_buffer!(source_buffers, source_systems)

    #--- assemble per-cell influence matrices and LU-factorize ---#

    old_strengths = save_strengths(source_buffers, source_systems)
    set_unit_strength!(source_buffers, source_systems)

    LF = typeof(lu(zeros(TF, 0, 0)))
    lu_factorizations = Vector{LF}(undef, length(cell_body_indices))
    cell_nonsingular = Vector{Bool}(undef, length(cell_body_indices))

    for (i_cell, body_indices) in enumerate(cell_body_indices)
        n_cell = length(body_indices)
        cell_matrix = zeros(TF, n_cell, n_cell)

        for (col, i_source_global) in enumerate(body_indices)
            i_sys, i_local = _global_to_system_index(source_systems, i_source_global)
            source_system = source_systems[i_sys]
            source_buffer = source_buffers[i_sys]

            for (row, i_target_global) in enumerate(body_indices)
                t_sys, t_local = _global_to_system_index(target_systems, i_target_global)
                target_buffer = target_buffers[t_sys]

                # reset target
                reset!(target_buffer, t_local:t_local)

                # compute direct influence of single source on single target
                direct!(target_buffer, t_local:t_local, derivatives_switches[t_sys], source_system, source_buffer, i_local:i_local)

                # extract influence value
                infl = zeros(TF, 1)
                influence!(infl, view(target_buffer, :, t_local:t_local), derivatives_switches[t_sys], source_system, view(source_buffer, :, i_local:i_local))
                cell_matrix[row, col] = infl[1]
            end
        end

        F = lu(cell_matrix; check=false)
        lu_factorizations[i_cell] = F
        cell_nonsingular[i_cell] = issuccess(F)
    end

    restore_strengths!(source_buffers, source_systems, old_strengths)

    return JacobiPreconditioner{TF,LF}(lu_factorizations, cell_body_indices, cell_nonsingular, n_bodies_total)
end

"""
    _global_to_system_index(systems, global_index)

Given a global body index (1-based across all systems), return `(system_index, local_index)`.
"""
function _global_to_system_index(systems, global_index)
    offset = 0
    for (i_sys, system) in enumerate(systems)
        nb = get_n_bodies(system)
        if global_index <= offset + nb
            return i_sys, global_index - offset
        end
        offset += nb
    end
    error("global_index $global_index out of range for systems with $(sum(get_n_bodies(s) for s in systems)) total bodies")
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::JacobiPreconditioner, x::AbstractVector)
    y .= x  # default: identity preconditioning for uncovered/singular cells
    for (k, body_indices) in enumerate(P.cell_body_indices)
        P.cell_nonsingular[k] || continue
        ldiv!(view(y, body_indices), P.lu_factorizations[k], view(x, body_indices))
    end
    return y
end

function residual!(residual_vector, self_matrices::Matrices, strengths::Vector, strengths_by_leaf::Vector{UnitRange{Int}})

    # initialize mean squared error
    mse = zero(eltype(residual_vector))
    mae = zero(eltype(residual_vector))
    n = 0

    # loop over self influence matrices
    for i_leaf in eachindex(self_matrices.rhs_offsets)
        # get matrix and rhs
        mat, rhs = get_matrix_vector(self_matrices, i_leaf)

        # unpack strengths
        leaf_strengths = view(strengths, strengths_by_leaf[i_leaf])

        # compute the residual
        vrange = 1:length(rhs)
        this_residual = view(residual_vector, vrange)
        this_residual .= rhs
        mul!(this_residual, mat, leaf_strengths, 1.0, -1.0)

        # aggregate mse residual
        for r in this_residual
            mse += r * r
            n += length(this_residual)
            mae = max(mae, abs(r))
        end
    end

    # compute the mean squared error
    mse /= n

    # mse = maximum(abs.(residual_vector)) # temporary check

    return mae
    # return mse
end
