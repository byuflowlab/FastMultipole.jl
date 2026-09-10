#------- near-field influence-matrix cache -------#

# For frozen relative geometry, every near-field direct interaction is a
# linear map from source strengths to target outputs. Cache each direct-list
# entry's dense block once (unit-strength column probing through `direct!`),
# then re-evaluate the whole near-field as packed BLAS matvecs.
#
# Validity contract: Tree identity (objectid) IS the validity key — callers
# own invalidation and must rebuild the cache for a new Tree. Only source
# STRENGTHS (buffer rows 5:4+strength_dims) may change between evaluations;
# any system whose kernel reads strength-like inputs outside those rows will
# produce wrong results — the exactness tests guard this per system family.

const NEARFIELD_CACHE_DEFAULT_MAX_BYTES = 4 * 1024^3   # 4 GiB

struct NearfieldInfluenceCache{TF,TDS}
    matrices::Matrices{TF}                 # one dense block per (direct-list entry × system pair), target-major order
    entries::Vector{SVector{4,Int32}}      # (i_target_branch, i_source_branch, i_target_system, i_source_system); branch 0 = treeless
    target_ranges::Vector{UnitRange{Int}}  # target-buffer body columns, per block
    source_ranges::Vector{UnitRange{Int}}  # source-buffer body columns, per block
    output_ranges::Vector{UnitRange{Int}}  # target-buffer output rows, per block
    n_out::Vector{Int}                     # output rows per target body, per block
    n_comp::Vector{Int}                    # strength components per source body, per block
    strengths_scratch::Vector{Vector{TF}}  # per-thread strength gather buffers
    derivatives_switches::TDS
    target_tree_id::UInt                   # objectid of the trees at build; 0 for treeless caches
    source_tree_id::UInt
    n_target_bodies::Int
    n_source_bodies::Int
    build_time::Float64
    bytes::Int
end

"""
    NearfieldInfluenceCache(target_systems, target_tree, source_systems, source_tree,
        direct_list, derivatives_switches; max_bytes, direct_conditioning=())

Build a dense near-field cache for the (target-sorted) `direct_list`: one block
per direct-list entry and system pair, probed column-by-column with unit
strengths through `direct!`. Blocks are stored in target-major order so
owner-partitioned parallel evaluation accumulates race-free and
deterministically.

Refuses to build when `direct_conditioning` rules are present (conditioning may
mutate non-strength buffer rows around evaluation) or when the estimated size
exceeds `max_bytes` (checked BEFORE allocation).
"""
function NearfieldInfluenceCache(target_systems::Tuple, target_tree::Tree,
        source_systems::Tuple, source_tree::Tree, direct_list,
        derivatives_switches::Tuple;
        max_bytes::Integer=NEARFIELD_CACHE_DEFAULT_MAX_BYTES,
        max_build_time::Real=Inf,
        n_threads::Integer=Threads.nthreads(),
        direct_conditioning=())

    _refuse_conditioning(direct_conditioning, "build")

    entries, target_ranges, source_ranges =
        _tree_block_specs(target_tree, source_tree, direct_list)

    return _build_nearfield_cache(entries, target_ranges, source_ranges,
        target_tree.buffers, source_systems, source_tree.buffers,
        derivatives_switches, max_bytes, max_build_time,
        objectid(target_tree), objectid(source_tree); n_threads)
end

# block specs in target-major (direct-list) order
function _tree_block_specs(target_tree::Tree, source_tree::Tree, direct_list)
    target_branches = target_tree.branches
    source_branches = source_tree.branches
    entries = SVector{4,Int32}[]
    target_ranges = UnitRange{Int}[]
    source_ranges = UnitRange{Int}[]
    for (i_target, i_source) in direct_list
        for i_target_system in eachindex(target_branches[i_target].bodies_index)
            target_index = target_branches[i_target].bodies_index[i_target_system]
            length(target_index) == 0 && continue
            for i_source_system in eachindex(source_branches[i_source].bodies_index)
                source_index = source_branches[i_source].bodies_index[i_source_system]
                length(source_index) == 0 && continue
                push!(entries, SVector{4,Int32}(i_target, i_source, i_target_system, i_source_system))
                push!(target_ranges, target_index)
                push!(source_ranges, source_index)
            end
        end
    end
    return entries, target_ranges, source_ranges
end

"""
    NearfieldInfluenceCache(target_systems, target_buffers, source_systems,
        source_buffers, derivatives_switches; max_bytes)

Degenerate no-tree form for standalone `direct!`: one block per system pair
over the full body index ranges. Body counts are the only validity guard.
"""
function NearfieldInfluenceCache(target_systems::Tuple, target_buffers,
        source_systems::Tuple, source_buffers, derivatives_switches::Tuple;
        max_bytes::Integer=NEARFIELD_CACHE_DEFAULT_MAX_BYTES,
        max_build_time::Real=Inf,
        n_threads::Integer=Threads.nthreads(),
        direct_conditioning=())

    _refuse_conditioning(direct_conditioning, "build")

    entries = SVector{4,Int32}[]
    target_ranges = UnitRange{Int}[]
    source_ranges = UnitRange{Int}[]
    for i_target_system in eachindex(target_systems)
        n_t = size(target_buffers[i_target_system], 2)
        n_t == 0 && continue
        for i_source_system in eachindex(source_systems)
            n_s = size(source_buffers[i_source_system], 2)
            n_s == 0 && continue
            push!(entries, SVector{4,Int32}(0, 0, i_target_system, i_source_system))
            push!(target_ranges, 1:n_t)
            push!(source_ranges, 1:n_s)
        end
    end

    return _build_nearfield_cache(entries, target_ranges, source_ranges,
        target_buffers, source_systems, source_buffers,
        derivatives_switches, max_bytes, max_build_time, UInt(0), UInt(0);
        n_threads)
end

# one size pass shared by the builder and the estimator (no drift):
# per-block (m, n), output metadata, total flop/probe counts, and bytes
function _nearfield_cache_size_pass(entries, target_ranges, source_ranges,
        derivatives_switches, source_systems, TF)
    n_blocks = length(entries)
    sizes = Vector{Tuple{Int,Int}}(undef, n_blocks)
    n_out = Vector{Int}(undef, n_blocks)
    n_comp = Vector{Int}(undef, n_blocks)
    output_ranges = Vector{UnitRange{Int}}(undef, n_blocks)
    total_mn = 0
    total_m = 0
    max_width = 0
    total_probe_pairs = 0   # the probe loop touches each (target, source) pair n_comp times
    for k in 1:n_blocks
        i_ts = entries[k][3]
        i_ss = entries[k][4]
        out_range = output_range(derivatives_switches[i_ts])
        sd = strength_dims(source_systems[i_ss])
        m = length(out_range) * length(target_ranges[k])
        n = sd * length(source_ranges[k])
        sizes[k] = (m, n)
        n_out[k] = length(out_range)
        n_comp[k] = sd
        output_ranges[k] = out_range
        total_mn += m * n
        total_m += m
        max_width = max(max_width, n)
        total_probe_pairs += sd * length(target_ranges[k]) * length(source_ranges[k])
    end
    bytes = sizeof(TF) * (total_mn + total_m)
    return (; sizes, n_out, n_comp, output_ranges, max_width,
              total_probe_pairs, bytes)
end

# per-(target,source)-pair kernel time from one warmed-up single-source
# direct! sample on the first block (min-of-3); leaves the sampled output
# rows zeroed
function _sample_probe_time(target_buffers, source_buffers, source_systems,
        derivatives_switches, entries, target_ranges, source_ranges,
        output_ranges)
    i_ts = entries[1][3]
    i_ss = entries[1][4]
    switch = derivatives_switches[i_ts]
    target_buffer = target_buffers[i_ts]
    source_buffer = source_buffers[i_ss]
    source_system = source_systems[i_ss]
    target_range = target_ranges[1]
    i_body = first(source_ranges[1]):first(source_ranges[1])
    # warm up (compile), then time
    direct!(target_buffer, target_range, switch, source_system, source_buffer, i_body)
    t = minimum(@elapsed direct!(target_buffer, target_range, switch,
            source_system, source_buffer, i_body)
        for _ in 1:3)
    @views target_buffer[output_ranges[1], target_range] .= zero(eltype(target_buffer))
    return t / length(target_range)
end

"""
    estimate_nearfield_cache(target_tree, source_tree, direct_list,
        derivatives_switches, source_systems; sample=true)

Estimate a [`NearfieldInfluenceCache`](@ref)'s cost WITHOUT building it:
returns `(; bytes, est_build_time, n_blocks, total_probe_pairs)`. `bytes`
uses the exact size-pass arithmetic the builder uses; `est_build_time` times
one warmed-up single-source kernel evaluation and scales it by the number of
probe pairs (`sample=false` skips the timing and reports `NaN`). Used by the
builder's `max_build_time` guard and by cached-near-field autotuning
feasibility checks (non-throwing by design).
"""
function estimate_nearfield_cache(target_tree::Tree, source_tree::Tree,
        direct_list, derivatives_switches::Tuple, source_systems::Tuple;
        sample::Bool=true, n_threads::Integer=1)
    entries, target_ranges, source_ranges =
        _tree_block_specs(target_tree, source_tree, direct_list)
    TF = promote_type(eltype.(source_tree.buffers)...)
    sp = _nearfield_cache_size_pass(entries, target_ranges, source_ranges,
        derivatives_switches, source_systems, TF)
    est_build_time = NaN
    if length(entries) == 0
        est_build_time = 0.0
    elseif sample
        t_per_pair = _sample_probe_time(target_tree.buffers,
            source_tree.buffers, source_systems, derivatives_switches,
            entries, target_ranges, source_ranges, sp.output_ranges)
        # the parallel build splits keys across up to n_threads tasks; assume
        # ideal scaling (this estimate feeds wall-clock guards, not results)
        est_build_time = t_per_pair * sp.total_probe_pairs / max(1, n_threads)
    end
    return (; bytes=sp.bytes, est_build_time, n_blocks=length(entries),
              total_probe_pairs=sp.total_probe_pairs)
end

@inline function _refuse_conditioning(direct_conditioning, stage::String)
    rules = normalize_direct_conditioning(direct_conditioning)
    has_direct_conditioning(rules) && throw(ArgumentError(
        "NearfieldInfluenceCache cannot be used with direct_conditioning rules " *
        "(conditioning mutates source buffers around near-field evaluation, " *
        "which the cached linear map cannot represent); remove the rules or " *
        "disable the cache ($stage)"))
    return nothing
end

# probe every source column owned by the keys in `key_range`, filling their
# blocks' matrix columns. `target_buffers` may be the caller's buffers (serial)
# or a thread-private copy (parallel build): the probe writes target output
# rows, so concurrent probes must not share them. Matrix-column writes are
# disjoint across keys (each block belongs to exactly one key) and strength
# pokes touch disjoint source-buffer columns, so the shared `matrices` and
# `source_buffers` need no synchronization.
function _probe_source_keys!(matrices, keys_sorted, key_range, blocks_by_source,
        entries, target_ranges, source_ranges, output_ranges,
        target_buffers, source_systems, source_buffers,
        derivatives_switches, ::Type{TF}) where TF
    for i_key in key_range
        key = keys_sorted[i_key]
        _, i_ss = key
        block_list = blocks_by_source[key]
        source_system = source_systems[i_ss]
        source_buffer = source_buffers[i_ss]
        sd = strength_dims(source_system)
        source_range = source_ranges[block_list[1]]
        for i_body in source_range
            for i_comp in 1:sd
                source_buffer[4+i_comp, i_body] = one(TF)
                j = (i_body - first(source_range)) * sd + i_comp
                for k in block_list
                    i_ts = entries[k][3]
                    switch = derivatives_switches[i_ts]
                    target_buffer = target_buffers[i_ts]
                    target_range = target_ranges[k]
                    out_range = output_ranges[k]
                    @views target_buffer[out_range, target_range] .= zero(TF)
                    direct!(target_buffer, target_range, switch, source_system,
                        source_buffer, i_body:i_body)
                    block, _ = get_matrix_vector(matrices, k)
                    @views block[:, j] .= vec(target_buffer[out_range, target_range])
                end
                source_buffer[4+i_comp, i_body] = zero(TF)
            end
        end
    end
    return nothing
end

# contiguous key-range chunks balanced by per-key probe work
function _chunk_keys_by_work(key_work::Vector{Int}, n_chunks::Int)
    n_keys = length(key_work)
    n_chunks = min(n_chunks, n_keys)
    n_chunks <= 1 && return [1:n_keys]
    total = sum(key_work)
    per_chunk = cld(total, n_chunks)
    chunks = UnitRange{Int}[]
    i_start = 1
    work = 0
    for i in 1:n_keys
        work += key_work[i]
        if work >= per_chunk && length(chunks) < n_chunks - 1
            push!(chunks, i_start:i)
            i_start = i + 1
            work = 0
        end
    end
    i_start <= n_keys && push!(chunks, i_start:n_keys)
    return chunks
end

function _build_nearfield_cache(entries, target_ranges, source_ranges,
        target_buffers, source_systems, source_buffers, derivatives_switches,
        max_bytes, max_build_time, target_tree_id::UInt, source_tree_id::UInt;
        n_threads::Integer=Threads.nthreads())

    start_time = time_ns()
    TF = promote_type(eltype.(source_buffers)...)

    # size pass + guards BEFORE allocation
    n_blocks = length(entries)
    sp = _nearfield_cache_size_pass(entries, target_ranges, source_ranges,
        derivatives_switches, source_systems, TF)
    sp.bytes > max_bytes && throw(ArgumentError(
        "NearfieldInfluenceCache would require $(sp.bytes) bytes " *
        "($(round(sp.bytes / 1024^3; digits=2)) GiB) for $n_blocks blocks, " *
        "exceeding max_bytes = $max_bytes; raise max_bytes or disable the cache"))

    # group blocks by (source branch, source system) so each unit-strength
    # probe fills every block that reads it; keys chunked by probe work for
    # the parallel build
    blocks_by_source = Dict{Tuple{Int,Int},Vector{Int}}()
    for k in 1:n_blocks
        key = (Int(entries[k][2]), Int(entries[k][4]))
        push!(get!(() -> Int[], blocks_by_source, key), k)
    end
    keys_sorted = sort!(collect(keys(blocks_by_source)))
    key_work = [sum(prod(sp.sizes[k]) for k in blocks_by_source[key])
                for key in keys_sorted]
    # finer chunks than workers so the atomic-counter pool below absorbs any
    # residual imbalance (measured ~8% on a cheap kernel; free for expensive
    # ones — buffer copies are per WORKER, not per chunk)
    chunks = _chunk_keys_by_work(key_work, 4 * n_threads)
    n_workers = min(Int(n_threads), length(chunks))

    if isfinite(max_build_time) && n_blocks > 0
        t_per_pair = _sample_probe_time(target_buffers, source_buffers,
            source_systems, derivatives_switches, entries, target_ranges,
            source_ranges, sp.output_ranges)
        # the probe parallelizes over key chunks; assume ideal scaling for the
        # guard (it is a wall-clock backstop, not a promise)
        est_build_time = t_per_pair * sp.total_probe_pairs / n_workers
        est_build_time > max_build_time && throw(ArgumentError(
            "NearfieldInfluenceCache build is estimated at " *
            "$(round(est_build_time; digits=2)) s (kernel sample " *
            "$(t_per_pair) s/pair × $(sp.total_probe_pairs) probe pairs / " *
            "$(length(chunks)) build threads), " *
            "exceeding max_build_time = $max_build_time s; raise " *
            "max_build_time or disable the cache"))
    end
    sizes, n_out, n_comp, output_ranges, max_width =
        sp.sizes, sp.n_out, sp.n_comp, sp.output_ranges, sp.max_width
    bytes = sp.bytes

    matrices = n_blocks == 0 ? EmptyMatrices(TF) : Matrices(sizes, TF)

    # save strengths, zero all strength rows
    old_strengths = Tuple(Matrix{TF}(undef, strength_dims(source_systems[i]),
        size(source_buffers[i], 2)) for i in eachindex(source_systems))
    for i in eachindex(source_systems)
        sd = strength_dims(source_systems[i])
        old_strengths[i] .= view(source_buffers[i], 5:4+sd, :)
        source_buffers[i][5:4+sd, :] .= zero(TF)
    end

    # column-by-column unit-strength probing, parallel over key chunks. Each
    # task probes with PRIVATE copies of both buffer sets: direct! writes
    # target output rows, and the strength pokes are NOT disjoint across keys
    # (the direct list can hold source branches at multiple tree levels, whose
    # body ranges overlap). Matrix writes are disjoint (each block belongs to
    # exactly one key) and every entry is computed independently, so the
    # result is bit-identical at any thread count.
    if n_workers <= 1
        _probe_source_keys!(matrices, keys_sorted, 1:length(keys_sorted),
            blocks_by_source, entries, target_ranges, source_ranges,
            output_ranges, target_buffers, source_systems, source_buffers,
            derivatives_switches, TF)
    else
        next_chunk = Threads.Atomic{Int}(0)
        @sync for _ in 1:n_workers
            Threads.@spawn begin
                local_targets = map(copy, target_buffers)
                local_sources = map(copy, source_buffers)
                while true
                    i = Threads.atomic_add!(next_chunk, 1) + 1
                    i > length(chunks) && break
                    _probe_source_keys!(matrices, keys_sorted, chunks[i],
                        blocks_by_source, entries, target_ranges,
                        source_ranges, output_ranges, local_targets,
                        source_systems, local_sources,
                        derivatives_switches, TF)
                end
            end
        end
    end

    # restore strengths; leave target output rows zeroed (callers reset anyway)
    for i in eachindex(source_systems)
        sd = strength_dims(source_systems[i])
        source_buffers[i][5:4+sd, :] .= old_strengths[i]
    end
    for k in 1:n_blocks
        i_ts = entries[k][3]
        @views target_buffers[i_ts][output_ranges[k], target_ranges[k]] .= zero(TF)
    end

    strengths_scratch = [Vector{TF}(undef, max_width) for _ in 1:Threads.nthreads()]
    n_target_bodies = sum(size(b, 2) for b in target_buffers; init=0)
    n_source_bodies = sum(size(b, 2) for b in source_buffers; init=0)
    build_time = (time_ns() - start_time) * 1e-9

    return NearfieldInfluenceCache{TF,typeof(derivatives_switches)}(matrices,
        entries, target_ranges, source_ranges, output_ranges, n_out, n_comp,
        strengths_scratch, derivatives_switches, target_tree_id, source_tree_id,
        n_target_bodies, n_source_bodies, build_time, bytes)
end

"""
    check_cache_trees(cache, target_tree, source_tree)

Throw unless `cache` was built from exactly these `Tree` objects (compared by
`objectid`). Tree identity is the whole validity key — callers own
invalidation and must rebuild the cache for a new `Tree`.
"""
function check_cache_trees(cache::NearfieldInfluenceCache, target_tree::Tree, source_tree::Tree)
    objectid(target_tree) == cache.target_tree_id &&
        objectid(source_tree) == cache.source_tree_id ||
        throw(ArgumentError("NearfieldInfluenceCache was built from different " *
            "Tree objects than those provided — the cache is keyed to Tree " *
            "identity; rebuild the cache for a new Tree"))
    return nothing
end

#------- donor / retarget: reuse a built cache across identical rebuilt trees -------#

# A cache is keyed to Tree identity, so rebuilding a plan at the SAME
# (leaf_size, multipole_acceptance) on frozen geometry — e.g. successive
# tuner candidates that only vary expansion order — normally pays the full
# probe cost again for bit-identical blocks. A donor snapshots everything the
# blocks depend on (block specs, output layout, and every buffer row a kernel
# may read); `retarget_nearfield_cache` verifies the new trees reproduce all
# of it EXACTLY and, only then, rebinds the donor's matrices to the new tree
# ids. Any mismatch returns `nothing` (caller builds fresh) — never a stale
# cache. The retargeted cache ALIASES the donor's storage (matrices and
# scratch), so the donor must not be evaluated concurrently with it.

struct NearfieldCacheDonor{TF,TC<:NearfieldInfluenceCache{TF}}
    cache::TC
    target_positions::Vector{Matrix{TF}}  # rows 1:3 per target buffer (tree-sorted order)
    source_buffers::Vector{Matrix{TF}}    # full copies, strength rows zeroed
end

"""
    NearfieldCacheDonor(cache, target_tree, source_tree, source_systems)

Snapshot `cache` (built from exactly these trees) together with the buffer
rows its blocks depend on, so a later plan rebuilt at the same knobs on the
same frozen geometry can adopt the blocks via
[`retarget_nearfield_cache`](@ref) instead of re-probing.
"""
function NearfieldCacheDonor(cache::NearfieldInfluenceCache{TF},
        target_tree::Tree, source_tree::Tree, source_systems::Tuple) where TF
    check_cache_trees(cache, target_tree, source_tree)
    target_positions = [Matrix{TF}(b[1:3, :]) for b in target_tree.buffers]
    source_buffers = [Matrix{TF}(copy(b)) for b in source_tree.buffers]
    for (i, b) in enumerate(source_buffers)
        sd = strength_dims(source_systems[i])
        b[5:4+sd, :] .= zero(TF)
    end
    return NearfieldCacheDonor{TF,typeof(cache)}(cache, target_positions,
        source_buffers)
end

# elementwise equality outside the given row range (the rows that may change)
function _rows_equal_except(a::AbstractMatrix, b::AbstractMatrix, skip::UnitRange{Int})
    size(a) == size(b) || return false
    @inbounds for j in axes(a, 2), i in axes(a, 1)
        i in skip && continue
        a[i, j] == b[i, j] || return false
    end
    return true
end

"""
    retarget_nearfield_cache(donor, target_tree, source_tree, direct_list,
        derivatives_switches, source_systems)

Rebind the donor's cache to freshly built trees, returning a valid
`NearfieldInfluenceCache` for them WITHOUT re-probing — or `nothing` when
anything the blocks depend on differs (block specs, output rows, strength
dims, or any non-strength buffer entry), in which case the caller must build
fresh. Exact-by-construction: the blocks are functions of exactly the
verified data. The returned cache's `build_time` is the retarget cost, not
the donor's original build time.
"""
function retarget_nearfield_cache(donor::NearfieldCacheDonor{TF},
        target_tree::Tree, source_tree::Tree, direct_list,
        derivatives_switches::Tuple, source_systems::Tuple) where TF
    start_time = time_ns()
    cache = donor.cache

    # block specs must be reproduced exactly
    entries, target_ranges, source_ranges =
        _tree_block_specs(target_tree, source_tree, direct_list)
    entries == cache.entries || return nothing
    target_ranges == cache.target_ranges || return nothing
    source_ranges == cache.source_ranges || return nothing

    # output layout and strength dims must be reproduced exactly
    for k in eachindex(entries)
        output_range(derivatives_switches[entries[k][3]]) == cache.output_ranges[k] ||
            return nothing
        strength_dims(source_systems[entries[k][4]]) == cache.n_comp[k] ||
            return nothing
    end

    # every buffer row a kernel may read must be bit-identical: target
    # positions (rows 1:3) and every non-strength source row (positions,
    # radii, vertices, ...)
    length(target_tree.buffers) == length(donor.target_positions) || return nothing
    for (i, b) in enumerate(target_tree.buffers)
        size(b, 2) == size(donor.target_positions[i], 2) || return nothing
        view(b, 1:3, :) == donor.target_positions[i] || return nothing
    end
    length(source_tree.buffers) == length(donor.source_buffers) || return nothing
    for (i, b) in enumerate(source_tree.buffers)
        sd = strength_dims(source_systems[i])
        _rows_equal_except(b, donor.source_buffers[i], 5:4+sd) || return nothing
    end

    retarget_time = (time_ns() - start_time) * 1e-9
    return NearfieldInfluenceCache{TF,typeof(cache.derivatives_switches)}(
        cache.matrices, cache.entries, cache.target_ranges,
        cache.source_ranges, cache.output_ranges, cache.n_out, cache.n_comp,
        cache.strengths_scratch, cache.derivatives_switches,
        objectid(target_tree), objectid(source_tree),
        cache.n_target_bodies, cache.n_source_bodies, retarget_time,
        cache.bytes)
end

#------- evaluation -------#

"""
    nearfield_matvec!(target_buffers, cache, source_buffers; n_threads)

Evaluate the cached near field: gather strengths from the source buffers,
one BLAS matvec per block, accumulate (`+=`, matching `direct!` semantics)
into the target-buffer output rows. Deterministic at any thread count: blocks
are owner-partitioned by target branch (partition cuts only at target-branch
boundaries, so all blocks writing one target branch run serially on one
owner, in fixed order).
"""
function nearfield_matvec!(target_buffers, cache::NearfieldInfluenceCache{TF},
        source_buffers; n_threads::Integer=Threads.nthreads()) where TF

    # cheap validity guards; Tree identity is checked by tree-based callers
    n_targets = sum(size(b, 2) for b in target_buffers; init=0)
    n_sources = sum(size(b, 2) for b in source_buffers; init=0)
    n_targets == cache.n_target_bodies && n_sources == cache.n_source_bodies ||
        throw(ArgumentError("NearfieldInfluenceCache body counts " *
            "($(cache.n_target_bodies) targets, $(cache.n_source_bodies) sources) " *
            "do not match the provided buffers ($n_targets targets, $n_sources " *
            "sources) — rebuild the cache for a new Tree"))

    n_blocks = length(cache.entries)
    n_blocks == 0 && return nothing

    n_threads = min(n_threads, length(cache.strengths_scratch))
    if n_threads <= 1
        _nearfield_matvec_range!(target_buffers, cache, source_buffers, 1:n_blocks, 1)
    else
        assignments = _make_cache_assignments(cache, n_threads)
        Threads.@threads :static for i_task in eachindex(assignments)
            _nearfield_matvec_range!(target_buffers, cache, source_buffers,
                assignments[i_task], i_task)
        end
    end

    return nothing
end

function _nearfield_matvec_range!(target_buffers, cache::NearfieldInfluenceCache{TF},
        source_buffers, block_range, i_scratch) where TF
    scratch = cache.strengths_scratch[i_scratch]
    for k in block_range
        i_ts = cache.entries[k][3]
        i_ss = cache.entries[k][4]
        target_buffer = target_buffers[i_ts]
        source_buffer = source_buffers[i_ss]
        target_range = cache.target_ranges[k]
        source_range = cache.source_ranges[k]
        out_range = cache.output_ranges[k]
        sd = cache.n_comp[k]
        n_out = cache.n_out[k]

        # gather strengths in column layout
        n = sd * length(source_range)
        x = view(scratch, 1:n)
        j = 0
        @inbounds for i_body in source_range
            for i_comp in 1:sd
                j += 1
                x[j] = source_buffer[4+i_comp, i_body]
            end
        end

        block, rhs = get_matrix_vector(cache.matrices, k)
        mul!(rhs, block, x)
        @views target_buffer[out_range, target_range] .+= reshape(rhs, n_out, :)
    end
    return nothing
end

# contiguous block-range assignments balanced by block flops, cutting only at
# target-branch boundaries (all blocks of one target branch share an owner)
function _make_cache_assignments(cache::NearfieldInfluenceCache, n_threads)
    n_blocks = length(cache.entries)
    total_work = 0
    for (m, n) in cache.matrices.sizes
        total_work += m * n
    end
    n_per_thread = cld(total_work, n_threads)

    assignments = UnitRange{Int}[]
    i_start = 1
    work = 0
    for k in 1:n_blocks
        m, n = cache.matrices.sizes[k]
        work += m * n
        branch_boundary = k == n_blocks || cache.entries[k+1][1] != cache.entries[k][1]
        if work >= n_per_thread && branch_boundary && length(assignments) < n_threads - 1
            push!(assignments, i_start:k)
            i_start = k + 1
            work = 0
        end
    end
    i_start <= n_blocks && push!(assignments, i_start:n_blocks)
    return assignments
end
