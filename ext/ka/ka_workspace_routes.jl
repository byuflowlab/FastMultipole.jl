#------- RESIDENT OPERATOR WORKSPACE (step vi-b) -------#

"""
    ka_radix_cache_workspace(backend, TF, basis_info, ell, h0, max_cells, max_nodes,
                             route_capacity, accepted_offsets, invariant;
                             ell_axes, first_level)

Build a device-resident [`ResidentOperatorWorkspace`](@ref) on any KA backend.

This is a thin forward to `FastMultipole._radix_cache_workspace`, which is already
backend-generic: every allocation in it goes through `similar(exemplar.phi, ...)`,
`_array_like_vector(exemplar.phi, ...)` or `DegreeMajorMaps(TF, P, exemplar.phi)`,
so handing it a KA-array exemplar returns a KA-resident workspace. There is
nothing to port.

The KA path builds the `ConcatenatedFixedZM2L` plan with
`MaterializedYRotationM2L`, the strategy gated bit-exact against the host; the
dense and factored strategies stay host-only.

same launch-overhead problem is the one-sync-per-driver discipline.

If this ever needs a keyword the generic builder does not already take, fix the
genericity in `src/translate_batched.jl` rather than branching here.
"""
function ka_radix_cache_workspace(backend, ::Type{TF},
        basis_info::FastMultipole.OperatorBasisInfo{B,LH}, ell::Integer, h0::TF,
        max_cells::Integer, max_nodes::Integer, route_capacity::Integer,
        accepted_offsets::Vector{SVector{3,Int}},
        invariant::FastMultipole.OperatorInvariantCache;
        ell_axes::SVector{3,Int}=SVector(Int(ell), Int(ell), Int(ell)),
        first_level::Integer=0) where {TF,B,LH}
    exemplar = _ka_flat_buffer(backend, TF, basis_info, 1)
    return FastMultipole._radix_cache_workspace(TF, basis_info, exemplar, Int(ell), h0,
        Int(max_cells), Int(max_nodes), Int(route_capacity), accepted_offsets, invariant,
        FastMultipole.ConcatenatedFixedZM2L(), FastMultipole.MaterializedYRotationM2L();
        compact_cuda_factored=false, ell_axes=ell_axes, first_level=Int(first_level))
end

#------- flat (uniform) radix route generation: KA port of
#         `_cuda_generate_radix_routes!` (src/translate_batched_cuda.jl:6174) -------#
#
# Second-to-last CUDA-only dependency of `update_cuda_radix_state!`'s uniform,
# `sfs=false` path. Five elementwise kernels (`cell_at` scatter, then
# flag/scan/compact for the accepted-offset routes and again for the rejected
# offsets' direct pairs) driven by the same chunked loop nest as the CUDA
# original, whose chunking exists so the flag/prefix buffers stay bounded by
# `class_chunk * max_cells` rather than by `naccept * n_cells`.
#
# `ctx` is a NamedTuple on the CUDA side already, so this driver is generic in
# it without a core/wrapper split: it reads `cell_at`, `class_chunk`,
# `d_accepted`, `d_rejected`, `route_flags`/`route_prefix`,
# `direct_flags`/`direct_prefix`, `host_scalar32`, `route_levels`,
# `route_offsets`, `route_targets`, `route_sources`, `direct_targets`,
# `direct_sources`; `grid` is read for `cell_keys` only.
#
# Host oracle: `build_radix_routes!` (src/interaction_list_batched.jl:966),
# which emits routes offset-class-major/cell-minor and direct pairs
# target-major/offset-minor -- exactly the two flat index decompositions below,
# so the gate compares elementwise rather than set-wise.

@kernel function ka_cell_at_scatter_kernel!(cell_at, @Const(cell_keys), n_cells, ell)
    c = @index(Global)
    @inbounds if c <= n_cells
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        cell_at[ix + 1, iy + 1, iz + 1] = Int32(c)
    end
end

@kernel function ka_route_flags_kernel!(flags, @Const(cell_at), @Const(cell_keys),
        @Const(offsets), kbase, n_cells, kn, ell)
    idx = @index(Global)
    @inbounds if idx <= kn * n_cells
        kloc = (idx - 1) ÷ n_cells + 1
        c = (idx - 1) % n_cells + 1
        k = kbase + kloc
        G = 1 << ell
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = Int32(0)
        if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
            src = cell_at[sx + 1, sy + 1, sz + 1]
        end
        flags[idx] = src == Int32(0) ? Int32(0) : Int32(1)
    end
end

@kernel function ka_route_compact_kernel!(route_levels, route_offsets, route_targets,
        route_sources, route_class, @Const(flags), @Const(prefix), @Const(cell_at),
        @Const(cell_keys), @Const(offsets), kbase, n_cells, kn, ell, leaf_offset, base)
    idx = @index(Global)
    @inbounds if idx <= kn * n_cells && flags[idx] == Int32(1)
        kloc = (idx - 1) ÷ n_cells + 1
        c = (idx - 1) % n_cells + 1
        k = kbase + kloc
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = cell_at[sx + 1, sy + 1, sz + 1]
        p = base + Int(prefix[idx])
        route_levels[p] = ell
        route_offsets[1, p] = Int(offsets[1, k])
        route_offsets[2, p] = Int(offsets[2, k])
        route_offsets[3, p] = Int(offsets[3, k])
        route_targets[p] = leaf_offset + c
        route_sources[p] = leaf_offset + Int(src)
        route_class[p] = Int32(k)
    end
end

@kernel function ka_direct_flags_kernel!(flags, @Const(cell_at), @Const(cell_keys),
        @Const(offsets), fbase, len, kn, ell)
    idx = @index(Global)
    @inbounds if idx <= len
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        G = 1 << ell
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = Int32(0)
        if 0 <= sx < G && 0 <= sy < G && 0 <= sz < G
            src = cell_at[sx + 1, sy + 1, sz + 1]
        end
        flags[idx] = src == Int32(0) ? Int32(0) : Int32(1)
    end
end

@kernel function ka_direct_compact_kernel!(direct_targets, direct_sources,
        @Const(flags), @Const(prefix), @Const(cell_at), @Const(cell_keys),
        @Const(offsets), fbase, len, kn, ell, base)
    idx = @index(Global)
    @inbounds if idx <= len && flags[idx] == Int32(1)
        g = fbase + idx
        c = (g - 1) ÷ kn + 1
        k = (g - 1) % kn + 1
        ix, iy, iz = ka_decode_morton_key(cell_keys[c], ell)
        sx = ix - offsets[1, k]
        sy = iy - offsets[2, k]
        sz = iz - offsets[3, k]
        src = cell_at[sx + 1, sy + 1, sz + 1]
        p = base + Int(prefix[idx])
        direct_targets[p] = c
        direct_sources[p] = Int(src)
    end
end

function ka_generate_radix_routes!(ctx, grid, n_cells::Int, leaf_offset::Int,
        ell::Int, route_class; workgroup=KA_AUTO_WORKGROUP)
    backend = KA.get_backend(ctx.cell_at)
    cell_keys = grid.cell_keys
    fill!(ctx.cell_at, Int32(0))
    if n_cells > 0
        scatter = _cached_kernel(ka_cell_at_scatter_kernel!, backend, workgroup)
        scatter(ctx.cell_at, cell_keys, n_cells, ell; ndrange=n_cells)
    end

    naccept = size(ctx.d_accepted, 2)
    n_routes = 0
    k0 = 1
    flags_kernel = _cached_kernel(ka_route_flags_kernel!, backend, workgroup)
    compact_kernel = _cached_kernel(ka_route_compact_kernel!, backend, workgroup)
    while k0 <= naccept && n_cells > 0
        kn = min(ctx.class_chunk, naccept - k0 + 1)
        used = kn * n_cells
        used <= length(ctx.route_flags) ||
            throw(AssertionError("device route flag buffer exceeded its capacity"))
        flags_kernel(ctx.route_flags, ctx.cell_at, cell_keys, ctx.d_accepted,
            k0 - 1, n_cells, kn, ell; ndrange=used)
        accumulate!(+, view(ctx.route_prefix, 1:used), view(ctx.route_flags, 1:used))
        # The one-scalar D2H per chunk is the same sync point the CUDA driver
        # takes: the compact launch needs `chunk_total` on the host to
        # bounds-check the route buffers.
        KA.synchronize(backend)
        copyto!(ctx.host_scalar32, 1, ctx.route_prefix, used, 1)
        chunk_total = Int(ctx.host_scalar32[1])
        if chunk_total > 0
            n_routes + chunk_total <= length(ctx.route_targets) ||
                throw(AssertionError("device route buffer exceeded its capacity"))
            compact_kernel(ctx.route_levels, ctx.route_offsets, ctx.route_targets,
                ctx.route_sources, route_class, ctx.route_flags, ctx.route_prefix,
                ctx.cell_at, cell_keys, ctx.d_accepted, k0 - 1, n_cells, kn, ell,
                leaf_offset, n_routes; ndrange=used)
        end
        n_routes += chunk_total
        k0 += kn
    end

    nreject = size(ctx.d_rejected, 2)
    n_direct = 0
    if n_cells > 0 && nreject > 0
        total = nreject * n_cells
        flag_capacity_direct = length(ctx.direct_flags)
        flag_capacity_direct > 0 ||
            throw(AssertionError("device direct flag buffer has zero capacity"))
        dflags_kernel = _cached_kernel(ka_direct_flags_kernel!, backend, workgroup)
        dcompact_kernel = _cached_kernel(ka_direct_compact_kernel!, backend, workgroup)
        f0 = 0
        while f0 < total
            len = min(flag_capacity_direct, total - f0)
            dflags_kernel(ctx.direct_flags, ctx.cell_at, cell_keys, ctx.d_rejected,
                f0, len, nreject, ell; ndrange=len)
            accumulate!(+, view(ctx.direct_prefix, 1:len), view(ctx.direct_flags, 1:len))
            KA.synchronize(backend)
            copyto!(ctx.host_scalar32, 1, ctx.direct_prefix, len, 1)
            chunk_total = Int(ctx.host_scalar32[1])
            if chunk_total > 0
                n_direct + chunk_total <= length(ctx.direct_targets) ||
                    throw(AssertionError("device direct pair buffer exceeded its capacity"))
                dcompact_kernel(ctx.direct_targets, ctx.direct_sources,
                    ctx.direct_flags, ctx.direct_prefix, ctx.cell_at, cell_keys,
                    ctx.d_rejected, f0, len, nreject, ell, n_direct; ndrange=len)
            end
            n_direct += chunk_total
            f0 += len
        end
    end
    KA.synchronize(backend)
    return n_routes, n_direct
end

