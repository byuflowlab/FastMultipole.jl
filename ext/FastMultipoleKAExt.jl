module FastMultipoleKAExt

using FastMultipole
using KernelAbstractions
using LinearAlgebra
using StaticArrays: SVector
using GPUArraysCore: AnyGPUMatrix, AnyGPUVector
const KA = KernelAbstractions

# Constructing a KA kernel object (`some_kernel!(backend, workgroup)`) redoes
# generic dispatch/partitioning work on every call; caching it per (kernel
# function, backend type, workgroup) avoids repeating that on the hot path.
const _KERNEL_CACHE = Dict{Tuple{Any,DataType,Int},Any}()
function _cached_kernel(f, backend, workgroup::Int)
    key = (f, typeof(backend), workgroup)
    return get!(() -> f(backend, workgroup), _KERNEL_CACHE, key)
end

# Backend-agnostic M2M building blocks (GPU-native, no host round-trip),
# mirroring FastMultipole's CUDA-only production kernels in
# src/translate_batched_cuda.jl (`_cuda_gather_rotate_z_kernel!`,
# `_cuda_rotate_z_scatter_accumulate_kernel!`, `_resident_mul!`) so they run
# on any KernelAbstractions backend (CUDA, Metal, ...).
#
# Verified against a from-scratch CPU reference derived directly from the
# CUDA source (not yet wired into FastMultipole's actual
# ResidentOperatorGroup/ResidentOperatorWorkspace structs -- see the plan
# file / session notes for what remains):
#   - ka_gather_rotate_z!           : max abs err 5.96e-8 vs CPU (n=8x6)
#   - ka_rotate_z_scatter_accumulate!: max abs err 1.19e-7 vs CPU (w/ KA.@atomic)
#   - full non-LH M2M group chain    : max abs err 2.14e-6 (relerr 4.2e-7),
#     gather_rotate_z -> stacked_y_dense -> rotate_z_scatter_accumulate
# All Float32, on Metal (MetalBackend, M1 Pro).

@kernel function ka_gather_rotate_z_kernel!(dst, @Const(src), @Const(flat_idx), @Const(cols),
                                             @Const(row_m), @Const(row_ssign), @Const(row_pair),
                                             @Const(phis), sgn)
    i = @index(Global)
    nrow = size(dst, 1)
    @inbounds begin
        row = (i - 1) % nrow + 1
        col = (i - 1) ÷ nrow + 1
        s, c = sincos(row_m[row] * phis[col])
        a = src[flat_idx[row], cols[col]]
        b = src[flat_idx[row_pair[row]], cols[col]]
        dst[row, col] = c * a + sgn * row_ssign[row] * s * b
    end
end

"""
    ka_gather_rotate_z!(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn; workgroup=64)

Backend-agnostic port of `_cuda_gather_rotate_z_kernel!`
(src/translate_batched_cuda.jl): fused flat-column gather + z-axis rotation
stage of the M2M/M2L source alignment. `sgn = inverse ? -1 : 1`, matching the
CUDA `_gather_rotate_z!` convention.
"""
function ka_gather_rotate_z!(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn; workgroup=64)
    length(dst) == 0 && return dst
    backend = KA.get_backend(dst)
    kernel = _cached_kernel(ka_gather_rotate_z_kernel!, backend, workgroup)
    kernel(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn; ndrange=length(dst))
    return dst
end

@kernel function ka_rotate_z_scatter_accumulate_kernel!(dest, @Const(slab), @Const(flat_idx),
                                                         @Const(col_targets), @Const(row_m),
                                                         @Const(row_ssign), @Const(row_pair), @Const(phis))
    i = @index(Global)
    nrow = size(slab, 1)
    @inbounds begin
        row = (i - 1) % nrow + 1
        col = (i - 1) ÷ nrow + 1
        s, c = sincos(row_m[row] * phis[col])
        v = c * slab[row, col] - row_ssign[row] * s * slab[row_pair[row], col]
        KA.@atomic dest[flat_idx[row], col_targets[col]] += v
    end
end

"""
    ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; workgroup=64)

Backend-agnostic port of `_cuda_rotate_z_scatter_accumulate_kernel!`
(src/translate_batched_cuda.jl): inverse z-rotation fused with an
atomic-accumulating scatter back into the flat coefficient buffer (the
M2M/M2L "return alignment" stage). Uses `KernelAbstractions.@atomic`,
confirmed working on Metal.
"""
function ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; workgroup=64)
    length(slab) == 0 && return dest
    backend = KA.get_backend(dest)
    kernel = _cached_kernel(ka_rotate_z_scatter_accumulate_kernel!, backend, workgroup)
    kernel(dest, slab, flat_idx, col_targets, row_m, row_ssign, row_pair, phis; ndrange=length(slab))
    return dest
end

"""
    ka_stacked_y_dense!(out_slab, in_slab, Ur, Vs, C, S, G, G2, ndof)

Backend-agnostic port of `_stacked_y_dense!` (src/translate_batched.jl):
the dense y-rotation application used by the M2M `SharedRotationM2M`
strategy. On CUDA this is `_resident_mul!` (== `CUBLAS.gemm!`) around an
elementwise C/S combine; here it is `LinearAlgebra.mul!` (which dispatches
to each backend's own GPU matmul -- Metal's MPS-backed `mul!` for
`MtlArray`, CUBLAS for `CuArray`) plus the same elementwise combine via
ordinary broadcasting. No custom `@kernel` is needed for this stage --
`mul!`/broadcast are already backend-generic.
"""
function ka_stacked_y_dense!(out_slab, in_slab, Ur, Vs, C, S, G, G2, ndof::Integer)
    mul!(G, Vs, in_slab)
    Gt = @view G[1:ndof, :]
    Gb = @view G[(ndof + 1):(2 * ndof), :]
    G2t = @view G2[1:ndof, :]
    G2b = @view G2[(ndof + 1):(2 * ndof), :]
    G2t .= C .* Gt .- S .* Gb
    G2b .= S .* Gt .+ C .* Gb
    mul!(out_slab, Ur, G2)
    return out_slab
end

@kernel function ka_gather_rows_kernel!(dst, @Const(src), @Const(rows))
    i = @index(Global)
    nrow = size(dst, 1)
    @inbounds begin
        row = (i - 1) % nrow + 1
        col = (i - 1) ÷ nrow + 1
        dst[row, col] = src[rows[row], col]
    end
end

@kernel function ka_gather_values_kernel!(dst, @Const(src), @Const(ids))
    i = @index(Global)
    @inbounds dst[i] = src[ids[i]]
end

"""
    ka_gather_values!(dst, src, ids; workgroup=64)

Backend-agnostic port of `_gather_values!` (src/translate_batched.jl): allocation-free
value gather `dst[i] = src[ids[i]]`, used by the M2L concat plan's per-chunk column
parameter gather (phi/theta/r/invr) from the per-class geometry tables.
"""
function ka_gather_values!(dst, src, ids; workgroup=64)
    length(dst) == 0 && return dst
    backend = KA.get_backend(dst)
    kernel = _cached_kernel(ka_gather_values_kernel!, backend, workgroup)
    kernel(dst, src, ids; ndrange=length(dst))
    return dst
end

"""
    ka_gather_rows!(dst, src, rows; workgroup=64)

Backend-agnostic port of `_gather_rows!` (src/translate_batched.jl): allocation-free
row gather `dst[i, :] = src[rows[i], :]`, used by the Lamb-Helmholtz row-mix stage of
`_resident_stage_group_apply!`.
"""
function ka_gather_rows!(dst, src, rows; workgroup=64)
    length(dst) == 0 && return dst
    backend = KA.get_backend(dst)
    kernel = _cached_kernel(ka_gather_rows_kernel!, backend, workgroup)
    kernel(dst, src, rows; ndrange=length(dst))
    return dst
end

#------- PRODUCTION PRIMITIVE DISPATCH -------#
#
# The resident stage drivers in src/translate_batched.jl
# (`_resident_stage_group_apply!`, `_launch_resident_m2l_concat!`) are already
# backend-agnostic: apart from views, broadcast, `mul!` and `fill!`, the only
# device work they do goes through the four primitives below. CUDA's "port" of
# the far field is exactly the same trick -- `_launch_cuda_resident_m2m!` and
# `_launch_cuda_resident_l2l!` are one-line passthroughs to those same generic
# drivers, and `src/translate_batched_cuda.jl` specializes only these
# primitives on `CUDA.AnyCuArray`.
#
# So overloading them on GPU arrays puts the KA kernels into the *production*
# call graph rather than alongside it, which is why the standalone
# `ka_resident_stage_group_apply!` / `ka_resident_m2l_concat_apply!` drivers
# (which re-implemented the generic drivers) are no longer needed.
#
# Dispatch handle is `GPUArraysCore.AnyGPU{Matrix,Vector}` -- the wrapper-aware
# analogue of `CUDA.AnyCuArray`. This matters: the drivers hand these
# primitives *views* (`_matrix_col_view`, `@view`), and a `SubArray` of an
# `MtlArray` is not itself an `AbstractGPUArray`. Verified: `MtlMatrix` and
# views of it are `AnyGPUMatrix`, host `Array` is not, and CUDA's
# `AnyCuArray` methods remain strictly more specific for every `CuArray`
# shape including views -- so the CUDA path is unchanged and unambiguous.

function FastMultipole._gather_rotate_z!(dst::AnyGPUMatrix, src::AnyGPUMatrix, flat_idx,
        cols, row_m, row_ssign, row_pair, phis, inverse::Bool)
    # `sgn` must be a float, never the raw `Bool`: `Bool * Number` would
    # silently zero the sin cross-term rather than negate it.
    TF = eltype(dst)
    sgn = inverse ? -one(TF) : one(TF)
    return ka_gather_rotate_z!(dst, src, flat_idx, cols, row_m, row_ssign, row_pair, phis, sgn)
end

function FastMultipole._rotate_z_scatter_accumulate!(dest::AnyGPUMatrix, slab, flat_idx,
        col_targets, row_m, row_ssign, row_pair, phis)
    return ka_rotate_z_scatter_accumulate!(dest, slab, flat_idx, col_targets,
                                           row_m, row_ssign, row_pair, phis)
end

function FastMultipole._gather_rows!(dst::AnyGPUMatrix, src::AnyGPUMatrix, rows)
    return ka_gather_rows!(dst, src, rows)
end

function FastMultipole._gather_values!(dst::AnyGPUVector, src::AnyGPUVector, ids)
    return ka_gather_values!(dst, src, ids)
end

@kernel function ka_fill_invperm_kernel!(invperm, @Const(perm), n)
    sorted_i = @index(Global)
    @inbounds if sorted_i <= n
        invperm[perm[sorted_i]] = sorted_i
    end
end

"""
    ka_fill_invperm!(invperm, perm; workgroup=64)

Backend-agnostic port of `_cuda_fill_invperm_kernel!` (src/translate_batched_cuda.jl):
scatter the inverse of the body sort permutation, `invperm[perm[i]] = i`, so a global
body ordinal maps back to its sorted slot. `invperm` is written over `1:length(perm)`
and must be at least that long; `perm` must be a genuine permutation of `1:n` (each
slot is written exactly once, so a non-permutation silently leaves stale entries --
the same contract CUDA carries).
"""
function ka_fill_invperm!(invperm, perm; workgroup::Int=64)
    n = length(perm)
    n == 0 && return invperm
    length(invperm) >= n || throw(ArgumentError(
        "invperm (length $(length(invperm))) is shorter than perm (length $n)"))
    backend = KA.get_backend(invperm)
    kernel = _cached_kernel(ka_fill_invperm_kernel!, backend, workgroup)
    kernel(invperm, perm, n; ndrange=n)
    return invperm
end

@kernel function ka_fill_single_system_attribution_kernel!(body_system, body_index, n)
    i = @index(Global)
    @inbounds if i <= n
        body_system[i] = 1
        body_index[i] = i
    end
end

"""
    ka_fill_single_system_attribution!(body_system, body_index, n; workgroup=64)

Fill the `DeviceRadixGrid` body-attribution arrays for a single source system:
`body_system[i] = 1`, `body_index[i] = i` over `1:n`. Both are indexed by *global*
(unsorted) body ordinal, so a sorted slot is resolved as `body_system[perm[slot]]`.

This is the attribution half of `_cuda_extract_matrix_positions_kernel!`
(src/translate_batched_cuda.jl), which CUDA fuses into position extraction. The KA
build takes an already-extracted `3 x n` position matrix, so there is nothing to
fuse with and the fill stands alone.

Multi-system attribution (`_cuda_extract_source_positions_kernel!`: one launch per
system, each writing `isys` and its local index into an offset slice) is not ported;
it belongs to the repack path rather than the octree build.
"""
function ka_fill_single_system_attribution!(body_system, body_index, n::Int; workgroup::Int=64)
    n == 0 && return body_system, body_index
    (length(body_system) >= n && length(body_index) >= n) || throw(ArgumentError(
        "body attribution arrays (lengths $(length(body_system)), $(length(body_index))) " *
        "are shorter than n=$n"))
    backend = KA.get_backend(body_system)
    kernel = _cached_kernel(ka_fill_single_system_attribution_kernel!, backend, workgroup)
    kernel(body_system, body_index, n; ndrange=n)
    return body_system, body_index
end

@kernel function ka_tree_routes_kernel!(m2m_parent, m2m_child, l2l_parent, l2l_child,
        @Const(parent_index), n_root_nodes, n_nodes)
    edge = @index(Global)
    node = edge + n_root_nodes
    @inbounds if node <= n_nodes
        parent = parent_index[node]
        m2m_parent[edge] = parent
        m2m_child[edge] = node
        l2l_parent[edge] = parent
        l2l_child[edge] = node
    end
end

"""
    ka_tree_routes!(m2m_parent, m2m_child, l2l_parent, l2l_child, parent_index, n_nodes;
                    n_root_nodes=1, workgroup=64)

Backend-agnostic port of `_cuda_tree_routes_kernel!`/`_cuda_radix_tree_routes`
(src/translate_batched_cuda.jl): materialize the parent-child edge list the resident
M2M and L2L passes walk. One edge per non-root node, in node order, so edge `e`
carries child `e + n_root_nodes` and its parent. M2M and L2L get identical arrays
(the passes differ in direction, not in topology) -- CUDA writes both rather than
aliasing, and this does the same so the two can diverge later without a data race.

`n_nodes` is the *logical* node count: `parent_index` is capacity-sized in the KA
context, so its `length` is not the extent. The four outputs are written over
`1:(n_nodes - n_root_nodes)` and must be at least that long. `n_root_nodes` is the
number of leading nodes that are roots (`parent_index == 0`) and therefore contribute
no edge; 1 for the adaptive tree, matching CUDA's call site.
"""
function ka_tree_routes!(m2m_parent, m2m_child, l2l_parent, l2l_child, parent_index,
        n_nodes::Int; n_root_nodes::Int=1, workgroup::Int=64)
    n_edges = max(n_nodes - n_root_nodes, 0)
    n_edges == 0 && return m2m_parent, m2m_child, l2l_parent, l2l_child
    n_nodes <= length(parent_index) || throw(ArgumentError(
        "n_nodes=$n_nodes exceeds parent_index (length $(length(parent_index)))"))
    for (name, arr) in (("m2m_parent", m2m_parent), ("m2m_child", m2m_child),
            ("l2l_parent", l2l_parent), ("l2l_child", l2l_child))
        length(arr) >= n_edges || throw(ArgumentError(
            "$name (length $(length(arr))) is shorter than the edge count $n_edges"))
    end
    backend = KA.get_backend(m2m_parent)
    kernel = _cached_kernel(ka_tree_routes_kernel!, backend, workgroup)
    kernel(m2m_parent, m2m_child, l2l_parent, l2l_child, parent_index, n_root_nodes,
        n_nodes; ndrange=n_edges)
    return m2m_parent, m2m_child, l2l_parent, l2l_child
end

@kernel function ka_pack_body_matrix_kernel!(body, @Const(source_buffer), @Const(perm),
        @Const(body_system), @Const(body_index), isys, n, nrows, nsys)
    sorted_i = @index(Global)
    @inbounds if sorted_i <= n
        global_i = perm[sorted_i]
        if body_system[global_i] == isys
            ibody = body_index[global_i]
            for row in 1:nsys
                body[row, sorted_i] = source_buffer[row, ibody]
            end
            for row in (nsys + 1):nrows
                body[row, sorted_i] = zero(eltype(body))
            end
        end
    end
end

"""
    ka_pack_body_matrix!(body, source_buffer, perm, body_system, body_index, n;
                         isys=1, workgroup=64)

Backend-agnostic port of `_cuda_pack_radix_body_kernel!`/`_pack_radix_body_matrix!`
(src/translate_batched_cuda.jl): gather one source system's bodies out of its
global-ordinal `source_buffer` into `body`, the sorted-order `dpb x n` matrix the
resident lifecycle reads. Column `sorted_i` of `body` takes buffer column
`body_index[perm[sorted_i]]`, and only for slots this system owns
(`body_system[perm[sorted_i]] == isys`).

Canonical all-rows packed layout (task 032): every source-buffer row is carried,
including radius row 4, and a system narrower than `body` is zero-padded. Call once
per system, as CUDA does -- with the single-system attribution of
[`ka_fill_single_system_attribution!`](@ref) one `isys=1` call fills every column.

`n` is the *logical* body count: `perm` is capacity-sized in the KA context, so its
`length` is not the extent. Columns beyond `n`, and columns this system does not own,
are left untouched.
"""
function ka_pack_body_matrix!(body, source_buffer, perm, body_system, body_index,
        n::Int; isys::Integer=1, workgroup::Int=64)
    n == 0 && return body
    size(body, 2) >= n || throw(ArgumentError(
        "body has $(size(body, 2)) columns, fewer than n=$n"))
    n <= length(perm) || throw(ArgumentError(
        "n=$n exceeds perm (length $(length(perm)))"))
    nrows = size(body, 1)
    nsys = min(size(source_buffer, 1), nrows)
    backend = KA.get_backend(body)
    kernel = _cached_kernel(ka_pack_body_matrix_kernel!, backend, workgroup)
    kernel(body, source_buffer, perm, body_system, body_index, Int(isys), n, nrows, nsys;
        ndrange=n)
    return body
end

# --- Integration with FastMultipole's real M2M call path ---
#
# `ka_resident_stage_group_apply!` mirrors `_resident_stage_group_apply!`
# (src/translate_batched.jl:2962) -- the real production M2M/L2L group-apply
# reached via `_launch_resident_m2m!` -- substituting the four KA building blocks
# above for its CPU/CUDA-specific primitives. It operates on the same
# `FlatCoefficientBuffer`/`ResidentOperatorGroup`/`ResidentOperatorWorkspace` types,
# so it can be dropped in wherever the CPU function is called, backed by any
# KernelAbstractions array (Metal, CUDA, or plain CPU Array).
#
# This DOES use the `count[]`-prefix views (`_vector_prefix_view`/`_matrix_col_view`),
# exactly as the CPU function does. An earlier version did not, on the reasoning that
# those helpers are FastMultipole-internal and that callers would size `group`/`ws` to
# exactly `group.count[]` columns -- true for the isolated per-group suite this backs
# (test/metal_env/ka_m2m_correctness.jl), and false everywhere else. In the production
# lifecycle `ka_lifecycle_body!` loops over `ws.m2m_groups`/`ws.l2l_groups` from a real
# workspace whose scratch is sized to `max_batch`, the max over levels, so most groups
# are SHORTER than the scratch. Without the views that is a `DimensionMismatch` where
# the shapes disagree and, worse, stale trailing columns scattered into `dest` where
# they happen to broadcast. Both helpers return the array unchanged when the size
# already matches, so the isolated suites are unaffected.
function ka_resident_stage_group_apply!(dest, src, group, ws, kind::Symbol)
    n = group.count[]
    n == 0 && return dest
    mult = kind === :m2m
    ystk = ws.ystk_phi
    Ur = mult ? ystk.mult_Ur : ystk.loc_Ur
    Vs = mult ? ystk.mult_Vs : ystk.loc_Vs
    ndof_phi = size(ws.aphi, 1)
    source_idx = FastMultipole._vector_prefix_view(group.source_idx, n)
    target_idx = FastMultipole._vector_prefix_view(group.target_idx, n)
    group_phis = FastMultipole._vector_prefix_view(group.phis, n)
    group_thetas = FastMultipole._vector_prefix_view(group.thetas, n)
    aphi = FastMultipole._matrix_col_view(ws.aphi, n)
    yphi = FastMultipole._matrix_col_view(ws.yphi, n)
    zphi = FastMultipole._matrix_col_view(ws.zphi, n)
    rphi = FastMultipole._matrix_col_view(ws.rphi, n)
    cphi = FastMultipole._matrix_col_view(ws.cphi, n)
    C = FastMultipole._matrix_col_view(ystk.Cy, n)
    S = FastMultipole._matrix_col_view(ystk.Sy, n)
    G = FastMultipole._matrix_col_view(ystk.G, n)
    G2 = FastMultipole._matrix_col_view(ystk.G2, n)
    thetas_row = transpose(group_thetas)
    C .= cos.(ystk.nu .* thetas_row)
    S .= sin.(ystk.nu .* thetas_row)
    ka_gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, source_idx,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis, one(eltype(aphi)))
    ka_stacked_y_dense!(yphi, aphi, Ur, Vs, C, S, G, G2, ndof_phi)
    mul!(zphi, group.phi_dense, yphi)
    ret_phi = zphi

    has_lh = size(dest.chi, 1) > 0
    if has_lh
        ystk_c = ws.ystk_chi
        Urc = mult ? ystk_c.mult_Ur : ystk_c.loc_Ur
        Vsc = mult ? ystk_c.mult_Vs : ystk_c.loc_Vs
        ndof_chi = size(ws.achi, 1)
        achi = FastMultipole._matrix_col_view(ws.achi, n)
        ychi = FastMultipole._matrix_col_view(ws.ychi, n)
        zchi = FastMultipole._matrix_col_view(ws.zchi, n)
        rchi = FastMultipole._matrix_col_view(ws.rchi, n)
        cchi = FastMultipole._matrix_col_view(ws.cchi, n)
        Cc = FastMultipole._matrix_col_view(ystk_c.Cy, n)
        Sc = FastMultipole._matrix_col_view(ystk_c.Sy, n)
        Gc = FastMultipole._matrix_col_view(ystk_c.G, n)
        G2c = FastMultipole._matrix_col_view(ystk_c.G2, n)
        Cc .= cos.(ystk_c.nu .* thetas_row)
        Sc .= sin.(ystk_c.nu .* thetas_row)
        ka_gather_rotate_z!(achi, src.chi, ws.chi_flat_idx, source_idx,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, group_phis, one(eltype(achi)))
        ka_stacked_y_dense!(ychi, achi, Urc, Vsc, Cc, Sc, Gc, G2c, ndof_chi)
        mul!(zchi, group.chi_dense, ychi)
        chi_rows = mult ? ws.maps_chi.row_down : ws.maps_chi.row_up
        ka_gather_rows!(yphi, zchi, ws.maps_phi.row_pair)
        ka_gather_rows!(ychi, zchi, chi_rows)
        cphi .= zphi .+ group.lh_phi_rows .* yphi
        cchi .= zchi .+ group.lh_chi_rows .* ychi
        ka_stacked_y_dense!(rchi, cchi, Urc, Vsc, Cc, Sc, Gc, G2c, ndof_chi)
        ka_rotate_z_scatter_accumulate!(dest.chi, rchi, ws.chi_flat_idx, target_idx,
            ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, group_phis)
        ret_phi = cphi
    end

    ka_stacked_y_dense!(rphi, ret_phi, Ur, Vs, C, S, G, G2, ndof_phi)
    ka_rotate_z_scatter_accumulate!(dest.phi, rphi, ws.phi_flat_idx, target_idx,
        ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, group_phis)
    KA.synchronize(KA.get_backend(dest.phi))
    return dest
end

# Build a single-group ResidentOperatorGroup + the (M2M-only) slice of a
# ResidentOperatorWorkspace on `exemplar`'s backend, sized to `nbatch` columns, with
# `source_idx = target_idx = 1:nbatch` (an isolated per-node correctness check has no
# tree, so there is no real parent/child relationship to reuse). M2L-only workspace
# fields (`m2l_sources`, `m2l_targets`, `m2l_groups`, `m2l_concat`, the whole-pass
# `y_mult_*`/`y_loc_*`, `nonleaf_idx`) are untouched by `_resident_stage_group_apply!`
# and left as `nothing`.
function _build_m2m_group_and_workspace(exemplar, ::Type{TF}, invariant_cache,
        phis_host::Vector{TF}, thetas_host::Vector{TF}, rs_host::Vector{TF}, ::Val{LH}) where {TF,LH}
    basis_info = invariant_cache.basis_info
    B = typeof(basis_info.basis)
    nbatch = length(phis_host)
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active

    group = FastMultipole._resident_group(exemplar, TF, basis_info, :m2m, 0,
        collect(1:nbatch), collect(1:nbatch), phis_host, thetas_host, rs_host)

    phi_flat_idx = FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_phi))
    chi_flat_idx = LH ?
        FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_active)) :
        FastMultipole._array_like_vector(exemplar, Int, Int[])
    maps_phi = FastMultipole.DegreeMajorMaps(TF, P_phi, exemplar)
    maps_chi = LH ? FastMultipole.DegreeMajorMaps(TF, P_active, exemplar) : maps_phi

    ndof_phi = FastMultipole.degree_major_dof(P_phi)
    ndof_chi = LH ? FastMultipole.degree_major_dof(P_active) : 0
    mkphi() = similar(exemplar, TF, ndof_phi, nbatch)
    mkchi() = similar(exemplar, TF, ndof_chi, LH ? nbatch : 0)
    aphi = mkphi(); yphi = mkphi(); zphi = mkphi(); rphi = mkphi(); cphi = mkphi()
    achi = mkchi(); ychi = mkchi(); zchi = mkchi(); rchi = mkchi(); cchi = mkchi()

    ystk_phi = FastMultipole.StackedYChannel(exemplar, TF, invariant_cache, P_phi, nbatch)
    ystk_chi = LH ? FastMultipole.StackedYChannel(exemplar, TF, invariant_cache, P_active, nbatch) : nothing

    ws = FastMultipole.ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        nothing, nothing, nothing, nothing, nothing, nbatch,
        nothing, nothing,
        aphi, yphi, zphi, rphi, achi, ychi, zchi, rchi, cphi, cchi,
        nothing, nothing, nothing, nothing, FastMultipole.ResidentOperatorGroup[], nothing, nothing,
        ystk_phi, ystk_chi,
    )
    return group, ws
end

"""
    FastMultipole.ka_m2m_operator_batch!(op, targets, sources, phis, thetas, rs, invariant_cache, scratch, lamb_helmholtz)

KernelAbstractions-backed M2M batch operator (stub declared in `src/FastMultipole.jl`).
On a CPU backend this just delegates to the existing `m2m_operator_batch!` (identical
math, no device dispatch needed). On a GPU backend it builds a single-group resident
operator workspace (`_build_m2m_group_and_workspace`) and runs `ka_resident_stage_group_apply!`
-- the KA-ported form of the real production resident M2M path (`_resident_stage_group_apply!`,
reached in production via `_launch_resident_m2m!`) -- against it. Numerically this should
agree with the (CPU, non-resident) `op`/`scratch`-based path up to floating point error:
both compute the same M2M translation, just through different (materialized-rotation vs.
factored-y dense-GEMM) factorizations of the same math.
"""
function FastMultipole.ka_m2m_operator_batch!(op, targets, sources, phis, thetas, rs,
        invariant_cache, scratch, lamb_helmholtz::Val{LH}) where LH
    backend = KA.get_backend(targets.phi)
    if backend isa KA.CPU
        return FastMultipole.m2m_operator_batch!(op, targets, sources, phis, thetas, rs,
            invariant_cache, scratch, lamb_helmholtz)
    end
    TF = eltype(targets.phi)
    group, ws = _build_m2m_group_and_workspace(targets.phi, TF, invariant_cache,
        TF.(collect(phis)), TF.(collect(thetas)), TF.(collect(rs)), lamb_helmholtz)
    return ka_resident_stage_group_apply!(targets, sources, group, ws, :m2m)
end

# --- M2L (horizontal pass) ---
#
# The real production GPU M2L path is `ConcatenatedFixedZM2L`/`_launch_resident_m2l_concat!`
# (src/translate_batched.jl:3828, `ResidentM2LConcatPlan`/`ConcatChannelOps`) -- confirmed
# by checking `RadixFMMCache`'s default/allowed M2L strategies (src/translate_batched_resident.jl),
# not the `FactoredRotationM2L`/`_resident_factored_m2l_group_apply!` path, whose per-degree
# y-rotation blocks are type-asserted as plain CPU `Matrix{TF}` (translate_batched.jl:3077-3080)
# and never dispatch to CUBLAS/Metal GPU matmul -- that path is CPU-only.
#
# `_launch_resident_m2l_concat!`'s primitives are, beyond the M2M-shared ones above:
#   - `_gather_values!` (1D per-chunk column-parameter gather) -- ported here as `ka_gather_values!`
#   - `_stacked_y_dense!` -- already backend-generic (`mul!` + broadcast); reused directly
#     from `ka_stacked_y_dense!` above (same math, real/imag stacked-block form)
#   - `_resident_mul!` (== `mul!`) for the fixed z-translation GEMM -- already backend-generic,
#     called directly
# No new `@kernel` beyond `ka_gather_values!` is needed; `ka_gather_rotate_z!`,
# `ka_gather_rows!`, and `ka_rotate_z_scatter_accumulate!` are reused unchanged from M2M.

# Build a `ResidentM2LConcatPlan` + the (M2L-only) slice of a `ResidentOperatorWorkspace`
# on `exemplar`'s backend for an isolated per-route correctness check: `nbatch` independent
# (source, target) = (i, i) pairs, one geometry class per route (arbitrary per-route
# (r, θ, φ), not the uniform-grid-stencil classes production groups routes into) so any
# random test geometry can be exercised without a tree. `accepted_offsets` here only sizes
# `nclasses` to `nbatch`; the real per-class (r, θ, φ) tables are overwritten right after
# construction with the caller's actual test angles.
function _build_m2l_concat_plan_and_workspace(exemplar, ::Type{TF}, invariant_cache,
        phis_host::Vector{TF}, thetas_host::Vector{TF}, rs_host::Vector{TF}, ::Val{LH}) where {TF,LH}
    basis_info = invariant_cache.basis_info
    B = typeof(basis_info.basis)
    nbatch = length(phis_host)
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active

    offsets = [SVector{3,Int}(i, 0, 0) for i in 1:nbatch]
    plan = FastMultipole.ResidentM2LConcatPlan(TF, basis_info, exemplar,
        FastMultipole.ConcatenatedFixedZM2L(nbatch), invariant_cache, offsets, one(TF), nbatch)
    copyto!(plan.phis, phis_host)
    copyto!(plan.thetas, thetas_host)
    copyto!(plan.rs, rs_host)
    copyto!(plan.invrs, inv.(rs_host))
    copyto!(plan.route_class, Int32.(1:nbatch))

    phi_flat_idx = FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_phi))
    chi_flat_idx = LH ?
        FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_active)) :
        FastMultipole._array_like_vector(exemplar, Int, Int[])
    maps_phi = FastMultipole.DegreeMajorMaps(TF, P_phi, exemplar)
    maps_chi = LH ? FastMultipole.DegreeMajorMaps(TF, P_active, exemplar) : maps_phi

    empty_sm() = similar(exemplar, TF, 0, 0)
    ws = FastMultipole.ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        nothing, nothing, nothing, nothing, nothing, nbatch,
        nothing, nothing,
        empty_sm(), empty_sm(), empty_sm(), empty_sm(),
        empty_sm(), empty_sm(), empty_sm(), empty_sm(),
        empty_sm(), empty_sm(),
        nothing, nothing, nothing, nothing, FastMultipole.ResidentOperatorGroup[], nothing, plan,
        nothing, nothing,
    )
    return plan, ws
end

"""
    ka_resident_m2l_concat_apply!(dest, src, ws, route_sources, route_targets, nroutes)

Backend-agnostic port of `_launch_resident_m2l_concat!` (src/translate_batched.jl:3828),
the real production `ConcatenatedFixedZM2L` resident M2L apply. Operates on `ws.m2l_concat`
(a `ResidentM2LConcatPlan`) plus `route_sources`/`route_targets` (GPU index vectors) directly,
rather than a full `DeviceResidentRadixState`, so it can be dropped into an isolated
per-route correctness check the same way `ka_resident_stage_group_apply!` was for M2M.
"""
function ka_resident_m2l_concat_apply!(dest, src, ws, route_sources, route_targets, nroutes::Int)
    nroutes == 0 && return dest
    plan = ws.m2l_concat
    LH = size(dest.chi, 1) > 0
    TF = eltype(dest.phi)
    for c0 in 1:plan.chunk:nroutes
        cols = c0:min(c0 + plan.chunk - 1, nroutes)
        n = length(cols)
        cls = @view plan.route_class[cols]
        phis = @view plan.col_phi[1:n]
        thetas = @view plan.col_theta[1:n]
        invr_col = @view plan.col_invr[1:n]
        ka_gather_values!(phis, plan.phis, cls)
        ka_gather_values!(thetas, plan.thetas, cls)
        ka_gather_values!(invr_col, plan.invrs, cls)
        invr_row = transpose(invr_col)
        if LH
            rs_col = @view plan.col_r[1:n]
            ka_gather_values!(rs_col, plan.rs, cls)
            rs_row = transpose(rs_col)
        end
        src_cols = @view route_sources[cols]
        tgt_cols = @view route_targets[cols]
        aphi = @view plan.aphi[:, 1:n]; yphi = @view plan.yphi[:, 1:n]
        zphi = @view plan.zphi[:, 1:n]; rphi = @view plan.rphi[:, 1:n]
        ops_phi = plan.ops_phi
        ndof_phi = size(plan.aphi, 1)
        Gphi = @view ops_phi.G[:, 1:n]; G2phi = @view ops_phi.G2[:, 1:n]
        Cphi = @view ops_phi.Cy[:, 1:n]; Sphi = @view ops_phi.Sy[:, 1:n]
        sphi = @view ops_phi.scale[:, 1:n]
        thetas_row = transpose(thetas)
        Cphi .= cos.(ops_phi.nu .* thetas_row)
        Sphi .= sin.(ops_phi.nu .* thetas_row)
        sphi .= invr_row .^ plan.rexp_phi
        ka_gather_rotate_z!(aphi, src.phi, ws.phi_flat_idx, src_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis, one(TF))
        ka_stacked_y_dense!(yphi, aphi, ops_phi.yU_mult, ops_phi.yV_mult,
            Cphi, Sphi, Gphi, G2phi, ndof_phi)
        yphi .*= sphi
        mul!(zphi, ops_phi.zD, yphi)
        zphi .*= sphi
        ret_phi = zphi

        if LH
            ops_chi = plan.ops_chi
            ndof_chi = size(plan.achi, 1)
            Gchi = @view ops_chi.G[:, 1:n]; G2chi = @view ops_chi.G2[:, 1:n]
            Cchi = @view ops_chi.Cy[:, 1:n]; Schi = @view ops_chi.Sy[:, 1:n]
            schi = @view ops_chi.scale[:, 1:n]
            achi = @view plan.achi[:, 1:n]; ychi = @view plan.ychi[:, 1:n]
            zchi = @view plan.zchi[:, 1:n]; rchi = @view plan.rchi[:, 1:n]
            cphi = @view plan.cphi[:, 1:n]; cchi = @view plan.cchi[:, 1:n]
            lhgp = @view plan.lhgp[:, 1:n]; lhgu = @view plan.lhgu[:, 1:n]
            Cchi .= cos.(ops_chi.nu .* thetas_row)
            Schi .= sin.(ops_chi.nu .* thetas_row)
            schi .= invr_row .^ plan.rexp_chi
            ka_gather_rotate_z!(achi, src.chi, ws.chi_flat_idx, src_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis, one(TF))
            ka_stacked_y_dense!(ychi, achi, ops_chi.yU_mult, ops_chi.yV_mult,
                Cchi, Schi, Gchi, G2chi, ndof_chi)
            ychi .*= schi
            mul!(zchi, ops_chi.zD, ychi)
            zchi .*= schi
            ka_gather_rows!(lhgp, zchi, ws.maps_phi.row_pair)
            ka_gather_rows!(lhgu, zchi, ws.maps_chi.row_up)
            cphi .= zphi .+ (plan.lh_arow_unit .* rs_row) .* lhgp
            cchi .= zchi .+ (plan.lh_brow_unit .* rs_row) .* lhgu
            ka_stacked_y_dense!(rchi, cchi, ops_chi.yU_loc, ops_chi.yV_loc,
                Cchi, Schi, Gchi, G2chi, ndof_chi)
            ka_rotate_z_scatter_accumulate!(dest.chi, rchi, ws.chi_flat_idx, tgt_cols,
                ws.maps_chi.row_m, ws.maps_chi.row_ssign, ws.maps_chi.row_pair, phis)
            ret_phi = cphi
        end

        ka_stacked_y_dense!(rphi, ret_phi, ops_phi.yU_loc, ops_phi.yV_loc,
            Cphi, Sphi, Gphi, G2phi, ndof_phi)
        ka_rotate_z_scatter_accumulate!(dest.phi, rphi, ws.phi_flat_idx, tgt_cols,
            ws.maps_phi.row_m, ws.maps_phi.row_ssign, ws.maps_phi.row_pair, phis)
    end
    KA.synchronize(KA.get_backend(dest.phi))
    return dest
end

"""
    FastMultipole.ka_m2l_operator_batch!(op, targets, sources, phis, thetas, rs, invariant_cache, scratch, lamb_helmholtz)

KernelAbstractions-backed M2L batch operator (stub declared in `src/FastMultipole.jl`).
On a CPU backend this delegates to the existing `m2l_operator_batch!` (identical math, no
device dispatch needed). On a GPU backend it builds a single-chunk resident concat plan
(`_build_m2l_concat_plan_and_workspace`) treating each (phis[i], thetas[i], rs[i]) as an
independent (source i -> target i) M2L pair, and runs `ka_resident_m2l_concat_apply!` --
the KA-ported form of the real production `ConcatenatedFixedZM2L` resident M2L path
(`_launch_resident_m2l_concat!`) -- against it.
"""
function FastMultipole.ka_m2l_operator_batch!(op, targets, sources, phis, thetas, rs,
        invariant_cache, scratch, lamb_helmholtz::Val{LH}) where LH
    backend = KA.get_backend(targets.phi)
    if backend isa KA.CPU
        return FastMultipole.m2l_operator_batch!(op, targets, sources, phis, thetas, rs,
            invariant_cache, scratch, lamb_helmholtz)
    end
    TF = eltype(targets.phi)
    plan, ws = _build_m2l_concat_plan_and_workspace(targets.phi, TF, invariant_cache,
        TF.(collect(phis)), TF.(collect(thetas)), TF.(collect(rs)), lamb_helmholtz)
    nbatch = length(phis)
    idx = FastMultipole._array_like_vector(targets.phi, Int, collect(1:nbatch))
    return ka_resident_m2l_concat_apply!(targets, sources, ws, idx, idx, nbatch)
end

# --- L2L (downward pass) ---
#
# The real production GPU L2L path (`_launch_resident_l2l!`, src/translate_batched.jl:3702)
# reaches the exact same `_resident_stage_group_apply!` group-apply function that M2M does,
# just with `kind=:l2l` (the function's `mult = kind === :m2m` branch alone selects the
# multipole vs. local y-rotation tables and LH row direction). `ka_resident_stage_group_apply!`
# above already threads `kind` through unchanged, so no new KA kernel or group-apply port is
# needed for L2L -- only a `:l2l` group/workspace builder and dispatch stub, mirroring M2M's.
function _build_l2l_group_and_workspace(exemplar, ::Type{TF}, invariant_cache,
        phis_host::Vector{TF}, thetas_host::Vector{TF}, rs_host::Vector{TF}, ::Val{LH}) where {TF,LH}
    basis_info = invariant_cache.basis_info
    B = typeof(basis_info.basis)
    nbatch = length(phis_host)
    P_phi = basis_info.orders.P_phi
    P_active = basis_info.orders.P_active

    group = FastMultipole._resident_group(exemplar, TF, basis_info, :l2l, 0,
        collect(1:nbatch), collect(1:nbatch), phis_host, thetas_host, rs_host)

    phi_flat_idx = FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_phi))
    chi_flat_idx = LH ?
        FastMultipole._array_like_vector(exemplar, Int, FastMultipole._degree_major_to_flat_indices(P_active)) :
        FastMultipole._array_like_vector(exemplar, Int, Int[])
    maps_phi = FastMultipole.DegreeMajorMaps(TF, P_phi, exemplar)
    maps_chi = LH ? FastMultipole.DegreeMajorMaps(TF, P_active, exemplar) : maps_phi

    ndof_phi = FastMultipole.degree_major_dof(P_phi)
    ndof_chi = LH ? FastMultipole.degree_major_dof(P_active) : 0
    mkphi() = similar(exemplar, TF, ndof_phi, nbatch)
    mkchi() = similar(exemplar, TF, ndof_chi, LH ? nbatch : 0)
    aphi = mkphi(); yphi = mkphi(); zphi = mkphi(); rphi = mkphi(); cphi = mkphi()
    achi = mkchi(); ychi = mkchi(); zchi = mkchi(); rchi = mkchi(); cchi = mkchi()

    ystk_phi = FastMultipole.StackedYChannel(exemplar, TF, invariant_cache, P_phi, nbatch)
    ystk_chi = LH ? FastMultipole.StackedYChannel(exemplar, TF, invariant_cache, P_active, nbatch) : nothing

    ws = FastMultipole.ResidentOperatorWorkspace{TF,B,LH}(
        basis_info, phi_flat_idx, chi_flat_idx, maps_phi, maps_chi,
        nothing, nothing, nothing, nothing, nothing, nbatch,
        nothing, nothing,
        aphi, yphi, zphi, rphi, achi, ychi, zchi, rchi, cphi, cchi,
        nothing, nothing, nothing, nothing, FastMultipole.ResidentOperatorGroup[], nothing, nothing,
        ystk_phi, ystk_chi,
    )
    return group, ws
end

"""
    FastMultipole.ka_l2l_operator_batch!(op, targets, sources, phis, thetas, rs, invariant_cache, scratch, lamb_helmholtz)

KernelAbstractions-backed L2L batch operator (stub declared in `src/FastMultipole.jl`).
On a CPU backend this just delegates to the existing `l2l_operator_batch!` (identical
math, no device dispatch needed). On a GPU backend it builds a single-group resident
operator workspace (`_build_l2l_group_and_workspace`) and runs `ka_resident_stage_group_apply!`
with `kind=:l2l` -- the same KA-ported group-apply M2M already uses, reached in production
via `_launch_resident_l2l!`.
"""
function FastMultipole.ka_l2l_operator_batch!(op, targets, sources, phis, thetas, rs,
        invariant_cache, scratch, lamb_helmholtz::Val{LH}) where LH
    backend = KA.get_backend(targets.phi)
    if backend isa KA.CPU
        return FastMultipole.l2l_operator_batch!(op, targets, sources, phis, thetas, rs,
            invariant_cache, scratch, lamb_helmholtz)
    end
    TF = eltype(targets.phi)
    group, ws = _build_l2l_group_and_workspace(targets.phi, TF, invariant_cache,
        TF.(collect(phis)), TF.(collect(thetas)), TF.(collect(rs)), lamb_helmholtz)
    return ka_resident_stage_group_apply!(targets, sources, group, ws, :l2l)
end

#------- Step 5/6: adaptive octree construction, Phase A (K_max frontier split) -------#
#
# Backend-agnostic port of src/tree_batched_cuda.jl's Phase A
# (`_adt_cuda_seed_root_kernel!`/`_adt_cuda_split_flags_kernel!`/
# `_adt_cuda_split_compact_kernel!`/`_cuda_adaptive_build_leaves!`): the
# level-synchronous K_max frontier split that builds the adaptive octree's leaf
# set from a device array of full-depth-sorted Morton keys, before 2:1 balance
# (tree_batched_cuda.jl Phase B, not yet ported) and finalize (Phase C, not yet
# ported). This is the first phase of the tree/grid construction subsystem that
# `RadixFMMCache(device=true)` currently requires CUDA for (see the ka-migration
# plan/memory note on why `fmm!()` can't reach the KA M2M/M2L/L2L path on Metal
# without this). `_cuda_lower_bound` is CUDA-lifecycle-gated (only defined once
# `load_cuda_radix_lifecycle!()` runs), so `ka_lower_bound` below is a
# self-contained duplicate, not a shared call -- consistent with how the rest of
# this ext never calls into the lazy CUDA-only kernel file.

@inline function ka_lower_bound(keys, first, stop, key)
    lo = first
    hi = stop + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if @inbounds(keys[mid]) < key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

# Child tuple of frontier cell `i`, child bits `c` (0-based): occupied range by
# binary search over the full-depth-sorted body keys. Matches
# `_adt_cuda_child_range` (tree_batched_cuda.jl) exactly.
@inline function ka_child_range(sorted_keys, alev, akey, alo, ahi, i, c, ell_max)
    @inbounds begin
        lc = Int(alev[i]) + 1
        shift = 3 * (ell_max - lc)
        ckey = (akey[i] << 3) | UInt64(c)
        startk = ckey << shift
        endk = startk + (UInt64(1) << shift)
        lo_c = ka_lower_bound(sorted_keys, Int(alo[i]), Int(ahi[i]), startk)
        hi_c = ka_lower_bound(sorted_keys, Int(alo[i]), Int(ahi[i]), endk) - 1
    end
    return lc, ckey, lo_c, hi_c
end

@kernel function ka_seed_root_kernel!(lev, key, lo, hi, n)
    i = @index(Global)
    @inbounds if i == 1
        lev[1] = Int32(0)
        key[1] = UInt64(0)
        lo[1] = Int32(1)
        hi[1] = Int32(n)
    end
end

# flags over the 8 x nact virtual child slots; want_leaf=1 flags children that
# become leaves, want_leaf=0 flags children that stay active (split again).
@kernel function ka_split_flags_kernel!(flags, @Const(sorted_keys), @Const(alev),
        @Const(akey), @Const(alo), @Const(ahi), nact, K_max, ell_max, want_leaf)
    j = @index(Global)
    @inbounds if j <= 8 * nact
        i = (j - 1) >> 3 + 1
        c = (j - 1) & 7
        lc, _, lo_c, hi_c = ka_child_range(sorted_keys, alev, akey, alo, ahi, i, c, ell_max)
        f = Int32(0)
        if lo_c <= hi_c
            isleaf = (hi_c - lo_c + 1 <= K_max) || (lc == ell_max)
            f = ((want_leaf == Int32(1)) == isleaf) ? Int32(1) : Int32(0)
        end
        flags[j] = f
    end
end

@kernel function ka_split_compact_kernel!(dlev, dkey, dlo, dhi, base, @Const(flags),
        @Const(prefix), @Const(sorted_keys), @Const(alev), @Const(akey), @Const(alo),
        @Const(ahi), nact, ell_max)
    j = @index(Global)
    @inbounds if j <= 8 * nact && flags[j] == Int32(1)
        i = (j - 1) >> 3 + 1
        c = (j - 1) & 7
        lc, ckey, lo_c, hi_c = ka_child_range(sorted_keys, alev, akey, alo, ahi, i, c, ell_max)
        idx = base + Int(prefix[j])
        dlev[idx] = Int32(lc)
        dkey[idx] = ckey
        dlo[idx] = Int32(lo_c)
        dhi[idx] = Int32(hi_c)
    end
end

# Inclusive scan of flags[1:m] into prefix[1:m], returning the total. Mirrors
# `_adt_cuda_scan_total!`; `accumulate!` is already backend-generic (KA/GPUArrays),
# so only the final host-scalar readback needs to be written out explicitly.
function _ka_scan_total!(flags, prefix, m::Int)
    m == 0 && return 0
    fv = view(flags, 1:m)
    pv = view(prefix, 1:m)
    accumulate!(+, pv, fv)
    return Int(Array(view(prefix, m:m))[1])
end

#------- Preallocated tree-build context (zero recurring allocation) -------#
#
# Mirrors CUDA-native's `DeviceAdaptiveCUDAContext`/`actx` convention
# (src/tree_batched_cuda.jl:60-160): every working buffer used by
# `ka_build_adaptive_tree!`'s phases is allocated once, here, at capacity, and
# reused across calls instead of being rebuilt via `KA.zeros(...)` on every
# invocation -- the source of the ~1.95GB/trial n=1e6 allocation pressure
# bisected in job 13506038 (see project_fastmultipole_ka_migration memory).
# Scoped to tree construction (Phases A/B/C/D); CUDA's DTR/interaction-list
# buffers (`u_capacity`/`v_capacity`/`wx_capacity`) have no KA counterpart yet.

struct KAAdaptiveTreeContext{B,G,NT<:NamedTuple}
    backend::B
    maxn::Int
    leaf_capacity::Int
    frontier_capacity::Int
    node_capacity::Int
    # The tree's public output, in the form the resident FMM path consumes
    # (`DeviceResidentRadixState.grid`). Allocated here at capacity and mutated
    # in place by the phases -- `bufs` aliases its arrays, so the phase code is
    # unchanged and there is no per-build tuple-to-grid conversion (which would
    # allocate and break the task-023 contract). Mirrors CUDA, where
    # `DeviceAdaptiveCUDAContext` owns the grid and every phase writes `grid.*`.
    grid::G
    bufs::NT
end

"""
    ka_allocate_adaptive_context(backend, TF, maxn; leaf_capacity, frontier_capacity, node_capacity)

Allocate a `KAAdaptiveTreeContext`: every scratch/output buffer
`ka_build_adaptive_tree!` and its phase functions need, sized once at
`maxn`/`leaf_capacity`/`frontier_capacity`/`node_capacity` and reused across
calls. Construct once per (backend, capacity) combination, outside any
trial/timestep loop.

The node- and cell-indexed outputs are allocated as the fields of a capacity-sized
`DeviceRadixGrid` (`actx.grid`), which `bufs` aliases; the geometry fields
(`x_min`/`h0`/`ell`) and the prefix lengths (`n_bodies`/`n_cells`) are placeholders
until `ka_build_adaptive_tree!` sets them from its build arguments, exactly as
`_cuda_refresh_adaptive_tree!` does. `grid.body_system`/`grid.body_index` are filled
for a single source system (see `ka_fill_single_system_attribution!`); multi-system
attribution belongs to the repack path rather than the octree build.
"""
function ka_allocate_adaptive_context(backend, ::Type{TF}, maxn::Int;
        leaf_capacity::Int, frontier_capacity::Int, node_capacity::Int) where TF
    LC, FC, NC = leaf_capacity, frontier_capacity, node_capacity
    grid = FastMultipole.DeviceRadixGrid(
        zero(SVector{3,TF}), one(TF), 0, 0, 0,
        KA.zeros(backend, Int, maxn), KA.zeros(backend, Int, maxn),
        KA.zeros(backend, UInt64, LC), KA.zeros(backend, Int, 2, LC),
        KA.zeros(backend, Int, maxn), KA.zeros(backend, Int, maxn),
        KA.zeros(backend, TF, 3, LC),
        KA.zeros(backend, Int, NC), KA.zeros(backend, UInt64, NC),
        KA.zeros(backend, Int, 3, NC), KA.zeros(backend, TF, 3, NC),
        KA.zeros(backend, Int, NC), KA.zeros(backend, Int, 2, NC),
        KA.zeros(backend, Int, LC),
    )
    bufs = (
        keys=KA.zeros(backend, UInt64, maxn), perm=grid.perm,
        invperm=grid.invperm,
        sorted_keys=KA.zeros(backend, UInt64, maxn),

        llev=KA.zeros(backend, Int32, LC), lkey=KA.zeros(backend, UInt64, LC),
        llo=KA.zeros(backend, Int32, LC), lhi=KA.zeros(backend, Int32, LC),
        a_lev=KA.zeros(backend, Int32, FC), a_key=KA.zeros(backend, UInt64, FC),
        a_lo=KA.zeros(backend, Int32, FC), a_hi=KA.zeros(backend, Int32, FC),
        b_lev=KA.zeros(backend, Int32, FC), b_key=KA.zeros(backend, UInt64, FC),
        b_lo=KA.zeros(backend, Int32, FC), b_hi=KA.zeros(backend, Int32, FC),
        bl_flags=KA.zeros(backend, Int32, FC), bl_prefix=KA.zeros(backend, Int32, FC),

        bal_shifted=KA.zeros(backend, UInt64, LC), bal_order=KA.zeros(backend, Int, LC),
        bal_scratch_starts=KA.zeros(backend, UInt64, LC), bal_marks=KA.zeros(backend, Int32, LC),
        bal_flags=KA.zeros(backend, Int32, LC), bal_prefix=KA.zeros(backend, Int32, LC),
        dlev=KA.zeros(backend, Int32, LC), dkey=KA.zeros(backend, UInt64, LC),
        dlo=KA.zeros(backend, Int32, LC), dhi=KA.zeros(backend, Int32, LC),

        fin_shifted=KA.zeros(backend, UInt64, LC), fin_order=KA.zeros(backend, Int, LC),
        fin_skey=KA.zeros(backend, UInt64, LC), fin_slev=KA.zeros(backend, Int32, LC),
        fin_cand=KA.zeros(backend, UInt64, LC),
        fin_flags=KA.zeros(backend, Int32, NC), fin_prefix=KA.zeros(backend, Int32, NC),
        node_keys=grid.node_keys, node_levels=grid.node_levels,
        node_coords=grid.node_coords, node_centers=grid.node_centers,
        node_lo=KA.zeros(backend, Int32, NC), node_hi=KA.zeros(backend, Int32, NC),
        parent_index=grid.parent_index, child_ranges=grid.child_ranges,
        leaf_index=KA.zeros(backend, Int32, NC), leaf_slot_of=KA.zeros(backend, Int32, NC),
        cell_ranges=grid.cell_ranges, cell_centers=grid.cell_centers,
        cell_keys=grid.cell_keys, leaf_to_node=grid.leaf_to_node,

        node_sigma=KA.zeros(backend, TF, NC),
    )
    return KAAdaptiveTreeContext(backend, maxn, leaf_capacity, frontier_capacity,
        node_capacity, grid, bufs)
end

"""
    ka_adaptive_build_leaves!(actx, sorted_keys, ell_max, K_max, n; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_leaves!` (tree_batched_cuda.jl):
builds the K_max leaf set (adaptive octree theory §1.2) from `sorted_keys`, a
KA-backend array of the `n` bodies' full-depth Morton keys in ascending sorted
order. Returns `(nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi)`, each a
backend array truncated logically to `1:nl` (allocated at `actx.leaf_capacity`).
`leaf_lo`/`leaf_hi` are 1-based inclusive ranges into `sorted_keys`. Scratch/
output buffers come from `actx` (see `KAAdaptiveTreeContext`) -- no allocation.
"""
function ka_adaptive_build_leaves!(actx::KAAdaptiveTreeContext, sorted_keys::AbstractVector{UInt64},
        ell_max::Int, K_max::Int, n::Int; workgroup::Int=64)
    backend = actx.backend
    b = actx.bufs
    leaf_capacity = actx.leaf_capacity
    frontier_capacity = actx.frontier_capacity
    llev, lkey, llo, lhi = b.llev, b.lkey, b.llo, b.lhi

    seedk = _cached_kernel(ka_seed_root_kernel!, backend, 1)

    if n <= K_max || ell_max == 0
        seedk(llev, lkey, llo, lhi, n; ndrange=1)
        KA.synchronize(backend)
        return 1, llev, lkey, llo, lhi
    end

    a_lev, a_key, a_lo, a_hi = b.a_lev, b.a_key, b.a_lo, b.a_hi
    b_lev, b_key, b_lo, b_hi = b.b_lev, b.b_key, b.b_lo, b.b_hi
    flags, prefix = b.bl_flags, b.bl_prefix

    seedk(a_lev, a_key, a_lo, a_hi, n; ndrange=1)
    KA.synchronize(backend)

    flagsk = _cached_kernel(ka_split_flags_kernel!, backend, workgroup)
    compactk = _cached_kernel(ka_split_compact_kernel!, backend, workgroup)

    nact = 1
    nl = 0
    round = 0
    while nact > 0
        round += 1
        round <= ell_max + 1 || error("adaptive KA K_max split failed to terminate")
        m = 8 * nact
        m <= frontier_capacity || error(
            "adaptive KA split frontier capacity $frontier_capacity exceeded")

        flagsk(flags, sorted_keys, a_lev, a_key, a_lo, a_hi, nact, K_max, ell_max,
            Int32(1); ndrange=m)
        KA.synchronize(backend)
        nleaf = _ka_scan_total!(flags, prefix, m)
        nl + nleaf <= leaf_capacity || error(
            "adaptive KA leaf capacity $leaf_capacity exceeded")
        if nleaf > 0
            compactk(llev, lkey, llo, lhi, nl, flags, prefix, sorted_keys, a_lev, a_key,
                a_lo, a_hi, nact, ell_max; ndrange=m)
            KA.synchronize(backend)
        end
        nl += nleaf

        flagsk(flags, sorted_keys, a_lev, a_key, a_lo, a_hi, nact, K_max, ell_max,
            Int32(0); ndrange=m)
        KA.synchronize(backend)
        nact2 = _ka_scan_total!(flags, prefix, m)
        nact2 <= frontier_capacity || error(
            "adaptive KA active frontier count $nact2 exceeded capacity $frontier_capacity")
        if nact2 > 0
            compactk(b_lev, b_key, b_lo, b_hi, 0, flags, prefix, sorted_keys, a_lev, a_key,
                a_lo, a_hi, nact, ell_max; ndrange=m)
            KA.synchronize(backend)
        end
        a_lev, b_lev = b_lev, a_lev
        a_key, b_key = b_key, a_key
        a_lo, b_lo = b_lo, a_lo
        a_hi, b_hi = b_hi, a_hi
        nact = nact2
    end
    return nl, llev, lkey, llo, lhi
end

#------- Phase B: 2:1 balance (Jacobi rounds over the leaf key set) -------#

@inline function ka_upper_bound(keys, first, stop, key)
    lo = first
    hi = stop + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if @inbounds(keys[mid]) <= key
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

@inline function ka_morton_key(ix, iy, iz, ell)
    key = UInt64(0)
    for bit in 0:(ell - 1)
        key |= (UInt64((ix >> bit) & 0x1) << (3 * bit))
        key |= (UInt64((iy >> bit) & 0x1) << (3 * bit + 1))
        key |= (UInt64((iz >> bit) & 0x1) << (3 * bit + 2))
    end
    return key
end

@inline function ka_decode_morton_key(key, ell)
    ix = 0
    iy = 0
    iz = 0
    for bit in 0:(ell - 1)
        ix |= Int((key >> (3 * bit)) & UInt64(0x1)) << bit
        iy |= Int((key >> (3 * bit + 1)) & UInt64(0x1)) << bit
        iz |= Int((key >> (3 * bit + 2)) & UInt64(0x1)) << bit
    end
    return ix, iy, iz
end

@kernel function ka_leaf_shifted_kernel!(shifted, @Const(lev), @Const(key), ell_max)
    i = @index(Global)
    @inbounds shifted[i] = key[i] << (3 * (ell_max - Int(lev[i])))
end

# Mark every leaf that violates 2:1 against the current leaf set (mirrors
# `_adt_cuda_balance_mark_kernel!`): leaf B at level lev emits its <=8 touching
# parent-level cells; a leaf A at level <= lev-2 whose interval contains the
# emitted cell start is marked. `marks` is pre-zeroed by the driver (a
# same-kernel clear would race with concurrent mark writes from other threads).
@kernel function ka_balance_mark_kernel!(marks, @Const(lev), @Const(key), nl,
        @Const(sorted_starts), @Const(order), @Const(slev), ell_max)
    i = @index(Global)
    @inbounds if i <= nl
        l = Int(lev[i])
        if l >= 2
            cx, cy, cz = ka_decode_morton_key(key[i], l)
            Gc = 1 << (l - 1)
            qx0 = (cx - 1) >> 1
            qy0 = (cy - 1) >> 1
            qz0 = (cz - 1) >> 1
            for dz in 0:1, dy in 0:1, dx in 0:1
                qx = qx0 + dx
                qy = qy0 + dy
                qz = qz0 + dz
                if 0 <= qx < Gc && 0 <= qy < Gc && 0 <= qz < Gc
                    qstart = ka_morton_key(qx, qy, qz, l - 1) << (3 * (ell_max - (l - 1)))
                    j = ka_upper_bound(sorted_starts, 1, nl, qstart) - 1
                    if j != 0
                        aid = Int(order[j])
                        la = Int(slev[aid])
                        if la <= l - 2
                            astart = sorted_starts[j]
                            alen = UInt64(1) << (3 * (ell_max - la))
                            if qstart < astart + alen
                                marks[aid] = Int32(1)
                            end
                        end
                    end
                end
            end
        end
    end
end

# Per-leaf emission count: unmarked leaves keep one slot; marked leaves emit
# their occupied children (theory §1.4 -- the split leaf is occupied, so >= 1).
@kernel function ka_balance_count_kernel!(cnt, @Const(marks), @Const(lev), @Const(key),
        @Const(lo), @Const(hi), nl, @Const(sorted_keys), ell_max)
    i = @index(Global)
    @inbounds if i <= nl
        if marks[i] == Int32(0)
            cnt[i] = Int32(1)
        else
            m = 0
            for c in 0:7
                _, _, lo_c, hi_c = ka_child_range(sorted_keys, lev, key, lo, hi, i, c, ell_max)
                if lo_c <= hi_c
                    m += 1
                end
            end
            cnt[i] = Int32(m)
        end
    end
end

@kernel function ka_balance_emit_kernel!(dlev, dkey, dlo, dhi, @Const(marks), @Const(prefix),
        @Const(lev), @Const(key), @Const(lo), @Const(hi), nl, @Const(sorted_keys), ell_max)
    i = @index(Global)
    @inbounds if i <= nl
        base = i == 1 ? 0 : Int(prefix[i - 1])
        if marks[i] == Int32(0)
            dlev[base + 1] = lev[i]
            dkey[base + 1] = key[i]
            dlo[base + 1] = lo[i]
            dhi[base + 1] = hi[i]
        else
            w = 0
            for c in 0:7
                lc, ckey, lo_c, hi_c = ka_child_range(sorted_keys, lev, key, lo, hi, i, c, ell_max)
                if lo_c <= hi_c
                    w += 1
                    dlev[base + w] = Int32(lc)
                    dkey[base + w] = ckey
                    dlo[base + w] = Int32(lo_c)
                    dhi[base + w] = Int32(hi_c)
                end
            end
        end
    end
end

"""
    ka_adaptive_balance!(actx, nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi, sorted_keys, ell_max;
                          workgroup=64)

Backend-agnostic port of `_cuda_adaptive_balance!` (tree_batched_cuda.jl): Jacobi
2:1-balance sweep (theory §1.4, Sundar-style) to the fixed point over the K_max
leaf set produced by `ka_adaptive_build_leaves!`. `leaf_levels`/`leaf_keys`/
`leaf_lo`/`leaf_hi` are backend arrays (allocated at `actx.leaf_capacity`, logically
truncated to `1:nl`); `sorted_keys` is the same full-depth-sorted body key
array `ka_adaptive_build_leaves!` was given. Returns
`(nl, n_balance_splits, leaf_levels, leaf_keys, leaf_lo, leaf_hi)` -- the final
leaf arrays may be either input array (ping-pong), not necessarily the ones
passed in. Scratch/output buffers come from `actx` -- no allocation.
"""
function ka_adaptive_balance!(actx::KAAdaptiveTreeContext, nl::Int, leaf_levels, leaf_keys,
        leaf_lo, leaf_hi, sorted_keys::AbstractVector{UInt64}, ell_max::Int; workgroup::Int=64)
    backend = actx.backend
    b = actx.bufs
    leaf_capacity = actx.leaf_capacity

    shifted, order, scratch_starts = b.bal_shifted, b.bal_order, b.bal_scratch_starts
    marks, flags, prefix = b.bal_marks, b.bal_flags, b.bal_prefix
    dlev, dkey, dlo, dhi = b.dlev, b.dkey, b.dlo, b.dhi

    shiftedk = _cached_kernel(ka_leaf_shifted_kernel!, backend, workgroup)
    markk = _cached_kernel(ka_balance_mark_kernel!, backend, workgroup)
    countk = _cached_kernel(ka_balance_count_kernel!, backend, workgroup)
    emitk = _cached_kernel(ka_balance_emit_kernel!, backend, workgroup)

    src = (leaf_levels, leaf_keys, leaf_lo, leaf_hi)
    dst = (dlev, dkey, dlo, dhi)
    total = 0
    round = 0
    while true
        round += 1
        round <= 2 * ell_max + 4 || error(
            "adaptive KA 2:1 balance failed to reach a fixed point")

        shiftedk(shifted, src[1], src[2], ell_max; ndrange=nl)
        KA.synchronize(backend)
        ov = view(order, 1:nl)
        sortperm!(ov, view(shifted, 1:nl))
        ka_gather_values!(view(scratch_starts, 1:nl), view(shifted, 1:nl), ov; workgroup=workgroup)

        fill!(view(marks, 1:nl), Int32(0))
        markk(marks, src[1], src[2], nl, scratch_starts, order, src[1], ell_max; ndrange=nl)
        KA.synchronize(backend)

        copyto!(view(flags, 1:nl), view(marks, 1:nl))
        nmark = _ka_scan_total!(flags, prefix, nl)
        nmark == 0 && break
        total += nmark

        countk(flags, marks, src[1], src[2], src[3], src[4], nl, sorted_keys, ell_max; ndrange=nl)
        KA.synchronize(backend)
        nl2 = _ka_scan_total!(flags, prefix, nl)
        nl2 <= leaf_capacity || error(
            "adaptive KA leaf capacity $leaf_capacity exceeded during the balance sweep")

        emitk(dst[1], dst[2], dst[3], dst[4], marks, prefix, src[1], src[2], src[3], src[4],
            nl, sorted_keys, ell_max; ndrange=nl)
        KA.synchronize(backend)

        src, dst = dst, src
        nl = nl2
    end
    return nl, total, src[1], src[2], src[3], src[4]
end

#------- Phase C: level-major node table finalize -------#

@kernel function ka_ancestor_flags_kernel!(flags, @Const(slev), nl, L)
    i = @index(Global)
    @inbounds if i <= nl
        flags[i] = Int(slev[i]) >= L ? Int32(1) : Int32(0)
    end
end

@kernel function ka_ancestor_compact_kernel!(cand, @Const(flags), @Const(prefix),
        @Const(slev), @Const(skey), nl, L)
    i = @index(Global)
    @inbounds if i <= nl && flags[i] == Int32(1)
        cand[Int(prefix[i])] = skey[i] >> (3 * (Int(slev[i]) - L))
    end
end

@kernel function ka_unique_flags_kernel!(flags, @Const(cand), m)
    i = @index(Global)
    @inbounds if i <= m
        flags[i] = (i == 1 || cand[i] != cand[i - 1]) ? Int32(1) : Int32(0)
    end
end

@kernel function ka_unique_compact_kernel!(node_keys, node_levels, base, @Const(flags),
        @Const(prefix), @Const(cand), m, L)
    i = @index(Global)
    @inbounds if i <= m && flags[i] == Int32(1)
        idx = base + Int(prefix[i])
        node_keys[idx] = cand[i]
        node_levels[idx] = L
    end
end

@kernel function ka_node_ranges_kernel!(node_lo, node_hi, @Const(node_keys),
        @Const(node_levels), n_nodes, @Const(sorted_keys), n, ell_max)
    i = @index(Global)
    @inbounds if i <= n_nodes
        shift = 3 * (ell_max - Int(node_levels[i]))
        startk = node_keys[i] << shift
        endk = startk + (UInt64(1) << shift)
        node_lo[i] = Int32(ka_lower_bound(sorted_keys, 1, n, startk))
        node_hi[i] = Int32(ka_lower_bound(sorted_keys, 1, n, endk) - 1)
    end
end

@kernel function ka_node_geometry_kernel!(node_coords, node_centers, @Const(node_keys),
        @Const(node_levels), n_nodes, x_min, h0)
    i = @index(Global)
    @inbounds if i <= n_nodes
        L = Int(node_levels[i])
        cx, cy, cz = ka_decode_morton_key(node_keys[i], L)
        node_coords[1, i] = cx
        node_coords[2, i] = cy
        node_coords[3, i] = cz
        TF = eltype(node_centers)
        width = (2 * h0) / (1 << L)
        node_centers[1, i] = x_min[1] + width * (TF(cx) + TF(0.5))
        node_centers[2, i] = x_min[2] + width * (TF(cy) + TF(0.5))
        node_centers[3, i] = x_min[3] + width * (TF(cz) + TF(0.5))
    end
end

@kernel function ka_parent_kernel!(parent_index, @Const(node_keys), base, count,
        base_prev, count_prev)
    i = @index(Global)
    @inbounds if i <= count
        node = base + i
        pk = node_keys[node] >> 3
        parent_index[node] = ka_lower_bound(node_keys, base_prev + 1,
            base_prev + count_prev, pk)
    end
end

@kernel function ka_children_kernel!(child_ranges, @Const(node_keys), base, count,
        base_next, count_next)
    i = @index(Global)
    @inbounds if i <= count
        node = base + i
        k = node_keys[node] << 3
        firstc = ka_lower_bound(node_keys, base_next + 1, base_next + count_next, k)
        endc = ka_lower_bound(node_keys, base_next + 1, base_next + count_next, k + UInt64(8))
        child_ranges[1, node] = endc > firstc ? firstc : 0
        child_ranges[2, node] = endc - firstc
    end
end

@kernel function ka_leaf_flags_kernel!(flags, @Const(child_ranges), n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes
        flags[i] = child_ranges[2, i] == 0 ? Int32(1) : Int32(0)
    end
end

@kernel function ka_leaf_compact_kernel!(leaf_index, leaf_slot_of, @Const(flags),
        @Const(prefix), n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes
        leaf_slot_of[i] = Int32(0)
        if flags[i] == Int32(1)
            slot = Int(prefix[i])
            leaf_index[slot] = Int32(i)
            leaf_slot_of[i] = Int32(slot)
        end
    end
end

@kernel function ka_cell_arrays_kernel!(cell_ranges, cell_centers, cell_keys, leaf_to_node,
        @Const(leaf_index), @Const(node_lo), @Const(node_hi), @Const(node_centers),
        @Const(node_keys), n_leaves)
    c = @index(Global)
    @inbounds if c <= n_leaves
        f = Int(leaf_index[c])
        leaf_to_node[c] = f
        cell_ranges[1, c] = Int(node_lo[f])
        cell_ranges[2, c] = Int(node_hi[f]) - Int(node_lo[f]) + 1
        cell_centers[1, c] = node_centers[1, f]
        cell_centers[2, c] = node_centers[2, f]
        cell_centers[3, c] = node_centers[3, f]
        cell_keys[c] = node_keys[f]
    end
end

"""
    ka_adaptive_finalize!(actx, nl, leaf_levels, leaf_keys, leaf_lo, leaf_hi, sorted_keys,
                           ell_max, n, x_min, h0::TF; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_finalize!` (tree_batched_cuda.jl): builds the
level-major node table (ancestor + leaf nodes, Morton-sorted within each level) from
the final (post 2:1-balance) leaf set produced by `ka_adaptive_balance!`, then resolves
node body-ranges, geometry, parent/child links, leaf compaction, and the leaf-indexed
cell presentation arrays. `leaf_levels`/`leaf_keys`/`leaf_lo`/`leaf_hi` are backend
arrays logically truncated to `1:nl`; `sorted_keys` is the same full-depth-sorted body
key array used throughout the adaptive-tree pipeline. `x_min`/`h0` are the tree's
bounding-box origin/half-width (plain scalars, e.g. an `SVector{3,TF}`/`TF`, not device
arrays -- mirrors CUDA's `grid.x_min`/`grid.h0` convention).

Returns a `NamedTuple` `(n_nodes, n_leaves, node_keys, node_levels, node_coords,
node_centers, node_lo, node_hi, parent_index, child_ranges, leaf_index, leaf_slot_of,
cell_ranges, cell_centers, cell_keys, leaf_to_node)`. The node-indexed arrays are
allocated at `actx.node_capacity` and logically truncated to `1:n_nodes`; the
leaf-indexed cell arrays are allocated at `actx.leaf_capacity` and logically truncated
to `1:n_leaves` (== `nl`, asserted). Scratch/output buffers come from `actx` -- no
allocation.
"""
function ka_adaptive_finalize!(actx::KAAdaptiveTreeContext, nl::Int, leaf_levels, leaf_keys,
        leaf_lo, leaf_hi, sorted_keys::AbstractVector{UInt64}, ell_max::Int, n::Int, x_min,
        h0::TF; workgroup::Int=64) where TF
    backend = actx.backend
    b = actx.bufs

    shifted, order = b.fin_shifted, b.fin_order
    skey, slev, cand = b.fin_skey, b.fin_slev, b.fin_cand
    flags, prefix = b.fin_flags, b.fin_prefix

    node_keys, node_levels = b.node_keys, b.node_levels
    node_coords, node_centers = b.node_coords, b.node_centers
    node_lo, node_hi = b.node_lo, b.node_hi
    parent_index, child_ranges = b.parent_index, b.child_ranges
    leaf_index, leaf_slot_of = b.leaf_index, b.leaf_slot_of

    shiftedk = _cached_kernel(ka_leaf_shifted_kernel!, backend, workgroup)
    ancflagsk = _cached_kernel(ka_ancestor_flags_kernel!, backend, workgroup)
    anccompactk = _cached_kernel(ka_ancestor_compact_kernel!, backend, workgroup)
    uflagsk = _cached_kernel(ka_unique_flags_kernel!, backend, workgroup)
    ucompactk = _cached_kernel(ka_unique_compact_kernel!, backend, workgroup)
    rangesk = _cached_kernel(ka_node_ranges_kernel!, backend, workgroup)
    geomk = _cached_kernel(ka_node_geometry_kernel!, backend, workgroup)
    parentk = _cached_kernel(ka_parent_kernel!, backend, workgroup)
    childrenk = _cached_kernel(ka_children_kernel!, backend, workgroup)
    leafflagsk = _cached_kernel(ka_leaf_flags_kernel!, backend, workgroup)
    leafcompactk = _cached_kernel(ka_leaf_compact_kernel!, backend, workgroup)
    cellk = _cached_kernel(ka_cell_arrays_kernel!, backend, workgroup)

    shiftedk(shifted, leaf_levels, leaf_keys, ell_max; ndrange=nl)
    KA.synchronize(backend)
    ov = view(order, 1:nl)
    sortperm!(ov, view(shifted, 1:nl))
    ka_gather_values!(view(skey, 1:nl), leaf_keys, ov; workgroup=workgroup)
    ka_gather_values!(view(slev, 1:nl), leaf_levels, ov; workgroup=workgroup)

    node_capacity = actx.node_capacity
    off = zeros(Int, ell_max + 2)
    n_nodes = 0
    for L in 0:ell_max
        off[L + 1] = n_nodes
        ancflagsk(flags, slev, nl, L; ndrange=nl)
        KA.synchronize(backend)
        me = _ka_scan_total!(flags, prefix, nl)
        me == 0 && continue
        anccompactk(cand, flags, prefix, slev, skey, nl, L; ndrange=nl)
        KA.synchronize(backend)
        uflagsk(flags, cand, me; ndrange=me)
        KA.synchronize(backend)
        mL = _ka_scan_total!(flags, prefix, me)
        n_nodes + mL <= node_capacity || error(
            "adaptive KA node capacity $node_capacity exceeded; raise node_capacity")
        ucompactk(node_keys, node_levels, n_nodes, flags, prefix, cand, me, L; ndrange=me)
        KA.synchronize(backend)
        n_nodes += mL
    end
    off[ell_max + 2] = n_nodes

    rangesk(node_lo, node_hi, node_keys, node_levels, n_nodes, sorted_keys, n, ell_max;
        ndrange=n_nodes)
    geomk(node_coords, node_centers, node_keys, node_levels, n_nodes, x_min, h0;
        ndrange=n_nodes)
    KA.synchronize(backend)

    fill!(view(parent_index, 1:n_nodes), 0)
    fill!(view(child_ranges, :, 1:n_nodes), 0)
    for L in 1:ell_max
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_prev = off[L]
        count_prev = off[L + 1] - off[L]
        parentk(parent_index, node_keys, base, count, base_prev, count_prev; ndrange=count)
    end
    for L in 0:(ell_max - 1)
        base = off[L + 1]
        count = off[L + 2] - off[L + 1]
        count > 0 || continue
        base_next = off[L + 2]
        count_next = off[L + 3] - off[L + 2]
        childrenk(child_ranges, node_keys, base, count, base_next, count_next; ndrange=count)
    end
    KA.synchronize(backend)

    leafflagsk(flags, child_ranges, n_nodes; ndrange=n_nodes)
    KA.synchronize(backend)
    n_leaves = _ka_scan_total!(flags, prefix, n_nodes)
    n_leaves == nl || throw(AssertionError(
        "adaptive KA finalize: leaf count mismatch ($n_leaves vs $nl)"))
    leafcompactk(leaf_index, leaf_slot_of, flags, prefix, n_nodes; ndrange=n_nodes)
    KA.synchronize(backend)

    cell_ranges = view(b.cell_ranges, :, 1:n_leaves)
    cell_centers = view(b.cell_centers, :, 1:n_leaves)
    cell_keys = view(b.cell_keys, 1:n_leaves)
    leaf_to_node = view(b.leaf_to_node, 1:n_leaves)
    cellk(cell_ranges, cell_centers, cell_keys, leaf_to_node, leaf_index, node_lo, node_hi,
        node_centers, node_keys, n_leaves; ndrange=n_leaves)
    KA.synchronize(backend)

    return (n_nodes=n_nodes, n_leaves=n_leaves, node_keys=node_keys, node_levels=node_levels,
        node_coords=node_coords, node_centers=node_centers, node_lo=node_lo, node_hi=node_hi,
        parent_index=parent_index, child_ranges=child_ranges, leaf_index=leaf_index,
        leaf_slot_of=leaf_slot_of, cell_ranges=cell_ranges, cell_centers=cell_centers,
        cell_keys=cell_keys, leaf_to_node=leaf_to_node, level_offsets=off)
end

#------- Phase D: per-node subtree sigma_max upward pass -------#

@kernel function ka_leaf_sigma_kernel!(node_sigma, @Const(node_lo), @Const(node_hi),
        @Const(child_ranges), @Const(source_bodies), sigma_row, n_nodes)
    i = @index(Global)
    @inbounds if i <= n_nodes && child_ranges[2, i] == 0
        TF = eltype(node_sigma)
        m = zero(TF)
        for r in Int(node_lo[i]):Int(node_hi[i])
            s = source_bodies[sigma_row, r]
            s > m && (m = s)
        end
        node_sigma[i] = m
    end
end

@kernel function ka_sigma_up_kernel!(node_sigma, @Const(child_ranges), base, count)
    i = @index(Global)
    @inbounds if i <= count
        node = base + i
        nc = Int(child_ranges[2, node])
        if nc != 0
            c0 = Int(child_ranges[1, node])
            TF = eltype(node_sigma)
            m = zero(TF)
            for c in c0:(c0 + nc - 1)
                s = node_sigma[c]
                s > m && (m = s)
            end
            node_sigma[node] = m
        end
    end
end

"""
    ka_adaptive_sigma_sweep!(actx, node_lo, node_hi, child_ranges, n_nodes, level_offsets,
                              ell_max, source_bodies, sigma_row; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_sigma_sweep!` (tree_batched_cuda.jl): computes,
for every node of the finalized level-major node table (see `ka_adaptive_finalize!`),
the max of `source_bodies[sigma_row, :]` over its subtree -- leaves take the max over
their own body range (`node_lo`/`node_hi`), interior nodes take the max over their
children's already-computed `node_sigma_max`, processed level-by-level from `ell_max - 1`
up to `0` using `level_offsets` (so children are always finalized before their parent is
visited). `source_bodies` must be indexed in the same sorted-body ordering as
`node_lo`/`node_hi` (i.e. the same `sorted_keys` passed to `ka_adaptive_finalize!`).

Returns `node_sigma_max`, an array sized `actx.node_capacity`, logically truncated to
`1:n_nodes`. Buffer comes from `actx` -- no allocation.
"""
function ka_adaptive_sigma_sweep!(actx::KAAdaptiveTreeContext, node_lo, node_hi, child_ranges,
        n_nodes::Int, level_offsets::Vector{Int}, ell_max::Int,
        source_bodies::AbstractMatrix{TF}, sigma_row::Int; workgroup::Int=64) where TF
    backend = actx.backend
    node_sigma = actx.bufs.node_sigma

    leafsigmak = _cached_kernel(ka_leaf_sigma_kernel!, backend, workgroup)
    upk = _cached_kernel(ka_sigma_up_kernel!, backend, workgroup)

    leafsigmak(node_sigma, node_lo, node_hi, child_ranges, source_bodies, sigma_row,
        n_nodes; ndrange=n_nodes)
    KA.synchronize(backend)

    for L in (ell_max - 1):-1:0
        base = level_offsets[L + 1]
        count = level_offsets[L + 2] - level_offsets[L + 1]
        count > 0 || continue
        upk(node_sigma, child_ranges, base, count; ndrange=count)
        KA.synchronize(backend)
    end

    return node_sigma
end

#------- Phase E: DTR interaction-list build (U/V/W/X) -------#
#
# Backend-agnostic port of `_cuda_adaptive_build_lists!` and its six kernels
# (src/tree_batched_cuda.jl:910-1139), the frontier dual-tree-recursion sweep of
# adaptive octree theory §2.7 with sticky sigma demotion (§5.2).
#
# Shape note: the CPU reference (`build_adaptive_interaction_lists!`,
# src/interaction_list_batched.jl:1182) is a DFS over an explicit pair stack,
# while the device version is a level-synchronous BFS over a pair frontier with
# the same flags/scan/compact decomposition as Phases A-C. Both enumerate the
# same pair SET; they emit it in different orders. Any comparison between them
# must canonicalize (sort) first -- element-wise equality is meaningless here.
#
# Precision note: CUDA's `_adt_cuda_classify` hardcodes Float64 for the sigma
# gate. Metal has no Float64 at all, so the KA port makes the gate arithmetic
# generic in a float type `TG` instead (the CUDA path is left exactly as it is).
# `ka_gate_float_type` picks a default that is always valid for the backend the
# tree lives on -- the sigma array's own float type, which is whatever that
# backend supports -- and any float type can be forced via the `gate_type`
# keyword: Float64 on CUDA to mirror the native path bit-for-bit, Float32 on
# Metal, Float16 or a custom AbstractFloat if some future backend wants it.
# Only the gate comparison `delta_min2*g2 < cut^2` is precision-sensitive, and
# only for pairs sitting within a rounding step of the threshold; everything
# else in the classification is exact integer lattice arithmetic, so the choice
# cannot change which pairs are geometrically near.
#
# `g2` is bounded by 3*(2^ell_max)^2, so it is exactly representable in Float32
# for ell_max <= 12 and in Float16 for ell_max <= 4 -- past those the gate
# arithmetic, not the tree, is what limits precision.

"""
    ka_gate_float_type(node_sigma) -> Type{<:AbstractFloat}

Default float type for the DTR sigma gate: the float type the tree's own sigma
array already uses, which is by construction one the backend supports. Override
with the `gate_type` keyword on `ka_adaptive_build_lists!` when a specific
precision is wanted (e.g. Float64 on CUDA, to match `_adt_cuda_classify`).
"""
ka_gate_float_type(node_sigma) = float(eltype(node_sigma))

const _KA_KIND_U = Int32(1)
const _KA_KIND_V = Int32(2)
const _KA_KIND_W = Int32(3)
const _KA_KIND_X = Int32(4)
const _KA_KIND_EXPAND = Int32(5)

@inline function ka_axis_clamp(ca::Int, la::Int, cb::Int, lb::Int)
    k = lb - la
    a0 = ca << k
    a1 = ((ca + 1) << k) - 1
    return cb < a0 ? a0 - cb : (cb > a1 ? cb - a1 : 0)
end

# Mirror of `_adt_cuda_classify`: returns (kind, dem_out, leaf_a, leaf_b).
@inline function ka_dtr_classify(node_levels, node_coords, child_ranges, node_sigma,
        ia::Int, ib::Int, dem::Bool, q::Int, gate::Bool, rho_t::TG,
        delta_min2::TG, ell_max::Int) where TG
    @inbounds begin
        la = Int(node_levels[ia]); lb = Int(node_levels[ib])
        ax = Int(node_coords[1, ia]); ay = Int(node_coords[2, ia]); az = Int(node_coords[3, ia])
        bx = Int(node_coords[1, ib]); by = Int(node_coords[2, ib]); bz = Int(node_coords[3, ib])
        local dx::Int, dy::Int, dz::Int
        if la == lb
            dx = bx - ax; dy = by - ay; dz = bz - az
        elseif la < lb
            dx = ka_axis_clamp(ax, la, bx, lb)
            dy = ka_axis_clamp(ay, la, by, lb)
            dz = ka_axis_clamp(az, la, bz, lb)
        else
            dx = ka_axis_clamp(bx, lb, ax, la)
            dy = ka_axis_clamp(by, lb, ay, la)
            dz = ka_axis_clamp(bz, lb, az, la)
        end
        near = dem || (dx * dx + dy * dy + dz * dz <= q)
        dem_out = dem
        if !near && gate
            # integer-exact squared AABB gap on the finest lattice (host mirror)
            sa = 1 << (ell_max - la)
            sb = 1 << (ell_max - lb)
            g2 = 0
            g = max(ax * sa - (bx * sb + sb), bx * sb - (ax * sa + sa), 0); g2 += g * g
            g = max(ay * sa - (by * sb + sb), by * sb - (ay * sa + sa), 0); g2 += g * g
            g = max(az * sa - (bz * sb + sb), bz * sb - (az * sa + sa), 0); g2 += g * g
            cut = rho_t * TG(node_sigma[ib])
            if delta_min2 * TG(g2) < cut * cut
                near = true
                dem_out = true
            end
        end
        leaf_a = child_ranges[2, ia] == 0
        leaf_b = child_ranges[2, ib] == 0
        local kind::Int32
        if !near
            kind = la == lb ? _KA_KIND_V : (la < lb ? _KA_KIND_W : _KA_KIND_X)
        elseif leaf_a && leaf_b
            kind = _KA_KIND_U
        else
            kind = _KA_KIND_EXPAND
        end
    end
    return kind, dem_out, leaf_a, leaf_b
end

@inline function ka_expand_count(node_levels, child_ranges, ia::Int, ib::Int,
        leaf_a::Bool, leaf_b::Bool)
    @inbounds begin
        la = Int(node_levels[ia]); lb = Int(node_levels[ib])
        na = Int(child_ranges[2, ia]); nb = Int(child_ranges[2, ib])
        if la == lb
            leaf_a && return nb
            leaf_b && return na
            return na * nb
        end
        return la < lb ? nb : na
    end
end

# j-th (1-based) child pair of an EXPAND pair, in the host's deterministic
# (ja-major, jb-minor) order.
@inline function ka_expand_get(node_levels, child_ranges, ia::Int, ib::Int,
        leaf_a::Bool, leaf_b::Bool, j::Int)
    @inbounds begin
        la = Int(node_levels[ia]); lb = Int(node_levels[ib])
        if la == lb
            if leaf_a
                return ia, Int(child_ranges[1, ib]) + j - 1
            elseif leaf_b
                return Int(child_ranges[1, ia]) + j - 1, ib
            else
                nb = Int(child_ranges[2, ib])
                return Int(child_ranges[1, ia]) + (j - 1) ÷ nb,
                    Int(child_ranges[1, ib]) + (j - 1) % nb
            end
        elseif la < lb
            return ia, Int(child_ranges[1, ib]) + j - 1
        else
            return Int(child_ranges[1, ia]) + j - 1, ib
        end
    end
end

@kernel function ka_dtr_seed_kernel!(fa, fb, fdem)
    i = @index(Global)
    @inbounds if i == 1
        fa[1] = Int32(1)
        fb[1] = Int32(1)
        fdem[1] = Int32(0)
    end
end

# want: 1..5 kind flags; 6 = demotion-trigger diagnostic count
@kernel function ka_dtr_flags_kernel!(flags, @Const(fa), @Const(fb), @Const(fdem), np,
        @Const(node_levels), @Const(node_coords), @Const(child_ranges), @Const(node_sigma),
        q, gate, rho_t, delta_min2, ell_max, want)
    p = @index(Global)
    @inbounds if p <= np
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, dem_out, _, _ = ka_dtr_classify(node_levels, node_coords, child_ranges,
            node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2, ell_max)
        if want == Int32(6)
            flags[p] = (dem_out && !dem) ? Int32(1) : Int32(0)
        else
            flags[p] = kind == want ? Int32(1) : Int32(0)
        end
    end
end

# Emit U/W/X node-id pairs at base offsets (deterministic scan order).
@kernel function ka_dtr_emit_pairs_kernel!(dst_t, dst_s, base, @Const(flags),
        @Const(prefix), @Const(fa), @Const(fb), np)
    p = @index(Global)
    @inbounds if p <= np && flags[p] == Int32(1)
        idx = base + Int(prefix[p])
        dst_t[idx] = fa[p]
        dst_s[idx] = fb[p]
    end
end

# Emit V pairs with the global class id; validity per the 025 phase-table
# membership (sticky-demotion invariant) via the violation flag.
@kernel function ka_dtr_emit_v_kernel!(vt, vs, vc, base, @Const(flags), @Const(prefix),
        @Const(fa), @Const(fb), np, @Const(node_levels), @Const(node_coords),
        @Const(offset_lut), @Const(level_class_of), reach, noffsets, first_m2l_level,
        violation_flags)
    p = @index(Global)
    @inbounds if p <= np && flags[p] == Int32(1)
        ia = Int(fa[p]); ib = Int(fb[p])
        la = Int(node_levels[ia])
        ox = Int(node_coords[1, ia]) - Int(node_coords[1, ib])
        oy = Int(node_coords[2, ia]) - Int(node_coords[2, ib])
        oz = Int(node_coords[3, ia]) - Int(node_coords[3, ib])
        k = 0
        if abs(ox) <= reach && abs(oy) <= reach && abs(oz) <= reach
            k = Int(offset_lut[ox + reach + 1, oy + reach + 1, oz + reach + 1])
        end
        phase = 1 + (Int(node_coords[1, ib]) & 1) + 2 * (Int(node_coords[2, ib]) & 1) +
            4 * (Int(node_coords[3, ib]) & 1)
        ok = k != 0 && la >= first_m2l_level &&
            level_class_of[phase, k, la + 1] != Int32(0)
        ok || (violation_flags[1] = Int32(1))
        idx = base + Int(prefix[p])
        vt[idx] = fa[p]
        vs[idx] = fb[p]
        vc[idx] = Int32((la - first_m2l_level) * noffsets + k)
    end
end

@kernel function ka_dtr_expand_count_kernel!(flags, @Const(fa), @Const(fb), @Const(fdem),
        np, @Const(node_levels), @Const(node_coords), @Const(child_ranges),
        @Const(node_sigma), q, gate, rho_t, delta_min2, ell_max)
    p = @index(Global)
    @inbounds if p <= np
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, _, leaf_a, leaf_b = ka_dtr_classify(node_levels, node_coords, child_ranges,
            node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2, ell_max)
        flags[p] = kind == _KA_KIND_EXPAND ?
            Int32(ka_expand_count(node_levels, child_ranges, ia, ib, leaf_a, leaf_b)) :
            Int32(0)
    end
end

@kernel function ka_dtr_expand_emit_kernel!(ga, gb, gdem, @Const(fa), @Const(fb),
        @Const(fdem), np, @Const(prefix), @Const(node_levels), @Const(node_coords),
        @Const(child_ranges), @Const(node_sigma), q, gate, rho_t, delta_min2, ell_max)
    p = @index(Global)
    @inbounds if p <= np
        ia = Int(fa[p]); ib = Int(fb[p]); dem = fdem[p] != Int32(0)
        kind, dem_out, leaf_a, leaf_b = ka_dtr_classify(node_levels, node_coords,
            child_ranges, node_sigma, ia, ib, dem, q, gate, rho_t, delta_min2, ell_max)
        if kind == _KA_KIND_EXPAND
            # exclusive prefix from the inclusive scan, as CUDA's emit kernel does
            base = p == 1 ? 0 : Int(prefix[p - 1])
            cnt = ka_expand_count(node_levels, child_ranges, ia, ib, leaf_a, leaf_b)
            d = dem_out ? Int32(1) : Int32(0)
            for j in 1:cnt
                ja, jb = ka_expand_get(node_levels, child_ranges, ia, ib, leaf_a, leaf_b, j)
                ga[base + j] = Int32(ja)
                gb[base + j] = Int32(jb)
                gdem[base + j] = d
            end
        end
    end
end

"""
    KAAdaptiveListsContext

Preallocated buffers for the DTR/list phases (E/F/G), mirroring the list portion
of CUDA's `DeviceAdaptiveCUDAContext`.

Constructed against the `KAAdaptiveTreeContext` whose tree it will build lists
for, and **shares that context's frontier-sized scratch** rather than allocating
a second copy: the DTR pair frontier aliases the tree-build frontier ping-pong
(`a_lev`/`a_lo`/`a_hi` and `b_lev`/`b_lo`/`b_hi`, all `Int32` at
`frontier_capacity`), and the DTR scan aliases `bl_flags`/`bl_prefix`. That is
safe because a tree build always completes before its list build, and CUDA's own
`actx` shares buffers the same way (the U CSR reusing the V sort scratch is the
same trick, kept below). At n=1e6 with `frontier_capacity` ~2e7 this is the
difference between ~640MB of duplicate frontier scratch and none.

**Ordering requirement**: do not interleave `ka_build_adaptive_tree!` and
`ka_adaptive_build_lists!` on the same pair of contexts -- finish the tree, then
build its lists. Rebuilding the tree invalidates any list built from it anyway.

`offset_lut`/`level_class_of` are the geometry LUTs from an
`AdaptiveInteractionLists` (moved to the backend by the caller).
"""
struct KAAdaptiveListsContext{B,L3,C3,NT<:NamedTuple}
    backend::B
    frontier_capacity::Int
    u_capacity::Int
    v_capacity::Int
    wx_capacity::Int
    offset_lut::L3
    level_class_of::C3
    lut_reach::Int
    noffsets::Int
    first_m2l_level::Int
    ell_max::Int
    nclasses::Int
    class_starts::Vector{Int}
    level_starts::Vector{Int}
    host_class_counts::Vector{Int32}
    bufs::NT
end

function ka_allocate_lists_context(actx::KAAdaptiveTreeContext, offset_lut,
        level_class_of; u_capacity::Int, v_capacity::Int, wx_capacity::Int,
        lut_reach::Int, noffsets::Int, first_m2l_level::Int, ell_max::Int=0,
        nclasses::Int=max(1, noffsets * (ell_max + 1 - first_m2l_level)),
        leaf_capacity::Int=actx.leaf_capacity, maxn::Int=actx.maxn)
    backend = actx.backend
    FC = actx.frontier_capacity
    UC, VC, WC = u_capacity, v_capacity, wx_capacity
    LC, MN = max(1, leaf_capacity), max(1, maxn)
    t = actx.bufs
    bufs = (
        # DTR pair frontier + scan: aliases of the tree-build frontier scratch
        # (see the docstring). Same element type and length; the tree build is
        # finished by the time any of these are read.
        fa=t.a_lev, fb=t.a_lo, fdem=t.a_hi,
        fa2=t.b_lev, fb2=t.b_lo, fdem2=t.b_hi,
        flags=t.bl_flags, prefix=t.bl_prefix,
        u_targets=KA.zeros(backend, Int32, UC), u_sources=KA.zeros(backend, Int32, UC),
        vstage_targets=KA.zeros(backend, Int32, VC),
        vstage_sources=KA.zeros(backend, Int32, VC),
        vstage_class=KA.zeros(backend, Int32, VC),
        w_targets=KA.zeros(backend, Int32, WC), w_sources=KA.zeros(backend, Int32, WC),
        x_targets=KA.zeros(backend, Int32, WC), x_sources=KA.zeros(backend, Int32, WC),
        # slot 1: V phase-table violation; slot 2: U endpoint not a leaf slot
        violation_flags=KA.zeros(backend, Int32, 2),

        # Phase F: V class partition into the CSR route stream
        vsort_keys=KA.zeros(backend, UInt64, VC), vsort_ix=KA.zeros(backend, Int, VC),
        route_targets=KA.zeros(backend, Int, VC), route_sources=KA.zeros(backend, Int, VC),
        route_class=KA.zeros(backend, Int32, VC),
        route_class_offset=KA.zeros(backend, Int32, VC),
        class_counts_dev=KA.zeros(backend, Int32, nclasses),

        # Phase G: U endpoints -> leaf slots, then target-major U CSR
        direct_targets=KA.zeros(backend, Int, UC),
        direct_sources=KA.zeros(backend, Int, UC),
        u_csr_offsets=KA.zeros(backend, Int32, LC + 1),
        u_csr_sources=KA.zeros(backend, Int32, UC),
        u_csr_body_leaf=KA.zeros(backend, Int32, MN),
    )
    # Guard the aliasing assumption rather than trusting it silently.
    all(x -> length(x) >= FC, (bufs.fa, bufs.fb, bufs.fdem, bufs.fa2, bufs.fb2,
        bufs.fdem2, bufs.flags, bufs.prefix)) || throw(ArgumentError(
        "tree context's frontier scratch is smaller than frontier_capacity=$FC"))
    return KAAdaptiveListsContext(backend, FC, UC, VC, WC, offset_lut, level_class_of,
        lut_reach, noffsets, first_m2l_level, ell_max, nclasses,
        zeros(Int, nclasses + 1), zeros(Int, ell_max + 2), zeros(Int32, nclasses), bufs)
end

"""
    ka_adaptive_build_lists!(lctx, node_levels, node_coords, child_ranges, node_sigma;
                             ell_max, near_radius2, gate, rho_t, delta_min2, workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_lists!`: the frontier DTR sweep
(theory §2.7) producing the U/V/W/X interaction lists from a finalized adaptive
octree. Returns `(n_u, n_v, n_w, n_x, n_dem)`; the lists themselves live in
`lctx.bufs`, valid over `1:n_*`.

`rho_t`/`delta_min2` may be any `Real`; they are converted to `gate_type`,
which defaults to `ka_gate_float_type(node_sigma)` and controls the sigma-gate
precision (see the precision note above). Pass `gate_type=Float64` on CUDA to
mirror `_adt_cuda_classify` bit-for-bit.
"""
function ka_adaptive_build_lists!(lctx::KAAdaptiveListsContext, node_levels, node_coords,
        child_ranges, node_sigma; ell_max::Int, near_radius2::Int, gate::Bool,
        rho_t::Real, delta_min2::Real,
        gate_type::Type{TG}=ka_gate_float_type(node_sigma),
        workgroup::Int=64) where {TG<:AbstractFloat}
    backend = lctx.backend
    b = lctx.bufs
    # Convert once, on the host: the kernels take these as scalar arguments, so
    # every pair sees the identical value and the gate stays in exactly TG.
    rho_t_g = TG(rho_t)
    delta_min2_g = TG(delta_min2)
    fill!(b.violation_flags, Int32(0))

    seedk = _cached_kernel(ka_dtr_seed_kernel!, backend, 1)
    flagk = _cached_kernel(ka_dtr_flags_kernel!, backend, workgroup)
    emitk = _cached_kernel(ka_dtr_emit_pairs_kernel!, backend, workgroup)
    emitvk = _cached_kernel(ka_dtr_emit_v_kernel!, backend, workgroup)
    ecountk = _cached_kernel(ka_dtr_expand_count_kernel!, backend, workgroup)
    eemitk = _cached_kernel(ka_dtr_expand_emit_kernel!, backend, workgroup)

    a = (b.fa, b.fb, b.fdem)
    bb = (b.fa2, b.fb2, b.fdem2)
    seedk(a[1], a[2], a[3]; ndrange=1)

    node_args = (node_levels, node_coords, child_ranges, node_sigma)
    gate_args = (near_radius2, gate, rho_t_g, delta_min2_g, ell_max)

    np = 1
    n_u = 0; n_v = 0; n_w = 0; n_x = 0; n_dem = 0
    rounds = 0
    while np > 0
        rounds += 1
        rounds <= 2 * ell_max + 3 || throw(AssertionError(
            "adaptive KA DTR failed to terminate"))

        # U / W / X share the plain pair-emit path; only the `want` code, the
        # running count and the destination buffers differ. Written out rather
        # than looped, mirroring `_cuda_adaptive_build_lists!`.
        emit_kind! = function (want, base, cap, dst_t, dst_s, label)
            flagk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args..., want;
                ndrange=np)
            m = _ka_scan_total!(b.flags, b.prefix, np)
            base + m <= cap || throw(AssertionError(
                "adaptive KA $label list capacity $cap exceeded"))
            m > 0 && emitk(dst_t, dst_s, base, b.flags, b.prefix, a[1], a[2], np; ndrange=np)
            return m
        end
        n_u += emit_kind!(_KA_KIND_U, n_u, lctx.u_capacity, b.u_targets, b.u_sources, "U")
        n_w += emit_kind!(_KA_KIND_W, n_w, lctx.wx_capacity, b.w_targets, b.w_sources, "W")
        n_x += emit_kind!(_KA_KIND_X, n_x, lctx.wx_capacity, b.x_targets, b.x_sources, "X")

        # V carries the geometry class id, so it needs its own emit kernel.
        flagk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args..., _KA_KIND_V;
            ndrange=np)
        m = _ka_scan_total!(b.flags, b.prefix, np)
        n_v + m <= lctx.v_capacity || throw(AssertionError(
            "adaptive KA V route capacity $(lctx.v_capacity) exceeded"))
        if m > 0
            emitvk(b.vstage_targets, b.vstage_sources, b.vstage_class, n_v, b.flags,
                b.prefix, a[1], a[2], np, node_levels, node_coords, lctx.offset_lut,
                lctx.level_class_of, lctx.lut_reach, lctx.noffsets, lctx.first_m2l_level,
                b.violation_flags; ndrange=np)
        end
        n_v += m

        # demotion diagnostic
        if gate
            flagk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args..., Int32(6);
                ndrange=np)
            n_dem += _ka_scan_total!(b.flags, b.prefix, np)
        end

        # expand: count child pairs per frontier entry, scan, emit next frontier
        ecountk(b.flags, a[1], a[2], a[3], np, node_args..., gate_args...; ndrange=np)
        np2 = _ka_scan_total!(b.flags, b.prefix, np)
        np2 <= lctx.frontier_capacity || throw(AssertionError(
            "adaptive KA DTR frontier capacity $(lctx.frontier_capacity) exceeded"))
        np2 > 0 && eemitk(bb[1], bb[2], bb[3], a[1], a[2], a[3], np, b.prefix,
            node_args..., gate_args...; ndrange=np)
        a, bb = bb, a
        np = np2
    end

    # One sync at the end, not one per launch. KA kernels on a backend are
    # ordered on that backend's own queue, so consecutive launches need no
    # explicit barrier, and each `_ka_scan_total!` already forces a sync via its
    # host readback of the scan total. Per-kernel `KA.synchronize` is what cost
    # the M2M port 53.6ms vs 20ms before it was removed there (commit ff14b7e).
    KA.synchronize(backend)
    Int(Array(view(b.violation_flags, 1:1))[1]) == 0 || throw(AssertionError(
        "adaptive KA V pair lies outside the task-025 phase-table class set — " *
        "the sticky demotion invariant (theory §2.4/§5.2) is violated"))

    return n_u, n_v, n_w, n_x, n_dem
end

#------- Phase F: V-list class partition into the CSR route stream -------#
#
# Port of `_cuda_adaptive_partition_v!` and its three kernels
# (src/tree_batched_cuda.jl:1143-1218). The V stage stream carries a class id
# per pair; the routes have to be grouped by class so each M2L class becomes one
# contiguous slab. Sorting on `(class << 32) | emission_index` makes the
# partition stable by construction -- ties inside a class keep DTR emission
# order -- so no separate stable-sort primitive is needed.

@kernel function ka_vsort_keys_kernel!(keys, @Const(vclass), n_v)
    i = @index(Global)
    @inbounds if i <= n_v
        keys[i] = (UInt64(vclass[i]) << 32) | UInt64(i)
    end
end

@kernel function ka_csr_gather_kernel!(route_targets, route_sources, route_class,
        route_class_offset, @Const(vsort_ix), @Const(vt), @Const(vs), @Const(vc),
        n_v, noffsets)
    p = @index(Global)
    @inbounds if p <= n_v
        i = Int(vsort_ix[p])
        route_targets[p] = Int(vt[i])
        route_sources[p] = Int(vs[i])
        c = Int(vc[i])
        route_class[p] = Int32(c)
        # per-offset id for the dense family (class_base = 0 convention)
        route_class_offset[p] = Int32(c - ((c - 1) ÷ noffsets) * noffsets)
    end
end

@kernel function ka_class_histogram_kernel!(counts, @Const(vclass), n_v)
    i = @index(Global)
    @inbounds if i <= n_v
        KA.@atomic counts[Int(vclass[i])] += Int32(1)
    end
end

"""
    ka_adaptive_partition_v!(lctx, n_v; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_partition_v!`: deterministically
partitions the `n_v` staged V pairs by class into the CSR route stream
(`lctx.bufs.route_*`, valid over `1:n_v`) and fills `lctx.class_starts` /
`lctx.level_starts` on the host. Returns `n_v` (the route count).
"""
function ka_adaptive_partition_v!(lctx::KAAdaptiveListsContext, n_v::Int;
        workgroup::Int=64)
    backend = lctx.backend
    b = lctx.bufs
    cc = lctx.host_class_counts

    if n_v == 0
        fill!(cc, Int32(0))
    else
        keysk = _cached_kernel(ka_vsort_keys_kernel!, backend, workgroup)
        gatherk = _cached_kernel(ka_csr_gather_kernel!, backend, workgroup)
        histk = _cached_kernel(ka_class_histogram_kernel!, backend, workgroup)

        keysk(b.vsort_keys, b.vstage_class, n_v; ndrange=n_v)
        # `sortperm!` dispatches to the backend's own sort (confirmed working on
        # Metal in Phase B), matching CUDA's `_cuda_sortperm_into!` convention.
        sortperm!(view(b.vsort_ix, 1:n_v), view(b.vsort_keys, 1:n_v))
        gatherk(b.route_targets, b.route_sources, b.route_class, b.route_class_offset,
            b.vsort_ix, b.vstage_targets, b.vstage_sources, b.vstage_class, n_v,
            lctx.noffsets; ndrange=n_v)
        fill!(b.class_counts_dev, Int32(0))
        histk(b.class_counts_dev, b.vstage_class, n_v; ndrange=n_v)
        copyto!(cc, Array(b.class_counts_dev))   # forces the sync
    end

    cs = lctx.class_starts
    cs[1] = 1
    @inbounds for c in 1:lctx.nclasses
        cs[c + 1] = cs[c] + Int(cc[c])
    end
    ls = lctx.level_starts
    first = lctx.first_m2l_level
    @inbounds for L in 0:lctx.ell_max
        ls[L + 1] = L < first ? 1 : cs[(L - first) * lctx.noffsets + 1]
    end
    ls[lctx.ell_max + 2] = cs[end]
    return n_v
end


#------- Phase G: U endpoints -> leaf slots, then the target-major U CSR -------#
#
# Port of `_adt_cuda_u_slots_kernel!` and `_cuda_adaptive_build_u_csr!` with its
# three kernels (src/tree_batched_cuda.jl:1222-1324). The DTR emits U pairs in
# frontier-major order, which is NOT target-major, so building the fused
# nearfield's target-owned CSR needs the same stable (target-slot, index) key
# sort the V partition uses.
#
# The offsets kernel writes the half-open slot range (tprev, t] for each CSR
# position, which fills in leaves that own no U pairs at all; slots past the
# last occupied target keep the `n_u + 1` prefill. Targets are sorted, so those
# ranges are disjoint and the concurrent writes never overlap.

@kernel function ka_u_slots_kernel!(direct_targets, direct_sources, @Const(u_targets),
        @Const(u_sources), @Const(leaf_slot_of), n_u, violation_flags)
    k = @index(Global)
    @inbounds if k <= n_u
        ts = leaf_slot_of[Int(u_targets[k])]
        ss = leaf_slot_of[Int(u_sources[k])]
        (ts == Int32(0) || ss == Int32(0)) && (violation_flags[2] = Int32(1))
        direct_targets[k] = Int(ts)
        direct_sources[k] = Int(ss)
    end
end

@kernel function ka_body_leaf_kernel!(body_leaf, @Const(cell_ranges), n_leaves)
    l = @index(Global)
    @inbounds if l <= n_leaves
        first = cell_ranges[1, l]
        last = first + cell_ranges[2, l] - 1
        i = first
        while i <= last
            body_leaf[i] = Int32(l)
            i += 1
        end
    end
end

@kernel function ka_usort_keys_kernel!(keys, @Const(direct_targets), n_u)
    i = @index(Global)
    @inbounds if i <= n_u
        keys[i] = (UInt64(direct_targets[i]) << 32) | UInt64(i)
    end
end

@kernel function ka_ucsr_gather_kernel!(u_csr_sources, @Const(usort_ix),
        @Const(direct_sources), n_u)
    p = @index(Global)
    @inbounds if p <= n_u
        u_csr_sources[p] = Int32(direct_sources[Int(usort_ix[p])])
    end
end

@kernel function ka_ucsr_offsets_kernel!(u_csr_offsets, @Const(usort_ix),
        @Const(direct_targets), n_u)
    p = @index(Global)
    @inbounds if p <= n_u
        t = Int(direct_targets[Int(usort_ix[p])])
        tprev = p == 1 ? 0 : Int(direct_targets[Int(usort_ix[p - 1])])
        s = tprev + 1
        while s <= t
            u_csr_offsets[s] = Int32(p)
            s += 1
        end
    end
end

"""
    ka_adaptive_u_slots!(lctx, leaf_slot_of, n_u; workgroup=64)

Map the `n_u` U pairs' node ids to leaf-cell slots, into
`lctx.bufs.direct_targets`/`direct_sources`. Throws if either endpoint of any
pair is not a leaf (which would mean the DTR produced a non-leaf U pair).
"""
function ka_adaptive_u_slots!(lctx::KAAdaptiveListsContext, leaf_slot_of, n_u::Int;
        workgroup::Int=64)
    n_u == 0 && return nothing
    backend = lctx.backend
    b = lctx.bufs
    k = _cached_kernel(ka_u_slots_kernel!, backend, workgroup)
    k(b.direct_targets, b.direct_sources, b.u_targets, b.u_sources, leaf_slot_of,
        n_u, b.violation_flags; ndrange=n_u)
    # the readback below is itself the sync point
    Int(Array(view(b.violation_flags, 2:2))[1]) == 0 || throw(AssertionError(
        "adaptive KA U pair endpoint is not a leaf cell slot"))
    return nothing
end

"""
    ka_adaptive_build_u_csr!(lctx, cell_ranges, n_leaves, n_u; workgroup=64)

Backend-agnostic port of `_cuda_adaptive_build_u_csr!`: builds the target-major
CSR (`u_csr_offsets` over `1:n_leaves+1`, `u_csr_sources` over `1:n_u`) from the
slot-mapped U list produced by `ka_adaptive_u_slots!`, plus the body -> leaf-slot
map for the dense body-packed nearfield shape. Reuses the Phase F sort scratch,
as CUDA does.
"""
function ka_adaptive_build_u_csr!(lctx::KAAdaptiveListsContext, cell_ranges,
        n_leaves::Int, n_u::Int; workgroup::Int=64)
    backend = lctx.backend
    b = lctx.bufs
    n_leaves + 1 <= length(b.u_csr_offsets) || throw(AssertionError(
        "KA U-CSR offsets capacity $(length(b.u_csr_offsets)) exceeded " *
        "(n_leaves=$n_leaves)"))
    fill!(view(b.u_csr_offsets, 1:(n_leaves + 1)), Int32(n_u + 1))

    if n_u > 0
        n_u <= length(b.vsort_keys) || throw(AssertionError(
            "KA U-CSR reuses the V sort scratch; n_u=$n_u exceeds v_capacity " *
            "$(length(b.vsort_keys))"))
        keysk = _cached_kernel(ka_usort_keys_kernel!, backend, workgroup)
        gatherk = _cached_kernel(ka_ucsr_gather_kernel!, backend, workgroup)
        offsk = _cached_kernel(ka_ucsr_offsets_kernel!, backend, workgroup)

        keysk(b.vsort_keys, b.direct_targets, n_u; ndrange=n_u)
        sortperm!(view(b.vsort_ix, 1:n_u), view(b.vsort_keys, 1:n_u))
        gatherk(b.u_csr_sources, b.vsort_ix, b.direct_sources, n_u; ndrange=n_u)
        offsk(b.u_csr_offsets, b.vsort_ix, b.direct_targets, n_u; ndrange=n_u)
    end

    if n_leaves > 0
        bodyk = _cached_kernel(ka_body_leaf_kernel!, backend, workgroup)
        bodyk(b.u_csr_body_leaf, cell_ranges, n_leaves; ndrange=n_leaves)
    end
    KA.synchronize(backend)     # single sync: results are caller-visible after this
    return nothing
end

#------- Harness front end: position -> full-depth key, and a full-build driver -------#
#
# Not a CUDA-parity phase (no `_cuda_*` counterpart is ported one-to-one here) --
# this stitches Phases A-D together into a single from-scratch tree build, the way
# `_cuda_refresh_adaptive_tree!` (tree_batched_cuda.jl:1387) stitches the hand-written
# CUDA stages, so the 4-way tree-build benchmark can drive local-Metal and HPC-KA off
# one shared, backend-agnostic entry point.

@kernel function ka_radix_keys_kernel!(keys, @Const(positions), x_min, h0, ell, n)
    i = @index(Global)
    @inbounds if i <= n
        G = 1 << ell
        delta = (2 * h0) / G
        px = positions[1, i]
        py = positions[2, i]
        pz = positions[3, i]
        ix = clamp(floor(Int, (px - x_min[1]) / delta), 0, G - 1)
        iy = clamp(floor(Int, (py - x_min[2]) / delta), 0, G - 1)
        iz = clamp(floor(Int, (pz - x_min[3]) / delta), 0, G - 1)
        keys[i] = ka_morton_key(ix, iy, iz, ell)
    end
end

"""
    ka_radix_keys!(keys, positions, x_min, h0, ell; workgroup=64)

Backend-agnostic port of `_cuda_radix_keys_checked_kernel!` (translate_batched_cuda.jl),
minus its out-of-bounds flag (bodies are assumed to already lie in the fixed root cube
`[x_min, x_min + 2*h0]^3` -- true by construction for a from-scratch benchmark build;
callers needing the OOB guard for a live refresh loop should add it at the call site).
Writes each body's full-depth (`ell`-level) Morton key from its position into `keys`.
`positions` is a `3 x n` backend matrix; `x_min` is a plain 3-tuple/SVector, not a
device array (mirrors CUDA's `grid.x_min` convention).
"""
function ka_radix_keys!(keys::AbstractVector{UInt64}, positions::AbstractMatrix,
        x_min, h0, ell::Int; workgroup::Int=64)
    n = length(keys)
    backend = KA.get_backend(keys)
    keysk = _cached_kernel(ka_radix_keys_kernel!, backend, workgroup)
    keysk(keys, positions, x_min, h0, ell, n; ndrange=n)
    KA.synchronize(backend)
    return keys
end

"""
    ka_build_adaptive_tree!(actx, positions, ell_max, K_max, balance, x_min, h0::TF;
                             sigma_row=0, source_bodies=nothing, workgroup=64)

From-scratch adaptive-octree build from raw body positions: full-depth Morton keys
(`ka_radix_keys!`) + sort (`sortperm!`/`ka_gather_values!`, same as Phases B/C) feeding
`ka_adaptive_build_leaves!` (Phase A) -> `ka_adaptive_balance!` (Phase B, if `balance`)
-> `ka_adaptive_finalize!` (Phase C) -> `ka_adaptive_sigma_sweep!` (Phase D, if
`sigma_row > 0` and `source_bodies` given). `positions` is a `3 x n` backend matrix
(`n <= actx.maxn`); `x_min`/`h0` are the fixed root cube (plain scalars, not device
arrays). Mirrors the stage order of `_cuda_refresh_adaptive_tree!` for a first
(non-incremental) build. All scratch/output buffers come from `actx` (see
`KAAdaptiveTreeContext`/`ka_allocate_adaptive_context`) -- zero recurring allocation,
matching CUDA-native's `actx` convention.

Returns Phase C's `NamedTuple` merged with `grid`, `perm`, `invperm`, `sorted_keys`,
`n_balance_splits`, and `node_sigma_max` (`nothing` if the sigma sweep was not armed).
"""
function ka_build_adaptive_tree!(actx::KAAdaptiveTreeContext, positions::AbstractMatrix,
        ell_max::Int, K_max::Int, balance::Bool, x_min, h0::TF; sigma_row::Int=0,
        source_bodies=nothing, workgroup::Int=64,
        stage_ns::Union{Nothing,Vector{UInt64}}=nothing) where TF
    backend = actx.backend
    n = size(positions, 2)
    n <= actx.maxn || error("adaptive KA context maxn=$(actx.maxn) exceeded by n=$n")
    b = actx.bufs

    # Optional per-stage timing (mirrors CUDA-native's actx.profile_stages/
    # stage_ns convention, tree_batched_cuda.jl:1387-1431) to root-cause the
    # n>=1e5 timing anomaly (project_fastmultipole_ka_migration memory).
    t0 = stage_ns !== nothing ? (KernelAbstractions.synchronize(backend); time_ns()) : UInt64(0)

    keys = view(b.keys, 1:n)
    ka_radix_keys!(keys, positions, x_min, h0, ell_max; workgroup=workgroup)

    perm = view(b.perm, 1:n)
    sortperm!(perm, keys)
    sorted_keys = view(b.sorted_keys, 1:n)
    ka_gather_values!(sorted_keys, keys, perm; workgroup=workgroup)
    invperm = view(b.invperm, 1:n)
    ka_fill_invperm!(invperm, perm; workgroup=workgroup)

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[1] = time_ns() - t0; t0 = time_ns()
    end

    nl, llev, lkey, llo, lhi = ka_adaptive_build_leaves!(actx, sorted_keys, ell_max, K_max, n;
        workgroup=workgroup)

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[2] = time_ns() - t0; t0 = time_ns()
    end

    n_balance_splits = 0
    if balance
        nl, n_balance_splits, llev, lkey, llo, lhi = ka_adaptive_balance!(actx, nl, llev, lkey,
            llo, lhi, sorted_keys, ell_max; workgroup=workgroup)
    end

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[3] = time_ns() - t0; t0 = time_ns()
    end

    fin = ka_adaptive_finalize!(actx, nl, llev, lkey, llo, lhi, sorted_keys, ell_max, n, x_min,
        h0; workgroup=workgroup)

    # Publish the build into the context's DeviceRadixGrid. The arrays were
    # written in place through the `bufs` aliases; only the scalars need setting,
    # the same five `_cuda_refresh_adaptive_tree!` assigns.
    grid = actx.grid
    grid.x_min = SVector{3,TF}(x_min[1], x_min[2], x_min[3])
    grid.h0 = h0
    grid.ell = ell_max
    grid.n_bodies = n
    grid.n_cells = fin.n_leaves
    ka_fill_single_system_attribution!(grid.body_system, grid.body_index, n;
        workgroup=workgroup)

    if stage_ns !== nothing
        KernelAbstractions.synchronize(backend)
        stage_ns[4] = time_ns() - t0
    end

    node_sigma_max = if sigma_row > 0 && source_bodies !== nothing
        ka_adaptive_sigma_sweep!(actx, fin.node_lo, fin.node_hi, fin.child_ranges, fin.n_nodes,
            fin.level_offsets, ell_max, source_bodies, sigma_row; workgroup=workgroup)
    else
        nothing
    end

    return merge(fin, (grid=grid, perm=perm, invperm=invperm, sorted_keys=sorted_keys,
        n_balance_splits=n_balance_splits, node_sigma_max=node_sigma_max))
end

#------- Step (v): handing the built tree to a DeviceResidentRadixState -------#
#
# The analogue is `_cuda_allocate_adaptive_lifecycle` (translate_batched_cuda.jl),
# the ADAPTIVE path's state constructor -- not `cuda_radix_state`, which serves the
# uniform path and assumes exact-length grid arrays (`length(grid.node_keys)` ==
# n_nodes). That distinction is what makes this cheap: the adaptive path already
# hands its own capacity-sized `actx.grid` straight into the state and carries the
# logical extents in `state.counts::RadixStepCounts`, so `actx.grid` goes in as-is,
# by reference, with no truncating views and no host-mirror cross-check.
#
# Scope: this constructs the state. It does not build the interaction list or the
# operator workspace (`ResidentOperatorWorkspace`, whose plan construction is
# CUDA-specific), and it does not run the lifecycle -- B2M and L2B have no KA port.
# `interaction_list` and `scratch` are therefore `nothing`, and the route arrays are
# allocated empty. Those are the next steps, not omissions this one papers over.

function _ka_flat_buffer(backend, ::Type{TF}, basis_info::FastMultipole.OperatorBasisInfo{B,LH},
        batch::Integer) where {TF,B,LH}
    phi = KA.zeros(backend, TF, basis_info.basis_dof_phi, batch)
    chi = LH ? KA.zeros(backend, TF, basis_info.basis_dof_chi, batch) :
        KA.zeros(backend, TF, 0, 0)
    return FastMultipole.FlatCoefficientBuffer{TF,typeof(phi),B,LH}(phi, chi, basis_info)
end

"""
    ka_refresh_adaptive_lists!(lctx, actx, build; near_radius2, ell_max,
                               rho_t=0, sigma_armed=false)

Backend-agnostic port of `tree_batched_cuda.jl`'s `_cuda_refresh_adaptive_lists!`:
run the DTR sweep, partition the V stream into CSR routes, map the U endpoints to
leaf slots and rebuild the target-major U CSR, in that order, against the tree
`ka_build_adaptive_tree!` most recently wrote into `actx.grid`.

Returns the NamedTuple `ka_radix_state`'s `lists` keyword expects. `n_direct` is
the U-pair count and `n_routes` the CSR route count -- the two logical extents that
land in `state.counts`; the remaining counts are diagnostics (`n_dem` is the
sticky-demotion trigger count, nonzero only when the sigma gate is armed).

The lists live in `lctx.bufs` and are valid over `1:n_*`; nothing is copied.
"""
function ka_refresh_adaptive_lists!(lctx::KAAdaptiveListsContext,
        actx::KAAdaptiveTreeContext, build; near_radius2::Int, ell_max::Int,
        rho_t::Real=0, sigma_armed::Bool=false, workgroup::Int=64)
    grid = actx.grid
    b = actx.bufs
    n_nodes = build.n_nodes
    n_leaves = build.n_leaves
    gate = sigma_armed && rho_t > 0
    # the finest-lattice cell width, squared; the gate compares an integer-exact
    # squared AABB gap against (rho_t * sigma)^2 in these units
    delta_min = 2 * Float64(grid.h0) / (1 << ell_max)

    n_u, n_v, n_w, n_x, n_dem = ka_adaptive_build_lists!(lctx,
        grid.node_levels, grid.node_coords, grid.child_ranges, b.node_sigma;
        ell_max=ell_max, near_radius2=near_radius2, gate=gate, rho_t=rho_t,
        delta_min2=delta_min * delta_min, workgroup=workgroup)

    n_routes = ka_adaptive_partition_v!(lctx, n_v; workgroup=workgroup)

    n_u <= length(lctx.bufs.direct_targets) || throw(AssertionError(
        "adaptive KA direct capacity $(length(lctx.bufs.direct_targets)) exceeded " *
        "by n_u=$n_u"))
    ka_adaptive_u_slots!(lctx, b.leaf_slot_of, n_u; workgroup=workgroup)
    ka_adaptive_build_u_csr!(lctx, grid.cell_ranges, n_leaves, n_u; workgroup=workgroup)

    return (lctx=lctx, n_direct=n_u, n_routes=n_routes,
        n_u=n_u, n_v=n_v, n_w=n_w, n_x=n_x, n_dem=n_dem, n_nodes=n_nodes)
end

"""
    ka_radix_state(actx, build, source_buffer, P, lamb_helmholtz=Val(false);
                   options, n_root_nodes=1, workgroup=64)

Build a `DeviceResidentRadixState` around the tree `ka_build_adaptive_tree!` just
wrote into `actx.grid`. `build` is that call's return value (its `n_nodes`/`n_leaves`
supply the logical extents, which the grid itself does not carry); `source_buffer` is
the `dpb x n` source buffer in *global* (unsorted) body order, which is gathered into
the state's sorted-order body matrix.

`actx.grid` is stored by reference, not copied: `state.grid === actx.grid`, so a later
rebuild through the same context is visible to the state without reconstructing it.
The grid's arrays stay capacity-sized; every logical extent lives in `state.counts`
(`n_bodies`, `n_cells`, `n_nodes`), exactly as on the CUDA adaptive path.

Pass `lists` -- a `ka_refresh_adaptive_lists!` return -- to wire the interaction
list: `interaction_list` becomes the lists context, the route/direct arrays alias
its device buffers, and `counts.n_routes`/`counts.n_direct` carry their extents.
Omit it and those stay `nothing`/empty with both counts zero.

Still not wired, and `nothing`/empty rather than silently wrong: `scratch` (the
operator workspace builds CUDA-specific M2L plans), the `route_levels`/`route_offsets`
pair (the adaptive path routes by CSR class instead), and the host node/route mirrors. The host *body* mirrors
(`host_body_perm`/`host_body_system_ids`/`host_body_indices`) are downloaded once here,
so they are correct for this build and go stale on the next one -- the CUDA path
re-downloads them per step in `_cuda_update_adaptive_radix_state!`.
"""
function ka_radix_state(actx::KAAdaptiveTreeContext, build, source_buffer,
        P::Integer, lamb_helmholtz::Val{LH}=Val(false);
        options::FastMultipole.CUDARadixLifecycleOptions, lists=nothing,
        scratch=nothing, output_rows::Int=4, n_root_nodes::Int=1,
        workgroup::Int=64) where LH
    backend = actx.backend
    grid = actx.grid
    TF = typeof(grid.h0)
    options.precision === TF || throw(ArgumentError(
        "options.precision=$(options.precision) does not match the KA context's " *
        "element type $TF; allocate the context and the options at the same precision"))
    n = grid.n_bodies
    n_cells = grid.n_cells
    n_nodes = build.n_nodes
    n_nodes <= actx.node_capacity || throw(ArgumentError(
        "build n_nodes=$n_nodes exceeds the context node_capacity=$(actx.node_capacity)"))
    n_cells == build.n_leaves || throw(ArgumentError(
        "grid.n_cells=$n_cells disagrees with build.n_leaves=$(build.n_leaves); " *
        "`build` must be the result of the most recent ka_build_adaptive_tree! on `actx`"))

    basis_info = FastMultipole.OperatorBasisInfo(FastMultipole.CompressedComplexBasis(),
        P, lamb_helmholtz)
    counters = FastMultipole.CUDARadixTransferCounters()

    # sorted-order body matrix. Single-system attribution (step iv), so one isys=1
    # pack call covers every column; the kernel keeps CUDA's system indirection so
    # it stays correct when multi-system attribution lands.
    dpb = size(source_buffer, 1)
    source_bodies = KA.zeros(backend, TF, dpb, actx.maxn)
    ka_pack_body_matrix!(source_bodies, source_buffer, grid.perm, grid.body_system,
        grid.body_index, n; isys=1, workgroup=workgroup)

    m2m_parent = KA.zeros(backend, Int, actx.node_capacity)
    m2m_child = KA.zeros(backend, Int, actx.node_capacity)
    l2l_parent = KA.zeros(backend, Int, actx.node_capacity)
    l2l_child = KA.zeros(backend, Int, actx.node_capacity)
    ka_tree_routes!(m2m_parent, m2m_child, l2l_parent, l2l_child, grid.parent_index,
        n_nodes; n_root_nodes=n_root_nodes, workgroup=workgroup)

    multipoles = _ka_flat_buffer(backend, TF, basis_info, actx.node_capacity)
    locals = _ka_flat_buffer(backend, TF, basis_info, actx.node_capacity)
    output = KA.zeros(backend, TF, output_rows, actx.maxn)

    # Route/direct wiring. With `lists` given (a `ka_refresh_adaptive_lists!`
    # return), the state ALIASES the lists context's device buffers rather than
    # copying them -- the task-023 zero-recurring-allocation contract, and the same
    # shape as the CUDA adaptive path, where the U-slot kernel writes straight into
    # `state.direct_targets`. `route_levels`/`route_offsets` stay empty on both
    # paths: the adaptive lifecycle routes by CSR class, not by (level, offset), so
    # a capacity-sized zero array there would only look meaningful.
    empty_iv = KA.zeros(backend, Int, 0)
    empty_im = KA.zeros(backend, Int, 3, 0)
    if lists === nothing
        route_targets = route_sources = direct_targets = direct_sources = empty_iv
        n_routes = n_direct = 0
        interaction_list = nothing
    else
        lb = lists.lctx.bufs
        route_targets, route_sources = lb.route_targets, lb.route_sources
        direct_targets, direct_sources = lb.direct_targets, lb.direct_sources
        n_routes, n_direct = lists.n_routes, lists.n_direct
        lists.n_nodes == n_nodes || throw(ArgumentError(
            "lists were built for n_nodes=$(lists.n_nodes) but this build has " *
            "n_nodes=$n_nodes; refresh the lists after the tree build"))
        interaction_list = lists.lctx
    end

    # one-shot download of the host body mirrors (see docstring on staleness)
    host_body_perm = Array{Int}(undef, n)
    host_body_system_ids = Array{Int}(undef, n)
    host_body_indices = Array{Int}(undef, n)
    copyto!(host_body_perm, 1, grid.perm, 1, n)
    copyto!(host_body_system_ids, 1, grid.body_system, 1, n)
    copyto!(host_body_indices, 1, grid.body_index, 1, n)
    counters.metadata_downloads += 3

    return FastMultipole.DeviceResidentRadixState{TF,FastMultipole.CompressedComplexBasis,LH}(
        grid, interaction_list, source_bodies, source_bodies,
        grid.perm, grid.body_system, grid.body_index,
        host_body_perm, host_body_system_ids, host_body_indices,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        grid.cell_centers, grid.cell_ranges,
        m2m_parent, m2m_child, l2l_parent, l2l_child,
        multipoles, locals,
        empty_iv, empty_im, route_targets, route_sources,
        direct_targets, direct_sources, output,
        FastMultipole.OperatorInvariantCache(TF, basis_info), scratch, counters, options,
        FastMultipole.RadixStepCounts(n, n_cells, n_nodes, n_routes, n_direct),
    )
end

#------- B2M (body -> multipole), step 3 -------#
#
# Port of `_cuda_b2m_vortex_leaf_nodes_kernel!` / `_cuda_b2m_leaf_nodes_kernel!`
# (src/translate_batched_cuda.jl): one workgroup per leaf cell, body-parallel
# accumulation, tree-reduced across the group.
#
# The per-body math is NOT reimplemented -- `_resident_vortex_phi_contrib` and
# `_resident_vortex_chi_contrib` live in src/translate_batched_resident.jl and
# are backend-agnostic `@inline` Julia shared with the CPU path. Only the
# reduction shape is ported.
#
# Workgroup size is 128 to match `CUDA_B2M_BLOCK`, not the 64 used by the tree
# kernels: the reduction is a halving tree, so its summation ORDER depends on
# the group size, and matching CUDA's block exactly is what makes this
# bit-exact against the CUDA reference rather than merely close. WG must be a
# power of two.
#
# Metal portability trap: the `@localmem` element type must be a compile-time
# constant reaching the `Val`-wrapped `SharedMemory` call, and computing it
# inside the kernel as `TF = eltype(phi)` does NOT qualify -- on Metal that
# compiles but raises a device-side "undefined variable error" at launch (KA's
# `@localmem` expansion, KernelAbstractions.jl:242). Pass the element type as a
# `::Type{TF}` kernel argument instead. `Val{WG}` dims are fine either way.
#
# Float32 discipline: every literal stays in TF. Apple GPUs reject Float64
# outright, and `_cuda_b2m_*` carries no Float64 of its own, so there is
# nothing to widen here -- but a stray `0.5` would silently promote and break
# Metal, so the accumulators are seeded with `zero(TF)`.

# The halving tree reduction is written out lexically in both places rather
# than factored into a helper: `@synchronize` must appear directly in the
# kernel body and in uniform control flow, so it can neither live inside a
# called function nor sit under a per-group `if`. (`ndrange = ncell * WG` gives
# exactly `ncell` groups, so no group guard is needed either.)
@kernel function ka_b2m_vortex_leaf_nodes_kernel!(phi, chi, @Const(source_bodies),
        @Const(cell_centers), @Const(cell_ranges), @Const(leaf_to_node),
        P_phi, P_chi, ncell, ::Type{TF}, ::Val{WG}) where {TF,WG}
    i_cell = @index(Group)
    tid = @index(Local)
    shre = @localmem TF (WG,)
    shim = @localmem TF (WG,)
    @inbounds begin
        first = cell_ranges[1, i_cell]
        count = cell_ranges[2, i_cell]
        cx = cell_centers[1, i_cell]
        cy = cell_centers[2, i_cell]
        cz = cell_centers[3, i_cell]
        node = leaf_to_node[i_cell]
        for n in 0:P_phi
            for m in 0:n
                acc_re = zero(TF); acc_im = zero(TF)
                k = first + tid - 1
                while k <= first + count - 1
                    re_, im_ = FastMultipole._resident_vortex_phi_contrib(
                        cx - source_bodies[1, k], cy - source_bodies[2, k],
                        cz - source_bodies[3, k], source_bodies[5, k],
                        source_bodies[6, k], source_bodies[7, k], n, m)
                    acc_re += re_; acc_im += im_
                    k += WG
                end
                shre[tid] = acc_re
                shim[tid] = acc_im
                @synchronize()
                s = WG >> 1
                while s >= 1
                    if tid <= s
                        shre[tid] += shre[tid + s]
                        shim[tid] += shim[tid + s]
                    end
                    @synchronize()
                    s >>= 1
                end
                if tid == 1
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    phi[row, node] = shre[1]
                    phi[row + 1, node] = shim[1]
                end
                @synchronize()
            end
        end
        for n in 1:P_chi
            for m in 0:n
                acc_re = zero(TF); acc_im = zero(TF)
                k = first + tid - 1
                while k <= first + count - 1
                    re_, im_ = FastMultipole._resident_vortex_chi_contrib(
                        cx - source_bodies[1, k], cy - source_bodies[2, k],
                        cz - source_bodies[3, k], source_bodies[5, k],
                        source_bodies[6, k], source_bodies[7, k], n, m)
                    acc_re += re_; acc_im += im_
                    k += WG
                end
                shre[tid] = acc_re
                shim[tid] = acc_im
                @synchronize()
                s = WG >> 1
                while s >= 1
                    if tid <= s
                        shre[tid] += shre[tid + s]
                        shim[tid] += shim[tid + s]
                    end
                    @synchronize()
                    s >>= 1
                end
                if tid == 1
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    chi[row, node] = shre[1]
                    chi[row + 1, node] = shim[1]
                end
                @synchronize()
            end
        end
    end
end

"""
    ka_launch_b2m!(state; workgroup=128)

Body-to-multipole for the KA lifecycle: mirror of `_launch_cuda_b2m!`. Zeroes
the multipole buffers, then runs one workgroup per leaf cell. Vortex sources
require the Lamb-Helmholtz channel, exactly as the CUDA launcher does.
"""
function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup::Int=128) where {TF,B,LH}
    return ka_launch_b2m!(state, state.options.body_type; workgroup)
end

function ka_launch_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ::Type{<:FastMultipole.Point{FastMultipole.Vortex}}; workgroup::Int=128) where {TF,B,LH}
    LH || throw(ArgumentError(
        "Point{Vortex} sources require the Lamb-Helmholtz channel; construct the " *
        "cache with lamb_helmholtz=true"))
    ispow2(workgroup) || throw(ArgumentError(
        "ka_launch_b2m! workgroup must be a power of two (halving tree reduction)"))
    fill!(state.multipoles.phi, zero(TF))
    fill!(state.multipoles.chi, zero(TF))
    ncell = state.counts.n_cells
    ncell == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    backend = KA.get_backend(state.multipoles.phi)
    kernel = _cached_kernel(ka_b2m_vortex_leaf_nodes_kernel!, backend, workgroup)
    kernel(state.multipoles.phi, state.multipoles.chi, state.source_bodies,
        state.cell_centers, state.cell_ranges, state.grid.leaf_to_node,
        orders.P_phi, orders.P_active, ncell, TF, Val(workgroup);
        ndrange=ncell * workgroup)
    return state
end

#------- L2B (local -> body output), step 4 -------#
#
# Port of `_cuda_l2b_output_kernel!` / `_cuda_l2b_output_hessian_kernel!`
# (src/translate_batched_cuda.jl). The per-body evaluation is NOT
# reimplemented: `_resident_local_eval_flat` and
# `_resident_local_eval_flat_hessian` (src/translate_batched_resident.jl) are
# backend-agnostic `@inline` Julia shared with the CPU path.
#
# Shape differs deliberately from CUDA's. CUDA runs a warp per cell (4 warps
# per 128-thread block) and strides bodies by 32; KA has no portable warp
# concept, so this runs one WORKGROUP per cell and strides by the workgroup
# size. That costs nothing and is safe: every body belongs to exactly one cell
# (`cell_ranges` partitions the sorted bodies) and each body writes only its
# own output column, so there is no reduction, no shared memory, no
# `@synchronize` and no atomic here. The value written for a given body is
# computed independently of the thread mapping, so this stays bit-exact
# against the CUDA kernel rather than merely close.

@kernel function ka_l2b_output_kernel!(output, @Const(source_bodies), @Const(cell_centers),
        @Const(cell_ranges), @Const(leaf_to_node), @Const(local_phi), @Const(local_chi),
        P_phi, P_active, ::Val{LHV}, ncell, ::Val{WG}) where {LHV,WG}
    cell = @index(Group)
    tid = @index(Local)
    @inbounds begin
        node = leaf_to_node[cell]
        first = cell_ranges[1, cell]
        last = first + cell_ranges[2, cell] - 1
        cx = cell_centers[1, cell]; cy = cell_centers[2, cell]; cz = cell_centers[3, cell]
        i = first + tid - 1
        while i <= last
            sp, gx, gy, gz = FastMultipole._resident_local_eval_flat(
                local_phi, local_chi, node,
                source_bodies[1, i] - cx, source_bodies[2, i] - cy,
                source_bodies[3, i] - cz, P_phi, P_active, Val(LHV))
            output[1, i] += sp
            output[2, i] += gx
            output[3, i] += gy
            output[4, i] += gz
            i += WG
        end
    end
end

@kernel function ka_l2b_output_hessian_kernel!(output, @Const(source_bodies),
        @Const(cell_centers), @Const(cell_ranges), @Const(leaf_to_node),
        @Const(local_phi), @Const(local_chi), P_phi, P_active, ::Val{LHV},
        ncell, ::Val{WG}) where {LHV,WG}
    cell = @index(Group)
    tid = @index(Local)
    @inbounds begin
        node = leaf_to_node[cell]
        first = cell_ranges[1, cell]
        last = first + cell_ranges[2, cell] - 1
        cx = cell_centers[1, cell]; cy = cell_centers[2, cell]; cz = cell_centers[3, cell]
        i = first + tid - 1
        while i <= last
            vals = FastMultipole._resident_local_eval_flat_hessian(
                local_phi, local_chi, node,
                source_bodies[1, i] - cx, source_bodies[2, i] - cy,
                source_bodies[3, i] - cz, P_phi, P_active, Val(LHV))
            Base.Cartesian.@nexprs 13 r -> (output[r, i] += vals[r])
            i += WG
        end
    end
end

"""
    ka_launch_l2b!(state; workgroup=64)

Local-to-body evaluation for the KA lifecycle: mirror of
`_launch_cuda_resident_l2b_only!`, minus the stream-event overlap (the KA
driver runs its stages in order). Selects the 13-row hessian variant on
`size(state.output, 1) >= 13`, the same gate the CUDA launcher uses; FLOWVPM
runs with `hessian=true` and therefore takes that branch.
"""
function ka_launch_l2b!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup::Int=64) where {TF,B,LH}
    ncell = state.counts.n_cells
    ncell == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    backend = KA.get_backend(state.output)
    args = (state.output, state.source_bodies, state.cell_centers, state.cell_ranges,
            state.grid.leaf_to_node, state.locals.phi, state.locals.chi,
            orders.P_phi, orders.P_active, Val(LH), ncell, Val(workgroup))
    if size(state.output, 1) >= 13
        kernel = _cached_kernel(ka_l2b_output_hessian_kernel!, backend, workgroup)
    else
        kernel = _cached_kernel(ka_l2b_output_kernel!, backend, workgroup)
    end
    kernel(args...; ndrange=ncell * workgroup)
    return state
end

#------- NEARFIELD (U-list direct pairs), step 5 -------#
#
# Port of the `:pairs` shape of `_cuda_direct_pairs_functor_kernel!`
# (src/translate_batched_cuda.jl). The per-pair math is NOT reimplemented:
# `_direct_pair_ug` / `_direct_pair_ugh` (src/translate_batched_resident.jl)
# are backend-agnostic and shared with the CPU path, so every kernel functor
# (PartitionedVortex, RegularizedVortex, SingularSource, ...) comes along for
# free.
#
# Deliberately ports ONLY the `:pairs` shape. The fused target-owned CSR
# shapes, the symmetric Newton-pair path, the binned split-vortex path and the
# g/h lookup table are all skipped (`ghv = Val(:shipped)`, `shlut = nothing`) --
# they are performance variants of the same physics. That leaves the KA U-CSR
# built in tree Phase G unused for now; it is correct and gated, and a fused
# shape can consume it later as a perf step.
#
# `_cuda_fast_rsqrt` is replaced by a plain `inv(sqrt(r2))`. That makes the KA
# kernel the MORE accurate side, so this stage is gated against the CPU
# reference, never against the CUDA kernel.
#
# Workgroup per pair, workitems striding the target bodies (CUDA uses a warp
# per pair striding by 32). Targets of different pairs overlap, so the output
# accumulation must stay atomic.

@kernel function ka_direct_pairs_functor_kernel!(kernel, output, @Const(source_bodies),
        @Const(cell_ranges), @Const(direct_targets), @Const(direct_sources),
        npairs, ::Type{T}, ::Val{HS}, ::Val{WG}) where {T,HS,WG}
    pair_i = @index(Group)
    tid = @index(Local)
    ep = FastMultipole._emits_potential(kernel)
    ghv = Val(:shipped)
    @inbounds begin
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        tfirst = cell_ranges[1, target_cell]
        tlast = tfirst + cell_ranges[2, target_cell] - 1
        sfirst = cell_ranges[1, source_cell]
        slast = sfirst + cell_ranges[2, source_cell] - 1
        i = tfirst + tid - 1
        while i <= tlast
            xi = source_bodies[1, i]; yi = source_bodies[2, i]; zi = source_bodies[3, i]
            u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
            h1 = zero(T); h2 = zero(T); h3 = zero(T)
            h4 = zero(T); h5 = zero(T); h6 = zero(T)
            h7 = zero(T); h8 = zero(T); h9 = zero(T)
            for j in sfirst:slast
                if i != j
                    dx = xi - source_bodies[1, j]
                    dy = yi - source_bodies[2, j]
                    dz = zi - source_bodies[3, j]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 > zero(r2)
                        invr = inv(sqrt(r2))
                        if HS
                            du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                                FastMultipole._direct_pair_ugh(kernel, dx, dy, dz, r2, invr,
                                    source_bodies, j, ghv)
                            u += du; gx += dgx; gy += dgy; gz += dgz
                            h1 += dh1; h2 += dh2; h3 += dh3
                            h4 += dh4; h5 += dh5; h6 += dh6
                            h7 += dh7; h8 += dh8; h9 += dh9
                        else
                            du, dgx, dgy, dgz = FastMultipole._direct_pair_ug(kernel,
                                dx, dy, dz, r2, invr, source_bodies, j, ghv)
                            u += du; gx += dgx; gy += dgy; gz += dgz
                        end
                    end
                end
            end
            if ep
                KA.@atomic output[1, i] += u
            end
            KA.@atomic output[2, i] += gx
            KA.@atomic output[3, i] += gy
            KA.@atomic output[4, i] += gz
            if HS
                KA.@atomic output[5, i]  += h1
                KA.@atomic output[6, i]  += h2
                KA.@atomic output[7, i]  += h3
                KA.@atomic output[8, i]  += h4
                KA.@atomic output[9, i]  += h5
                KA.@atomic output[10, i] += h6
                KA.@atomic output[11, i] += h7
                KA.@atomic output[12, i] += h8
                KA.@atomic output[13, i] += h9
            end
            i += WG
        end
    end
end

"""
    ka_launch_nearfield!(state; workgroup=64, clear=true)

U-list direct nearfield for the KA lifecycle: mirror of
`_launch_cuda_nearfield_kernel!` restricted to the `:pairs` shape. Zeroes
`state.output` (this is the first stage of the lifecycle, as on CUDA) unless
`clear=false`.
"""
function ka_launch_nearfield!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup::Int=64, clear::Bool=true) where {TF,B,LH}
    clear && fill!(state.output, zero(TF))
    npairs = state.counts.n_direct
    npairs == 0 && return state
    hs = size(state.output, 1) >= 13
    backend = KA.get_backend(state.output)
    kern = _cached_kernel(ka_direct_pairs_functor_kernel!, backend, workgroup)
    kern(state.options.direct_kernel, state.output, state.source_bodies,
         state.cell_ranges, state.direct_targets, state.direct_sources,
         npairs, TF, Val(hs), Val(workgroup); ndrange=npairs * workgroup)
    return state
end

#------- S2L (X list: source bodies -> local), step 6 -------#
#
# Port of `_cuda_adaptive_s2l_vortex_kernel!` (src/translate_batched_cuda.jl).
# The adaptive tree emits W and X pairs, so omitting these stages does not just
# cost accuracy -- it silently drops physics.
#
# The per-thread harmonics scratch is a slice of a GLOBAL array indexed by
# global thread id (`H = view(Hall, :, slot:slot, :)`), not shared memory, so
# it ports across directly: no `@localmem`, no `@synchronize`. `Hall` must be
# sized `(2, n_groups * workgroup, nH2)`.
#
# All the harmonic helpers (`cartesian_to_spherical`, `irregular_harmonics!`,
# `harmonic_index`, `flat_basis_index`, `_adt_S_re/_im`) live in non-CUDA src
# files and are reused unchanged.
#
# The element type is taken as a `::Type{TF}` argument rather than
# `eltype(lp)`: not strictly required here (there is no `@localmem`), but it
# keeps every kernel in this file on one convention.

@kernel function ka_adaptive_s2l_vortex_kernel!(lp, lc, @Const(source_bodies),
        @Const(cell_ranges), @Const(leaf_slot_of), @Const(node_centers),
        @Const(x_targets), @Const(x_sources), n_x, Hall, ::Type{TF},
        ::Val{P_phi}, ::Val{P_active}, ::Val{NG}, ::Val{WG}) where {TF,P_phi,P_active,NG,WG}
    grp = @index(Group)
    tid = @index(Local)
    gslot = (grp - 1) * WG + tid
    H = view(Hall, :, gslot:gslot, :)
    k = grp
    @inbounds while k <= n_x
        ia = Int(x_targets[k])
        ib = Int(x_sources[k])
        slot = Int(leaf_slot_of[ib])
        first = cell_ranges[1, slot]
        count = cell_ranges[2, slot]
        cx = node_centers[1, ia]; cy = node_centers[2, ia]; cz = node_centers[3, ia]
        sB = first + tid - 1
        while sB <= first + count - 1
            dx = source_bodies[1, sB] - cx
            dy = source_bodies[2, sB] - cy
            dz = source_bodies[3, sB] - cz
            wx = source_bodies[5, sB]; wy = source_bodies[6, sB]; wz = source_bodies[7, sB]
            r, theta, phi = FastMultipole.cartesian_to_spherical(dx, dy, dz)
            FastMultipole.irregular_harmonics!(H, r, theta, phi, P_phi + 2)
            for n in 1:P_phi
                _1_n = isodd(n) ? -one(TF) : one(TF)
                n_inv = inv(TF(n))
                for m in 0:n
                    _1_m = isodd(m) ? -one(TF) : one(TF)
                    i = FastMultipole.harmonic_index(n, m)
                    local Spre::TF, Spim::TF, Smre::TF, Smim::TF
                    if m < n
                        Spre = -_1_m * FastMultipole._adt_S_re(H, i + 1)
                        Spim = _1_m * FastMultipole._adt_S_im(H, i + 1)
                    else
                        Spre = zero(TF); Spim = zero(TF)
                    end
                    Sre = _1_m * FastMultipole._adt_S_re(H, i)
                    Sim = -_1_m * FastMultipole._adt_S_im(H, i)
                    if m == 0
                        Smre = -_1_m * Spre; Smim = _1_m * Spim
                    else
                        Smre = -_1_m * FastMultipole._adt_S_re(H, i - 1)
                        Smim = _1_m * FastMultipole._adt_S_im(H, i - 1)
                    end
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    KA.@atomic lp[row, ia] += -_1_n * n_inv * (
                        (n - m) * TF(0.5) * (wx * Spre - wy * Spim) -
                        (n + m) * TF(0.5) * (wx * Smre + wy * Smim) +
                        wz * m * Sim)
                    KA.@atomic lp[row + 1, ia] += -_1_n * n_inv * (
                        (n - m) * TF(0.5) * (wx * Spim + wy * Spre) -
                        (n + m) * TF(0.5) * (wx * Smim - wy * Smre) -
                        wz * m * Sre)
                end
            end
            for n in 0:P_active
                _1_np1 = isodd(n + 1) ? -one(TF) : one(TF)
                np1_inv = inv(TF(n + 1))
                for m in 0:n
                    _1_m = isodd(m) ? -one(TF) : one(TF)
                    i_np1 = FastMultipole.harmonic_index(n + 1, m)
                    Sp1pre = -_1_m * FastMultipole._adt_S_re(H, i_np1 + 1)
                    Sp1pim = _1_m * FastMultipole._adt_S_im(H, i_np1 + 1)
                    Sp1re = _1_m * FastMultipole._adt_S_re(H, i_np1)
                    Sp1im = -_1_m * FastMultipole._adt_S_im(H, i_np1)
                    local Sp1mre::TF, Sp1mim::TF
                    if m == 0
                        Sp1mre = -_1_m * Sp1pre; Sp1mim = _1_m * Sp1pim
                    else
                        Sp1mre = -_1_m * FastMultipole._adt_S_re(H, i_np1 - 1)
                        Sp1mim = _1_m * FastMultipole._adt_S_im(H, i_np1 - 1)
                    end
                    row = FastMultipole.flat_basis_index(n, m, 1)
                    KA.@atomic lc[row, ia] += _1_np1 * np1_inv * (
                        TF(0.5) * (wy * Sp1mre - wx * Sp1mim) -
                        TF(0.5) * (wy * Sp1pre + wx * Sp1pim) - wz * Sp1re)
                    KA.@atomic lc[row + 1, ia] += _1_np1 * np1_inv * (
                        TF(0.5) * (wy * Sp1mim + wx * Sp1mre) -
                        TF(0.5) * (wy * Sp1pim - wx * Sp1pre) - wz * Sp1im)
                end
            end
            sB += WG
        end
        k += NG
    end
end

#------- M2T (W list: multipole -> target bodies), step 6b -------#
#
# Port of `_cuda_adaptive_m2t_kernel!`. Like S2L this uses the GLOBAL
# per-thread harmonics scratch (no shared memory), and the evaluation itself
# is the shared `_resident_multipole_eval_flat[_hessian]` from
# src/translate_batched_resident.jl -- so only the traversal is ported.
#
# Note the asymmetry with S2L: here `leaf_slot_of` is indexed by the TARGET
# (`ia`) and the expansion centre comes from the SOURCE node (`ib`).

@kernel function ka_adaptive_m2t_kernel!(output, @Const(source_bodies), @Const(cell_ranges),
        @Const(leaf_slot_of), @Const(node_centers), @Const(w_targets), @Const(w_sources),
        n_w, @Const(ph), @Const(ch), Hall, ::Type{TF}, ::Val{P_phi}, ::Val{P_active},
        ::Val{LH}, ::Val{HS}, ::Val{NG}, ::Val{WG}) where {TF,P_phi,P_active,LH,HS,NG,WG}
    grp = @index(Group)
    tid = @index(Local)
    gslot = (grp - 1) * WG + tid
    H = view(Hall, :, gslot:gslot, :)
    k = grp
    @inbounds while k <= n_w
        ia = Int(w_targets[k])
        ib = Int(w_sources[k])
        slot = Int(leaf_slot_of[ia])
        first = cell_ranges[1, slot]
        count = cell_ranges[2, slot]
        cx = node_centers[1, ib]; cy = node_centers[2, ib]; cz = node_centers[3, ib]
        i = first + tid - 1
        while i <= first + count - 1
            dx = source_bodies[1, i] - cx
            dy = source_bodies[2, i] - cy
            dz = source_bodies[3, i] - cz
            r, theta, phi = FastMultipole.cartesian_to_spherical(dx, dy, dz)
            FastMultipole.irregular_harmonics!(H, r, theta, phi, P_phi + 2)
            if HS
                vals = FastMultipole._resident_multipole_eval_flat_hessian(ph, ch, ib, H,
                    P_phi, P_active, Val(LH))
                Base.Cartesian.@nexprs 13 r_ -> (KA.@atomic output[r_, i] += vals[r_])
            else
                u, gx, gy, gz = FastMultipole._resident_multipole_eval_flat(ph, ch, ib, H,
                    P_phi, P_active, Val(LH))
                KA.@atomic output[1, i] += u
                KA.@atomic output[2, i] += gx
                KA.@atomic output[3, i] += gy
                KA.@atomic output[4, i] += gz
            end
            i += WG
        end
        k += NG
    end
end

"""
    ka_launch_adaptive_m2t!(state, lists, actx, harmonics_scratch; workgroup=128)

W-list multipole-to-target for the KA lifecycle: mirror of
`_launch_cuda_adaptive_m2t!`. Selects the 13-row hessian variant on
`size(state.output, 1) >= 13`, as CUDA does.
"""
function ka_launch_adaptive_m2t!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        lists, actx, harmonics_scratch; workgroup::Int=128) where {TF,B,LH}
    n_w = lists.n_w
    n_w == 0 && return state
    orders = state.invariant_cache.basis_info.orders
    hs = size(state.output, 1) >= 13
    ng = min(n_w, KA_S2L_GROUPS)
    backend = KA.get_backend(state.output)
    b = lists.lctx.bufs
    kern = _cached_kernel(ka_adaptive_m2t_kernel!, backend, workgroup)
    kern(state.output, state.source_bodies, state.cell_ranges, actx.bufs.leaf_slot_of,
         state.grid.node_centers, b.w_targets, b.w_sources, n_w,
         FastMultipole.phi_slab(state.multipoles), FastMultipole.chi_slab(state.multipoles),
         harmonics_scratch, TF, Val(orders.P_phi), Val(orders.P_active), Val(LH),
         Val(hs), Val(ng), Val(workgroup); ndrange=ng * workgroup)
    return state
end

const KA_S2L_GROUPS = 512   # mirrors _ADT_CUDA_HARMONIC_BLOCKS

"""
    ka_launch_adaptive_s2l!(state, lists, actx, harmonics_scratch; workgroup=128)

X-list source-to-local for the KA lifecycle: mirror of
`_launch_cuda_adaptive_s2l!`, vortex channel. The X pair arrays live on the
lists context (`lists.lctx.bufs`) and `leaf_slot_of` on the tree context
(`actx.bufs`), so both are needed. `harmonics_scratch` must be
`(2, n_groups * workgroup, nH2)` -- see `ka_allocate_harmonics_scratch`.
"""
function ka_launch_adaptive_s2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        lists, actx, harmonics_scratch; workgroup::Int=128) where {TF,B,LH}
    n_x = lists.n_x
    n_x == 0 && return state
    LH || throw(ArgumentError("adaptive S2L (vortex) requires the Lamb-Helmholtz channel"))
    orders = state.invariant_cache.basis_info.orders
    ng = min(n_x, KA_S2L_GROUPS)
    backend = KA.get_backend(state.locals.phi)
    b = lists.lctx.bufs
    kern = _cached_kernel(ka_adaptive_s2l_vortex_kernel!, backend, workgroup)
    kern(FastMultipole.phi_slab(state.locals), FastMultipole.chi_slab(state.locals),
         state.source_bodies, state.cell_ranges, actx.bufs.leaf_slot_of,
         state.grid.node_centers, b.x_targets, b.x_sources, n_x,
         harmonics_scratch, TF, Val(orders.P_phi), Val(orders.P_active),
         Val(ng), Val(workgroup); ndrange=ng * workgroup)
    return state
end

"""
    ka_allocate_harmonics_scratch(backend, TF, P_phi; groups=KA_S2L_GROUPS, workgroup=128)

Per-thread irregular-harmonics scratch for the S2L/M2T kernels, shaped
`(2, groups * workgroup, nH2)`. Mirrors CUDA's
`CUDA.zeros(TF, 2, _ADT_CUDA_HARMONIC_BLOCKS * 128, nH2)`.
"""
function ka_allocate_harmonics_scratch(backend, ::Type{TF}, P_phi::Integer;
        groups::Int=KA_S2L_GROUPS, workgroup::Int=128) where TF
    nH2 = FastMultipole.harmonic_index(P_phi + 2, P_phi + 2)
    return KA.zeros(backend, TF, 2, groups * workgroup, nH2)
end

#------- KA LIFECYCLE DRIVER (step 7a) -------#
#
# Mirror of `_cuda_lifecycle_body!` (src/translate_batched_cuda.jl) for the
# UNIFORM radix lifecycle -- the one FLOWVPM actually runs.
#
# Why uniform and not adaptive: `_radix_cache_device_step!` branches on
# `cache.adaptive === nothing`, and FLOWVPM builds its `RadixFMMCache` with
# `ell`/`near_radius2`/`window_classes` and NO adaptive policy, so it takes
# `run_cuda_radix_lifecycle!`. It could not take the adaptive path anyway: that
# lifecycle rejects `PartitionedVortex` (FLOWVPM's shipped default kernel, a
# task-040 deferral) and the Lamb-Helmholtz M2T hessian throws outright.
#
# The uniform per-step body is only three stages and rebuilds no tree (the
# lattice is fixed at cache construction):
#     nearfield -> B2M -> [M2M -> M2L -> L2L -> L2B]
#
# This driver deliberately calls the ext's STANDALONE `ka_*` stage drivers
# rather than FastMultipole's generic ones. The generic drivers dispatch their
# primitives on array type, which means they run KA kernels on Metal but NATIVE
# CUDA kernels on CuArrays -- correct for production, useless for a KA-vs-native
# A/B on one GPU. Going through the standalone drivers makes "KA" unambiguous on
# every backend, so the same state can be run both ways and compared.
#
# One `KA.synchronize` at the END of the driver, never per stage: per-kernel
# syncs cost 1.37-2.7x in earlier measurements on this code.

"""
    ka_lifecycle_body!(state; workgroup_b2m=128, workgroup=64, sync=true)

Run the uniform radix lifecycle over `state` entirely with KA kernels. `state`
may be resident on any KA backend, including a `CuArray` state built by the
existing `RadixFMMCache(device=true)` -- which is how the KA-vs-native
comparison runs both arms over identical data with no second cache build.
"""
function ka_lifecycle_body!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH};
        workgroup_b2m::Int=128, workgroup::Int=64, sync::Bool=true) where {TF,B,LH}
    ws = state.scratch
    ws isa FastMultipole.ResidentOperatorWorkspace || throw(ArgumentError(
        "ka_lifecycle_body! requires a ResidentOperatorWorkspace in state.scratch"))

    # 1. nearfield (clears state.output, as CUDA's fill+nearfield does)
    ka_launch_nearfield!(state; workgroup=workgroup, clear=true)

    # 2. B2M
    ka_launch_b2m!(state; workgroup=workgroup_b2m)

    # 3. far field: M2M -> M2L -> L2L, then L2B
    FastMultipole._zero_resident_nonleaf_multipoles!(state)
    for group in ws.m2m_groups
        ka_resident_stage_group_apply!(state.multipoles, state.multipoles, group, ws, :m2m)
    end
    ka_launch_m2l!(state, ws)
    for group in ws.l2l_groups
        ka_resident_stage_group_apply!(state.locals, state.locals, group, ws, :l2l)
    end
    ka_launch_l2b!(state; workgroup=workgroup)

    sync && KA.synchronize(KA.get_backend(state.output))
    return state
end

# Selected by `set_radix_setting!(:RADIX_KA_LIFECYCLE, true)`; see the stub in
# src/radix_settings.jl. Keyword defaults are the driver's own -- the setting is
# a boolean arm switch, not a tuning surface.
FastMultipole.ka_radix_lifecycle!(state::FastMultipole.DeviceResidentRadixState) =
    ka_lifecycle_body!(state)

"""
    ka_launch_m2l!(state, ws)

M2L stage of [`ka_lifecycle_body!`](@ref), branching on the resident
interaction context exactly as `_launch_cuda_resident_m2l!` does: a
`DeviceHierarchicalM2LContext` generates and applies route windows here, and
anything else is the flat whole-route concat apply.

The branch is not optional. FLOWVPM's `RadixFMMCache` carries the hierarchical
policy, and on that context `state.counts.n_routes` holds only the LAST
window's route count -- a flat apply would silently translate a fraction of the
V list and be wrong rather than merely slow.
"""
function ka_launch_m2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        ws) where {TF,B,LH}
    hctx = state.interaction_list
    hctx isa FastMultipole.DeviceHierarchicalM2LContext &&
        return ka_hierarchical_m2l!(state, hctx, ws)
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    nroutes = state.counts.n_routes
    nroutes > 0 && ka_resident_m2l_concat_apply!(state.locals, state.multipoles, ws,
        state.route_sources, state.route_targets, nroutes)
    return state
end

"""
    ka_hierarchical_m2l!(state, hctx, ws)

KA arm of `_launch_cuda_hierarchical_m2l!`: the same `(level, offset-class
window)` loop nest, with the per-window concat apply run by KA kernels instead
of `_launch_resident_m2l_concat!`.

**Window GENERATION stays native.** `_cuda_hier_generate_window!` is the
flag/scan/compact that fills `state.route_targets`/`route_sources` for one
window; it is a CUDA-only function and is deliberately shared by both arms.
That makes this an A/B of the M2L *apply* -- the rotation/translation math,
which is where the time is -- and not of the route bookkeeping. Any timing
comparison built on this driver must be reported that way.

Concat plans only. The dense plan is CUDA-specific (`ResidentM2LDenseCUDAPlan`,
and only it is window-cacheable), and precomputed-y needs a per-window refresh
that has no KA port; both are outside the pin recorded in
[`ka_radix_cache_workspace`](@ref).

Because the generator is CUDA-only, this function is reachable only on a
CUDA-resident state and cannot be gated on Metal.
"""
function ka_hierarchical_m2l!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        hctx::FastMultipole.DeviceHierarchicalM2LContext, ws) where {TF,B,LH}
    plan = hctx.apply_plan
    plan isa FastMultipole.ResidentM2LConcatPlan || throw(ArgumentError(
        "ka_hierarchical_m2l! requires a ResidentM2LConcatPlan (build the cache " *
        "with m2l_strategy = ConcatenatedFixedZM2L); got $(typeof(plan))"))
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    route_class = plan.route_class
    noffsets = hctx.noffsets
    K = hctx.window_classes
    total = 0
    fill!(hctx.routes_per_level, 0)
    for L in hctx.first_m2l_level:hctx.ell
        level_total = 0
        class_base = (L - hctx.first_m2l_level) * noffsets
        for first_offset in 1:K:noffsets
            last_offset = min(first_offset + K - 1, noffsets)
            n = FastMultipole._cuda_hier_generate_window!(state, hctx, route_class, L,
                first_offset, last_offset, class_base)
            hctx.last_window_routes = n
            state.counts.n_routes = n
            n > 0 && ka_resident_m2l_concat_apply!(state.locals, state.multipoles, ws,
                state.route_sources, state.route_targets, n)
            level_total += n
        end
        hctx.routes_per_level[L + 1] = level_total
        total += level_total
    end
    hctx.total_routes = total
    state.counts.n_routes = total
    return state
end

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

The KA path is deliberately pinned to `ConcatenatedFixedZM2L` +
`MaterializedYRotationM2L` -- the strategy already gated bit-exact on H200. That
choice is what keeps this a wrapper: `compact_cuda_factored` routes only
`DenseTranslationM2L`, `PrecomputedFactoredYM2L` and `FactoredRotationM2L`
through the CUDA-specific whole-pass setups, so with concat the builder falls
through to `ResidentM2LConcatPlan` and reaches no `_cuda_*` function at all.
Hence `compact_cuda_factored=false` below, and hence no KA port of
`_cuda_factored_whole_pass_setup!` / `_cuda_precomputed_y_whole_pass_setup!` /
`_build_cuda_dense_m2l_plan`.

Cost of that pin: the KA path forgoes the dense-fused M2L, CUDA's fastest H200
configuration. That is a reversible performance ceiling, not a correctness gap.

Graph capture is likewise not ported, and does not need to be:
`_cuda_adaptive_graph_eligible` requires a `ResidentM2LDenseCUDAPlan`, so the
concat plan is never graph-eligible *even on CUDA*. The KA substitute for the
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

end
