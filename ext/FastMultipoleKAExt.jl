module FastMultipoleKAExt

using FastMultipole
using KernelAbstractions
using LinearAlgebra
using StaticArrays: SVector
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
# Unlike the private CPU function, this does not use the `count[]`-prefix views
# (`_vector_prefix_view`/`_matrix_col_view`) since those are FastMultipole-internal;
# callers are expected to size `group`/`ws` to exactly `group.count[]` columns (true
# for the isolated per-group correctness test this backs -- test/metal_env/ka_m2m_correctness.jl).
function ka_resident_stage_group_apply!(dest, src, group, ws, kind::Symbol)
    n = group.count[]
    n == 0 && return dest
    mult = kind === :m2m
    ystk = ws.ystk_phi
    Ur = mult ? ystk.mult_Ur : ystk.loc_Ur
    Vs = mult ? ystk.mult_Vs : ystk.loc_Vs
    ndof_phi = size(ws.aphi, 1)
    source_idx = group.source_idx
    target_idx = group.target_idx
    group_phis = group.phis
    group_thetas = group.thetas
    aphi = ws.aphi; yphi = ws.yphi; zphi = ws.zphi; rphi = ws.rphi; cphi = ws.cphi
    C = ystk.Cy; S = ystk.Sy; G = ystk.G; G2 = ystk.G2
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
        achi = ws.achi; ychi = ws.ychi; zchi = ws.zchi; rchi = ws.rchi; cchi = ws.cchi
        Cc = ystk_c.Cy; Sc = ystk_c.Sy; Gc = ystk_c.G; G2c = ystk_c.G2
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

end
