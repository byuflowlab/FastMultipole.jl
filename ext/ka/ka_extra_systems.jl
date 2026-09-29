#------- extra source systems carried by the resident tree (KA) -------#
#
# Host mirror: src/resident_extra_tree.jl. The binning and the multipoles are
# computed on the host -- an extra system is small next to the resident field,
# and the multipole recursions are the octree's own -- and only two kernels run
# on the device: one adds the per-cell coefficients into the leaf multipoles
# before M2M, the other sweeps the near cell pairs.

"""
    ka_extra_tree_prepare(state, system)

Bin `system` onto the resident grid and pack everything the two device kernels
need: the binned bodies and their per-cell ranges, the multipole columns and
the nodes they belong to, and the bodies held out of the tree.
"""
function ka_extra_tree_prepare(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        system) where {TF,B,LH}
    n_cells = Int(state.counts.n_cells)
    binned, loose = FastMultipole.bin_resident_extra_source(TF, system, state.grid, n_cells)
    orders = state.invariant_cache.basis_info.orders
    rows_phi = size(FastMultipole.phi_slab(state.multipoles), 1)
    rows_chi = LH ? size(FastMultipole.chi_slab(state.multipoles), 1) : rows_phi
    nodes, phi, chi = FastMultipole.resident_extra_multipole_columns(TF, system,
        binned.buffer, binned.cell_ranges, Array(state.cell_centers),
        Array(state.grid.leaf_to_node), orders.P_phi, orders.P_active, n_cells,
        Val(LH), rows_phi, rows_chi)
    backend = KA.get_backend(state.output)
    up(A) = (d = KA.allocate(backend, eltype(A), size(A)...); copyto!(d, A); d)
    return (; buffer = up(binned.buffer), cell_ranges = up(binned.cell_ranges),
            nodes = up(nodes), phi = up(phi), chi = up(chi), loose,
            kernel = FastMultipole.direct_kernel(system))
end

# one thread per (row, touched cell): the nodes are distinct, so no atomics
@kernel function ka_extra_tree_add_multipoles_kernel!(ph, ch, @Const(phi_add), @Const(chi_add),
        @Const(nodes), nrows_phi, nrows_chi, nrows, ncols, ::Val{LH}) where LH
    idx = @index(Global)
    # phi and chi are ragged (phi to P_phi, chi to P_active): stride over the
    # taller of the two and guard each channel by its own row count
    @inbounds if idx <= nrows * ncols
        row = (idx - 1) % nrows + 1
        col = (idx - 1) ÷ nrows + 1
        node = nodes[col]
        if row <= nrows_phi
            ph[row, node] += phi_add[row, col]
        end
        if LH && row <= nrows_chi
            ch[row, node] += chi_add[row, col]
        end
    end
end

"""
    ka_extra_tree_b2m!(state, prepared; workgroup)

Add the extra bodies' multipoles to the leaf multipoles. Must run after the
resident B2M and before M2M.
"""
function ka_extra_tree_b2m!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        prepared; workgroup=KA_AUTO_WORKGROUP) where {TF,B,LH}
    ncols = size(prepared.phi, 2)
    ncols == 0 && return state
    ph = FastMultipole.phi_slab(state.multipoles)
    ch = FastMultipole.chi_slab(state.multipoles)
    nrows_phi = size(prepared.phi, 1)
    nrows_chi = LH ? size(prepared.chi, 1) : nrows_phi
    nrows = max(nrows_phi, nrows_chi)
    backend = KA.get_backend(state.output)
    wg = resolve_workgroup(backend, workgroup)
    kern = _cached_kernel(ka_extra_tree_add_multipoles_kernel!, backend, wg)
    n = nrows * ncols
    kern(ph, ch, prepared.phi, prepared.chi, prepared.nodes, nrows_phi, nrows_chi,
         nrows, ncols, Val(LH); ndrange=cld(n, wg) * wg)
    return state
end

# one workgroup per near cell pair, threads striding over the target cell's
# bodies; a target body is written by several pairs, hence the atomics
@kernel function ka_extra_tree_near_kernel!(kernel, output, @Const(bodies), @Const(cell_ranges),
        @Const(ex_buffer), @Const(ex_ranges), @Const(direct_targets), @Const(direct_sources),
        n_direct, ::Type{T}, ::Val{HS}, ::Val{WG}, ::Val{EP}) where {T,HS,WG,EP}
    tid = @index(Local)
    pair_i = @index(Group)
    # `kernel` here is the caller's own direct kernel, not one of the cache's
    # functors, so querying it on the device is a dynamic dispatch that fails
    # to compile; the host resolves it and passes the answer as a type parameter
    ep = EP
    @inbounds if pair_i <= n_direct
        target_cell = direct_targets[pair_i]
        source_cell = direct_sources[pair_i]
        scount = ex_ranges[2, source_cell]
        if scount > 0
            sfirst = ex_ranges[1, source_cell]
            tfirst = cell_ranges[1, target_cell]
            tcount = cell_ranges[2, target_cell]
            i = tfirst + tid - 1
            while i <= tfirst + tcount - 1
                xi = bodies[1, i]; yi = bodies[2, i]; zi = bodies[3, i]
                u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
                h1 = zero(T); h2 = zero(T); h3 = zero(T)
                h4 = zero(T); h5 = zero(T); h6 = zero(T)
                h7 = zero(T); h8 = zero(T); h9 = zero(T)
                for j in sfirst:(sfirst + scount - 1)
                    if HS
                        du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                            FastMultipole._extra_pair_ugh(kernel, xi, yi, zi, ex_buffer, j)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                        h1 += dh1; h2 += dh2; h3 += dh3
                        h4 += dh4; h5 += dh5; h6 += dh6
                        h7 += dh7; h8 += dh8; h9 += dh9
                    else
                        du, dgx, dgy, dgz = FastMultipole._extra_pair_ug(kernel, xi, yi, zi, ex_buffer, j)
                        u += du; gx += dgx; gy += dgy; gz += dgz
                    end
                end
                if ep
                    KA.@atomic output[1, i] += u
                end
                KA.@atomic output[2, i] += gx
                KA.@atomic output[3, i] += gy
                KA.@atomic output[4, i] += gz
                if HS
                    # the velocity gradient of the near pairs: without it only the
                    # far field carried the extra source's gradient
                    KA.@atomic output[5, i] += h1
                    KA.@atomic output[6, i] += h2
                    KA.@atomic output[7, i] += h3
                    KA.@atomic output[8, i] += h4
                    KA.@atomic output[9, i] += h5
                    KA.@atomic output[10, i] += h6
                    KA.@atomic output[11, i] += h7
                    KA.@atomic output[12, i] += h8
                    KA.@atomic output[13, i] += h9
                end
                i += WG
            end
        end
    end
end

"""
    ka_extra_tree_near!(state, prepared; workgroup)

Sweep the near cell pairs, summing the binned extra bodies of each source cell
against the resident bodies of the paired target cell.
"""
function ka_extra_tree_near!(state::FastMultipole.DeviceResidentRadixState{TF},
        prepared; workgroup::Int=128) where TF
    n_direct = Int(state.counts.n_direct)
    n_direct == 0 && return state
    backend = KA.get_backend(state.output)
    wg = resolve_workgroup(backend, workgroup)
    hs = size(state.output, 1) >= 13
    dkernel = _ka_device_direct_kernel(prepared.kernel, TF, 0)
    kern = _cached_kernel(ka_extra_tree_near_kernel!, backend, wg)
    kern(dkernel, state.output, state.source_bodies, state.cell_ranges,
         prepared.buffer, prepared.cell_ranges, state.direct_targets, state.direct_sources,
         n_direct, TF, Val(hs && FastMultipole._extra_pair_has_hessian(dkernel)), Val(wg),
         Val(FastMultipole._emits_potential(prepared.kernel)); ndrange=n_direct * wg)
    return state
end

# The prepared form of an extra tree source is a pure function of its bodies
# and of the resident grid's occupied-cell set (binning, cell centers, leaf to
# node map). While the source reports the same `source_revision` and the
# occupancy epoch has not moved, the previous call's arrays are reused: the
# RK3 stages of a solver freeze the bodies over the step, and preparing them
# on the host every stage measured 16% of a sixty-four-rotor step. A source
# with revision `nothing` is prepared every call.
const _KA_EXTRA_TREE_CACHE_MAX = 64

function _ka_extra_tree_prepared!(cache::FastMultipole.RadixFMMCache, sys)
    rev = FastMultipole.source_revision(sys)
    rev === nothing && return ka_extra_tree_prepare(cache.state, sys)
    ctx = cache.device_ctx
    key = objectid(sys)
    hit = get(ctx.extra_tree_cache, key, nothing)
    kernel = FastMultipole.direct_kernel(sys)     # the prepared form carries the kernel: a changed kernel is a miss
    epoch = ctx.epoch_id[]
    # `system === sys`: an objectid can be reused by another object after GC
    if hit !== nothing && hit.system === sys && hit.revision == rev &&
            hit.epoch == epoch && hit.kernel == kernel
        ctx.extra_tree_hits[] += 1
        return hit.prepared
    end
    ctx.extra_tree_misses[] += 1
    # entries from an older epoch can never hit again; dropping them releases
    # their device arrays and the systems they hold
    filter!(kv -> kv.second.epoch == epoch, ctx.extra_tree_cache)
    length(ctx.extra_tree_cache) >= _KA_EXTRA_TREE_CACHE_MAX && empty!(ctx.extra_tree_cache)
    prepared = ka_extra_tree_prepare(cache.state, sys)
    ctx.extra_tree_cache[key] = (; system = sys, revision = rev, epoch, kernel, prepared)
    return prepared
end

"""
    ka_radix_cache_device_step!(cache, targets, switches; nearfield_pass=nothing,
        workgroup=KA_AUTO_WORKGROUP, extra_targets=(), extra_target_switches=(),
        extra_sources=(), extra_tree_sources=(), self_induce=true)

Backend-agnostic device step: refresh the device state, run the uniform
lifecycle body (or the all-pairs direct arm when `:RADIX_DIRECT_ARM` is set),
apply the extra systems and any `nearfield_pass`, and scatter the output back
into the target systems. The keywords mirror the host `fmm!` radix path.
"""
function ka_radix_cache_device_step!(cache::FastMultipole.RadixFMMCache,
        targets::Tuple, switches::Tuple; nearfield_pass=nothing,
        workgroup=KA_AUTO_WORKGROUP, extra_targets::Tuple=(),
        extra_target_switches::Tuple=(), extra_sources::Tuple=(),
        extra_tree_sources::Tuple=(), self_induce::Bool=true)
    # construction-locked settings must not have drifted: a late flip is
    # baked-in-silently otherwise (buffers sized at construction)
    FastMultipole.verify_locked_radix_settings(cache.locked_settings)
    # With the direct arm armed, the whole lifecycle is replaced by one
    # all-pairs kernel, and the grid/route refresh is skipped with it. Same
    # finalize: only the U/J evaluation differs (a `nearfield_pass` is refused,
    # since the arm builds no direct pairs).
    direct_arm = FastMultipole.radix_setting(:RADIX_DIRECT_ARM)
    # The extra-sources-only call needs no routes, but it DOES need the bodies
    # repacked and the permutation refreshed: `finalize` de-permutes through
    # the state's body metadata, and the body count changes between calls in a
    # shedding solver; skipping the refresh would leave a stale permutation and
    # scatter the result onto the wrong particles.
    # per-stage timers (see _KA_UPDATE_TIMERS): `outside` is the host time
    # since the previous device call ended, i.e. everything the caller did
    _utick!(:outside, KA.get_backend(cache.state.output))
    ka_update_radix_state!(cache, targets; workgroup, direct_only=direct_arm)
    state = cache.state
    _utick!(:update_total, KA.get_backend(state.output))
    if !self_induce
        # Sources-only: the resident bodies are targets but not sources, so
        # there is no resident field to build and `extra_tree_sources` are
        # applied directly, which is what the host path does (src/fmm.jl).
        # Carrying them in the tree instead is possible -- zero the leaf
        # multipoles, skip the resident near pairs, run the pipeline -- but it
        # costs a second full far-field pass to save an all-pairs sweep that
        # measured 0.88 s of a 9.9 s step at four rotors and 1.94 s of 34 s at
        # sixteen. It lost both times, and by more at the larger size.
        fill!(state.output, zero(eltype(state.output)))
        ka_extra_sources_into_output!(state, extra_tree_sources; workgroup)
    elseif direct_arm
        # the all-pairs arm writes rows 2:4 (and 5:13) but only writes row 1 for
        # a potential-emitting kernel, so it starts from zero like the host; tree
        # sources have no leaves to join here and are applied all-pairs
        fill!(state.output, zero(eltype(state.output)))
        ka_direct_body!(state; workgroup)
        ka_extra_sources_into_output!(state, extra_tree_sources; workgroup)
    else
        prepared = isempty(extra_tree_sources) ? () :
            Tuple(_ka_extra_tree_prepared!(cache, sys) for sys in extra_tree_sources)
        isempty(prepared) || _utick!(:extra_prepare, KA.get_backend(state.output))
        ka_lifecycle_body!(state; extra_tree=prepared)
        for p in prepared
            ka_extra_tree_finish!(state, p; workgroup)
        end
        isempty(prepared) || _utick!(:extra_finish, KA.get_backend(state.output))
    end
    if nearfield_pass !== nothing
        # the consumer's own pass over the U-list direct pairs (e.g. an SFS
        # estimator through `radix_nearfield`), at the same point as on the
        # host (src/fmm.jl); the direct arm builds no pairs
        direct_arm && throw(ArgumentError(
            "nearfield_pass is not supported on the all-pairs direct arm " *
            "(radix setting :RADIX_DIRECT_ARM): it runs over the U-list direct pairs, which this arm does not build"))
        nearfield_pass(cache)
        _utick!(:nearfield_pass, KA.get_backend(state.output))
    end
    # after the nearfield pass, as on the host (src/fmm.jl): an SFS estimator
    # there reads the velocity gradient of the resident bodies and the tree
    # sources only
    ka_extra_sources_into_output!(state, extra_sources; workgroup)
    isempty(extra_sources) || _utick!(:extra_sources, KA.get_backend(state.output))
    ka_finalize_radix_output!(state, targets; derivatives_switches=switches,
        host_output_staging=cache.device_ctx.host_output,
        target_buffers=FastMultipole._radix_cache_target_buffers!(cache, switches),
        device_target_buffers=cache.device_ctx.device_target_buffers)
    _utick!(:finalize, KA.get_backend(state.output))
    # Extra targets are summed all-pairs against the packed resident bodies,
    # which reads no local expansion or near list, so it is also correct on
    # the direct arm (whose lifecycle state is stale).
    self_induce &&
        ka_extra_targets_evaluate!(state, extra_targets, extra_target_switches; workgroup)
    isempty(extra_targets) || _utick!(:extra_targets, KA.get_backend(state.output))
    return cache
end

#------- extra target / source systems (src/radix_extra_systems.jl) -------#
#
# Device counterparts of `_radix_extra_sources_into_output!` and
# `_radix_extra_targets_evaluate!`: the extra systems are host objects packed
# on the host, uploaded, evaluated by a thread-per-target rectangular kernel,
# and (for extra targets) downloaded and scattered through the host
# `buffer_to_target!`. Nothing here is on the resident lifecycle's zero-copy
# contract: the extras are a few hundred bodies per call.

# thread per (extra target, source chunk): a target loops over one chunk of the
# packed main bodies with the cache's nearfield functor and writes its partial
# sums to `part[:, i, c]`; the wrapper reduces over chunks. A thread per target
# alone leaves the device idle for the usual few hundred targets, and the wall
# time is then the serial loop over every body (~25 ms per call at 44k bodies
# on an H200, whatever the target count).
@kernel function ka_extra_targets_from_main_kernel!(kernel, part, @Const(xt), nt,
        @Const(source_bodies), nbodies, chunk, ::Type{T}, ::Val{HS}) where {T,HS}
    i, c = @index(Global, NTuple)
    ep = FastMultipole._emits_potential(kernel)
    ghv = Val(:shipped)
    jlo = (c - 1) * chunk + 1
    jhi = min(c * chunk, nbodies)
    @inbounds if i <= nt && jlo <= jhi
        xi = xt[1, i]; yi = xt[2, i]; zi = xt[3, i]
        u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
        h1 = zero(T); h2 = zero(T); h3 = zero(T)
        h4 = zero(T); h5 = zero(T); h6 = zero(T)
        h7 = zero(T); h8 = zero(T); h9 = zero(T)
        for j in jlo:jhi
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
        part[1, i, c] = ep ? u : zero(T)
        part[2, i, c] = gx; part[3, i, c] = gy; part[4, i, c] = gz
        if HS
            part[5, i, c] = h1; part[6, i, c] = h2; part[7, i, c] = h3
            part[8, i, c] = h4; part[9, i, c] = h5; part[10, i, c] = h6
            part[11, i, c] = h7; part[12, i, c] = h8; part[13, i, c] = h9
        end
    end
end

# thread per resident body (positions in rows 1:3 of the packed bodies, slot
# order), loop over an extra source's packed buffer through the source's own
# functor. ACCUMULATES into the resident output.
@kernel function ka_targets_from_extra_source_kernel!(kernel, out, @Const(xt), nt,
        @Const(source_buffer), ns, ::Type{T}, ::Val{HS}) where {T,HS}
    i = @index(Global)
    @inbounds if i <= nt
        xi = xt[1, i]; yi = xt[2, i]; zi = xt[3, i]
        u = zero(T); gx = zero(T); gy = zero(T); gz = zero(T)
        h1 = zero(T); h2 = zero(T); h3 = zero(T)
        h4 = zero(T); h5 = zero(T); h6 = zero(T)
        h7 = zero(T); h8 = zero(T); h9 = zero(T)
        for j in 1:ns
            if HS
                du, dgx, dgy, dgz, dh1, dh2, dh3, dh4, dh5, dh6, dh7, dh8, dh9 =
                    FastMultipole._extra_pair_ugh(kernel, xi, yi, zi, source_buffer, j)
                u += du; gx += dgx; gy += dgy; gz += dgz
                h1 += dh1; h2 += dh2; h3 += dh3
                h4 += dh4; h5 += dh5; h6 += dh6
                h7 += dh7; h8 += dh8; h9 += dh9
            else
                du, dgx, dgy, dgz = FastMultipole._extra_pair_ug(kernel, xi, yi, zi,
                    source_buffer, j)
                u += du; gx += dgx; gy += dgy; gz += dgz
            end
        end
        out[1, i] += u
        out[2, i] += gx; out[3, i] += gy; out[4, i] += gz
        if HS
            out[5, i] += h1; out[6, i] += h2; out[7, i] += h3
            out[8, i] += h4; out[9, i] += h5; out[10, i] += h6
            out[11, i] += h7; out[12, i] += h8; out[13, i] += h9
        end
    end
end

_ka_upload(backend, host::AbstractMatrix{TF}) where TF =
    copyto!(KA.allocate(backend, TF, size(host)), host)

function _ka_launch_extra_source!(backend, wg, out, xt, nt::Int, system, ::Type{TF},
        hs::Bool) where TF
    ns = FastMultipole.get_n_bodies(system)
    ns == 0 && return out
    buffer = _ka_upload(backend, FastMultipole._radix_extra_source_buffer(TF, system))
    # the typed device mirror for the regularized vortex kernels (Float64 fields
    # do not compile on Metal); other kernels pass through unchanged. The extra
    # buffer carries no inverse-sigma row, so the mirror divides (inv_sigma_row 0).
    kernel = _ka_device_direct_kernel(FastMultipole.direct_kernel(system), TF, 0)
    kern = _cached_kernel(ka_targets_from_extra_source_kernel!, backend, wg)
    kern(kernel, out, xt, nt, buffer, ns, TF,
         Val(hs && FastMultipole._extra_pair_has_hessian(kernel)); ndrange=cld(nt, wg) * wg)
    return out
end

"""
    ka_points_from_extra_source(backend, xt, system, TF; hessian=false, workgroup) -> Matrix

Velocity of an extra source system at arbitrary points `xt` (3 x n, host),
summed all-pairs on the device with the system's own direct kernel; returns
a host `(4 or 13) x n` matrix in the radix output layout (row 1 potential,
rows 2:4 velocity). Independent of any resident cache: a device form of a
host all-pairs loop, for callers whose source count times point count has
outgrown the host (the wake ring rows onto every body's control points are
O(bodies^2) and were 17% of a step at sixty-four rotors).
"""
function ka_points_from_extra_source(backend, xt_h::AbstractMatrix, system, ::Type{TF};
        hessian::Bool=false, workgroup=KA_AUTO_WORKGROUP) where TF
    nt = size(xt_h, 2)
    rows = hessian ? 13 : 4
    out = KA.allocate(backend, TF, rows, nt); fill!(out, zero(TF))
    nt == 0 && return Array(out)
    wg = resolve_workgroup(backend, workgroup)
    xt = _ka_upload(backend, TF.(xt_h))
    _ka_launch_extra_source!(backend, wg, out, xt, nt, system, TF, hessian)
    KA.synchronize(backend)
    return Array(out)
end

# the bodies held out of the tree: a packed buffer rather than a whole system
function _ka_launch_extra_buffer!(backend, wg, out, xt, nt::Int, host_buffer, kernel,
        ::Type{TF}, hs::Bool) where TF
    ns = size(host_buffer, 2)
    ns == 0 && return out
    buffer = _ka_upload(backend, host_buffer)
    kern = _cached_kernel(ka_targets_from_extra_source_kernel!, backend, wg)
    kern(kernel, out, xt, nt, buffer, ns, TF,
         Val(hs && FastMultipole._extra_pair_has_hessian(kernel)); ndrange=cld(nt, wg) * wg)
    return out
end

"""
    ka_extra_tree_finish!(state, prepared; workgroup)

The near sweep and the held-out bodies, after the lifecycle has run.
"""
function ka_extra_tree_finish!(state::FastMultipole.DeviceResidentRadixState{TF},
        prepared; workgroup=KA_AUTO_WORKGROUP) where TF
    ka_extra_tree_near!(state, prepared; workgroup)
    n = Int(state.counts.n_bodies)
    if n > 0 && size(prepared.loose, 2) > 0
        backend = KA.get_backend(state.output)
        wg = resolve_workgroup(backend, workgroup)
        _ka_launch_extra_buffer!(backend, wg, state.output, state.source_bodies, n,
            prepared.loose, _ka_device_direct_kernel(prepared.kernel, TF, 0), TF, size(state.output, 1) >= 13)
    end
    return state
end

"""
    ka_extra_sources_into_output!(state, extra_sources; workgroup)

Apply every extra source system to the resident bodies, accumulating into
`state.output` in slot order (slot positions are rows 1:3 of
`state.source_bodies`), after the lifecycle body and before finalize.
"""
function ka_extra_sources_into_output!(state::FastMultipole.DeviceResidentRadixState{TF},
        extra_sources::Tuple; workgroup=KA_AUTO_WORKGROUP) where TF
    isempty(extra_sources) && return state
    n = state.counts.n_bodies
    n == 0 && return state
    hs = size(state.output, 1) >= 13
    backend = KA.get_backend(state.output)
    wg = resolve_workgroup(backend, workgroup)
    for system in extra_sources
        _ka_launch_extra_source!(backend, wg, state.output, state.source_bodies, n,
            system, TF, hs)
    end
    KA.synchronize(backend)
    return state
end

"""
    ka_extra_targets_evaluate!(state, extra_targets, switches; workgroup)

Evaluate every extra target system from the resident bodies on the device,
then scatter through the target's switch on the host.
"""
function ka_extra_targets_evaluate!(state::FastMultipole.DeviceResidentRadixState{TF,B,LH},
        extra_targets::Tuple, switches::Tuple; workgroup=KA_AUTO_WORKGROUP) where {TF,B,LH}
    isempty(extra_targets) && return state
    n = Int(state.counts.n_bodies)
    backend = KA.get_backend(state.output)
    wg = resolve_workgroup(backend, workgroup)
    dkernel = _ka_device_direct_kernel(state.options.direct_kernel, TF, 0)
    allpairs = _cached_kernel(ka_extra_targets_from_main_kernel!, backend, wg)
    for (system, switch) in zip(extra_targets, switches)
        hs = !isempty(FastMultipole.hessian_range(switch))
        hs && size(state.output, 1) < 13 && throw(ArgumentError(
            "hessian output requested for an extra target system but the cache " *
            "was built with hessian=false"))
        xt_h = FastMultipole._radix_extra_target_positions(TF, system)
        nt = size(xt_h, 2)
        nt == 0 && continue
        rows = hs ? 13 : 4
        if n == 0
            FastMultipole._radix_scatter_extra_target!(TF, system, switch, zeros(TF, rows, nt))
            continue
        end
        xt = _ka_upload(backend, xt_h)
        nchunk = max(1, min(cld(n, 256), cld(65_536, nt)))
        chunk = cld(n, nchunk)
        # The kernel writes part[:, i, c] only for chunks that hold a
        # body. With nchunk chosen first and chunk rounded up, the last
        # chunk can be EMPTY ((nchunk-1)*chunk >= n), and its slab of an
        # uninitialized buffer would be summed in: pool garbage that differs
        # every call and looks like a race. So size the chunk count from the
        # chunk, and zero the buffer.
        nchunk = cld(n, chunk)
        part = KA.allocate(backend, TF, rows, nt, nchunk)
        fill!(part, zero(TF))
        allpairs(dkernel, part, xt, nt, state.source_bodies, n, chunk, TF, Val(hs);
                 ndrange=(cld(nt, wg) * wg, nchunk))
        KA.synchronize(backend)
        out = nchunk == 1 ? reshape(part, rows, nt) : dropdims(sum(part; dims=3); dims=3)
        KA.synchronize(backend)
        FastMultipole._radix_scatter_extra_target!(TF, system, switch, Array(out))
    end
    return state
end



