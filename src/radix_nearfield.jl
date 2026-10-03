#------- consumer passes over the resident near field -------#
#
# A consumer with its own pairwise physics over the near field (a vortex
# method's subfilter-scale estimator, a vorticity reconstruction, ...) runs it
# through `fmm!(...; nearfield_pass = f)`: `f(cache)` is called once the
# lifecycle has completed the standard outputs of the resident bodies and the
# tree-carried sources, before extra sources are added and before delivery.
# The pass reads the sorted data through `radix_nearfield(cache)` and owns its
# scratch, its kernels (host loops or KernelAbstractions kernels on the
# device) and its delivery. FastMultipole holds no consumer physics.

"""
    radix_nearfield(cache::RadixFMMCache) -> NamedTuple

The resident lifecycle's sorted near-field data, for a consumer pass run
through `fmm!(...; nearfield_pass = f)`; host arrays on a host cache, device
arrays on a device cache (`device` tells which). Fields:

- `source_bodies`: the packed sources in sorted order (`data_per_body` rows:
  position 1:3, radius 4, strength 5:`4+strength_dims`, the consumer's extra
  rows after);
- `output`: the sorted outputs (row 1 potential, 2:4 gradient, 5:13 hessian
  when the cache carries it); a pass may read it and, for a re-evaluation,
  write it;
- `cell_ranges` (`2 x n_cells`: first sorted body and body count of each cell),
  `direct_targets`, `direct_sources` (the `n_direct` near cell pairs of the
  U-list as ordered (target <- source) pairs: both (A, B) and (B, A) are
  listed, the self pair (A, A) once), `n_cells`, `n_bodies`;
- `body_perm`, `body_system_ids`, `body_indices`: sorted position -> global
  body (system id and index), and their host copies `host_body_perm`,
  `host_body_system_ids`, `host_body_indices`;
- `masked`: the bodies the evaluation took out of the tree (`nothing`, or
  `(; idx, buffer)`; see [`radix_set_masked!`](@ref) and [`radix_masked_grid`](@ref)).

The arrays are the cache's own: capacity-sized, valid over the live prefix,
and reused every step.
"""
function radix_nearfield(cache::RadixFMMCache)
    state = cache.state
    c = state.counts
    return (; source_bodies = state.source_bodies, output = state.output,
              cell_ranges = state.cell_ranges,
              direct_targets = state.direct_targets, direct_sources = state.direct_sources,
              n_direct = Int(c.n_direct), n_cells = Int(c.n_cells), n_bodies = Int(c.n_bodies),
              body_perm = state.body_perm, body_system_ids = state.body_system_ids,
              body_indices = state.body_indices,
              host_body_perm = state.host_body_perm, host_body_system_ids = state.host_body_system_ids,
              host_body_indices = state.host_body_indices,
              masked = cache.masked,
              device = cache.device)
end

#------- the masked bodies in the near field (moved from FLOWVPM, 2026-10-03) -------#
#
# Bodies an oversize policy took out of the tree are packed with zero strength and
# core, so the near cell pairs never carry them. A consumer pass that needs their
# pairs reads the masked set from `radix_nearfield(cache).masked` and finds, for
# each target, the masked bodies within its own cutoff through a uniform grid of
# them ([`radix_masked_grid`](@ref)) and their sorted slots
# ([`radix_masked_slots`](@ref); the KA extension has the device method).

"""
    radix_set_masked!(cache, masked)

Record the bodies the evaluation took out of the tree: `nothing`, or
`(idx, buffer)` with `idx` the global indices into system 1 and `buffer` their
packed columns. Exposed as `radix_nearfield(cache).masked`.
"""
radix_set_masked!(cache::RadixFMMCache, masked) =
    (cache.masked = masked === nothing ? nothing : (; idx = masked[1], buffer = masked[2]); cache)

"""
    radix_masked_grid(masked, TF, cutoff; core_row=8) -> NamedTuple

The masked bodies binned on a uniform grid whose cell is at least the largest
`cutoff * core` (so the 27 cells around a target hold every masked body within
the cutoff of it): `cols` (their packed columns, sorted by cell), `mx` (positions),
`ms` (cores), `mp` (global indices), `offsets` (`offsets[c]:offsets[c+1]-1` are
cell `c`'s bodies), `origin`, `h`, `dims`, and `psorted`/`korder` mapping a global
index back to its sorted position. The sort is stable, so the order is fixed.
"""
function radix_masked_grid(masked, ::Type{TF}, cutoff::Real; core_row::Int=8) where TF
    idx, buf = masked.idx, masked.buffer
    K = length(idx)
    rc = Float64(cutoff)
    h = rc * maximum(Float64(buf[core_row, k]) for k in 1:K)
    lo = [minimum(Float64(buf[d, k]) for k in 1:K) for d in 1:3]
    hi = [maximum(Float64(buf[d, k]) for k in 1:K) for d in 1:3]
    dims = [floor(Int, (hi[d] - lo[d]) / h) + 1 for d in 1:3]
    while prod(dims) > 1 << 22                  # a coarser grid only adds candidates
        h *= 2; dims = [floor(Int, (hi[d] - lo[d]) / h) + 1 for d in 1:3]
    end
    cell(k) = 1 + min(floor(Int, (buf[1, k] - lo[1]) / h), dims[1] - 1) +
              dims[1] * (min(floor(Int, (buf[2, k] - lo[2]) / h), dims[2] - 1) +
              dims[2] * min(floor(Int, (buf[3, k] - lo[3]) / h), dims[3] - 1))
    keys = [cell(k) for k in 1:K]
    order = sortperm(keys)                      # stable: fixed summation order
    ncells = prod(dims)
    offsets = zeros(Int32, ncells + 1)
    for k in order; offsets[keys[k] + 1] += 1; end
    offsets[1] = 1
    for c in 2:ncells + 1; offsets[c] += offsets[c - 1]; end
    mp = idx[order]
    pord = sortperm(mp)
    return (; K, cols = buf[:, order], mx = TF.(buf[1:3, order]), ms = TF.(buf[core_row, order]),
              mp, psorted = mp[pord], korder = Int32.(pord), offsets,
              origin = TF.(lo), h = TF(h), dims = Int32.(dims))
end

"""
    radix_masked_slots(grid, nf) -> Vector{Int}

The sorted slot of each masked body of `grid` (in the grid's order; 0 if it is not
resident), from the host permutation of `nf = radix_nearfield(cache)`.
"""
function radix_masked_slots(m, nf)
    mslot = zeros(Int, m.K)
    perm = nf.host_body_perm; sys = nf.host_body_system_ids; bidx = nf.host_body_indices
    @inbounds for s in 1:nf.n_bodies
        g = perm[s]
        sys[g] == 1 || continue
        t = searchsortedfirst(m.psorted, bidx[g])
        (t <= m.K && m.psorted[t] == bidx[g]) && (mslot[m.korder[t]] = s)
    end
    return mslot
end

# device method: the KA extension
radix_masked_grid_device(m, nf, backend) = throw(ArgumentError(
    "radix_masked_grid_device needs the KernelAbstractions extension"))

"""
    radix_masked_foreach(f, m, mslot, xi, yi, zi)

Host iteration over the masked bodies in the 27 grid cells around `(xi, yi, zi)`:
calls `f(t, s)` with `t` the body's position in the grid order and `s` its slot.
"""
@inline function radix_masked_foreach(f, m, mslot, xi, yi, zi)
    cx = floor(Int, (xi - m.origin[1]) / m.h)
    cy = floor(Int, (yi - m.origin[2]) / m.h)
    cz = floor(Int, (zi - m.origin[3]) / m.h)
    nx, ny, nz = m.dims
    @inbounds for kz in max(cz - 1, 0):min(cz + 1, nz - 1),
                  ky in max(cy - 1, 0):min(cy + 1, ny - 1),
                  kx in max(cx - 1, 0):min(cx + 1, nx - 1)
        c = 1 + kx + nx * (ky + ny * kz)
        for t in m.offsets[c]:m.offsets[c + 1] - 1
            f(t, mslot[t])
        end
    end
    return nothing
end
