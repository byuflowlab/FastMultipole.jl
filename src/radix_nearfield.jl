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
  `host_body_system_ids`, `host_body_indices`.

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
              device = cache.device)
end
