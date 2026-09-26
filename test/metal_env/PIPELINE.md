# The radix-FMM pipeline: CUDA spine and KA status

CUDA is the oracle; KA is the deliverable. This file records CUDA's end-to-end
execution order once, as a numbered spine, so work walks *down the pipeline*
instead of chasing whichever suite happens to be red. Mapped 2026-08-29
(session 17); every `file:line` below was spot-checked against the tree at that
time.

Sides:

* CUDA — `src/translate_batched_resident.jl` (construction) and
  `src/translate_batched_cuda.jl` (per-step; runtime-`include()`d, so everything
  in it is CUDA-only regardless of content).
* KA — `ext/FastMultipoleKAExt.jl` (cited below as `ext:N`).

Verdicts: **SAME** (KA does the same thing), **MISSING-IN-KA**, or **DIFFERS**.

---

## A. Once, at construction

CUDA: `src/translate_batched_resident.jl:2426-2591` (`RadixFMMCache`).

| # | Stage | CUDA | KA | Verdict |
|---|---|---|---|---|
| 0 | User entry (`fmm!` / cache ctor) | `resident:2426` | `register_radix_device_backend!` (`ext:6831`) | SAME (registry, not a method overwrite) |
| 1 | Argument + trait validation | `resident:2445-2519`, `:2549` | `ext:5561` (`ka_validate_radix_arguments`) | SAME (three noted deviations) |
| 2 | Geometry: `_radix_bounds` → center/box/`h0`/`x_min`/`box_extent` | `resident:2521-2535` | `ext:5693` (`ka_radix_geometry`) | SAME (one noted deviation) |
| 3 | Non-cubic axis resolution `_resolve_radix_ell_axes` | `resident:2190-2214`, called at `:2534` | shared host helper, called by `ka_radix_geometry` | SAME |
| 4 | `ell` selection | `resident:2428` — a plain kwarg, default 4 | kwarg, default 4 | SAME |
| 5 | Stencil policy + hierarchical tables (`root_level`, `first_m2l_level`, accepted/rejected offsets) + options/`dk` finalization | `resident:2546-2578`, `:2594-2626` | `ext:5769` (`ka_radix_stencil_policy`) | SAME (two noted deviations, one reordering) |
| 6 | Capacity sizing: `max_cells`, `max_nodes`, `max_level_nodes`, `route_capacity`, `direct_capacity` | `resident:2580-2592` | `ext:5877` (`ka_radix_capacities`) | SAME |
| 7 | Device ctx allocation | | `ext:5226` | SAME (hierarchical + `ConcatenatedFixedZM2L` only) |
| 8 | First state refresh | inside the ctor, `cuda:6564`, then `built = true` (`resident:2691`) | inside `ka_radix_cache_device_build` (`ext:5406`), then `built = true` | SAME |

**Stage 1 landed 2026-08-30** as `ka_validate_radix_arguments` (`ext:5561`): a
statement-for-statement port of the constructor's pre-geometry prologue, in
CUDA's order and with CUDA's error text, returning the resolved traits instead
of assigning constructor locals. Three deviations, all documented at the site:
device availability is a KA-backend check rather than `cuda_radix_available()`;
`m2l_strategy=PrecomputedFactoredYM2L` is refused where CUDA accepts it (no KA
plan exists -- see the skipped set in section C); and CUDA's post-options `dk` checks (`resident:2600-2626`) read
`options.direct_kernel` AFTER the policy-dependent substitution, so they belong
to stage 5. Gated by `pipeline_stage_bench.jl`, which asserts KA's `LH`, `TF`,
`n0` and `maxn` against the host cache FLOWVPM built for the same wake, and by
the deletion of the faked `LH` from `pipeline_device_args.jl`.

**Stages 2 and 3 landed 2026-08-30** as `ka_radix_geometry` (`ext:5693`): a
statement-for-statement port of the constructor's geometry block
(`resident:2521-2535`), returning `(; x_min, h0, ell_axes, box_extent)`. It
shares CUDA's own `_radix_bounds` and `_resolve_radix_ell_axes` — backend-
independent host code producing host scalars, so a KA copy would be duplication
rather than a port — which is why stage 3 lands with it. One deviation, a
subtraction: CUDA's derived branch also names `center` and `box`, dead at the
end of the block and not returned. The `bounds === nothing` branch walks bodies
through `get_position` (scalar indexing on a device-backed system) exactly as
CUDA does; production never reaches it, because FLOWVPM always passes explicit
`bounds` (`FLOWVPM_fmm_radix.jl:521-523`, `:539`, `:560-562`). Gated by
`pipeline_stage_bench.jl`, which asserts all four outputs against the host
cache, and by the deletion of `x_min`, `h0`, `ell_axes` and `box_extent` from
`pipeline_device_args.jl`.

**Stages 5 and 6 landed 2026-08-30** as `ka_radix_stencil_policy` (`ext:5769`)
and `ka_radix_capacities` (`ext:5877`). Stage 5 ports the policy block
(`resident:2546-2578`) together with the options/`dk` tail (`:2594-2626`) that
stage 1 explicitly deferred to it, because that tail cannot run until the class
count exists. Two deviations, both refusals in stage 1's style: `adaptive` is
refused outright (KA has no adaptive octree lifecycle), and `device=true` is
passed unconditionally to `_default_radix_policy` and
`_assert_device_kernel_policy`, a KA cache being device-resident by
construction — which is also what selects `RADIX_DEVICE_WINDOW_CLASSES`. One
reordering: CUDA runs the capacity block BETWEEN the tables and the options
tail; the two are data-independent, so KA runs the tail first and leaves the
capacities whole for stage 6. Stage 6 is a port with no deviations. Both share
CUDA's own host helpers rather than copying them, as stage 2 does.

**`pipeline_device_args.jl` no longer reads a single field off the host cache.**
`device_build_args` — the copy of `ka_device_cache_correctness.jl:105-109` —
is gone, and so are the five hand-fed capacities. What the file still carries is
the CALLER side (`production_caller_args`): the `bounds`, `ell`,
`near_radius2`, `window_classes` and `level_radii2` FLOWVPM itself computes and
hands `RadixFMMCache`, which are inputs to the front end, not part of it. The
host cache is still built for every size, purely as the gate.

This closes the two consequences this document recorded while stages 2-6 were
missing: KA now has a front end, and the sizing it performs is gated against an
independently-built host cache rather than copied from one.

**Stage 8 landed 2026-08-30** (session 22, recorded session 23): the tail of
`ka_radix_cache_device_build` calls `ka_update_radix_state!(cache, sources)` and
then sets `cache.built = true`, the same two statements in the same order as the
CUDA constructor's `update_cuda_radix_state!` + `resident:2691`. No code was
written this session; the row above was stale. It is gated implicitly by
`pipeline_stage_bench.jl`, which goes straight from `ka_radix_cache_device_build`
to `ka_fmm!` with no caller-side refresh in between and scores relerr(U) ~1e-6 --
a cache whose first refresh had not run would have no grid at all.

Nothing is outstanding at construction. **Stage 0 landed** via
`register_radix_device_backend!` (`src/FastMultipole.jl:213`, KA side `ext:6831`):
an extension cannot overwrite a parent-module method the way CUDA's runtime
`include()` does, so `fmm!` consults a registry hook instead. The bench and the
suites may still call `ext.ka_fmm!` directly, but production no longer has to.

There is no auto-`ell` derivation on *either* side. FLOWVPM's
`max(2, floor(log2(np)/3))` rule lives in the caller
(`FLOWVPM/src/FLOWVPM_fmm_radix.jl:492`), not in FastMultipole. Do not import it
into these gates: it would make KA diverge from CUDA rather than match it.

## B. Every step, through grid build

| # | Stage | Verdict |
|---|---|---|
| 9 | Step entry + locked-settings verify | SAME |
| 10 | Refresh entry; adaptive branch | DIFFERS — CUDA dispatches, KA `throw`s (`ext:5030-5034`) |
| 11 | Source refresh + position gather | SAME |
| 12 | Morton keys + OOB check | SAME |
| 13 | Sort (counting sort vs `sortperm`, same branch condition, both unstable in counting-sort mode) | SAME |
| 14 | Cell compression + capacity assert | SAME |
| 15 | Occupancy-epoch detection | DIFFERS in control flow only — **REVISIT** |
| 16 | `cell_centers` | SAME |
| 17 | Per-level node keys + `level_offsets` | SAME |
| 18 | Node topology | SAME |
| 19 | Nearfield sub-Morton subsort | SAME (`ka_nearfield_subsort!`) |
| 20 | Pack bodies + geometry gate | SAME |
| 21 | Host mirrors | SAME |
| 22 | Tree m2m/l2l route edges | SAME |
| 23 | Stage-group refresh | SAME |

**Stage 15 — REVISIT.** CUDA early-returns when occupancy is unchanged
(`cuda:6672-6676`); KA instead wraps stages 16-18 in `if occ_changed`
(`ext:5082`). These are *argued* to be semantically equivalent; nothing gates it.
Stage 15's occupancy-static/changed alternation across steps is listed as
uncovered below, so the equivalence is unverified. Do not read this row as
closed.

**Stage 19 landed 2026-08-30** as `ka_nearfield_subsort!` plus two kernels
(`ka_subsort_keys_kernel!`, `ka_subsort_cell_sort_kernel!`), mirroring
`_cuda_nearfield_subsort!` (`cuda:6803-6810`) under the same
`radix_setting(:CUDA_NEARFIELD_SUBSORT)` and direct-kernel gate. The
`subsort_keys` buffer was already allocated. It composes into `perm` before
packing and so changes same-cell summation order -- last-bit changes, not
bit-equality -- and `perm` cannot be gated elementwise on the production path
(counting sort is unstable, see [[reference-ka-counting-sort-port]]), so it is
gated end-to-end on U instead. This was the last post-tree divergence that moved
results.

> **Stage 15 is still open** (stage 19 closed 2026-08-30). The SAME/DIFFERS
> verdicts elsewhere in section B should not be taken to imply that grid build as
> a whole is closed until stage 15's control-flow equivalence has a gate.

## C. After grid build

Refresh stages, then B2M -> M2M -> M2L -> L2L -> L2B, with nearfield, SFS and
finalize. Same order and same gates on both sides. Operator by operator:

| CUDA | KA | Verdict |
|---|---|---|
| `run_cuda_radix_lifecycle!` -> `_cuda_lifecycle_body!` | `ka_lifecycle_body!` | SAME |
| `_cuda_graph_eligible` / `_run_cuda_radix_lifecycle_graph!` | — | SKIPPED — no KA graph capture, and it requires the dense plan |
| `_launch_cuda_nearfield_async!` (`CUDA_OVERLAP_NEARFIELD`) | `ka_launch_nearfield!`, synchronous | SKIPPED — no KA stream overlap |
| `_launch_cuda_b2m!` | `ka_launch_b2m!` | SAME |
| resident m2m stage groups | `ka_resident_stage_group_apply!(:m2m)` | SAME |
| `_launch_cuda_resident_m2l!` | `ka_launch_m2l!` / `ka_hierarchical_m2l!` / `..._cached_concat!` | SAME |
| resident l2l stage groups | `ka_resident_stage_group_apply!(:l2l)` | SAME |
| `_launch_cuda_resident_l2b!` | `ka_launch_l2b!` | SAME |
| `_launch_cuda_resident_l2b_only!` (overlap variant) | — | SKIPPED with the overlap path |
| `_launch_cuda_sfs!` (4 kernels, 2 launchers) | `ka_launch_sfs!` | SAME |
| `finalize_cuda_radix_output!` | `ka_finalize_radix_output!` | SAME |
| `finalize_cuda_radix_sfs_output!` | `ka_finalize_radix_sfs_output!` | SAME |
| `_cuda_adaptive_radix_lifecycle!` | — | SKIPPED — adaptive |
| M2T with `LH` | — | SKIPPED by design; no CUDA version exists to copy (see [[reference-lh-m2t-hessian-deferral]]) |

### The deliberately-skipped set, in full

Nothing on FLOWVPM's path is unported. What KA does not do, and why:

* **The adaptive lifecycle** (stage 10, `_cuda_update_adaptive_radix_state!` and
  `_cuda_adaptive_radix_lifecycle!`). A different tree, not a different backend
  for the same tree: variable depth, per-step child-range reshaping, rebuilt
  interaction lists. Production never reaches it -- only a GPU-resident field
  reaches radix at all, and FLOWVPM always builds with explicit `bounds` +
  `window_classes`, i.e. the uniform hierarchical path. KA `throw`s
  (`ext:5030-5034`). The `ka_adaptive_*` suites gate the octree entry, not this.
* **The factored and precomputed-Y M2L route refreshes**
  (`_cuda_refresh_factored_m2l_routes!` `cuda:3569`,
  `_cuda_refresh_precomputed_y_m2l_routes!` `cuda:4047`). Both are plan-typed:
  they exist only to histogram `route_class` and prefix-sum it into the
  per-class ranges a `ResidentM2LFactoredPlan` / `ResidentM2LPrecomputedYPlan`
  apply needs. `ka_radix_cache_workspace` pins
  `ConcatenatedFixedZM2L`/`MaterializedYRotationM2L`, so neither plan is
  constructible on the KA path, and the concat apply needs no class-start table
  at all. Dead code on this configuration, not deferred work -- which is also
  why the skip has no gate.
* **CUDA graph capture** and **nearfield/L2B stream overlap**. Performance
  paths with no KA analogue.

### Ported but off the production path

`ka_generate_radix_routes!` (`ext:3972`) is the faithful port of CUDA's **flat**
route generator: it decodes each leaf Morton key, subtracts each accepted
stencil offset, and looks the source up in `ctx.cell_at`, a dense 3-D array
indexed by leaf coordinate. FLOWVPM's `window_classes` selects
`HierarchicalRigidStencil`, so `hierarchical_ctx` is non-`nothing` and the
refresh takes `ka_hier_generate_direct_pairs!` / `ka_hier_generate_window_core!`
instead -- which index `node_at` at `level_base_L` by the leaf node's stored
`node_coords`, per level. Different table, different indexing; neither is a
special case of the other. **The trap:** `ctx.cell_at` and the flat route
buffers are zero-sized on a production cache, so any probe or gate that calls
the flat generator is measuring code the real step never runs
([[reference-flowvpm-radix-cache-is-hierarchical]]).

**The M2L window cache is live in KA, on the concat plan.** CUDA's
`_cuda_windows_cacheable` (`cuda:6895`) is gated on the dense FUSED family, and
KA's dense apply is the GEMM reference driver CUDA explicitly excludes -- so
porting CUDA's cached-window M2L was never a standalone port. But the concat
apply consumes only `(class, source, target, count)` and takes no level
argument, which makes concat MORE cacheable on KA than on CUDA: one apply per
epoch. `ka_hierarchical_m2l_cached_concat!` does that, and it removes the
per-window D2H and syncs along with the generation. Measured at np=8192, 120
interleaved warm trials: concat 50.4 -> 37.0 ms, and concat is now 2x dense
(dense 72.6 ms). See [[reference-ka-concat-window-cache]].

## The cache object

Matches field for field within KA's declared envelope (hierarchical +
`ConcatenatedFixedZM2L`, no adaptive): identical ctx fields, identical
size formulas, identical epoch predicates (`epoch_have`, `prev_n`,
`prev_n_cells`, `epoch_cell_keys`, `epoch_id`, `win_valid`), identical capacity
guards. Remaining differences — unpinned host mirrors, no
`enable_synchronization!`, zero-length symmetric-nearfield arrays, no nearfield
bin context — are perf/memory only. Everything CUDA can do that KA cannot is an
explicit `throw`, never a silent fallback.

## Gate coverage, mapped onto the spine

| Stages | Gate |
|---|---|
| 11 | `ka_source_positions_correctness.jl` |
| 12-14 | `ka_grid_keys_cells_correctness.jl` |
| 12-16 | `ka_grid_epoch_centers_correctness.jl` |
| 13 | `ka_counting_sort_correctness.jl` |
| 17 | `ka_grid_level_nodes_correctness.jl` |
| 17-18 | `ka_grid_node_topology_correctness.jl` |
| 23 | `ka_stage_groups_correctness.jl` |
| 7-10, 20-22 | `ka_device_cache_correctness.jl`, `ka_production_driver_correctness.jl` — end-to-end only, no per-stage assertion |
| 0-1 | `pipeline_stage_bench.jl` — real rotor wake, host-cache trait comparison |

**Uncovered:** stage 22 (transitively only); stage 15's
occupancy-static/changed alternation across steps. Stages 2-6 are now gated by
`pipeline_stage_bench.jl` against an independently-built host cache; stage 19 is
gated end-to-end on U by `pipeline_uj_fmm.jl`.

**Misfiled — do not mistake these for grid coverage.**
`ka_tree_build`, `ka_tree_finalize`, `ka_tree_leaves`, `ka_tree_balance`,
`ka_tree_lists`, `ka_sigma_sweep_correctness.jl` gate the **adaptive** octree
(`ext:2372`), a path KA `throw`s on in production (`ext:5030-5034`). Six suites
and a large share of total suite runtime, gating something the production path
never takes.

## Two coverage facts worth asserting

* `first_m2l_level <= ell`. On an anisotropic `ell_axes`, `first_m2l_level` can
  exceed `ell` and yield a silently pure-direct schedule
  (`src/interaction_list_batched.jl:384-399`) — a case that passes while testing
  no M2L at all.
* `level_radii2`, the coarse multi-radius schedule, is built only at `ell >= 3`
  (`resident:2323-2330`), so any `ell=2` case exercises the uniform policy only.

## Running the suites

`bash test/metal_env/run_suites.sh` — one line per suite, full output under
`logs/`. Use `julia -t auto`; a bare `julia` gives `threads = 1` and silently
un-threads the CPU comparison arm.
