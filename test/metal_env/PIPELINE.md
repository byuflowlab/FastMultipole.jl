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
| 0 | User entry (`fmm!` / cache ctor) | `resident:2426` | — | MISSING-IN-KA |
| 1 | Argument + trait validation | `resident:2445-2519`, `:2549` | `ext:5561` (`ka_validate_radix_arguments`) | SAME (three noted deviations) |
| 2 | Geometry: `_radix_bounds` → center/box/`h0`/`x_min`/`box_extent` | `resident:2521-2535` | `ext:5693` (`ka_radix_geometry`) | SAME (one noted deviation) |
| 3 | Non-cubic axis resolution `_resolve_radix_ell_axes` | `resident:2190-2214`, called at `:2534` | shared host helper, called by `ka_radix_geometry` | SAME |
| 4 | `ell` selection | `resident:2428` — a plain kwarg, default 4 | kwarg, default 4 | SAME |
| 5 | Stencil policy + hierarchical tables (`root_level`, `first_m2l_level`, accepted/rejected offsets) + options/`dk` finalization | `resident:2546-2578`, `:2594-2626` | `ext:5769` (`ka_radix_stencil_policy`) | SAME (two noted deviations, one reordering) |
| 6 | Capacity sizing: `max_cells`, `max_nodes`, `max_level_nodes`, `route_capacity`, `direct_capacity` | `resident:2580-2592` | `ext:5877` (`ka_radix_capacities`) | SAME |
| 7 | Device ctx allocation | | `ext:5226` | SAME (hierarchical + `ConcatenatedFixedZM2L` only) |
| 8 | First state refresh | inside the ctor, `cuda:6564` | left to the caller | DIFFERS |

**Stage 1 landed 2026-08-30** as `ka_validate_radix_arguments` (`ext:5561`): a
statement-for-statement port of the constructor's pre-geometry prologue, in
CUDA's order and with CUDA's error text, returning the resolved traits instead
of assigning constructor locals. Three deviations, all documented at the site:
device availability is a KA-backend check rather than `cuda_radix_available()`;
`sfs=true` is refused outright (KA hardcodes `sfs_ctx=nothing`) instead of
validated; and CUDA's post-options `dk` checks (`resident:2600-2626`) read
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

Still outstanding at construction: **stage 0** (there is no `fmm!` dispatch that
reaches KA; the bench and the suites call `ext.ka_fmm!` / 
`ext.ka_radix_cache_device_step!` directly) and **stage 8** (the first state
refresh, left to the caller).

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
| 19 | Nearfield sub-Morton subsort | **MISSING-IN-KA** — **REVISIT** |
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

**Stage 19 — REVISIT.** `_cuda_nearfield_subsort!`, `cuda:6803-6810`, gated on
`radix_setting(:CUDA_NEARFIELD_SUBSORT)` and a `PartitionedVortex`/`TwoPassVortex`
direct kernel; KA's placeholder comment is at `ext:5102-5103`. Locality only, but
it composes into `perm` before packing and so changes same-cell summation order —
the last post-tree divergence that moves results. Expect last-bit changes when it
lands, not bit-equality, and note that `perm` cannot be gated elementwise on the
production path (counting sort is unstable — see [[reference-ka-counting-sort-port]]).

> **Both stage 15 and stage 19 are open, not settled** (flagged 2026-08-30). The
> SAME/DIFFERS verdicts elsewhere in section B should not be taken to imply that
> grid build as a whole is closed until these two are resolved with a gate.

## C. After grid build

Refresh stages, then B2M → M2M → M2L → L2L → L2B, with nearfield and finalize.
Same order and same gates on both sides. One live divergence:

**The M2L window cache is dead in KA.** `ext:5148-5156` adds
`apply_plan isa ResidentM2LDenseCUDAPlan && DENSE_CUDA_FUSED` on top of CUDA's
condition at `cuda:6895`, and KA pins the Concat plan — so `win_valid` is never
set, `n_routes` is forced to 0, and every window regenerates every step. Prime
suspect for the M2L cost profiled in
[[project-m2l-cost-is-route-count-and-bandwidth]].

## The cache object

Matches field for field within KA's declared envelope (hierarchical +
`ConcatenatedFixedZM2L`, no SFS, no adaptive): identical ctx fields, identical
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

**Uncovered:** stages 2-6 (nothing to gate — no implementation); 19
(intentional, unported); 22 (transitively only); stage 15's
occupancy-static/changed alternation across steps.

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
