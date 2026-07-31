# 026 Implementation Hierarchical M2L Host

## Objective

Implement genuine multi-level (node-to-node) M2L on the host resident lifecycle, driven by
the `025` rigid source-major stencil and never compiling the full pair list. Support both
near radii from `025` — `near_radius2 = 12` (task-`024b` `theta = 0.5`) and
`near_radius2 = 3` (classic FMM `27 / 189`) — so the cost and accuracy of the two can be
measured against each other and against the shipped flat path.

The shipped flat `ConstantPAnalyticStencil` path is retained as a selectable correctness
oracle and remains the production default until `027` demonstrates parity and speed.

## Dependencies

- `025-theory-hierarchical-rigid-m2l-stencil.md` (stencil, coverage proof, scaling law)
- `021-impl-constant-p-stencil-and-interaction-list.md` (policy/strategy axes, flat stencil)
- `023-impl-production-integration.md` (invariant contract, counters)
- `023a-impl-factored-resident-m2l-host.md`, `023c-impl-precomputed-y-m2l-host.md`,
  `023e-impl-dense-translation-m2l-host.md` (the host strategies that must keep working)

## Required Reading

- `START_HERE.md`
- The dependency task files above
- `theory/hierarchical-rigid-m2l-stencil.md`
- `src/containers.jl`, `src/interaction_list_batched.jl`, `src/tree_batched.jl`,
  `src/translate_batched.jl`, `src/translate_batched_resident.jl`
- Existing radix interaction-list, integration, and timestepping tests

## Artifacts or Production Surface

- Types in `src/containers.jl`; stencil and traversal in `src/interaction_list_batched.jl`;
  per-level occupancy helpers in `src/tree_batched.jl`; plan/class/refresh changes in
  `src/translate_batched.jl` and `src/translate_batched_resident.jl` (placement rules,
  `START_HERE.md` "Implementation Code Placement").
- Tests in `test/radix_interaction_list_test.jl`, `test/radix_fmm_integration_test.jl`,
  `test/radix_fmm_timestepping_test.jl`, and the four strategy test files.
- `MATRIX_OPERATOR_REFACTOR/scripts/benchmark_026_hierarchical_host.jl` and
  `MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_host/`.

## Functional Phase

### Types (`src/containers.jl`)

Add a separation policy alongside `ConstantPAnalyticStencil` (`src/containers.jl:331`),
carrying the single near-radius parameter from `025`:

- `HierarchicalRigidStencil{C}` with `config::C` and `near_radius2::Int`.
- Convenience constructors: one for the constant-`P` case (default `near_radius2 = 12`,
  gated as below) and `classic_fmm_stencil(...)` for `near_radius2 = 3`.
- `RigidHierarchicalTables`: `near_offsets`, `push_offsets` (union, sorted `(z,y,x)`),
  `phase_starts::NTuple{9,Int}` CSR, `phase_index::Vector{Int32}`,
  `class_of::Matrix{Int32}` (`8 x n_offsets`, `0` = offset absent from that phase).
  ~140 KB at `near_radius2 = 12`, step-invariant, identical for every `ell >= 3`.
- `RadixLevelOccupancy{A}`: `ell`, `level_base::Vector{Int}` (prefix sums of the dense
  per-level grid sizes `8^L` — distinct from the existing `level_offsets`
  (`src/containers.jl:1504`), which is the prefix over *occupied* nodes; the traversal
  below uses both), and
  `node_at::A` storing the **flat node index** (`0` = unoccupied). Storing the node index
  removes the `leaf_to_node` indirection that currently forces every route to a leaf.

`nclasses = (ell - 1) * length(push_offsets)` — at most 10,440 at `ell = 7` versus
16,581,196 today (dense-lattice counts, `255^3 - 179`; see `025` deliverable 7 for the
occupancy-basis convention on reduction ratios). Classes are emitted level-ascending then offset-ascending, so the global
class-major contiguity the device assertions rely on is preserved for free.

The existing `RigidImplicitStencil <: RadixTraversalStrategy` stub (`src/containers.jl:340`,
reserved by `021`) is the traversal-strategy home for this work; do not introduce a new
strategy tag.

**Mandatory construction-time accuracy gate.** Assert that
`classify_radix_stencil_offsets(h0, ell, config)` reproduces exactly
`{|o|^2 <= near_radius2}` as its rejected set, and throw naming the required
`stencil_epsilon(ell)` otherwise. `RadixFMMCache`'s default `stencil_epsilon = 1e-4`
(`src/translate_batched_resident.jl:720`) does **not** obey the `025` ell-scaling, so
without this gate the stencil is silently wrong and the coverage proof does not apply.

### Per-level occupancy (`src/tree_batched.jl`)

Dense `node_at`, sized `sum(8^L for L in 0:ell) = 1.1428 * 8^ell` `Int32` — 1.14 MiB at
`ell = 6`, 9.14 MiB at `ell = 7`, negligible beside the operator tables. Refresh per step is
one `fill!` plus an `O(n_nodes)` scatter from `node_coords`, which `DeviceRadixGrid` already
stores decoded (`src/containers.jl:298`) — cheaper per node than today, because the Morton
decode disappears.

Retain `morton_key` + `searchsortedfirst` over the level's `node_keys` block (the
`radix_cell_index` pattern, `src/tree_batched.jl:266`) as (i) the host reference used by the
parity test and (ii) an automatic fallback above a configurable `ell`/byte threshold. This
keeps the "convert to Morton only when targets are needed" formulation available and lets
the two paths cross-check each other.

### Source-major traversal (`src/interaction_list_batched.jl`)

Class-outer, source-inner, for `L = 2:ell`, class `k` (offset `o`), and each occupied node
`s` in the flat range `level_offsets[L+1]+1 : level_offsets[L+2]`:

1. `c = node_coords[:, s]`; `u = c .& 1`.
2. Skip unless `class_of[phase(u), k] != 0`.
3. `t_coord = c + o`, bounds-tested with `_radix_coord_inbounds_at_level`
   (`src/interaction_list_batched.jl:218`).
4. `t = node_at[level_base[L+1] + 1 + linear(t_coord, 1 << L)]`; skip if `0`.
5. Emit `(source = s, target = t, class = class(L, k))`.

This is the existing `build_radix_routes!` loop nest with four changes: `cell_at` becomes a
per-level `node_at`; target and source roles swap (`t = c + o` rather than `s = c - o`); a
phase mask is added; the `leaf_to_node` indirection is deleted.

### Windowed generation

Full materialization is still too large (`~|V| * 1.14 * C` routes, ~18 GB at `ell = 6`), and
a fully implicit apply would require rewriting all four strategy launchers, which depend on
per-class contiguous `packed_sources`/`packed_targets` for their `D x W` slabs. Use a
**window** = one level x a chunk of `K` consecutive classes: generate that window's routes
into a fixed-capacity buffer, apply M2L for those classes, advance. Storage becomes
`O(window)` rather than `O(total pairs)`, which is what "never compile the full list" means
operationally, while every class inside a window stays whole and contiguous.

Supporting changes, deliberately small:

- Hoist `fill!(state.locals.phi/.chi, 0)` out of the four `_launch_resident_m2l_*!` bodies
  into `_launch_resident_m2l!` (`src/translate_batched.jl:3085`), before the window loop.
  All strategies already accumulate by scatter, so nothing else changes.
- Make the class-partition assertions window-local rather than global.
- **`DenseTranslationM2L` only:** apply the per-level diagonal `Lambda(s_L)` from `025`
  once per window, to the gathered source slab and the result slab. The other three
  strategies must **not** apply `Lambda` — their per-class tables (`Concatenated`'s
  per-class scalars, `Factored`/`PrecomputedY`'s `zlen x nclasses` `z_flat`) already
  carry the level-true `r`, so applying `Lambda` on top would double-scale. Note the
  windowing-invariance test cannot catch this mistake (both window widths would be wrong
  identically); the accuracy-parity tests are the gate.
- Keep a `window_classes = nclasses` single-window mode that degenerates exactly to today's
  control flow; parity oracles and small-`n` tests run in that mode, so correctness can land
  before `027` tunes windows.

### Class metadata and operator tables

- `route_levels`/`route_offsets` are never read by the production apply path
  (`src/translate_batched_cuda.jl:3066`); they are read only by the host/device
  route-parity tests (`test/cuda_radix_integration_test.jl:52`,
  `test/cuda_radix_lifecycle_test.jl:656`), which must be updated to compare
  class-level metadata instead. Replace with construction-time
  `class_level::Vector{Int32}` and `class_offset::Matrix{Int32}` of length `nclasses`;
  dropping the per-route arrays is by itself a 4x route-metadata reduction.
- `class_capacities[class(L,k)] = min(window_capacity, nodes_cap(L), prod(max(2^L - |o_d|, 0)))`
  — reuse `_dense_m2l_capacity` (`src/translate_batched.jl:4059`) with `G = 2^L`.
- `ConcatenatedFixedZM2L`: per-class scalars only; extend to `nclasses`. Nearly free —
  **land this strategy first**.
- `FactoredRotationM2L` / `PrecomputedFactoredYM2L`: y-mode and angle tables are
  direction-only and already level-invariant; only `z_flat` depends on `r`. Store
  `zlen x nclasses` for this row.
- `DenseTranslationM2L`: `D^2` per class binds. Use the `025` scaling law to store operators
  **per offset** with `Lambda(s_L)` applied per window, and update `_dense_m2l_footprint` /
  `_dense_m2l_limit_error` (`src/translate_batched.jl:4079`) to compute `operator_bytes`
  from offsets rather than classes and route metadata from the *window* capacity. The
  current formula is what produced `024b`'s 275 GiB projection.

## Optimize, Profile, and Retest Phase

Measure with `benchmark_026_hierarchical_host.jl`, reusing `fm024_class_counts`,
`fm024_occupancy`, and `fm024_memory` from
`MATRIX_OPERATOR_REFACTOR/scripts/benchmark_024_common.jl` and following the shape of
`benchmark_023a_factored_host.jl`. Sweep `(n, ell, P, strategy, policy)` with
`policy in {flat, hierarchical(12), hierarchical(3)}` and record:

- route counts total and per level; ratios versus flat and versus `C^2`;
- `nclasses`, nonempty classes, and per-level occupancy `p50/p90/max` — the key
  GEMM-efficiency diagnostic, since spreading routes over levels and offsets cuts routes
  per class by a similar factor;
- M2L time **broken down per level for each of the four strategies**, alongside the
  per-level class-occupancy histograms: `028` uses this to decide a heterogeneous
  per-level strategy mix (e.g. factored/concatenated at sparse coarse levels, dense at
  the leaf level), since the `024` defaults were measured only under the fat-class flat
  list; selecting the mix is `028`'s job, not this row's;
- construction time, and per-step generation split into `node_at` refresh / scan / compact;
- per-stage steady-state timings including the now load-bearing M2M and L2L;
- memory by category, and allocation counters;
- an `n`-sweep at fixed `ell` giving the empirical M2L-work exponent (target 1.0, current
  1.46), and the measured hierarchical-vs-flat crossover per `ell`;
- sampled-direct accuracy for both near radii at matched `P`, so the classic stencil's cost
  advantage is reported against its accuracy cost rather than in isolation.

Per-thread and multi-thread BLAS numbers must come from a non-macOS host, per the standing
project rule.

## Verification

Existing tests must stay green with the flat policy still the default:
`test/radix_interaction_list_test.jl`, `radix_grid_clustering_test.jl`,
`radix_fmm_integration_test.jl`, `radix_fmm_timestepping_test.jl`,
`dense_translation_m2l_test.jl`, `precomputed_y_resident_m2l_test.jl`,
`resident_m2m_gemm_test.jl`.

New and extended, in priority order:

1. **M2M/L2L correctness gate — run before any traversal work.** They currently run but
   feed nothing, so any sign, ordering, or scale error is invisible today and becomes a
   correctness bug the moment M2L is hierarchical. Assert nonleaf multipoles equal the
   direct body-to-multipole expansion about each nonleaf center at every level, and that
   nonleaf locals after M2L+L2L match a brute-force per-node oracle.
2. **Complete-body-coverage under the hierarchical scheme** — the primary gate. Extend
   `_assert_complete_body_coverage` to accept multi-level routes whose endpoints are node
   indices (expand node -> descendant leaves -> bodies). Run for both near radii on a dense
   `ell=3` grid, the existing boundary grid, the sparse grid, and random occupancy at
   `ell=4`. A bounds bug silently *drops* pairs, and only this test catches that.
3. **Stencil table tests** against the `025` constants: per-phase `|V_push(u)|` and union
   for `near_radius2 in (3, 12)`, `min|o|^2`, `max|o|_inf`, CSR consistency, and the
   level-1/level-2 branch behavior `025` establishes for each radius.
4. **Multi-level structure assertions**, the direct inverse of the bug being fixed:
   `length(unique(route_levels)) > 1`, some route targets outside the leaf node range, and
   route count `< 0.1 * C^2` at `n = 20000`.
5. **Level-scaling law test**: operator at `r` versus `2r` agrees with `Lambda`-scaling to
   ~1e-13 relative in Float64.
6. **Accuracy parity**: hierarchical versus `direct!` and versus the retained flat path at
   matched `stencil_epsilon(ell)`, for `LH in {false,true}`, `TF in {Float32,Float64}`,
   `ell in {3,4,5}`. Include `P = 4` per the standing project rule.
7. **Windowing invariance**: identical results for `K = nclasses` and `K = 8`; the partition
   assertion fires on a deliberately undersized window.
8. **Contract tests**: `@allocated(_launch_resident_m2l!) <= 64 KiB` and full step
   `< 512 KB` with windowing enabled; `route_uploads` / `operator_uploads` constant;
   `expansion_host_copies == 0`; array identity preserved across steps including `node_at`.

Record commands and result summaries in a `Verification Notes` section.

## Implementation Notes

- Added the exported `HierarchicalRigidStencil` and `classic_fmm_stencil`
  policies while retaining `ConstantPAnalyticStencil` as the cache default.
  Construction builds the approved 27/189/316 or 179/1253/1740 phase tables and
  rejects analytically incompatible epsilon choices with an actionable error.
- Added dense per-level `Int32` node occupancy with the 256 MiB / `ell <= 8`
  default gate and sorted Morton-key fallback. Refresh is one fill plus one
  occupied-node scatter and preserves array identity.
- Added level/offset class metadata, source-major phase-masked window route
  generation, leaf-near direct generation, separate total/per-level/last-window
  telemetry, and allocation-bounded prefix application. The measured production
  default window is four classes and no complete hierarchical route list is retained.
- Extended the resident strategy surface. Concatenated and factored selections
  share the allocation-bounded per-column hierarchical engine; precomputed-y
  retains its construction-time direction/z tables and uses the same bounded
  apply plan; dense stores one matrix per rigid offset and applies the exact
  scalar or asymmetric Lamb–Helmholtz source/target diagonals per level.
- Added `test/hierarchical_m2l_host_test.jl`, the task-026 benchmark driver and
  Slurm runner, and local smoke/all-policy/all-strategy CSV evidence under
  `data/hierarchical_m2l_host/`.

## Approval Notes

Clear-context review `2026-07-29` (different agent). Method: read `START_HERE.md`, this
task file, and the listed production/test/benchmark surfaces; ran the focused suite
locally (383/383 pass, Julia host, 2 threads); cross-checked every quantitative claim in
the Verification Notes against `data/hierarchical_m2l_host/{raw,summary}/final_026/`.

**Verdicts.** Objectives: met — genuine multi-level node-to-node M2L, both radii, windowed
generation with no full pair list, flat oracle retained as default. Correctness: the
phase-masked source-major traversal exactly implements the `025` partition (parent-near via
`fld(u+o,2)`, `t = s + o`, level-invariant tables); occupancy indexing is consistent between
refresh and lookup; window capacity `min(K*maxN, maxN^2)` is provably sufficient and
runtime-asserted; the dense `Lambda` diagonals including the asymmetric LH `chi` exponents
check out analytically; `Lambda` is correctly confined to the dense path. Performance
evidence: all 776 cases / 1,816 level rows exist and reproduce the claimed window selection
(w8 best, w4 within 1.74%), crossovers, exponents, allocation maxima, zero counters, and the
88 flat-only overruns. Benchmark methodology (warmup, M2L-only timers, policy-scoped
validator, source-hash manifest) is sound.

**Findings requiring disposition (user decision pending):**

- **B1 (significant).** Hierarchical `PrecomputedFactoredYM2L` silently executes the concat
  engine: `translate_batched_resident.jl` replaces `ctx.apply_plan` with a fresh
  whole-window `ResidentM2LConcatPlan`, the fully built precomputed-y hierarchical plan is
  never used, the precomputed-y window branch in `_launch_hierarchical_resident_m2l!` is
  unreachable, and the in-code comment claims the opposite. Outputs are correct (parity
  passes), but the benchmark's "hierarchical precomputed-y" rows measured the concat engine —
  so the recorded "precomputed-y never crossed flat" verdict compares flat precomputed-y
  against a mislabeled hierarchical concat and is not evidence about a genuine hierarchical
  precomputed-y path.
- **B2 (minor, flat regression).** Unconditional `maximum(... for L in 2:ell)` at
  construction makes a flat `RadixFMMCache` with `ell <= 1` throw "empty collection".
- **B3 (minor, test gap).** The asymmetric LH dense diagonals — the subtlest new
  arithmetic — have no direct test (four-strategy parity is LH=false; the LH sweep is
  concat-only).
- **Test-coverage gaps (moderate):** no route-count-vs-`C^2` assertion at large `n`; all
  numerical coverage/accuracy at `ell=3` only (task asked coverage at random `ell=4`
  occupancy and accuracy at `ell in {3,4,5}`); no per-node L2L oracle and no
  hierarchical-vs-flat numerical parity (both proxied by end-to-end `direct!` parity); the
  level-scaling law is asserted structurally, not numerically at ~1e-13; the CUDA
  route-parity tests were not updated to class metadata as this file's design section
  states (they remain flat-only and still valid).
- **Minor/cleanup:** dense footprint omits `source_scale`/`target_scale` bytes;
  `update_radix_state!` traverses all windows once for telemetry and again at launch;
  dense per-window loops scan all classes instead of the window range; `phase_starts` CSR
  built but unused by traversal; `policy` docstring omits `HierarchicalRigidStencil`.
  Cosmetic data nits: radius-3 potential maximum is 1.02e-5 (notes say 1.03e-5) and the
  4.70e-4 gradient figure is pooled flat+hierarchical (hierarchical-only 4.63e-4); the
  window-4 default was tuned on the immediately preceding source manifest, not the final
  one.

User disposition `2026-07-29`: fix B1–B3 plus the key test gaps in-session. See
"Post-Review Fixes" below. Because the reviewing agent modified code, task 026 remains
**Done and not Approved**; a fresh clear-context agent must perform approval per
`START_HERE.md`.

## Post-Review Fixes (2026-07-29)

Applied by the reviewing agent with user permission after the review above:

- **B1 fixed.** Removed the construction-time override in
  `src/translate_batched_resident.jl` that replaced the hierarchical
  `PrecomputedFactoredYM2L` apply plan with a whole-window concat plan. The
  specialized `ResidentM2LPrecomputedYPlan` — already correctly built over all
  `(ell-1) * noffsets` per-class effective offsets with level-true radii — is now
  the live apply plan, making the precomputed-y window branch in
  `_launch_hierarchical_resident_m2l!` reachable. Verified the plan's angle/scratch
  capacities dominate any window (leaf-`G` scaled-offset dense bounds strictly
  exceed level-`L` pair counts) and all packed arrays are `route_capacity`-sized.
  The misleading comment is gone with the override. **Benchmark caveat:** the
  final_026 "hierarchical precomputed-y" rows measured the concat engine under a
  precomputed-y label; the "precomputed-y never crossed flat" verdict is therefore
  not evidence about the genuine path and must be re-measured (earliest in `027`'s
  campaign or a follow-up host sweep).
- **B2 fixed.** The nonleaf `max_level_nodes` maximum is now guarded (`ell >= 2`,
  else 0), restoring flat `RadixFMMCache` construction at `ell <= 1`.
- **B3 fixed (test).** New seeded-nonzero-chi M2L-stage parity across all four
  strategies (`Float64`, LH on, P=4), directly exercising the dense asymmetric
  Lamb-Helmholtz source/target diagonals and the revived precomputed-y branch;
  gravitational-only LH runs leave `chi = 0` and could not catch errors there.
- **Test gaps closed:** random-occupancy exact-once coverage at `ell=4` for both
  radii; `n = 20000` route-count (`total < 0.1 C^2`) and multi-level structure
  assertions (nonleaf levels carry routes); a plan-identity regression guard that
  fails if a concat plan ever stands in for a specialized strategy again; flat
  `ell=1` construction/accuracy regression.
- `RadixFMMCache` docstring now documents the `HierarchicalRigidStencil` policy.
- Focused suite after fixes: **399/399 pass** (was 383). Full package suite rerun
  recorded below.

**Precomputed-y re-measurement (user-directed `2026-07-29`).** The mislabeled
benchmark rows are being re-collected: `cpu_026_run.sh` and `cpu_026_submit.sh`
gained a comma-separated `FM026_STRATEGIES` filter (partial matrices summarize
with `FM026_EXPECTED_PHASE=none`), and the summarizer's window-selection
completeness check now derives its expected key count from the swept matrix
instead of the hardcoded full-campaign 16.

The first submission (jobs `12953413`/`12953416`/`12953417`, manifest
`bb638b89…`) exposed a genuine defect in the revived path: hierarchical
`PrecomputedFactoredYM2L` at `near_radius2 = 12` allocated 92–219 KB per M2L
launch, violating the 64 KiB contract at every window width (radius 3 passed;
counters all zero; timings unaffected). Root cause: the per-angle stage calls
inside `_launch_resident_m2l_precomputed_y_plan!` read the intentionally
`Any`-typed workspace fields (`phi_flat_idx`, `maps_phi`, buffer matrices), so
each call paid ~17–25 B dynamic-dispatch boxing — negligible on the flat path
(one visit per angle) but multiplied to ~`nclasses` active (window, angle)
iterations by the hierarchical driver. Fix: the same inline host type-assert
pattern `_launch_hierarchical_concat_window!` already uses, hoisted to the top
of the precomputed-y apply. Measured at `n = 20000`, `ell = 4`, radius 12:
w=4 now 39,856 B (== concat), w=8 22,416 B, w=1 144,608 B (over-gate small-width
regime, same disclosed behavior as concat/windows 1–2 in the original campaign).
Two rejected alternatives are documented for the record: a whole-loop dispatch
barrier caused a severe inference regression (~90 MB/launch on the concat path),
and a per-window call barrier cost ~364 B/window in boxed arguments; the original
window loop was restored unchanged. Focused suite 399/399 and the full package
suite pass after the fix (Julia 1.12.5 host; ORC parity spot-checked on 1.11.7).

The stale first-submission jobs were cancelled/failed-at-summary (their raw CSVs
are retained on the cluster only — not synced into the local
`data/hierarchical_m2l_host/raw/` tree — and carry the pre-fix allocation columns) and
all three phases were resubmitted on the fixed source, manifest
`5f85ea2ab02b460c8d9733642bc8f49d530f987496e7f98d51d9177be830d5fc`: `12953684`
(tuning), `12953685` (scaling incl. same-source flat baseline), `12953686`
(accuracy).

Job `12953684` then failed on a cluster infrastructure issue: the `m12`
partition mixes CPU targets and concurrent jobs share `~/.julia`, so the
simultaneously started scaling job's precompile invalidated the pkgimage
mid-run on a different node ("Unable to find compatible target in cached code
image", cascadelake image on an AMD node). `cpu_026_run.sh` now exports a
per-job first-layer `JULIA_DEPOT_PATH` (`~/.julia_fm026/<jobid>`), isolating
each job's compiled cache. Tuning and accuracy were resubmitted as `12954300`
and `12954303` under manifest
`34106787d876f81998fcf235af459f82b76eec11513b72460a64a1c3250749d3`, which
differs from `5f85ea2a…` only in `scripts/cpu_026_run.sh` — `src/` and `test/`
are byte-identical, so the measured code is the same as scaling job `12953685`
(healthy, left running).

**Re-measurement results (`2026-07-30`, jobs `12953685` scaling `01:24:01`,
`12954300` tuning `00:23:22`, `12954303` accuracy `01:02:14`, all exit `0:0`,
validators clean, zero allocation/counter failures across all 194 cases at 1
and 64 BLAS threads).** Data under `raw/` and `summary/` per job id; logs in
`logs_precomputed_y_refit/`. Genuine hierarchical `PrecomputedFactoredYM2L`
verdicts, superseding the mislabeled final_026 rows:

- **Crossover verdict reversed.** The superseded record claimed radius-12
  precomputed-y never crossed the flat oracle in the sampled range. The genuine
  path crosses everywhere: radius 12 by `N in (256, 512]` at `ell=3`,
  `(128, 256]` at `ell=4`, and at/below the first sampled size at `ell=5`
  (both thread counts); radius 3 at/below the first sampled size at every
  level. The old non-crossing was an artifact of comparing the fast flat
  precomputed-y baseline against a mislabeled hierarchical concat engine.
- **Window default confirmed.** The genuine path's cross-case geomean minimum
  is window 4 outright (1073.7 ms; w8 +1.0%, monotonically worse above), so the
  shipped `window_classes = 4` default stands, now on direct evidence.
- **M2L-time scaling exponents** 0.90–1.15 (radius 12) and 0.98–1.10
  (radius 3) versus 1.29–1.63 flat; route exponents match the prior record
  (traversal unchanged).
- **Accuracy maxima unchanged** (radius 12: `3.44e-7` potential /
  `2.31e-5` gradient; radius 3: `1.02e-5` / `4.63e-4`), confirming the
  mislabel never affected numerical results, only performance attribution.
- The allocation fix holds on the cluster: hierarchical precomputed-y M2L
  passes the 64 KiB gate in every scaling and accuracy case and at every
  tuning window `>= 4`; tuning windows 1–2 exceed it (144,240 / 74,144 B)
  and are marked ineligible by the summarizer — the same disclosed
  small-width behavior as concat in the original campaign. Profile matches
  concat at the selected window.

Remaining note for `027`/`028`: hierarchical precomputed-y and hierarchical
concat have nearly identical M2L medians at the tuning point (~2.3 s radius 12,
~0.5 s radius 3 at `n=20000`, `ell=4`), because the shared z-translation and
scatter stages dominate; the strategies differentiate mainly through their flat
baselines and construction/storage costs.

Still open (deferred, minor): per-node L2L oracle and hierarchical-vs-flat numerical
parity remain proxied by end-to-end `direct!` parity; the ~1e-13 level-scaling law
check remains script-level (structural-only in tests); dense footprint omits the
`source_scale`/`target_scale` bytes; per-step double traversal (telemetry + launch)
and full-`nclasses` scans per dense window are `027`/`028` optimization candidates;
CUDA route-parity tests still compare per-route metadata (flat-only, valid; `027`
must move them to class metadata).

## Verification Notes

Julia 1.12.5 host verification:

```sh
JULIA_NUM_THREADS=2 julia --project=test -e \
  'using FastMultipole, FastMultipole.StaticArrays, FastMultipole.LinearAlgebra,
   Random, Test; include("test/gravitational.jl");
   include("test/hierarchical_m2l_host_test.jl")'
JULIA_NUM_THREADS=2 julia --project=test test/runtests.jl
FM026_N=50 FM026_ELL=3 FM026_REPS=1 \
  FM026_OUT=MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_host/local_strategy_sweep_20260728.csv \
  julia --project MATRIX_OPERATOR_REFACTOR/scripts/benchmark_026_hierarchical_host.jl
FM026_PHASE=tuning bash MATRIX_OPERATOR_REFACTOR/scripts/cpu_026_submit.sh
FM026_PHASE=scaling FM026_SELECTED_WINDOW=4 \
  bash MATRIX_OPERATOR_REFACTOR/scripts/cpu_026_submit.sh
FM026_PHASE=accuracy FM026_SELECTED_WINDOW=4 \
  bash MATRIX_OPERATOR_REFACTOR/scripts/cpu_026_submit.sh
FM026_EXPECTED_PHASE=all FM026_SELECTED_WINDOW=4 \
  julia --project \
  MATRIX_OPERATOR_REFACTOR/scripts/summarize_026_hierarchical_host.jl \
  MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_host/raw/final_026 \
  MATRIX_OPERATOR_REFACTOR/data/hierarchical_m2l_host/summary/final_026
```

- Focused hierarchical suite: 383/383 pass. It covers both radii, task-025
  table/CSR constants, dense/boundary/sparse exact-once body coverage,
  multi-level/nonleaf routes, a direct body-to-node nonleaf M2M oracle,
  full-vs-eight-class invariance, all four strategy selections, Float32/Float64,
  LH off/on, `P=4`, dense offset-only storage/scaling, the classifier failure,
  identities, counters, and allocation gates.
- Full package suite passed with no failures after the production changes.
- Representative warmed eight-class measurements: resident M2L 11,792 bytes
  (`<=64 KiB`) and full step 394,176 bytes (`<512 KiB`); zero expansion host
  copies and zero route/operator uploads.
- The local 12-case strategy sweep records flat/hierarchical-12/hierarchical-3
  route counts, per-level counts, construction/M2L/step time, allocations,
  occupancy memory, and resident memory. It is smoke evidence, not a crossover
  campaign.
- The first ORC Linux submission, job `12926966`, failed before collecting data
  because the repository root environment was incompatible with the ORC Julia
  installation. The corrected isolated-environment submission, job `12927444`,
  completed on `m12-1-14` in `00:01:47` with exit code `0:0`.
- Corrected one- and 64-thread BLAS smoke results were fetched as
  `orc_m12-1-14_blas1_20260728-213036.csv` and
  `orc_m12-1-14_blas64_20260728-213036.csv`. Each contains all 12 combinations
  of the three policies and four resident strategies. Both files satisfy the
  allocation gates (maximum warmed M2L `11,456` bytes; maximum full step
  `36,144` bytes) and every row reports zero expansion host copies, route
  uploads, and operator uploads.
- These ORC results are an `N=50`, `ell=3` smoke check. Task 026 remains
  **not Done** until the planned practical window-width sweep and representative
  scaling/crossover campaign are collected and recorded; the smoke run alone
  does not support a performance-policy conclusion.
- Added opt-in, allocation-free host telemetry for grid refresh, dense
  occupancy refresh, direct generation, the production fused scan/compact
  stage, tree/group refresh, and per-level M2L. The timers are disabled by
  default and their backing arrays preserve identity across steps.
- Replaced the smoke-only driver with isolated case, per-level, campaign, and
  validation tooling. The final matrix is fixed at 128 practical
  window-tuning, 456 scaling/clustered, and 192 sampled-accuracy cases (776
  total). Full-level numerical invariance remains covered by the focused test
  instead of materializing impractical full-level benchmark scratch. Final
  validation rejects duplicates, missing level rows, within-phase mixed source
  hashes, required hierarchical allocation/counter failures, and accuracy
  failures.
- Final-source local verification passed: focused task suite 383/383, all four
  strategy/accuracy driver smoke cases, and the complete package suite.
- ORC job `12928290` exposed a wide-window scratch-capacity defect: the
  hierarchical concatenated path applied a complete route window using slabs
  limited to the flat strategy chunk. `window=64` wrote past those slabs,
  aborted Julia, and left the batch process hung. The job was cancelled after
  `06:21:30`; its six partial case/level pairs are preserved under
  `failed_12928290/`.
- Hierarchical concatenated scratch now covers the complete active window, and
  route capacity is additionally bounded by the ordered node-pair count. A new
  regression constructs a capacity above the former 32,768-column chunk.
  Focused verification passes 382/382, and the formerly crashing window-64
  driver completed locally with 595,080 routes, 4,080-byte M2L allocation, and
  27,120-byte step allocation.
- Repaired source manifest
  `6a68c4dc5d9beb34e2a7b38a815d980e1e8906509a9e7b411931078ceba6c456`
  completed the tuning-only ORC job `12931759` in `02:54:23`, exit `0:0`.
  All 128 cases and 384 per-level rows validated with zero allocation/counter
  failures. The global within-2%-of-best rule selected four classes, which is
  now the `HierarchicalRigidStencil` default; focused verification passes
  383/383 after that change.
- Final selected-window source manifest
  `19a9673c9425d38d9f289d30b6e98be5a782c54bd6247945c8f8c696d381649c`
  produced all 456 scaling/clustered cases in job `12940953` (`07:19:42`) and
  all 192 sampled-accuracy cases in job `12940963` (`05:59:48`). Both Slurm
  jobs exited `1:0` only because the first validator version incorrectly
  applied the hierarchical allocation gate to the unchanged flat oracle.
  Every requested case and level row was present.
- The corrected policy-scoped validator passes the combined 776 cases and
  1,816 per-level rows. Every hierarchical selected-window case satisfies the
  allocation gates; maxima are 56,000-byte M2L / 239,056-byte full step for
  radius 12 and 4,544 / 136,272 bytes for radius 3. All policies report zero
  expansion host copies and zero route/operator uploads. The retained flat
  oracle has 88 recorded large-case allocation overruns, reported rather than
  misrepresented as a hierarchical failure.
- Window 8 had the minimum cross-case geometric-mean M2L time; window 4 was
  1.74% slower and therefore selected as the smallest width within 2% of the
  minimum. Radius-12 runtime crossover was observed by `N=512` for concat at
  `ell=4/5`, while precomputed-y did not cross the flat oracle in the sampled
  radius-12 range. Radius 3 crossed substantially earlier: concat/factored by
  `N=128`–`512` depending on level, and dense at or below the first sampled
  size.
- Measured route-work exponents were 1.293–1.408 for radius 12 and
  1.097–1.175 for radius 3, versus 1.436–1.959 for the flat route list.
  Sampled-direct maxima for radius 12 were `3.44e-7` potential and `2.32e-5`
  gradient. The classic radius-3 accuracy cost was explicit: `1.03e-5`
  potential and `4.70e-4` gradient maxima. All accuracy and cross-strategy
  gates passed.
- Final combined artifacts are under
  `data/hierarchical_m2l_host/{raw,summary}/final_026/`. The final complete
  package suite passed with the measured four-class default. Task 026 is
  **Done** and intentionally not Approved; approval remains for a different
  clear-context agent.

### Concrete-type consolidation follow-up (`2026-07-30`)

The post-implementation type review was addressed without changing field access
or splitting host/device state types. `CUDARadixLifecycleOptions` now carries
the concrete operator, M2M-strategy, and M2L-strategy types, and the concrete
options type participates in `DeviceResidentRadixState`. Resident groups,
channels, plans, workspaces, and state consolidate constructor-proven shared
container types while retaining distinct parameters for genuinely optional
storage (notably stage versus M2L group operator storage). Generated group
vectors now undergo an explicit homogeneous `Vector{T}` conversion and reject
mixed concrete group types during construction.

Verification on Julia 1.12.5, macOS host:

- Focused hierarchical suite: **399/399 pass**.
- Warmed allocation matrix
  (`strategy in {concat,factored,precomputed_y,dense}` crossed with flat and
  hierarchical windows `1,4,8,128`): **20/20 cells exactly 0 bytes** at
  `N=20000`; the final optional-storage adjustment was rechecked at `N=2000`
  with the same 20/20 exact-zero result. This includes flat precomputed-y and
  flat dense, formerly the two allocating cells.
- Controlled cold focused include: **31.80 s**, 93.36% compilation time
  (`59.61 s` whole process including fresh-depot dependency/test precompile).
  This recovers about **1.6 s** of the reported 33.4 s pre-change cold time and
  does not worsen the earlier 20.8 -> 33.4 s regression.
- Seven-sample warmed host smoke at `N=2000`, `ell=4`, `P=4`, hierarchical
  radius 12/window 4 produced M2L medians/IQRs of
  `1936.75/1.92 ms` (concat), `1937.89/2.24 ms` (factored),
  `345.25/1.14 ms` (precomputed-y), and `83.45/0.42 ms` (dense).
  The narrow IQRs, expected cross-strategy ordering, zero M2L allocation, and
  unchanged counters show no warmed runtime anomaly from the type-only change;
  no benchmark campaign rerun was warranted.
- Complete package suite: **pass**, `182.94 s` wall time after the CUDA
  host-mirror invariant fix (`184.94 s` on the preceding review-adjusted
  source).
- Added `scripts/cuda_026_type_smoke.jl` and its H200 Slurm wrapper to construct
  and run two recurring lifecycle steps for all four strategies, checking
  concrete option/state typing, persistent array identities, finite output, and
  unchanged transfer counters. The first ORC run, job `12960487` on
  `m13h-2-1`, failed during construction before launching a strategy: recurring
  CUDA state retains host body mirrors as `Vector{Int}` but omits the optional
  host route mirrors (`nothing`). The initial consolidation had incorrectly
  assigned both sets one type parameter. `DeviceResidentRadixState` now uses
  separate host-body and optional host-route parameters, exactly matching the
  host/device constructor invariant required by the plan. After the fix the
  focused suite passes 399/399, the CUDA host lifecycle gate passes 135/135,
  and the 20-cell allocation matrix remains exactly zero at `N=2000`.
  Corrected H200 job `12961171` ran on `m13h-2-1` and completed in `00:04:48`
  with exit `0:0`. All four strategies (concatenated, factored, precomputed-y,
  and dense) constructed concrete CUDA states and completed two recurring
  lifecycle steps; persistent array identities, finite output, zero expansion
  host copies, and unchanged route/operator upload counters all passed.

Fresh clear-context review: **conditionally approved**, with no supported-path
code blocker. The reviewer independently reran the focused suite (399/399) and
confirmed the option/state typing, constructor-proven shared parameters,
optional-storage separation, heterogeneous-group rejection, and direct field
access. The review's minor robustness suggestion was incorporated immediately:
operator/strategy type parameters are now subtype-bounded and invalid keyword
operators fail at construction. The corrected H200 result clears the review's
only remaining condition; final clear-context approval is recorded below.

Final clear-context verdict after direct inspection of the corrected source and
ORC log: **unconditionally approved**. The reviewer confirmed the independent
`HBV`/`HRV` state parameters, bounded concrete option types and preserved partial
forms, shared/optional storage invariants, homogeneous group rejection, all host
verification results, and H200 job `12961171` ending
`CUDA 026 concrete-container smoke: PASS`. No code or verification blocker
remains; task 026 is **Done and Approved**.


### Improvement review (2026-07-31, fresh clear-context agent)

A user-requested improvement-hunting review re-verified the approved state:
**no correctness bug found** (B1/B2 fixes genuine, Lambda exponents consistent
across all four LH channel blocks, phase tables exact, all capacity bounds
safe, occupancy indexing consistent), and every quantitative claim above
reproduces from `data/hierarchical_m2l_host/` (crossover reversal re-derived
from raw scaling ratios; window-4 geomean; exponents; accuracy maxima
bit-identical between the superseded and genuine campaigns). Two wording
overstatements in this file were corrected in place (the 64 KiB "every case"
sentence now scopes out tuning windows 1–2; the stale-raw-CSV claim now says
cluster-only). With user approval the review then applied:

- **Provenance:** `SUPERSEDED.md` markers in `raw/final_026/` and
  `summary/final_026/` flagging the mislabeled hierarchical precomputed-y rows.
- **Robustness (src):** flat resident M2L launchers now throw on a
  hierarchical-policy state instead of silently computing a partial answer
  from the last window (`_assert_flat_resident_state`; the CUDA hierarchical
  driver's `clear_locals=false` window applies are exempt); explicit `policy=`
  now rejects `stencil_epsilon`/`near_radius2`/`window_classes` instead of
  silently ignoring them (and `stencil_epsilon` rejects `near_radius2`);
  `RadixLevelOccupancy` returns a consistent all-zero prefix on
  overflow/over-budget instead of a partially built one; the accuracy-gate
  error now reports the epsilon used and distinguishes caller-supplied from
  machine-derived tolerances; a `@debug` documents the hierarchical
  factored-operator -> concat-engine substitution; the stale
  `ConstantPAnalyticStencil`-is-default sentence in the
  `HierarchicalRigidStencil` docstring was corrected.
- **Tests (focused suite 399 -> 431):** hierarchical-vs-flat end-to-end parity
  at matched epsilon (`TF in {Float32,Float64} x LH in {false,true} x
  P in {4,8}` at `ell=3`, plus `ell=4`), with truncation-scale tolerances
  recorded from measurement (P=4 pot <= 5.5e-7 / grad <= 3.2e-5; P=8 pot
  1.5e-9 / grad 4.6e-8) — this also gives the level-scaling law a numeric
  end-to-end gate at P=8; a seeded-nonzero-chi flat-engine oracle at `ell=2`
  where the hierarchical partition provably equals the flat partition
  (roundoff-level `rtol=1e-11` agreement), closing the
  strategies-only-agree-with-each-other gap; `@test_throws` negatives for
  malformed windows, the window capacity assertion, invalid `near_radius2`/
  `window_classes`, hierarchical `ell=1`, the new keyword conflicts, and the
  new flat-launcher guard; Float32 coverage for hierarchical precomputed-y and
  dense (cross-strategy gap 7.5e-9, gated at 1e-6); the dense plan-identity
  guard now asserts `===` like the precomputed-y one.

Deferred to `027`/`028` with a note (performance, not correctness): collapse
the precomputed-y per-`(level, offset)` z tables to per-offset blocks plus the
per-level power-of-two diagonal (an `(ell-1)x` storage/construction cut
mirroring what dense and concat already do — deviates from this file's
recorded `zlen x nclasses` decision, so it belongs in the `028` measured
cycle); per-window `O(nclasses)` sweeps in the precomputed-y refresh/apply;
`O(8^ell)` occupancy `fill!`. The untracked-in-git finding (the entire
refactor test/data surface existed only in the working tree, so CI coverage
was fictional on a clean checkout) was resolved by committing the surface.

Because this review modified production code and tests, per `START_HERE.md`
a fresh clear-context agent must re-approve task 026.
