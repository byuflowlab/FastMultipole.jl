# 037 Design Record — Rectangular Isotropic-Cell Radix Grid

Produced 2026-08-13 by the 037 design agent (Claude Fable 5) after a full
exploration of the radix grid, routing, and CUDA lifecycle code; saved
verbatim as the implementation plan of record. Referenced from the 037 task
file's work record.

## 0. Summary of the chosen design

**Virtual-cube embedding with per-axis leaf depths, plus construction-time
active-level trimming with a flat-top root level.**

The rectangular grid is specified by per-axis leaf depths
`ell_axes = (ell_x, ell_y, ell_z)` with `ell = maximum(ell_axes)` and a
single shared cell width `Δ = 2h0 / 2^ell` (`h0` remains the *virtual* cubic
half-width). The physical domain is the rectangular box
`x_min .+ Δ .* (2 .^ ell_axes)`; cells are **exactly** cubic at every
retained level, per-axis cell counts differ. Morton keying, sorting,
run-length compression, node building (`key >> 3`), parent resolution,
`morton_decode`, occupancy lookups, the window/compact route kernels, and
all operator math are **unchanged** — rectangularity enters only through
(a) the contractual per-axis in-box assertion, (b) capacity sizing from
per-axis counts, and (c) a construction-time active level range `[R, ell]`
with `R = max(ell - minimum(ell_axes), L_allnear)`, where level `R` runs a
"flat-top" M2L table (every non-near root-grid offset) and levels `R+1:ell`
run the existing task-025 V-list transition tables.

For equal per-axis depths this degenerates to `R = L_allnear = 1` with an
*empty* flat-top table (level-1 pairs are all near), i.e. bitwise the legacy
cubic route set — which is what makes the cube-regression gate exact rather
than approximate.

The performance mechanism matches the 035 handoff: the number of coarse
levels above the leaf drops from `ell` to `min(ell_axes) (+1)`, removing ~2
levels of M2M/L2L/route-window launch floor on the wake, while occupancy
compaction (already in place) continues to absorb empty-cell costs.

## 1. Theory verdict (required statement)

**No new exact-once or error-bound theory is required. Provably:**

1. **Error geometry.** Every retained level has physically cubic cells of
   width `Δ·2^(ell-L)` (the hierarchy stops at `R = ell - min(ell_axes)`
   *before* any axis saturates, so no anisotropic cells ever exist). Node
   centers, kernel distances, and the per-offset analytic bound
   (`constant_p_stencil_bound`, src/interaction_list_batched.jl:456) are
   computed exactly as today from `(x_min, h0, ell)`; there is no coordinate
   scaling. The flat-top classes at level `R` are classified by the *same*
   analytic classifier at root cell width — the level-scaling of the bound
   is exactly the task-002/025 `2^(ell-L)` rescaling already used by
   `_hierarchical_class_metadata`'s `effective_offsets = scale * o`
   (src/interaction_list_batched.jl:340-362). The construction gate
   `_verify_hierarchical_classifier!` (src/interaction_list_batched.jl:307)
   extends to verify the root level with zero new math.

2. **Exact-once coverage.** The task-025 proof assigns each well-separated
   leaf pair to the unique level at which its ancestor offset first leaves
   the (non-increasing-with-depth) near ball; its only use of the topmost
   levels is the base case "at level 1 every pair offset is near." The
   rectangular scheme replaces that base case with "at level `R`, every pair
   whose offset is non-near is emitted by the flat-top table" — which is the
   degenerate transition table `q_parent = ∞` bounded by the root grid box.
   The induction is otherwise untouched: ancestor offsets are monotone under
   coordinate halving regardless of per-axis counts (halving is per-axis
   independent), the near-radius schedule remains non-increasing over
   `R:ell`, and occupancy restriction to a sub-lattice cannot create or
   destroy coverage (lookups already return 0 for absent coordinates). This
   is a *boundary case* of the existing proof, not new theory. It should be
   recorded as a short lemma in this record and enforced by an exhaustive
   small-grid coverage test (Stage 3), but it does not warrant a theory
   task. The one condition to validate at construction: `R ≥ L_allnear` is
   not required — `R` may be any level in `[L_allnear, ell-0]`; what is
   required is that the flat-top enumeration at `R` covers the full
   root-grid offset range (a bounded box, enumerated at construction) and
   the schedule levels are re-anchored to `R:ell`.

**Explicit flag:** if a future pass wants coarse levels *above* `R` with
anisotropic (saturated-axis) cells to shrink the flat-top class count on
extreme aspect ratios (>~32:1), *that* would need a non-cubic-cell error
bound — new theory. The design below caps root aspect instead (guard in
Stage 3) and defers that case.

## 2. What the exploration established (anchor map)

- **Grid construction/keying**: `RadixGrid`/`_radix_grid` and all key math
  in src/tree_batched.jl (`radix_cell_coord` :203, `morton_key` :216,
  `morton_decode` :235, in-place refresh `update_radix_grid!` :456, node
  rebuild `_refresh_radix_nodes!` :370 — loops `level in 0:ell`,
  `key >> shift`), types `RadixGrid` (src/containers.jl:244),
  `DeviceRadixGrid` (:283).
- **Cache + invariant contract**: `RadixFMMCache` struct
  src/containers.jl:2041 (fields `ell, x_min, h0`); constructor
  src/translate_batched_resident.jl:1935 (bounds handling :2009-2021,
  capacities :2049-2058, policy :2034); `_assert_radix_positions_in_box`
  :1721; refresh `update_radix_state!` :2466; `recenter!` :2362 (cubes the
  derived bounds at :2377).
- **Capacity**: `_radix_level_node_capacity` src/translate_batched.jl:3953
  (`min(1 << 3L, max_cells)` — cubic); consumers resident:2049-2052 and
  translate_batched.jl:4519-4527 (workspace groups per level `(ell-1):-1:0`
  / `1:ell`).
- **Routing**: scheduled tables `_hierarchical_scheduled_tables`
  src/interaction_list_batched.jl:262 (levels hardwired `2:ell`), transition
  tables `_rigid_transition_tables` :212, class metadata :340, window
  builder :387 (reads `level_class_of[phase,k,L+1]` — mask-driven,
  level-agnostic), direct pairs :426 (leaf-only), flat-path
  `build_radix_routes!` :771; host route loop `2:ell` at resident:2294.
- **CUDA**: keys kernel + rectangular check point
  `_cuda_radix_keys_checked_kernel!` src/translate_batched_cuda.jl:133;
  node/parent/child kernels :326-450 (pure key shifts — unchanged); device
  build `_radix_cache_device_build` :5300; in-place grid update :5581; state
  update `update_cuda_radix_state!` :5754; stage groups :5145; occupancy
  scatter/window/compact/direct kernels :5968-6110 (all use cubic
  `linear = x + G(y + Gz)` — remains *correct* on a rectangular sub-lattice,
  only oversized); M2L level loops `for L in 2:ell` at :6345, :6889, :6965,
  :7009; dense per-level scale columns `D x (ell-1)` :4132; memory preflight
  :4065/4113; graph capture :4800-4885 (level structure must be
  construction-fixed — the trimming design satisfies this).
- **Consumer**: ../FLOWVPM.jl/src/FLOWVPM_fmm_radix.jl —
  `_radix_derive_bounds` :257 (**cubes** the tight bounds),
  `_radix_auto_geometry` :273, cache construction :339 (`bounds=(x_min, L)`
  scalar).
- **Key correctness observation**: because occupied coordinates on a
  rectangular domain simply never exceed `2^ell_a` per axis, *every*
  existing lookup (`_radix_cell_at`, `_hierarchical_node_lookup`, `node_at`
  scatter, window bounds checks `0 ≤ t < G`) is already correct on the
  sub-lattice — cubic `G` bounds are supersets and unoccupied coords return
  0. This is what makes Stage 1-2 small.

## 3. Staged implementation plan

### Stage 1 — Rectangular geometry contract, host path (no trimming)

*Lands green on CPU-only CI; pure additive.*

**Files / changes**

- src/containers.jl:2041 `RadixFMMCache`: add fields
  ```julia
  ell_axes::SVector{3,Int}      # per-axis leaf depths; cubic caches: (ell, ell, ell)
  box_extent::SVector{3,TF}     # physical extents Δ .* 2 .^ ell_axes (redundant, precomputed)
  ```
  Keep `ell`, `h0` with their existing (virtual-cube) meaning so every
  downstream consumer is untouched.
- src/translate_batched_resident.jl:1935 constructor:
  - Accept `bounds=(x_min, box_size)` with
    `box_size::Union{Real,SVector{3},NTuple{3}}`. Scalar → legacy cubic
    (`ell_axes = (ell,ell,ell)`), identical numerics.
  - Vector extents: `Δ = maximum(L)/2^ell`;
    `ell_a = clamp(ceil(Int, log2(L_a/Δ)), 0, ell)`; snap
    `box_extent = Δ .* 2 .^ ell_axes` (pad up, never shrink);
    `h0 = maximum(L)/2` unchanged. New helper:
    ```julia
    _resolve_radix_ell_axes(box_size, ell::Int, ::Type{TF}) -> (ell_axes::SVector{3,Int}, h0::TF, box_extent::SVector{3,TF})
    ```
  - Capacity: generalize src/translate_batched.jl:3953 to
    ```julia
    _radix_level_node_capacity(level::Integer, ell_axes::SVector{3,Int}, ell::Integer, max_cells::Integer)
    # = min(prod(1 << max(ell_a - ell + level, 0)), max_cells), overflow-guarded
    ```
    keeping the old 2-arg method delegating with `ell_axes=(L,L,L)`. Update
    call sites resident:2049-2052 and translate_batched.jl:4519-4527.
    `max_cells = min(prod-of-leaf-counts, maxn)`.
- src/translate_batched_resident.jl:1721 `_assert_radix_positions_in_box`:
  new method taking `box_extent::SVector{3,TF}` (per-axis upper bounds);
  call site :2479 passes `cache.box_extent`.
- `_radix_cell_at` (src/interaction_list_batched.jl:493) has a latent
  cubic-size assumption (`G = size(cell_at, 1)` checks all three axes) —
  fix to per-dimension `size(cell_at, d)` now (harmless for cubic, required
  if cell_at is ever sized rectangular).

**Untouched**: grid keying/sort/compress/node build, all route generation,
all operator pipelines, all CUDA code, flat-policy path, public defaults
(scalar bounds remain the default; derived bounds remain cubic in this
stage).

**Tests** (extend test/radix_fmm_integration_test.jl,
test/radix_grid_clustering_test.jl):
- Cube regression: `bounds=(x_min, (L,L,L))` cache vs `bounds=(x_min, L)`
  cache — identical `ell_axes`, routes, direct pairs, and outputs bitwise,
  at `expansion_order=3` (literature P=4) and one higher order.
- Rectangular host cache on an elongated cloud: capacity accounting
  (`max_cells`, `max_nodes` per-axis products), snap-up of non-power-of-two
  aspect, per-axis out-of-box `ArgumentError` (body inside the virtual cube
  but outside the rectangular box must throw).
- U/J accuracy vs sampled direct on the elongated cloud, same gate as the
  existing suite.

**Risks**: capacity formula overflow at large `ell_axes` sums (guard as the
existing `3*level >= 62` branch does); `max_level_nodes` (resident:2051)
must use the generalized capacity or route windows over-allocate (benign) /
under-allocate (assertion at resident window builder :411 catches it).

### Stage 2 — Device parity of the rectangular contract

**Files / changes**

- src/translate_batched_cuda.jl:133 `_cuda_radix_keys_checked_kernel!`:
  take per-axis extents (three scalars or an `SVector`) instead of
  `two_h0`; key quantization math unchanged. Thread `box_extent` through
  `_radix_cache_device_build` (:5300) and `_cuda_update_radix_grid_in_place!`
  (:5581).
- Device memory preflight (:4065, :4113): use the generalized capacities.
- `recenter!` (resident:2362): when `cache.ell_axes` is non-uniform, derive
  per-axis tight extents from `_recenter_union_bounds` (already per-axis,
  :2428) and rebuild with vector bounds preserving `ell_axes` resolution
  semantics; cubic caches keep the current cube derivation verbatim.
- No kernel indexing changes anywhere (see §2 key observation).

**Untouched**: sort (counting-sort histogram stays `1 << 3ell`), node
kernels, occupancy, routes, graph capture, 023 transfer counters.

**Tests** (test/cuda_radix_lifecycle_test.jl,
test/cuda_radix_interface_test.jl, test/device_system_interface_test.jl):
- Host/device parity of the rectangular cache (outputs + route/direct
  counts) at P=4, F32 and F64.
- Zero-allocation recurring refresh and counter contract on a rectangular
  device cache (reuse the existing lifecycle harness assertions).
- Rectangular oob flag on device; `recenter!` on a rectangular device cache
  preserves rectangularity and step counts restart.
- Graph-capture eligibility unchanged (existing cuda_radix_graph_test.jl
  case duplicated with vector bounds).

**Risks**: `recenter!` transient 2x memory already documented; nothing new.
The device counting sort keys span `2^{3ell}` even though only the
sub-lattice is occupied — unchanged behavior, no correctness risk.

### Stage 3 — Active-level trimming with flat-top root (the performance win)

**Design**: construction computes
```julia
_radix_root_level(ell_axes, ell, coarse_q) -> R   # max(ell - minimum(ell_axes), L_allnear)
```
with `L_allnear` = largest `L` such that `sum((N_a(L)-1)^2) ≤ coarse_q`
(cubic: 1). Guard: if the flat-top class count at `ell - min(ell_axes)`
exceeds a cap (proposal: 4096 offsets, matching the device window default),
lower `R` until it fits; if `R` reaches `L_allnear` the flat-top table is
empty and behavior is the legacy full hierarchy over `[L_allnear, ell]`.

**Files / changes**

- src/interaction_list_batched.jl:
  - New `_rigid_flat_top_tables(q_top, root_counts::SVector{3,Int})`:
    enumerate `{o : |o|² > q_top, |o_a| < N_a(R)}`, all 8 phases admitted.
    Emitted through the *existing* `level_class_of` mask mechanism — the
    window builder (:387) and CUDA window kernels (:5984/:6029) need
    **zero changes**.
  - `_hierarchical_scheduled_tables` (:262): signature gains
    `(policy, ell, R, root_counts)`; builds level tables for `R:ell`
    (flat-top at `R`, transitions `R+1:ell`); schedule length/validation
    re-anchored to levels `R:ell` (public `level_radii2` semantics: "coarse
    to fine over the active M2L levels"; legacy `2:ell` length still
    accepted for cubic caches).
  - `_hierarchical_class_metadata` (:340): loop `2:ell` → `R:ell` (skip if
    flat-top empty at `R`); `nclasses` shrinks → smaller dense-M2L payload.
  - `_verify_hierarchical_classifier!` (:307): additionally verify the
    root-level classifier boundary (same enumeration at root extent).
- src/tree_batched.jl:370 `_refresh_radix_nodes!`: new arg
  `first_level::Int` (loop `R:ell`, `level_offsets[1:R+1] .= 0`,
  `parent_index = 0` at level `R`). Same for the CUDA node build (level
  kernels take `max_level`; add `min_level`;
  translate_batched_cuda.jl:326-450, :558-575, :5678-5701).
- Tree edges: `_refresh_radix_tree_routes!` (resident:2260) and
  `_cuda_tree_routes_kernel!` (cuda:452) currently assume
  `node ≥ 2 ⇒ parent exists`. Generalize: iterate nodes
  `level_offsets[R+2]+1 : n_nodes` (children of retained levels),
  `n_edges = n_nodes - n_root_nodes`. **Audit every `n_nodes - 1` use**
  (resident:2534, cuda:5810, state counts consumers).
- Stage groups: workspace construction (translate_batched.jl:4519-4527) and
  refresh (resident:2191, cuda:5145) build M2M parent levels `(ell-1):-1:R`
  and L2L child levels `R+1:ell` only.
- M2L level loops: host resident:2294 (`2:ell` → active levels with
  nonempty class lists), CUDA :6345, :6889, :6965, :7009; store
  `first_m2l_level::Int` (+ per-level class ranges) on
  `HostHierarchicalM2LContext`/`DeviceHierarchicalM2LContext`
  (containers.jl:511/:544) at construction — step-invariant, so CUDA-graph
  capture and the 029 window cache are unaffected.
- Dense plan per-level scale columns (cuda:4132): size by active level
  count.

**Untouched**: leaf direct-pair generation, nearfield binning, B2M/L2B,
sort, occupancy lookup structure, flat `ConstantPAnalyticStencil` policy
(leaf-only; rectangular caches may use it, no trimming applies),
`recenter!` semantics.

**Tests**:
- **Exhaustive exact-once coverage**: small rectangular grids (e.g.
  `ell_axes` ∈ {(4,2,2),(3,3,1),(4,4,2)}), random sparse occupancy — assert
  every occupied leaf pair appears exactly once across
  `direct ∪ (M2L class at exactly one level)`; run for uniform q and a
  level schedule; compare summed potentials against the flat analytic
  oracle within the classifier tolerance.
- **Cube regression (exact)**: cubic cache with trimming = legacy route set
  and outputs bitwise (`R = 1`, empty flat-top).
- Host/device parity of the trimmed rectangular cache; zero-allocation
  refresh; graph capture + replay across occupancy epochs; P=4 coverage
  throughout; hierarchical telemetry (`routes_per_level`) consistency.
- 033 wake reference check: rectangular wake cache vs the sha256-checksummed
  033 references (scripts/benchmark_033_common.jl harness) at shipped
  defaults, plus the sampled-direct 1e-3 velocity gate.

**Risks** (the load-bearing stage):
1. Multi-root parent bookkeeping — most regression-prone; the `n_nodes - 1`
   audit must be exhaustive (grep across resident/cuda/tests).
2. `level_offsets` zero-prefix below `R`: any consumer assuming node 1 is
   *the* root (e.g. one-shot `host_resident_radix_grid`, resident:8 —
   deliberately left untrimmed as the oracle; keep cache-owned grids the
   only trimmed ones).
3. Push-offset union growth from flat-top offsets enlarges `class_of`
   (8 × noffsets × (ell+1)) and per-window flag buffers; cap guard bounds
   it; assert the window capacity math (resident:2053) uses the enlarged
   union.
4. Schedule semantics change for `level_radii2` — document carefully;
   validate both lengths; keep cubic default behavior byte-identical.
5. The 029 cached-window epoch logic (cuda:5869) must treat
   `first_m2l_level` as construction-fixed — it is, by design.

### Stage 3b (optional, measured) — Per-axis dense occupancy and cell_at sizing

Shrink `RadixLevelOccupancy` (tree_batched.jl:505) and `node_at`
(cuda:4113) from `8^L` to `prod(N_a(L))` per level, changing the linear
index to `x + Nx(y + Ny z)` in: tree_batched.jl:544,
interaction_list_batched.jl:364, cuda:5968/:5984/:6029/:6066/:6089. Only
worth it when the cubic dense budget forces the Morton-search fallback
(`ell > dense_occupancy_max_ell` on the long axis); measure first.
Cube-parity: identical indices when counts are equal. This is deliberately
separated because it touches five device kernels for a memory-only win.

### Stage 4 — FLOWVPM consumer wiring (coupling surface only; 035 owns the campaign)

- ../FLOWVPM.jl/src/FLOWVPM_fmm_radix.jl: `_radix_derive_bounds` (:257)
  gains a rectangular mode (per-axis tight extents + per-face padding, no
  cubing); `_radix_auto_geometry` (:273) unchanged (`L = max extent` drives
  `ell` and q via σ-adequacy — the leaf width `h_leaf = L/2^ell` is what
  the gate constrains and it is unchanged); cache construction (:339)
  passes vector bounds. Off by default behind a new `rectangular::Bool=false`
  setting so the legacy path is preserved; automatic `recenter!` keeps
  rectangular bounds.
- Test: FLOWVPM-side smoke via the existing radix integration test pattern
  (fmm-radix settings round-trip), plus the coupling's oob-retry path with
  rectangular bounds.

### Stage 5 — H200 comparison (acceptance evidence for 035)

Extend MATRIX_OPERATOR_REFACTOR/scripts/benchmark_035_gpu.jl (or a
`benchmark_037_gpu.jl` sibling) to run the wake n=1e5 (and n=1e6 control)
at shipped defaults, cubic vs rectangular, reporting per complete resident
step: grid refresh (023 stage timers, `update_stage_ns`), route
generation/window cache, M2M/L2L/M2L per-level (`m2l_level_ns`), nearfield,
finalize; plus persistent device memory (preflight report), per-step
allocations (must be zero), transfer counters, and the sampled-direct
velocity gate + 033 refcheck on both paths. Acceptance per the task file:
measured end-to-end wake benefit (target band 11-23% at n=1e5 from the 035
handoff; <2% expected at n=1e6 — report both), decision recorded in 035.

## 4. New public surface (complete list)

- `RadixFMMCache(...; bounds=(x_min, box_size))` where `box_size` may be a
  scalar (legacy, default) or 3-vector (rectangular path).
- `cache.ell_axes`, `cache.box_extent` (readable).
- `level_radii2` documented over active M2L levels for rectangular caches.
- Everything else — `fmm!`, `recenter!`, `update_radix_state!`, options,
  policies, traits — unchanged.

## Critical Files for Implementation

- src/translate_batched_resident.jl (cache constructor :1935, bounds :2009,
  capacities :2049, box assertion :1721, refresh :2466, recenter :2362)
- src/interaction_list_batched.jl (scheduled/transition/flat-top tables
  :212-:362, classifier gate :307, window builder :387)
- src/tree_batched.jl (keying :203-:247, node rebuild :370, occupancy
  :505-:561)
- src/translate_batched_cuda.jl (keys kernel :133, node kernels :326-:450,
  device build :5300, state update :5754, level loops
  :6345/:6889/:6965/:7009)
- src/containers.jl (RadixFMMCache :2041, DeviceRadixGrid :283,
  hierarchical contexts :511/:544)
