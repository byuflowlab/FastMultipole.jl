# 037 Rectangular Isotropic-Cell Radix Grid

## Status and Entry Gate

**In progress `2026-08-13`** (entry gate satisfied `2026-08-12`: 035 GO
handoff below + 036 Done and clear-context approved). Design record complete:
`037-implementation-plan.md` — virtual-cube embedding with per-axis leaf
depths + construction-time active-level trimming with a flat-top root level;
theory verdict: no new exact-once/error-bound theory required (boundary case
of the 025 proof; verdict argument recorded in the design record §1), with
the anisotropic->32:1-aspect generalization explicitly flagged as future
theory if ever needed. Five implementation stages; work record below.

Entry gate: `035` must complete the wake-versus-cube profile and recommend
implementation, with an expected end-to-end U/J-solve gain of at least 5% (or
an explicitly documented reason to proceed below that threshold). The
Integration Phase milestone review `036` must also be complete.

**035 handoff (2026-08-12, campaign concluded): GO.** The final wake n=1e5
solve is 7.98 ms (F32, shipped P5 defaults, ℓ=5 with only ~32 transverse-
plane occupied nodes above the leaf level); the ~2 spare transverse coarse
levels cost ~0.9-1.8 ms of M2M/L2L/launch floor = **11-23% expected
end-to-end U/J gain at n=1e5** (F64 similar; <2% at n=1e6 where the
nearfield dominates). Occupancy compaction already absorbs the empty-cell
cost (B2M/nearfield scale with occupied cells), so the win is coarse-level
count, not occupancy — size the acceptance case accordingly. Full profile
evidence: 035 Final Report §3 and the cycle-3D stage tables
(`fm035_cycle3d.csv`).

## Work Record

### 2026-08-13 — Stage 1: host rectangular geometry contract (implemented)

Per `037-implementation-plan.md` §3 Stage 1. `RadixFMMCache` gains
`ell_axes::SVector{3,Int}` + `box_extent::SVector{3,TF}` (cubic caches carry
`(ell,ell,ell)`; `ell`/`h0` keep their virtual-cube meaning);
`_resolve_radix_ell_axes` resolves scalar (legacy, bit-identical) or
3-vector `box_size` (`Δ = 2h0/2^ell`, per-axis `ell_a = clamp(⌈log2(L_a/Δ)⌉,
0, ell)` with an fp snap-up guard that never shrinks the extent);
`_radix_level_node_capacity` generalized to per-axis products
(overflow-guarded, 2-arg form delegates); `_assert_radix_positions_in_box`
takes per-axis bounds; `_radix_cell_at` per-dimension sizes. Guards: device +
vector bounds and rectangular `recenter!` both throw loudly until Stage 2
lifts them. Only non-plan touch: the device build's cache-construction call
passes the cubic values for the two new fields (struct arity).

Tests: new "rectangular radix geometry" (27 asserts) + "radix rectangular
bounds" (46) testsets — cube-regression gate is **bitwise** (vector
`(L,L,L)` ≡ scalar `L` in geometry, capacities, route/direct counts, and
`fmm!` outputs) at literature P=4 and P=9(order 8); elongated 4:1:1 cloud
resolves `(4,2,2)`, capacity 256/295, passes the sampled-direct accuracy
gates at both orders; snap-up, out-of-box throw, and refusal paths covered.
Full host suites green (radix integration 89, timestepping 51k,
hierarchical host M2L 877, dense/precomputed-y M2L, device_system_interface
35.4k). Note for Stage 2: host route-buffer prefixes are windowed scratch —
route parity must compare telemetry counts, not buffers. Committed
`7b9c69d`.

### 2026-08-13 — Stage 2: device parity of the rectangular contract (implemented)

Per plan §3 Stage 2, committed `d73da44`. `_cuda_radix_keys_checked_kernel!`
checks per-axis `box_extent` (quantization unchanged);
`_radix_cache_device_build` carries `ell_axes`/`box_extent` and sizes device
stage groups with the per-axis capacities; the Stage-1 device+vector-bounds
guard is removed; preflight needed no change (capacities arrive generalized;
cubic `cell_at`/occupancy terms intentionally stay cubic supersets).
`recenter!` now preserves rectangularity for derived bounds (per-axis tight
extents, same margin convention; cubic path verbatim; explicit bounds are
caller-final — scalar rebuilds cubic by design).

Tests: host rectangular `recenter!` section (61 asserts, integration suite);
device additions gated behind `FASTMULTIPOLE_REQUIRE_CUDA_TESTS` — interface
section 10 (parity F64/F32 × P∈(3,8) with telemetry-count comparison, 023
counter contract + stable warmed allocation over 3 steps, per-axis device
oob, rectangular device recenter), lifecycle rectangular case, and a
vector-bounds graph-capture testset. Local: all host suites green; CUDA
files parse and skip cleanly.

**H200 device validation PASSED (job 13160006, 2026-08-13, exit 0):**
interface 1317/1317, lifecycle 227/227 (incl. the rectangular device case),
rectangular vector-bounds graph capture 32/32, cached-window/graph parity
112/112, integration + clustering suites green. Three test-scaffolding
fixes were needed en route (jobs 13159955/13159996/13160001 — all in the
new CUDA-gated tests, no production change): route_capacity is
residency-specific (host K=4 vs device K=256) so the cross-residency
equality was dropped; the scalar lifecycle test system needed explicit
`lamb_helmholtz=false` and a `has_vector_potential` overload. Stage 3
unblocked.

### 2026-08-13 — Stage 3: active-level trimming with flat-top root (implemented)

Per plan §3 Stage 3. `_radix_root_level` (cap-guarded, `RADIX_FLAT_TOP_CLASS_CAP
= 4096`), `_rigid_flat_top_tables` emitting through the existing class-mask
mechanism (window builder + CUDA window kernels unchanged), scheduled
tables/metadata/classifier re-anchored to active levels `first:ell`
(legacy `2:ell` schedules accepted everywhere via identity-on-cubic
slicing), node build + stage groups + the four CUDA M2L loops trimmed to
`R:ell`, multi-root tree edges (`n_edges = n_nodes - n_root_nodes`; the
exhaustive `n_nodes - 1` audit found 6 sites: 2 fixed, 2 benign capacity
bounds, 2 untrimmed one-shot oracle paths by design). `first_m2l_level` is
construction-fixed on both hierarchical contexts (graph capture + 029
window cache safe). **Bitwise cube-regression gate PASSED** (old-vs-new
probe: routes, direct pairs, telemetry, outputs identical; only `n_nodes`
drops by 1 — the never-consumed virtual level-0 root is trimmed on cubic
caches too). New `test/radix_trimming_test.jl` (122 asserts, in runtests):
exhaustive exact-once coverage on {(4,2,2),(3,3,1),(4,4,2),(3,3,3)},
flat-vs-hierarchical oracle, cross-strategy parity, multi-root invariants,
cap guard, schedule anchoring — all literature P=4. All host suites green
locally. Documented deviations: leaf-radius `L_allnear` (conservative);
legacy-anchored schedules accepted on all caches; one-sided root classifier
gate; two test-expectation updates (max_nodes 295→292; cubic node-index
shift in the ell=2 oracle alignment). Open H200 risks recorded in the
Stage-3 report (device flat-top path, trimmed graph capture, counter
contract) — gated by the device job before Stage 4 lands.

**Stage 3 H200 device validation PASSED (job 13160427, 2026-08-13, exit 0):**
trimming 122/122 on hardware, interface 1333/1333 (incl. the device flat-top
path, multi-root tree-route kernel, and trimmed graph capture/replay),
lifecycle 227/227, all remaining suites green. Committed `c874818`.

### 2026-08-13 — Stage 4: FLOWVPM rectangular coupling mode (implemented)

FLOWVPM `gpu-full` commit `98d4c45`, coupling file + Part A tests only.
`RadixFMMSettings.rectangular::Bool=false`; rectangular `_radix_derive_bounds`
keeps per-axis tight extents (same per-face padding and `4σ_max` degenerate
floor, applied per axis); vector `box_size` passes through explicit bounds
(user-owned, no recenter); automatic recenter preserves rectangularity.
**Auto-geometry provably unchanged** (long axis equals the cubic derived
side ⇒ identical `ell`, `q`, leaf width between modes — asserted by test).
Part A green locally: +27 asserts (rectangular bounds 22, rectangular
recenter 5), existing testsets unchanged; wake u_rel_rms 1.87e-4
(cubic 1.85e-4). Part B device mirror deferred to the Stage-5 job.

### 2026-08-13 — Stage 5: pre-registered comparison (submitted)

Harness: `benchmark_035_gpu.jl` gains a `rectangular` case token and
`rectangular`/`ell_axes` CSV columns (new output file `fm037_stage5.csv`).
Pre-registered grid `scripts/fm037_cases_stage5.txt` (10 rows, committed
`dde3aa3` before the run): wake 1e5/1e6 × F32/F64 cubic-vs-rectangular at
the shipped 035 defaults (identical (ℓ,q,leaf width) between arms —
attribution is trimming + per-axis capacity only), RK3 at 1e5, cube 1e5
neutrality control. Pre-registered expectation (035 handoff): wake 1e5 gain
in the 11-23% band, wake 1e6 <2%, cube neutral. Job 13160439 (full
preflights incl. FLOWVPM Part A rectangular testsets + Part B, 033 refcheck
at shipped cubic defaults).

## Objective

Add an optional rectangular radix-grid path for elongated domains while
retaining approximately cubic *physical* cells. The wake should use unequal
cell counts along its axes rather than embedding the entire domain in a cubic
root box. The existing cubic path remains available and unchanged as the
default fallback.

## Ownership and scope

This item owns FastMultipole production changes only. `035` owns the benchmark
campaign, parameter selection, and performance verdict; FLOWVPM changes remain
on its `gpu-full` branch and are limited to consuming the supported FastMultipole
configuration.

The first design pass should target a fixed-resolution rectangular leaf grid.
Do not generalize the full hierarchy until measurements show that leaf-grid
support alone is insufficient. If the selected design requires a new exact-once
hierarchical interaction proof or error bound, add that theory work before the
corresponding production implementation.

## High-level deliverables

- Rectangular root geometry with per-axis cell counts and approximately equal
  physical cell widths.
- Generalized coordinate quantization and stable radix/Morton-compatible keys.
- Compact occupied-cell metadata without cubic dense-grid assumptions.
- Rectangular-grid interaction routing with cubic-cell error geometry.
- CUDA/device-resident construction and recurring refresh with the existing
  capacity and zero-allocation contract.
- Host/device parity, cube-regression, wake accuracy, and sampled-direct
  velocity-error tests.
- End-to-end H200 comparison against the cubic wake baseline, including grid
  refresh, routing, M2L/M2M/L2L, nearfield, memory, and allocation costs.

## Design constraints

- Physical cells should remain approximately cubic; coordinate scaling must not
  silently change kernel distances or the error model.
- Source and target grids used by one stencil must share the same physical
  domain and quantization convention.
- Preserve the legacy cubic path and public API compatibility.
- Preserve fixed-domain cache behavior, capacity sizing, and recurring
  device-residency guarantees.
- Do not claim a speedup from nominal cell-count reduction alone; report the
  complete resident step and the sampled-direct accuracy gate.

## Acceptance

The rectangular path is eligible for production only if it passes the existing
correctness, accuracy, allocation, CPU-compatibility, and CUDA lifecycle gates
and demonstrates a measured end-to-end benefit on the wake. The final decision
and comparison belong in `035`; this item records the implementation evidence
and any limitations or deferred hierarchy work.
