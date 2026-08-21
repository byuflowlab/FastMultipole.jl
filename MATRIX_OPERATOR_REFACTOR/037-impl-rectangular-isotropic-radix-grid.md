# 037 Rectangular Isotropic-Cell Radix Grid

## Status and Entry Gate

**DONE `2026-08-13`** (all five stages implemented, H200-validated, and
benchmarked; the 035-handoff expectation did not materialize — the
rectangular path ships as a validated opt-in, default stays cubic, verdict
in the Stage-5 work record). Clear-context approval pending.
Originally in progress `2026-08-13` (entry gate satisfied `2026-08-12`: 035 GO
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

**Default-flip pre-authorization (user, 2026-08-13, given before the job's
results were known):** if the rectangular grid performs within 10% of the
legacy cubic grid in ALL measured cases, AND outperforms legacy by MORE
than 10% on the higher-aspect-ratio (wake) cases, the shipped FLOWVPM
coupling default flips to `rectangular=true`. After 037 is Done and
clear-context approved, STOP (no further roadmap rows this session).

**Stage 5 result (job 13160439, H200, exit 0) — expectation NOT met;
default stays cubic.** All preflights green (incl. the FLOWVPM rectangular
Part A testsets on hardware), 033 refcheck PASSED, 10/10 rows gate-passing
with flat counters. U/J medians (cubic → rectangular):

| case | n | TF | cubic (ms) | rect (ms) | delta | rect ell_axes |
|---|---:|---|---:|---:|---|---|
| wake | 1e5 | F32 | 7.764 | 7.936 | +2.2% (slower) | (3,3,5) |
| wake | 1e5 | F64 | 14.411 | 14.896 | +3.4% (slower) | (3,3,5) |
| wake | 1e6 | F32 | 83.560 | 82.154 | −1.7% | (4,4,6) |
| wake | 1e6 | F64 | 178.975 | 175.316 | −2.0% | (4,4,6) |
| cube | 1e5 | F32 | 11.384 | 11.373 | neutral ✓ | (4,4,4) |

Errors comparable (wake 1e5 rect u=3.69e-4 vs 3.30e-4; 1e6 rect slightly
better); RK3 mirrors U/J. **Mechanism verification:** trimming worked as
designed — M2M+L2L isolated stages drop 2.14→1.62 ms (1e5) and 2.67→2.17
(1e6), n_nodes and (at 1e6) n_routes shrink — but the saving does not
reach the wall clock because the production pipeline already overlaps the
coarse far-field levels under the dominant nearfield (the same
overlap-over-crediting the 035 campaign documented for its cycle-2 B2M
model), and at 1e5 the flat-top root adds ~0.2 ms of M2L (n_routes
168200→186472). Root cause of the missed 11-23% band: (i) the AR=5 wake's
padded transverse extent needs ell_a = ell−2 (6.4 cells → 8), so only 1-2
coarse levels trim — and Stage 3 already gave the cubic path the level-0
trim for free; (ii) the 035 estimate priced the coarse-level launch floor
as critical-path time, which the overlap hides.

**Verdict per the pre-authorization (mechanical):** criterion (a) within
10% everywhere — PASS (worst +3.4%); criterion (b) >10% gain on
high-aspect — FAIL (best −2.0%). **The shipped default remains cubic
(`rectangular=false`); the rectangular path ships as a validated opt-in.**
Margins are far outside measurement noise (medians of 15 warmed reps;
criterion (b) misses by 5x). Where it should pay: aspect ratios well
beyond 5 (real rotor wakes), where more transverse levels trim; the
launch-floor share also grows as the nearfield shrinks (e.g. future
smaller-n or post-nearfield-optimization regimes). Recorded as guidance,
not a claim. Deferred: Stage 3b per-axis occupancy shrink (memory-only,
nothing binds); anisotropic >32:1 hierarchy (new theory).

**Task 037 status: DONE** (all five stages; production evidence recorded;
decision recorded here and as a 035 addendum). Awaiting clear-context
approval.

## Clear-Context Approval

**Date:** 2026-08-13. **Reviewing agent:** clear-context approval subagent
(Claude Fable 5), per the `START_HERE.md` clear-context protocol.

**Checked:** `START_HERE.md`, this task file in full, the design record
`037-implementation-plan.md`, the Stage 1-3 diffs (`7b9c69d`, `d73da44` +
the three test-scaffolding fixes, `c874818`), the Stage-5 prereg/verdict
commits (`dde3aa3`, `1b19c10`, `5bd5dca`), FLOWVPM `gpu-full` `98d4c45`
(`rectangular=false` default confirmed in `src/FLOWVPM_fmm_radix.jl`), the
Stage-5 artifacts in `data/flowvpm_gpu_campaign/`, and the 035 addendum.

**Verified:**

- *Correctness/tests.* `test/radix_trimming_test.jl` does what the work
  record claims: exhaustive exact-once audit (ordered occupied-leaf-pair
  hit matrix built from direct pairs plus production
  `build_hierarchical_routes_window!` routes expanded to leaf descendants,
  every entry asserted == 1) on {(4,2,2),(3,3,1),(4,4,2),(3,3,3)} at
  literature P=4, under default and uniform-q schedules, plus flat-oracle,
  cross-strategy parity, cap-guard, schedule-anchoring, and multi-root
  sections. The bitwise cube-regression gate exists in
  `test/radix_fmm_integration_test.jl` (vector `(L,L,L)` vs scalar `L`:
  geometry/capacity/route-count/direct-buffer equality and bitwise `fmm!`
  outputs at P=4 and order 8). Both host suites re-run green locally
  (trimming 122/122; integration 89/89 + rectangular bounds 63/63,
  4 threads). Stage 3 source (`_radix_root_level`, `_radix_flat_top_count`,
  `_rigid_flat_top_tables`, re-anchored class metadata) matches the design
  record §1/§3; the flat-top table reuses the existing `level_class_of`
  mask mechanism with window kernels unchanged, as designed.
- *H200 evidence.* `fm035-13160439.out` (exit clean, archived): device
  interface 1333/1333, lifecycle 227/227, FLOWVPM rectangular Part A
  testsets + Part B device suites on hardware, 033 refcheck all OK — this
  also substitutes for the unarchived intermediate job logs
  (`fm037t-13160006`/`13160427`, cluster-only; the counts recorded in the
  work record match the suites re-run here). Minor note, non-blocking.
- *Stage-5 verdict.* `fm037_stage5.csv` sha256 matches
  (`6c40437e…`); `gate_pass=true` and `counters_flat=true` on all 10 rows;
  every number in the verdict table recomputed and confirmed, including
  `ell_axes` (3,3,5)/(4,4,6)/(4,4,4). Verdict arithmetic is mechanical per
  the pre-authorization: (a) within 10% everywhere PASS (worst +3.4%);
  (b) >10% on high-aspect FAIL (best −2.0%) → default stays cubic,
  rectangular ships opt-in. Root-cause story is supported by the CSV:
  isolated stage sums exceed the eval wall (9.26 ms vs 7.37 ms at wake 1e5
  F32), confirming coarse-level overlap; M2M+L2L 2.14→1.62 / 2.67→2.17 ms;
  n_routes 168200→186472 with m2l +0.16 ms; transverse `ell_a = ell−2`
  consistent with the nodes_per_level columns. Honest reporting of the
  missed 035 expectation is plain and quantified.
- *035 addendum.* Purely additive (no removed lines in `5bd5dca`'s diff of
  the 035 file) and consistent with this record.

**Verdict: APPROVED.** No significant issues; consistency, correctness,
performance evidence, robustness, minimal invasiveness, and readability
all satisfied.

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
