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
files parse and skip cleanly. **Device tests not yet run on hardware — one
FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 H200 run (interface, lifecycle, graph,
integration) gates Stage 3.**

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
