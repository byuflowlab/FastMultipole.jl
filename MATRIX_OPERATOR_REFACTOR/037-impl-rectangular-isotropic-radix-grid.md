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
