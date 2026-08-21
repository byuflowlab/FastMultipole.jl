# 043 Theory: Partitioned-Tree Multi-GPU Decomposition

## Status and Entry Gate

Staged `2026-08-11` at the closure of `029`. Blocked by `042` (Adaptive
Octree Phase milestone): the decomposition must be derived against the
final (adaptive) tree machinery. Derivation row: artifacts under `theory/`,
`scripts/`, and `data/` only; no production `src/` changes.

## Objective

Derive a partitioned-tree multi-GPU decomposition of the device-resident
lifecycle that removes the replicated-work floor which falsified the `029`
mirrored-tree scheme, targeting `<= 1 ms` (goal) / `<= 2 ms` (win) per
resident step for the fixed 1M-body literature-`P=4` workload on up to
8 H200s.

## Evidence Base (read the `029` task file P2 verdict + Closure sections; do not re-derive)

- Mirrored 2-GPU: 3.993 ms, 58.3% efficiency vs the 4.657 ms single-GPU
  record; zero-comm wall floor ~3.8 ms from replicated refresh (0.51 ms),
  level-dependent B2M/M2M/L2L chains, finalize (0.24 ms + host tail).
- Validated exchange: work-list slicing + bitwise allreduce, comm+orch
  0.190 ms at n=1e6; perm-aware scatter-add; requires the
  `cuMemPoolSetAccess` pool P2P grant (CUDACore leaves pool memory
  P2P-unmapped; 32 -> 234 GB/s, exchange 0.905 -> 0.092 ms).
- Hard constraints: the device counting sort is non-deterministic across
  caches (atomic within-cell order) — cross-cache sorted-frame comparisons
  are invalid; the cached window stream is a half-enumeration with
  Morton-first-half target ranges — target-range filtering of cached
  windows is semantically void.
- Floors: ~0.87 ms n-independent per-GPU control floor (cycle-1 n=1e3
  control, graph replay, 28 launches + 1 graph launch); perfect 8-way
  compute split ~0.58 ms; hence ~1.0–1.2 ms naive 8-GPU estimate.

## Deliverables

1. **Ownership**: costed Morton-range leaf ownership (cost model, not equal
   volume — must load-balance the wake case), on the adaptive leaf set.
2. **Split-level scheme**: local subtrees below a split level `L_s`,
   replicated + allreduced coarse levels above; rule for choosing `L_s`.
3. **Halo sets**: per-level multipole halo (M2L stencil reach across
   partition boundaries, after per-level M2M) and body halo (boundary
   nearfield), each with an exact-once coverage proof on the adaptive
   U/V/W/X lists and a size model (surface-scaling argument).
4. **Migration policy**: ownership migration on occupancy epochs, hooked on
   the existing epoch detection; graph re-record rule.
5. **Comm/graph plan**: per-level exchanges inside captured graphs reusing
   the validated `029` P2 mechanism; count the added sync points in the
   M2M/L2L chains.
6. **Cost model + kill switch**: per-stage 2/4/8-GPU model calibrated to
   the measured floors above, including the per-GPU launch-floor budget.
   **Acceptance gate: modeled 8-GPU step `<= 2 ms` with an identified,
   itemized path to `<= 1 ms` (launch-floor shaving included); otherwise
   the deliverable is a recommendation NOT to proceed, and `044`/`045`
   are re-scoped or dropped by user decision.**

## Verification

A small script (`scripts/`) evaluating the cost model against the recorded
`029`/`041` per-stage data; halo-size enumeration on the three standard
cases (cube, wake, multi-scale) at 2/4/8 partitions.

## Work Record

(To be filled by the executing agent.)

## Approval Notes

To be filled by a different agent after this task is complete.
