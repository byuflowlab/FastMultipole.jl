# 028 Phase A — Consolidated evidence (verified 2026-07-31)

Reuse this; do not re-read the dependency task files except at the cited lines.
`MOR/` = `MATRIX_OPERATOR_REFACTOR/`.

## Best measured points (H200, P=4 i.e. expansion_order=3)

| What | Result | Source |
|---|---|---|
| n=1e6 flat dense ell=4, F64 | step_min **0.4246 s**, construct 27.9 s, grad rel RMS 1.1873e-4 | `MOR/data/cpu_gpu_scaling/gpu_m13h-1-1_12894403.csv` |
| n=1e6 flat dense ell=4, F32 | step_min **0.3213 s**, construct 31.5 s, grad rel RMS 1.1889e-4 | same |
| n=1e6 flat precomputed_y ell=5 (OOM fallback), F64 | 3.279 s, grad err 5.61e-5 | same |
| n=2e5 hier3 dense ell=4 K=1740, F64 | **full step 29.98 ms** (M2L 0.57, route_gen 0.26, L2B+near 9.6), 167.8 MB | `MOR/data/hierarchical_m2l_cuda/ksweep_m13h-2-1_12992039.csv` |
| n=2e5 hier12 same config | full step 48.56 ms | same |
| n=2e5 hier3 ell=5 K=64 stage budget | grid 2.5 + occ 0.02 + direct_gen 0.11 + B2M 0.74 + M2M 0.99 + M2L 4.59 + L2L 0.99 + L2B+near 1.44 = **15.6 ms instrumented of a 60.8 ms step; ~45 ms = host allocation/GC** (~2.6 KB/cell/step ≈ 1 MB/ms) | `MOR/data/hierarchical_m2l_cuda/depth_m13h-2-1_12992039.csv`; `MOR/027-...md:418-437` |
| n=1e5 flat concat post-019 tuning | exec 0.116 s (b2m 14, m2m 8, m2l 66, l2l 8, l2b+near 50 ms) | `MOR/data/operator_performance_tuning/m13h-1-1/cuda_019_phaseE_throughput.csv` |

Flat-path scaling anomaly (024b): 1e5→3.16e5 is 2.29x for 3.16x n, but 3.16e5→1e6 is
**5.44x** — grid-resolution-limited, not compute (`MOR/024b-...md:245-249`); the eight
ell=6/7 grids that failed there (up to 223.5 GiB requested vs 139.8 GiB) **now
construct on the hierarchical path**: ell=6 = 15.97 ms M2L / 1.61 GB, ell=7 =
22.41 ms / 1.78 GB at n=2e5 (`MOR/027-...md:405-416`) — but full step blows up at
ell≥6 (319/558 ms) because host per-cell allocation dominates at ~1-2 bodies/cell.
Best full-step ell at n=2e5 was 4-5.

## Established bound classifications (reuse; anything pre-026/027 is stale for M2L structure)

- **Hierarchical route generation**: launch/latency-bound — one blocking sync D2H
  copy per `(level, window)`, **46-61 µs/window** independent of n/ell/strategy;
  cured by large K (M2L at hier3/ell=5/n=2e5: K=4→1740 gives 20.8→3.64 ms, floor
  ~3.6 ms arithmetic). `MOR/027-...md:324-346, :374-387`.
- **Flat concat M2L post-019**: ~6x above its ~10 ms memory-bandwidth roofline at
  n=1e5 (was ~100x). `MOR/019-...md:183`.
- **Per-class (grouped-GEMM) M2L**: launch-bound, loses 20-500x to whole-pass at
  flat occupancies (023b/023d) — **never re-tested under hierarchical thin-class
  shape**. Occupancy shape at n=2e5/ell=5 (hier3): levels 2-4 mean 9.8/169/1080
  routes/class, leaf level mean ~11k (`MOR/data/hierarchical_m2l_cuda/*.classes.csv`)
  → the per-level strategy-mix lever named by the 028 task file.
- **L2B is fused with nearfield direct** in one launcher
  (`_launch_cuda_resident_l2b!`, `src/translate_batched_cuda.jl:2722`) — every
  recorded `l2b_ms` includes nearfield. 019 deferred L2B as largest device stage
  (50 ms at n=1e5 flat, flat in P; `MOR/019-...md:246-248`).
- Strategy rule at P=4 (flat, fat classes): **dense wins every M2L case** (e.g.
  0.046 ms vs precomputed_y 0.136 at N=2e4/ell=3); dense construction is the cost
  (break-even 542-18k steps). `MOR/024-...md:373-398`,
  `MOR/data/operator_ab_benchmark/summary/case_rankings.csv`.

## Open items on the verdict path

1. **Measurement gaps (primary)**: no hierarchical n=1e6, no hierarchical Float32,
   no hierarchical LH-on data.
2. **Host alloc/GC ~2.6 KB/cell/step** — 027 named it "a task 028 bottleneck item";
   at n=1e6 (ell≥5 → ≥1e5 cells) could alone exceed the 10 ms budget.
3. Fused L2B+nearfield cost at the target `(n, ell)` point.
4. Float32 `OperatorInvariantCache` construction anomaly (`MOR/023d-...md:229-237`)
   — construction-only, not per-step.
5. `accumulate!` scan scratch in hierarchical M2L is pool-served, not zero-alloc;
   preallocated block scan is a named optimize candidate (`MOR/027-...md:227-232`).
6. **K-default trap**: an explicitly constructed `HierarchicalRigidStencil` gets
   constructor default `window_classes=4` even on device; K=256 flows only through
   `_default_radix_policy` (`MOR/027-...md:694-697`). Always pass K explicitly.
7. 026 deferrals: precomputed-y per-(level,offset) z-table collapse, per-window
   O(nclasses) sweeps, O(8^ell) occupancy fill! (`MOR/026-...md:662-670`).

## Accuracy numbers (Float32 admissibility gate)

At n=1e6/ell=4/dense/flat (`gpu_m13h-1-1_12894403.csv`): F64 grad rel RMS
**1.1873e-4** (= the P=4 truncation error), F32 **1.1889e-4** (+0.13%) → F32
admissible at that point; re-verify hierarchically. Reference: 512-sample direct,
`MOR/data/cpu_gpu_scaling/references/direct_reference_n1000000.csv` (checksummed).
CPU1 n=1e6 grad rel RMS 6.07e-4. Hierarchical at n=2e4: hier3 ~6e-4 (**below the
gate**), hier12 ~3e-5 (passes) (`MOR/027-...md:348-353`) → **hier12 presumptive
verdict config**; report hier3 as faster-but-below-gate unless its n=1e6 error
passes.

## Environment records

Fullest: `MOR/data/operator_performance_tuning/m13h-1-1/env.md` (H200 sm_90
139.8 GiB, driver 580.159.4, CUDA runtime 12.8.0 local toolkit, CUDA.jl 6.2.0,
Julia 1.11.7). 027 ran Julia 1.11.7 / CUDA runtime 13.3 / H200 140.4 GiB. 027 CSVs
carry per-row `manifest/job/host/gpu/julia/cuda/blas_threads/seed` — copy that.

## Progress

- [ ] (nothing to execute in this file — reference only)
