# 019 Implementation Operator Performance Tuning

## Objective

Improve and document performance, allocation behavior, and retained storage of
completed operator paths after M2M, M2L, L2L, flat buffers, and real-basis
transform parity are implemented.

## Dependencies

- `008c-implementation-performance-baseline.md`
- `017-impl-flat-coefficient-buffers.md`
- `018-impl-real-solid-harmonic-basis.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Stable M2M, M2L, and L2L operator implementation notes
- Stable flat-buffer implementation notes
- Stable real-basis transform notes
- Existing benchmark scripts and results, including `008c` baseline notes

## Artifacts or Production Surface

- Production operator paths for M2M, M2L, L2L, and flat buffers
- Benchmark scripts or benchmark test files
- Benchmark result artifacts under `MATRIX_OPERATOR_REFACTOR/data/` if needed
- Parity tests covering optimized paths

## Deliverables

- Whole-operator benchmark suite and recorded results
- Allocation and memory-footprint benchmarks for completed M2M, M2L, L2L, and
  flat-buffer paths
- Bottleneck notes for completed operator paths
- Data-structure review identifying avoidable allocations, over-retained
  operator/cache data, duplicate transforms, poor scratch reuse, and storage
  layouts that block batching or GPU-friendly execution
  - **Explicit flag (from `016b`/`016a` watch item 5):** the `O(P^4)` `S_pos`/`S_neg`
    swap cache (`013`) is a known over-retained operator-cache item — revisit its
    storage cost here as part of the cache review.
- Scoped CPU single-thread optimizations where justified by benchmarks
- Scoped CPU multithread optimizations where justified by benchmarks
- Scoped storage/allocation improvements where justified by benchmarks and
  compatible with approved parity requirements
- GPU-oriented layout, batching, or execution notes where relevant
- Preserved parity with approved reference behavior
- Before/after benchmark summaries for chosen optimizations
- Before/after allocation counts, memory summaries, and retained-storage
  rationale for chosen optimizations
- Deferred alternatives and remaining performance or memory risks

## Verification

Rerun relevant parity tests plus benchmark commands. Record commands,
environment notes, before/after timing summaries, before/after allocation and
memory summaries, chosen optimizations, retained-storage rationale, deferred
alternatives, and remaining performance or memory risks.

## Starting Worklist from the 022 H200 Runs (2026-07-11)

Task-local implementation notes carried out of item `022` (user-directed scope
decision, 2026-07-11): the `ConcatenatedFixedZM2L` whole-pass M2L validated on the
H200 (see the `022` task file's "H200 Rerun 2026-07-11" section) but retains known
headroom. Measured context at n=1e5/`ell=4`/P=4: M2L stage ~1.7 s vs ~10 ms of
memory-bound slab traffic (~100x roofline gap), ~89 chunks x ~150 kernel launches,
fresh device allocations per chunk; host `build_radix_interaction_list` ~5.5 s;
M2M/L2L ~0.15-0.26 s each. Ranked worklist:

1. **Dense fixed-stage GEMMs in `_launch_resident_m2l_concat!`**
   (`src/translate_batched.jl`): materialize the block-diagonal `U`/`V` y-mode
   matrices and the full fixed factorial matrix `C` as dense dof x dof operators in
   `ResidentM2LConcatPlan`, so the per-chunk chain becomes ~5 whole-slab GEMMs plus
   ~5 element-wise ops per channel, replacing the per-degree y GEMMs (4 stages) and
   the per-m z GEMMs with fancy-index row gathers. Est. ~10x fewer launches; the
   ~2.7x wasted-flop factor of dense-over-block is irrelevant at these sizes.
2. **Preallocate all chunk temporaries + fused gather/scatter kernels**: move
   `Gre`/`Gim`, trig tables, scale tables, and gather outputs into the plan; add
   fused `gather+Z_phi` and `Z_phi^-1+scatter` custom kernels. Removes per-chunk
   device allocations — the observed allocator/GC jitter where the single-shot M2L
   stage timing (1.39 s) exceeded the min-of-3 whole-lifecycle time (0.99 s) at
   n=1e4 — and closes the Float32 allocation watch item (016b item 5) for the GPU
   path.
3. **Chunk-size sweep**: `ConcatenatedFixedZM2L` default 2^17; try 2^18/2^19 on the
   H200 (plan buffers grow to a few hundred MB; fine on 140 GB).
4. **Benchmark hygiene**: per-stage timings in
   `MATRIX_OPERATOR_REFACTOR/scripts/cuda_022_validation.jl` (`_cuda_stage_times`)
   should be min-of-reps like `exec_time`.
5. **Host list build** (`src/interaction_list_batched.jl`): for
   `ConstantPAnalyticStencil`, replace the O(cells^2) direct-complement loop in
   `foreach_radix_direct_pair` with enumeration of the bounded rejected-offset set
   (~1.3k offsets per target at P=4/eps=1e-4), and replace the M2L batcher Dict in
   `build_radix_interaction_list` with per-offset vectors plus `sizehint!`.
   5.5 s -> est. 1-2 s.
6. **Apply the concat treatment to M2M/L2L** if they dominate after items 1-2
   (same per-group host-driven launch pattern, but only tens of groups — unique
   radius per level — so lower priority).
7. **Deferred (feeds `024`)**: fully dense per-offset-class M2L via grouped/batched
   GEMM — one dof x dof matrix per offset class (~27k classes ~ 135 MB at P=4),
   executed as grouped GEMM over class-sorted routes; the `015`-deferred option.

## Completion Notes (2026-07-11)

Work executed as phases A–F against the ranked worklist above; user-directed
emphasis on the GPU items, plus a user-directed redesign of worklist item 5:
the interaction structure must not be built pair-by-pair (implicit stencil).

### Chosen optimizations

1. **Dense fixed-stage GEMMs in the concat M2L** (worklist item 1,
   `src/translate_batched.jl`): stacked block-diagonal factored-y operators
   (`_ymode_stacked_dense`: `Ur = [blockdiag(Ure) -blockdiag(Uim)]`,
   `Vs = [blockdiag(Vre); blockdiag(Vim)]`; applied by `_stacked_y_dense!` as
   2 whole-slab GEMMs around a paired e^{iνθ} rotation on contiguous [re; im]
   halves) and a dense dof×dof fixed factorial z matrix
   (`_m2l_dense_factorial_matrix`), replacing the per-degree y loop
   (~4 GEMMs × (P+1)) and the per-m z loop with row gathers. Held in
   `ConcatChannelOps` inside `ResidentM2LConcatPlan`.
2. **Zero per-chunk allocations + fused kernels** (item 2): all chunk
   temporaries preallocated in the plan (trig tables and the r^-(n+1/2) scale
   table computed once per chunk and reused across stages — the pre/post z
   scalings are identical); new dual-dispatch fused stages
   `_gather_rotate_z!` (gather + Z_φ), `_rotate_z_scatter_accumulate!`
   (Z_φ^{-1} + atomic scatter), `_gather_rows!`, `_gather_values!` with
   generic host fallbacks and `CUDA.@cuda` grid-stride kernels
   (`src/translate_batched_cuda.jl`). Launches per chunk: ~150 → ~13 (non-LH).
   Closes the 016b item-5 Float32/allocation watch item for the GPU path.
3. **Chunk sweep + benchmark hygiene** (items 3, 4):
   `scripts/cuda_019_tuning.jl` + `scripts/cuda_019_run.sh` (+ local
   `cuda_019_submit.sh`/`cuda_019_fetch.sh` drivers); per-stage timings are
   warm + min-of-reps (022's were single-shot), with an `exec_max` jitter
   column; chunk sweep 2^17/2^18/2^19 at n=1e5, P∈{4,8}. Result: chunk width
   changes the recorded lifecycle minimum by at most ~6% after items 1–2. The
   2^17 default stands because the timing differences are small/noisy and it uses
   the least scratch, not because it is the fastest row in every sweep.
4. **Implicit stencil classification** (item 5, redesigned per user direction):
   `RadixImplicitStencil` (`src/containers.jl`)
   = sorted accepted offsets + bounded rejected complement + dense G³
   `cell_at` occupancy map; constant-P enumerators and a specialized
   Dict-free `build_radix_interaction_list` in
   `src/interaction_list_batched.jl` (offset-class-major; O(cells·|offsets|)
   with O(1) lookups replaces the O(cells²) complement scan; single stencil
   build serves both passes; canonical batch order by construction). The lookup
   classification is implicit, but the current builder still materializes every
   matching M2L target/source route and direct pair for the resident lifecycle.
   Eliminating those route arrays remains deferred to device-side generation. The
   concat plan now stores **per-class geometry** (nclasses (r,θ,φ) tables +
   Int32 `route_class`, gathered per chunk) instead of 4 per-route device
   vectors — ~371 MB → ~47 MB at n=1e5/P=4 — with a per-route fallback for
   sub-leaf-level (ParentNeighborM2L) batches.
5. **Whole-slab M2M/L2L** (item 6, triggered by the phase-C data):
   `_resident_stage_group_apply!` runs each per-(level, radius) group through
   the same fused chain (gather+Z_φ kernel, stacked-y GEMMs via the new
   workspace `StackedYChannel`, one dense per-group z GEMM from
   `_z_dense_matrix_like`, LH row mix with prealloc gathers, fused
   Z_φ^{-1}+atomic scatter). The M2M scatter matrix, the per-degree execute
   functions, and the dead m2m/l2l staging buffers were deleted; M2M groups
   store per-column parents and accumulate atomically.

### H200 before/after (node m13h-1-1, H200, CUDA 12.8 local, Julia 1.11.7; CSVs under `data/operator_performance_tuning/`)

n=1e5, ℓ=4, P=4, ConstantPAnalyticStencil(4, 1e-4), concat chunk 2^17,
min-of-3 timings (022 baseline stage times were single-shot):

| metric | 022 baseline | post A+B+D | post E (final) | total |
|---|---|---|---|---|
| M2L stage | 1.698 s | 0.067 s | 0.066 s | **~25x** |
| M2M stage | 0.152 s | 0.168 s | 0.0080 s | **~19x** |
| L2L stage | 0.150 s | 0.172 s | 0.0080 s | **~19x** |
| L2B stage | 0.051 s | 0.050 s | 0.050 s | 1x |
| whole lifecycle (min of 3) | 1.863 s | 0.433 s | **0.116 s** | **~16x** |
| host list build | 5.55 s | 0.36 s | 0.35 s | **~16x** |
| state build | 1.31 s | 0.62 s | 0.62 s | ~2x |

Other cases: n=1e4/P=4 lifecycle 0.995 → 0.054 s (~18x; host mirror 65.2 →
46.9 s — the CPU parity path also gains from the dense GEMMs); n=1e5/P=8
lifecycle 4.273 → 0.270 s (~16x; M2L 4.055 → 0.188 s); tiny parent-neighbor
case lifecycle 14.34 s (shared) / 22.7 ms (022 concat) → 2.28 ms. The severe
022 allocator anomaly is absent, but lifecycle jitter is not uniformly within
10%: final `exec_max` spreads range from ~0.2% to ~27.5%, with the n=1e4/P=4
and n=1e5/P=4/chunk=2^18 rows above 20%. The M2L stage
now sits ~6x above the ~10 ms memory-bandwidth roofline estimate (was ~100x).
The largest remaining stage at n=1e5/P=4 is L2B (50 ms, flat across P).

### Allocation and retained-storage summary

- Reproducible host-path `@allocated` after warmup (n=2000, ℓ=3, P=4):
  concat M2L 6,528 B/chunk (LH 14,096 B/chunk), M2M/L2L 832–2,070 B/group
  across Float64/Float32 × LH off/on — all host-side
  dynamic-dispatch/view boxes from the `::Any`-typed plan/workspace fields
  (pre-existing style); **zero device-array temporaries** remain in the M2L
  chunk loop and the M2M/L2L group loop (before: fancy-index gathers, trig
  broadcast temps, ~8×(P+1) per-degree y temps, per-m z gathers, scatter GEMM
  temporaries, per chunk/group).
- Command and recorded CSV: `julia --project=.
  MATRIX_OPERATOR_REFACTOR/scripts/operator_performance_allocations.jl` and
  `data/operator_performance_tuning/local_macos_allocations_storage.csv`.
- Concat plan geometry: per-route `phis/thetas/rs/invrs` (4 × nroutes F64,
  ~371 MB at 11.6M routes) → per-class tables + Int32 route_class (~47 MB).
- Deleted retained storage: M2M scatter matrices, m2m/l2l staging buffers
  (4 × ndof × max_batch per channel), per-m z block tables on M2M/L2L groups
  (replaced by one dense dof×dof per group), and the whole **device mirror of
  `OperatorInvariantCache`** — the resident lifecycle reads only
  `basis_info` metadata from it, so `_upload_operator_cache` was removed
  (this includes the flagged O(P⁴) `S_pos`/`S_neg` blocks on device).
- `S_pos`/`S_neg` host retention (the explicit 016b/016a watch item): still
  built eagerly in `OperatorInvariantCache` for the materialized-`Ts` path
  (`build_Ts_from_S!` also consumes them at cache build for `T_y_pos90/neg90`).
  Measured host footprint (F64): 2.2 KiB at P=4, 18 KiB at P=8, 443 KiB at
  P=20 — O(P⁴) growth is real but absolutely small next to the factored
  `y_*_U/V` mode data (10/61/771 KiB at P=4/8/20). Rationale: keep eager host
  construction (cheap, needed by the 013/materialized parity surface until the
  `024` A/B decision retires one variant); the device-side copy is gone.

### Verification

- Local (macOS, no CUDA; Julia 1.12.5): full `Pkg.test()` passes after all
  phases; `test/cuda_radix_lifecycle_test.jl` 91+19;
  `test/radix_interaction_list_test.jl` 61,076 (includes new implicit-stencil
  invariants and specialized-builder order/oracle parity tests);
  `m2l_operator_test.jl` 9,908; `m2m_l2l_operator_test.jl` 23,260;
  `resident_m2m_gemm_test.jl` 104+48.
- H200 (sbatch `scripts/cuda_019_run.sh`, user-submitted): `CUDA_019_RESULT
  PASS`, `VALIDATION_EXIT=0`; the Phase-A/B/D environment record reports lifecycle
  tests 204/204 + concat parity 19/19, while the later Phase-E completion record
  reports 205/205 + 19/19 after one device-residency assertion update;
  `LIFECYCLE_TEST_EXIT=0`; parity matrix Float64/Float32 × Val(false)/Val(true);
  transfer counters exact (host-origin 1 upload/1 download, device-origin 0/0,
  `expansion_host_copies == 0`). One stale device-gated test was updated for
  the removed staging buffers (asserts `ystk_phi` residency instead).

### Deferred alternatives and remaining risks

- **Per-offset-class dense M2L via grouped GEMM** (worklist item 7, feeds
  `024`): the phase-D class-major route ordering and per-class geometry are
  the stepping stone; not materialized here (~27k classes ≈ 135 MB at P=4).
- **Device-side pair generation**: generate `route_targets/sources` on device
  from (`cell_at`, accepted offsets) with count+scan compaction, dropping the
  host route arrays entirely; deferred here because the list flows through
  `cuda_radix_state` construction, counters, and tests. Natural companion to
  item 7. **Update 2026-07-11:** promoted to a requirement of `023` (see the
  time-stepping fast-path requirement recorded there) — the production target
  is recurring per-step cost with moving particles, where the host list build
  would otherwise dominate the 0.116 s evaluation.
- **L2B (50 ms) is now the largest device stage** at n=1e5/P=4, flat in P —
  candidate for the same fusion treatment in a follow-up (per-body evaluation
  kernel occupancy / output-write coalescing).
- Host mirror of the concat path still runs ~47 s at n=1e4 (single-thread
  reference, not the tuned legacy CPU FMM); CPU-side threading of the concat
  chain was not in scope.
- Stage-timing reps=3 is modest; most rows are stable, but two final lifecycle
  rows have >20% min-to-max spread and a longer-rep rerun on a quiet node would
  tighten the conclusion as well as the P=8 B2M noise (8–26 ms across sweep rows).
- Risk: the `::Any`-typed plan/workspace fields keep per-launch dynamic
  dispatch on the host (~KB boxing per chunk/group, µs-scale) — irrelevant at
  current sizes, worth typed containers only if launch rates grow ~100x.
- **Deferred (approval review 2026-07-11, user-directed): host list capacity
  over-retention.** The specialized constant-`P` `build_radix_interaction_list`
  hints every per-offset batch to `ncells`
  (`src/interaction_list_batched.jl`, `sizehint!(targets/sources, ncells)`),
  so at n=1e5/ℓ=4/P=4 the returned `RadixInteractionList` retains ~1.96 GB of
  host memory for ~186 MB of payload (measured: 27,322 batches, avg ~430 pairs
  each, capacity 4096 each); the dead capacity scales like G⁶ with depth.
  Deferred into `023`, whose device-side pair-generation requirement removes
  the host route arrays from the recurring path; if the host builder survives
  there, replace the hint with the exact per-offset bound (or drop it).

## Approval Notes

Review remediation (2026-07-11): the clear-context reviewer found that the
allocation/storage figures lacked a reproducible artifact, two timing conclusions
overstated the retained CSVs, the word "implicit" obscured continued route-pair
materialization, and raw-log/test-count provenance was inconsistent. With user
permission, that reviewer added the allocation/storage benchmark and CSV, corrected
the conclusions and terminology, and reconciled the external H200 evidence notes.

Local verification after remediation: the allocation/storage script completed and
matched the corrected figures; focused tests passed (interaction list 61,076; M2L
9,908; M2M/L2L 23,260; resident buffer/operator 43+625+104+48; lifecycle/concat
91+19), and full `Pkg.test()` passed. The H200 results remain recorded external
evidence and were not rerun locally.

**Not approved by this reviewer.** These are substantive benchmark/evidence edits,
so a different agent must perform a fresh clear-context review before checking the
item-019 `Approved` box.

Fresh clear-context review (2026-07-11, different agent): **Approved.**

- Evidence reconciled: every figure in the H200 before/after table, the
  jitter/`exec_max` claims, the chunk-sweep ≤~6% conclusion, and the
  allocation/retained-storage summary match the retained CSVs under
  `data/operator_performance_tuning/` (022 baseline, phase ABD, phase E, local
  allocations). Rerunning
  `scripts/operator_performance_allocations.jl` locally reproduced
  `local_macos_allocations_storage.csv` bit-identically.
- Tests reproduced locally (macOS, Julia 1.12.5), matching the recorded counts
  exactly: radix interaction list 61,076; M2L operator 9,908; M2M/L2L 23,260;
  resident batch parity 625+104+48; CUDA lifecycle host path 91 + concat parity
  19. H200 gates remain recorded external evidence (raw-log caveat noted in the
  data README).
- Code inspection: all claimed structures exist and match the completion notes
  (`ConcatChannelOps`, `_stacked_y_dense!`, `_m2l_dense_factorial_matrix`,
  `StackedYChannel`, `_resident_stage_group_apply!`, fused CUDA kernels,
  `RadixImplicitStencil` + the Dict-free specialized builder); placement follows
  the `_batched` rules.
- One significant finding: per-offset `sizehint!` capacity over-retention in the
  specialized host list builder (~1.96 GB vs ~186 MB payload at n=1e5). Per user
  direction, approved as-is with the fix deferred into `023` (recorded above
  under deferred alternatives and remaining risks).
