# 019b Exploratory Small-P Fallback And Channel Layout

## Objective

Resolve two performance-policy questions that `008c` and the Lamb-Helmholtz theory
left deliberately open, via an exploratory benchmark followed by a **required
user-discussion decision**, then implement the chosen policy:

1. **Small-`P` / tiny-batch fallback.** In regimes where dense packing, BLAS launch,
   or fused-kernel overhead dominate (`P <= 3`, `batch == 1`, and nearby), the
   recurrence/compiled-loop may beat the dense operator path.
2. **Padded-uniform vs ragged `chi` layout.** The `P_chi = P_phi + 1` order rule is a
   proven accuracy floor and is not in question; the open question is the layout —
   uniform padded active basis (`P_active = P_chi`, with `phi` rows above `P_phi` as
   scratch) vs ragged per-channel matrices.

## Dependencies

- `008c-implementation-performance-baseline.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md` (order rule fixed; layout open)
- `008b-implementation-replan.md`
- `015-impl-axis-swap-benchmarks.md`
- `019-impl-operator-performance-tuning.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- The `019` performance-tuning evidence and the `015` batching decision

## Artifacts or Production Surface

- Benchmark scripts under `MATRIX_OPERATOR_REFACTOR/scripts/` and result artifacts
  under `MATRIX_OPERATOR_REFACTOR/data/`.
- After the user decision: production code implementing the chosen fallback policy and
  channel layout, kept swappable behind the `009` order accessors.
- Parity tests covering the chosen policy.

## Deliverables

- **Exploratory phase:** measure the dense-vs-recurrence crossover across `(P, batch)`
  and the padded-vs-ragged `chi` cost (storage and channel-coupled LH application) on
  the baseline machines. Surface concrete options to the user:
  always-dense / recurrence-fallback-below-threshold / per-stage-hybrid for the
  fallback; padded vs ragged for the layout.
- **Decision phase:** record the user's chosen policies and rationale here.
- **Implementation phase:** implement the chosen fallback dispatch and channel layout.

## Verification

Record benchmark commands, environment, and the crossover/layout evidence. After
implementation, rerun operator parity tests and confirm no accuracy regression.
Record the user decision and result summaries.

## Exploratory Results (2026-07-11 .. 2026-07-14)

### Harness and commands

- CPU: `MATRIX_OPERATOR_REFACTOR/scripts/impl_019b_smallp_layout.jl` (015-style
  self-contained `timeit`, machine-tagged CSVs, production-parity pre-gate plus a
  per-configuration concat-vs-recurrence stage parity gate). Run twice per host with
  BLAS threads pinned at process start (008c caveat):

  ```sh
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_019b_smallp_layout.jl
  OPENBLAS_NUM_THREADS=<ncores> OMP_NUM_THREADS=<ncores> \
    julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_019b_smallp_layout.jl
  ```

  On the cluster both regimes run via `scripts/cpu_019b_run.sh` (sbatch, 64 cores);
  submission/fetch via `scripts/cuda_019b_submit.sh [gpu|cpu|all]` /
  `scripts/cuda_019b_fetch.sh`.
- GPU: `scripts/cuda_019b_tuning.jl` via `scripts/cuda_019b_run.sh` (H200, sbatch).
  Rows stream as `CUDA_019B_ROW,` lines so a wall-time kill keeps partial data
  (the first submission died at a 1 h limit with buffered-only output; the second
  spent 3 h timing `SharedRotationM2L` at a 100k-route config — both fixed).
- Data: `data/smallp_fallback_layout/{m12-1-7, m13h-1-1, mecsrs-MacBook-Pro-188.local, raw}/`.
  Environments: EPYC 7763 (m12-1-7, Julia 1.11.7, OpenBLAS, blas1 + blas64), H200 on
  m13h-1-1 (job fm019b-12748386, `PASS`, all GPU-vs-host parity checks passed),
  Apple M2 laptop reference (blas1 only — per user direction 2026-07-11, BLAS-regime
  evidence comes from HPC nodes, not the Mac).

### Crossover evidence (question 1)

- **Per-column 014 operators vs recurrence** (isolated, `crossover_isolated_*`):
  recurrence wins below `P ≈ 8`–12 at *every* batch width, both BLAS regimes, both
  hosts (EPYC blas1 materialized: 0.53–0.93× for P ≤ 8, 1.03–1.05× at P = 12; LH is
  lower everywhere). Batch width barely moves these — the per-column kernels are
  scalar, confirming 015.
- **Tuned whole-slab concat host path vs per-route legacy recurrence**
  (`crossover_stage_*`, same states, parity-gated): the shipping dense form wins
  small P outright — EPYC blas1 φ-only: 7.6× (P=1), 3.7× (P=2), 1.7× (P=4), 1.16×
  (P=6–8); LH: 2.8× (P=1) down to ~1.0 (P=4), then recurrence wins at P ≥ 6
  (0.66–0.78×). Multithread BLAS (blas64) degrades concat at small route counts
  (0.27–0.39× at P=6–8 tiny config) — the remedy is the standing 008c rule (pin
  BLAS to 1 thread inside operators), not a fallback.
- **GPU (H200, `m13h-1-1/gpu_smallp.csv`): dense concat never loses.** At the
  smallest corner (n=128, ell=2, P=1) the concat M2L stage is 0.28 ms vs 3.9 ms
  per-route host recurrence (14×) and the whole GPU lifecycle (3.0 ms) already
  matches the host lifecycle (3.8 ms); by n=512 the GPU is 3–14× faster end-to-end,
  and 400–800× at n=1e4. `SharedRotationM2L` is uniformly catastrophic at these
  sizes (1.7–9.5 s vs 3–6 ms) — concat remains the GPU default.

### Layout evidence (question 2)

`layout_lh_*` simulates the dominant concat dense chain (production builders
`_ymode_stacked_dense` / `_m2l_dense_factorial_matrix` / `_stacked_y_dense!`) in
three variants. Single-thread BLAS (both hosts): padded costs +10–28% per
expansion at P ≤ 12 and channel-merging recovers nothing; blas64 is noisy in both
directions with no robust merged win. Storage (`layout_storage.csv`): padded
overhead is largest exactly at the small P the GPU path runs — +17–33%
(FlatCoefficientBuffer) / +18–39% (DegreeMajorRealBuffer) per column at P ≤ 4.
Padding's only theoretical upside (halving channel GEMM launches) buys nothing
because the GPU M2L is no longer launch-bound after 019's fusion (~13
launches/chunk, ~6× above roofline).

## Decision (user, 2026-07-14, in-session)

1. **Small-P / tiny-batch fallback: always-dense.** No fallback dispatch is added.
   Rationale: the GPU dense path never loses, the CPU concat host path wins the
   small-P regime the fallback was hypothesized for, and the only regimes where the
   recurrence wins on CPU (LH at P ≥ 6; multithreaded-BLAS small-batch) are covered
   respectively by the legacy octree path remaining the production CPU default
   (023) and by the 008c single-thread-BLAS operator rule. Thresholds are recorded
   here rather than encoded as a second code path.
2. **Chi layout: keep ragged** (φ at `P_phi`, χ at `P_active`), rejecting the padded
   single-array alternative on the measurements above. The `phi_slab` / `chi_slab` /
   `phi_physical_view` accessors remain the swap surface if this is ever revisited.

## Implementation (per decision)

- `src/containers.jl`: the two "deferred to task 019b" markers on
  `FlatCoefficientBuffer` now record the decided ragged policy and its rationale.
  No dispatch or layout code changes — always-dense means the existing dense paths
  simply serve every order.
- Parity tests extended to pin the decision at the smallest orders:
  `test/m2l_operator_test.jl` adds `P = 1` to both `Val(false)` and `Val(true)`
  per-column parity sweeps (9,908 → 10,272 assertions);
  `test/cuda_radix_lifecycle_test.jl` concat host parity adds `P = 1` / `P = 2`
  rows (ParentNeighborM2L and a loosened `ConstantPAnalyticStencil(2, 1e-2)`) and
  runs the synthetic-χ LH concat parity at `P_phi = 1` as well as 3 (19 → 37
  assertions).

## Verification

- Benchmark artifacts, commands, and environments recorded above; raw sbatch logs
  under `data/smallp_fallback_layout/raw/`.
- Focused suites after implementation (macOS, Julia 1.12.5): M2L operator parity
  10,272/10,272; CUDA radix lifecycle host gate 91/91 + concat host parity 37/37.
- Full `Pkg.test()` after implementation: **passed** (macOS, Julia 1.12.5,
  2026-07-14; "Testing FastMultipole tests passed", 89 test-summary groups, no
  failures; the single Broken group is pre-existing).
- No accuracy regression: all parity tests are oracle comparisons against
  production `multipole_to_local!` / the shared-rotation resident path, and the
  benchmark's own stage parity gates passed on every measured configuration.

## Approval Notes

Clear-context review, 2026-07-14, by a separate agent per `START_HERE.md` step 6.
**Verdict: APPROVED.**

Inspected: this task file; `scripts/impl_019b_smallp_layout.jl` (629 lines) and
`scripts/cuda_019b_tuning.jl` plus the sbatch/fetch wrappers;
`data/smallp_fallback_layout/{m12-1-7, m13h-1-1, mecsrs-MacBook-Pro-188.local, raw}/`;
the `src/containers.jl` decision markers; the `test/m2l_operator_test.jl` and
`test/cuda_radix_lifecycle_test.jl` extensions.

1. **Objectives:** all three phases evidenced — exploratory benchmarks with recorded
   commands/environments, a recorded in-session user decision with rationale
   (always-dense; ragged χ), and an implementation that matches it (no new dispatch,
   no layout change, `phi_slab`/`chi_slab`/`phi_physical_view` retained as the swap
   surface).
2. **Correctness:** re-ran the focused suites locally (macOS, Julia 1.12.5,
   2026-07-14): M2L operator parity 10,272/10,272; CUDA radix lifecycle host gate
   91/91; ConcatenatedFixedZM2L host parity 37/37 — all matching the recorded counts.
3. **Evidence spot-checks:** CSVs reproduce the task's claims — EPYC blas1 stage
   crossover P=1 φ-only 3.13e-7 vs 2.38e-6 s/route (7.6×) and LH 2.8×; H200
   `gpu_smallp.csv` smallest corner concat M2L 0.28 ms vs 3.9 ms host recurrence
   (14×), lifecycle 3.0 vs 3.8 ms, `SharedRotationM2L` 1.7–9.5 s (catastrophic);
   n=1e4 exec 0.035–0.128 s vs host 25–81 s (within the claimed 400–800×); layout
   padded +28% at P_phi=2 with merged-GEMM recovering ~2%; storage overhead
   33%/38% at P=1. Raw sbatch log `fm019b-12748386.out` ends `CUDA_019B_RESULT PASS`.
4. **Methodology:** BLAS threads pinned via env at process start (008c caveat noted
   in-script), machine-tagged CSVs with env records, a production-parity pre-gate
   (abort at >1e-6) plus a per-configuration concat-vs-recurrence stage parity gate,
   blas1+blas64 regimes taken on HPC nodes per the standing rule.
5. **Minimally invasive:** the only `src/` change attributable to 019b is the two
   comment markers on `FlatCoefficientBuffer` in `containers.jl` (confirmed by grep);
   behavioral surface unchanged.
6. Minor note (no action required): the `ConstantPAnalyticStencil(2, 1e-2)` test row's
   loosened tolerance is self-evident from context but carries no explanatory comment;
   `env_blas1.md` records `git HEAD: unknown`. Neither affects the decision evidence.
