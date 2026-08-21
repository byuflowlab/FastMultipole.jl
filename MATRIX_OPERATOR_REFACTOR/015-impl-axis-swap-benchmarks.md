# 015 Implementation Axis Swap Benchmarks

## Objective

Benchmark the two near-term full-M2L operator variants over the stable `014`
interfaces.

## Dependencies

- `004-theory-axis-swap-conventions.md`
- `005-theory-full-m2l-composition.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `014-impl-full-m2l-operator-pipeline.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Existing benchmark scripts and full M2L implementation notes

## Artifacts or Production Surface

- Benchmark scripts or benchmark test files
- Benchmark result artifacts under `MATRIX_OPERATOR_REFACTOR/data/` if needed

## Deliverables

- Benchmarks comparing only the two near-term variants:
  1. `MaterializedYRotationM2L`: materialized `Ts(theta)` M2L using the building
     blocks from `013`.
  2. `FactoredRotationM2L`: explicit factored `Z/S/Z/S` M2L using the `013c` stages,
     whose `S`/`S_inv` are the cached fixed per-degree mode matrices `V_n`/`U_n`
     (`y_mult_U/V`, `y_loc_U/V`) — not the ζ-dressed `013b` primitives (see the
     `2026-06-23` Plain-H amendment in `START_HERE.md` and `013c`).
- Result notes comparing both explicit operator paths with current production M2L.
- Harness sweeps over representative expansion orders `P`, offset-class counts,
  shared-direction counts, shared-norm counts, and batch sizes; record
  allocation/storage/cache footprint as well as timing.
- Record, but do not benchmark as `015` deliverables, these deferred options for
  later final implementation/performance tasks: fully dense per-offset M2L matrix,
  partially folded hybrids around `K_z`, alternate z-translation cache/scaling
  policies, real-basis operator execution, per-`m` block-batched z-translation, and
  non-rotation operator families.
- **Separate CPU and GPU recommendations.** Record an explicit recommendation for
  each target, justified by the benchmark evidence. `024` remains the definitive
  end-to-end comparison after integration.

## Verification

Run benchmarks with recorded commands, environment notes, and result summaries.
Include enough detail to reproduce the comparison.

For the M2L variant benchmark, record the harness commands, environment, timing
and allocation/storage summaries across the `P` / offset-class / batch-size sweep,
and the rationale for the separate CPU and GPU recommendations.

## Results (Done)

- **Harness:** `MATRIX_OPERATOR_REFACTOR/scripts/impl_015_m2l_variants.jl` (CPU only,
  no `src/` changes; self-contained `timeit`, machine-tagged output, two BLAS
  regimes). It includes a pre-timing correctness gate that aborts unless
  `MaterializedYRotationM2L` reproduces production `multipole_to_local!` to `<1e-6`
  (observed `0.00e+00`).
- **Run commands and environment:** see the reproduction block and environment
  section in `MATRIX_OPERATOR_REFACTOR/015-results.md` (host Apple M2, Julia 1.12.5,
  OpenBLAS, single- and multi-thread regimes).
- **Artifacts** (`MATRIX_OPERATOR_REFACTOR/data/axis_swap/<host>/`): `env.md`,
  `m2l_variants_blas1.csv`, `m2l_variants_blas8.csv`, `footprint.csv`,
  `batch_composition_blas1.csv`, `batch_composition_blas8.csv` (the composition CSVs
  carry true realized `n_dir, n_norm, n_offset_class` distinct counts, swept over
  P/batch/LH).
- **Full write-up, tables, and recommendations:** `MATRIX_OPERATOR_REFACTOR/015-results.md`.

Key results:

1. **Order crossover:** both variants beat production only at higher order
   (`Val(false)` wins at `P ≳ 12`; `P ≤ 8` favors the legacy recurrence) — motivates
   the `019b` small-`P` fallback.
2. **Channel crossover:** factored wins φ-only (`Val(false)`, 1.75× vs production at
   P=20); materialized wins Lamb-Helmholtz (`Val(true)`, 1.23× at P=20, and the only
   variant that beats production under LH — factored is slowest there).
3. **No batched-GEMM speedup yet:** `blas1 ≈ blas8` and per-expansion time is
   batch-independent — the current `014` CPU operators are scalar per-column
   kernels, so the `008c` dense projections are unrealized on CPU (future `019`/`022`).
4. **Footprint:** steady-state call allocation is **0 bytes** for both variants;
   static cache ≈1.3–1.5 MB and scratch ≈58–63 MB at P=20/batch=4096 (shared types).
5. **Batch composition** (swept over P∈{8,20}, batch∈{64,4096}, true distinct
   direction/norm/offset-class counts, both LH): materialized per-expansion time is
   flat across all diversity axes (<0.2%); factored rises mildly (~5–6%) with the
   distinct-direction count. Critically, even at the fully-shared corner
   (`n_dir=n_norm=1`) both pay near the full per-column cost — the current API does
   not exploit offset-class angle sharing (recorded headroom for a future
   per-offset-class materialized form).

**CPU recommendation:** retain both behind dispatch and select by LH policy
(factored for φ-only high-P, materialized for LH); if forced to one default, choose
`MaterializedYRotationM2L` (robust across LH and P). Gate on order above the `P≈8–12`
crossover.

**GPU recommendation (analytical, staged for `022`/`024`):** `MaterializedYRotationM2L`
is the recommended GPU candidate — fewer, larger dense operators per offset class
(better occupancy / fewer launches) vs the launch-bound small-stage factored path,
per the `008c` GPU baseline. No GPU operator exists yet; `022` implements the
device-resident form and `024` confirms the per-platform winner on real hardware.

Deferred options (recorded, not benchmarked here): fully dense per-offset M2L,
partially folded hybrids around `K_z`, alternate z-translation cache/scaling
policies, real-basis operator execution, per-`m` block-batched z-translation,
non-rotation operator families, and per-offset-class shared-angle reuse.

## Approval Notes

Approved by clear-context review on 2026-06-24.

Reviewed `START_HERE.md`, this completed task file, the benchmark harness, the
result write-up, and the generated axis-swap data artifacts. The harness compares
only the two required `014` variants (`MaterializedYRotationM2L` and
`FactoredRotationM2L`) against production recurrence, records the required
P/Lamb-Helmholtz/batch/diversity timing sweeps, and captures allocation/cache/
scratch footprint. The CSV artifacts support the summarized high-order crossover,
channel-dependent CPU recommendation, zero steady-state call allocation, and the
current lack of offset-class reuse.

Minor review notes, not blockers: the benchmark-local preflight parity check is
limited to materialized `Val(false)` at the min/max swept P, relying on the
approved `014` parity suite for the full factored and Lamb-Helmholtz surface; and
`env.md` is overwritten by the second BLAS-regime run, while thread-specific CSVs
remain preserved as `*_blas1.csv` and `*_blas8.csv`.
