# 028 Performance Feasibility: 1,000,000 Particles in 10 ms

## Objective

Determine, with measured evidence, whether an accurate 1,000,000-particle solve can
run in at most 0.01 s per time step on a single H200 — and improve the production
path toward that target, not merely describe the gap. Consolidate the existing
benchmark record, fill material evidence gaps, classify each stage's binding
constraint, then implement the highest-value justified optimizations, verify
correctness and accuracy, and repeat the measured profile/optimize/retest cycle
until the target is reached or the remaining barriers and next steps are
rigorously quantified.

## Target Definition (fixed by user decision, `2026-07-28`)

- **Problem:** `n = 1,000,000` bodies, resident CUDA lifecycle, single H200.
- **Accuracy:** literature `P = 4` (`expansion_order = 3`), measured with the
  `024b` sampled-direct error methodology. **Float32 qualifies** for the verdict
  if its sampled error is within the `P = 4` truncation error measured at
  Float64; Float64 results are always reported alongside.
- **Verdict timing boundary — per-time-step cost:** one full resident evaluation
  (B2M → M2M → M2L → L2L → L2B plus nearfield direct) **plus everything a real
  time step repeats**: the device-side convection update and the tree/route
  refresh under the chosen reuse policy. No per-step H2D/D2H of body state.
- Two additional boundaries are measured and reported in every end-to-end table
  but do not decide the verdict:
  1. *Steady-state evaluation only* — bodies resident, tree/routes/operators
     already built; one evaluation.
  2. *Including transfers* — adds per-step H2D upload of positions/strengths and
     D2H download of influence, the boundary that applies if particle state
     lives on the host.
- Record the exact GPU model, driver, CUDA toolkit, CUDA.jl, and Julia versions
  with every dataset.

## Dependencies

- `025-theory-hierarchical-rigid-m2l-stencil.md`, `026-impl-hierarchical-m2l-host.md`,
  `027-impl-hierarchical-m2l-cuda.md` (the hierarchical path is the presumptive
  dominant lever at `n = 1e6`)
- `024b-impl-cpu-gpu-scaling-benchmark.md` (the `n = 1e6` baseline and error
  methodology), `024-impl-operator-ab-benchmark.md`, `024a-impl-benchmark-visualization.md`
- `019-impl-operator-performance-tuning.md`, `022-impl-gpu-device-resident-m2l.md`,
  `023-impl-production-integration.md`

## Required Reading

- `START_HERE.md`
- The dependency task files above and their Verification Notes
- `MATRIX_OPERATOR_REFACTOR/data/cpu_gpu_scaling/` (`024b`),
  `data/operator_ab_benchmark/` (`024`, including `summary/unresolved_notes.md`),
  `data/operator_performance_tuning/` (`019`), `data/hierarchical_m2l_host/` (`026`),
  `data/hierarchical_m2l_cuda/` (`027`)
- `src/translate_batched_resident.jl`, `src/translate_batched_cuda.jl`,
  `test/radix_fmm_timestepping_test.jl`

## Artifacts or Production Surface

- Study data and reports in `MATRIX_OPERATOR_REFACTOR/data/feasibility_1m_10ms/`.
- Benchmark/profiling scripts in `MATRIX_OPERATOR_REFACTOR/scripts/`
  (`benchmark_028_feasibility.jl` plus the `cuda_028_{submit,run,fetch}.sh`
  triplet, following the existing cluster pattern).
- Figures extending the `024a` set via the established TikZ/pgfplots + CSV
  toolchain.
- Production optimizations (Phase B) touch only files already owned by the
  resident lifecycle rows, under the `START_HERE.md` placement rules; every
  optimization lands with tests and H200 before/after data.

## Phase A: Evidence Consolidation and Gap Measurement

Start from the existing record; re-measure only where evidence is missing or
stale (anything predating `026`/`027` is stale for M2L structure):

1. **Consolidate** the `024b` `n = 1e6` point (the 94x-speedup regression whose
   cause the hierarchical rows address), the `024` four-strategy lifecycle data,
   the `019`/`022` stage breakdowns and roofline gap, and the `026`/`027`
   hierarchical-vs-flat results, into one end-to-end and per-stage time budget
   at the target configuration.
2. **Per-stage bound classification.** For every stage ≥ ~5% of the step
   (sort/tree refresh, route generation, B2M, M2M, M2L, L2L, L2B, nearfield,
   convection update): attribute compute-bound, memory-bandwidth-bound,
   transfer-bound, launch/latency-bound, or otherwise constrained, using
   achieved FLOP/s and GB/s against H200 peaks, kernel counts x measured launch
   overhead, and CUDA-event timings. State the attribution method with the data
   so the claims are reproducible.
3. **Transfer and residency costs.** Measure H2D and D2H costs separately from
   resident compute (positions/strengths up; influence down; construction-time
   uploads amortized separately). Using the per-step workload below, compare
   permanent device residency against round-tripping body state each step, and
   answer explicitly whether particle state must remain permanently on the GPU
   across convection and other time-stepping operations to meet the target.
4. **Per-step workload (convection proxy).** The repo has no production time
   stepper, so fix the representative per-step workload as: device-side Euler
   update of positions from the computed gradient (`x .+= v * dt`), then the
   tree refresh policy — measure both full re-sort/re-tree per step and any
   cheaper reuse/refit policy the resident cache supports. Build on the
   existing `test/radix_fmm_timestepping_test.jl` harness.
5. **Levers to evaluate** (extend as evidence directs; each gets an expected
   gain, confidence, and risk):
   - hierarchical stencil at `n = 1e6` (from `026`/`027`; expected dominant —
     the flat path's measured work exponent is 1.46);
   - **per-level M2L strategy mix** (user-raised `2026-07-28`): the `024`
     defaults were measured under the flat leaf-only list, where offset classes
     are fat — the regime favoring per-class dense GEMM. The hierarchical
     stencil thins classes to `(level, offset)` and coarse levels have few
     occupied nodes: the small-class regime where `023b` measured per-class
     launches as launch-bound and where the factored strategy's batch-shared
     `U_n`/`V_n` allow concatenating all classes of a level (or all levels)
     into a few large GEMMs. Evaluate a heterogeneous mix — e.g.
     factored/concatenated at coarse levels, fused dense at the leaf level —
     driven by the per-level class-occupancy histograms `026`/`027` record.
     No existing benchmark covers this mix; do not assume the `024` defaults
     transfer to the hierarchical path;
   - the L2B and grouped-GEMM M2L residuals deferred by `019`;
   - the Float32 invariant-cache construction anomaly (`023d`, shared
     infrastructure) — now on the verdict path since Float32 is admissible;
   - launch-bound small-class behavior and kernel-fusion opportunities;
   - nearfield direct cost at the target `(n, ell)` operating point.
6. **Report.** State whether one stage bottlenecks the solve or time is broadly
   distributed; quantify the remaining gap and its uncertainty at all three
   timing boundaries; deliver prioritized recommendations with expected gains
   and risks. **Pause for user approval of the optimization list before Phase B**
   (precedent: the `019b` user-discussion decision).

## Phase B: Optimize, Verify, Retest

Implement the approved optimizations in priority order. For each: verify
correctness and accuracy (tests must include `P = 4` per the standing project
rule; the `023` counter/allocation contracts and existing radix/CUDA tests stay
green), then repeat the end-to-end and per-stage measurements at the target
configuration and record realized versus expected gain.

Repeat Phase A → Phase B until the target is met at the verdict boundary, or
the best remaining lever's projected gain is quantified as insufficient to
close the gap. Each cycle ends with a user checkpoint; do not begin a new
optimization cycle without one.

## Verification

- All production changes covered by tests, including `P = 4`; existing radix,
  lifecycle, precision, and counter tests green; sampled-direct error at the
  target configuration within the `P = 4` Float64 truncation-error gate (and
  the Float32 admissibility check recorded whenever Float32 is used for the
  verdict).
- Final report reproduces: per-stage budget table at all three boundaries,
  bound classification with method, residency answer, realized gains per
  optimization with before/after H200 data, and the final verdict (target met,
  or quantified remaining gap plus next steps).

Record commands, job ids, and result summaries in a `Verification Notes`
section.

## Implementation Notes

To be filled by the implementing agent.

## Approval Notes

To be filled by a different agent after this task is complete.
