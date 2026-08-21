# 016a Milestone Review Implementation 013-016

## Objective

Review Implementation tasks `013` through `016` against the background design
and coordination rules before the final Implementation tasks begin.

## Dependencies

- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `013-impl-axis-swap-operators.md`
- `014-impl-full-m2l-operator-pipeline.md`
- `015-impl-axis-swap-benchmarks.md`
- `016-impl-m2m-and-l2l-operator-pipelines.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Completed task files listed above
- Artifacts and production files listed by the completed task files

## Artifacts or Production Surface

Review the production files, tests, benchmarks, and generated artifacts listed
by tasks `013` through `016`.

## Deliverables

- Roadmap-alignment notes recorded in this file
- Any required coordination-document fixes identified before later work starts

## Verification

Confirm completed work matches the background design, hard phase gate, and task
ordering. If `START_HERE.md`, a task file, and `../MATRIX_OPERATOR_REFACTOR.md`
disagree, stop and require a coordination-document fix.

## Review Notes (2026-06-24)

Milestone Review performed per the `START_HERE.md` protocol. Required reading
completed: all of `../MATRIX_OPERATOR_REFACTOR.md`, `START_HERE.md`, and the
completed task files `013`, `013b`, `013c`, `014`, `015`, `016` with their listed
production/test/benchmark artifacts. Scope is the Implementation work since the
previous milestone (`012a`, which covered `009`–`012`).

### Status of reviewed tasks

All six reviewed tasks are marked both `Done` and `Approved` in `START_HERE.md`,
and each approval was recorded by a clear-context agent distinct from the
implementer (verified per task file). `013c` is notable: it was correctly held
through two `NOT APPROVED` review cycles for substituting the materialized
arithmetic under factored names, and only approved after the Plain-H fixed-mode
(`U_n`/`V_n`) factorization delivered a genuine `O(P^3)`/column staged operator.
This is the milestone's main correctness story and it was handled per the
CLAUDE.md root-cause mandate (a spike established why the ζ-dressed `013b`
primitives cannot compose into `R_y(θ)`), not worked around.

### Artifact existence and test wiring (independently verified)

- Production surface exists and follows the Implementation Code Placement rules:
  operators in `src/rotate_batched.jl` and `src/translate_batched.jl`; new
  structs/tags in `src/containers.jl`; additive includes + exports in
  `src/FastMultipole.jl` (`rotate_batched`/`translate_batched`,
  `AbstractM2L/M2M/L2L Operator`, `Materialized*`/`Factored*` tags, `*Scratch`).
- Tests are wired into `test/runtests.jl` (lines 57, 62, 67, 68, 69:
  `operator_cache_types_test`, `rotate_batched_test`, `translate_batched_test`,
  `m2l_operator_test`, `m2m_l2l_operator_test`) — they cannot silently rot.
- `015` benchmark artifacts present:
  `MATRIX_OPERATOR_REFACTOR/015-results.md`, the harness
  `scripts/impl_015_m2l_variants.jl`, and the CSV/env data under
  `data/axis_swap/tmpfac-126-17.et.byu.edu/`.

### Verification reproduced independently

```
julia --project=. -e '...; include("test/rotate_batched_test.jl");
  include("test/m2l_operator_test.jl"); include("test/m2m_l2l_operator_test.jl")'
# fixed y-swap primitives (013b):            548 / 548
# factored rotation alignment (013c):        389 / 389
# z-rotation operators (batched):          25684 / 25684
# axis-swap y-rotation operators (batched): 6819 / 6819
# M2L operator pipeline (task 014):         9908 / 9908
# M2M/L2L operator pipelines (task 016):   23260 / 23260

julia --project=. -e '...; include("test/operator_cache_types_test.jl")'
# operator cache support types: 212/212; side-effect-free: 5/5; legacy parity: 4/4
```

All counts match the numbers recorded in the task files exactly.

### Design / phase-gate / ordering alignment

1. **Hard Phase Gate respected.** All Theory rows (incl. addenda `008d`–`008h`)
   and the re-plan rows `008b`/`008c` are `Done`+`Approved`; this Implementation
   work post-dates the gate.
2. **Non-invasive, parity-only (`008b` re-plan + background Non-Goals).** Confirmed
   `src/translate.jl`, `src/rotate.jl`, `src/fmm.jl`, and
   `src/evaluate_expansions.jl` are untouched (`git status` clean for those). The
   operator layer is side-by-side and does not replace production
   `multipole_to_local!` / `multipole_to_multipole!` / `local_to_local!`. The
   mathematical translation algorithm and the rotate–translate–rotate
   decomposition are preserved, as the background design requires.
3. **Two-variant A/B roadmap intact.** `MaterializedYRotationM2L` (`013`,
   reconstructed `Ts(θ)`) and `FactoredRotationM2L` (`013c`, fixed `U_n`/`V_n`
   modes) are both composed in `014`, benchmarked in `015`, and remain genuinely
   distinct strategies (the `013c` re-reviews specifically prevented their
   collapse). `024` remains the definitive post-integration comparison.
4. **Plain-H amendment consistently propagated.** The `2026-06-23` amendment is
   reflected coherently across `START_HERE.md`, `013b` (superseded-as-building-block
   note), `013c`, `014`, `015`, and `theory/axis-swap-conventions.md`: the factored
   swaps are the plain mode matrices, not the ζ-dressed `013b` `T_y_*90` primitives
   (which remain valid only as a ±π/2 parity reference for the materialized path).
5. **`Val(true)` accuracy-order policy (`008h`).** `P_chi = P_phi + 1` with φ
   padding zeroed before z-translation and before return alignment is implemented
   and tested in both `014` (M2L) and `016` (M2M/L2L).

### Coordination-document consistency

No blocking disagreement among `START_HERE.md`, the task files, and
`../MATRIX_OPERATOR_REFACTOR.md`. The background design doc remains intentionally
high-level ("Background/design rationale only"); its Operator Layer section
already anticipates the factored axis-swap → z-rotation → inverse-swap chain that
`013c` realizes, and its mention of the materialized `update_Ts!` path is the
other benchmarked variant — both are consistent with the shipped work. The
operational evolution (the `013a` spike, the `2026-06-18`/`-19`/`-23` roadmap
amendments, and the two-variant decision) is captured in `START_HERE.md`, which
is the designated operational source of truth. No coordination fix is required
before downstream work.

### Non-blocking observations for later tasks (carried forward, not gating)

- `015` showed **no batched-GEMM speedup yet** on CPU (`blas1 ≈ blas8`,
  per-expansion time batch-independent) and **no offset-class angle reuse** even
  at the fully-shared corner. The crossover means both variants beat production
  only at `P ≳ 12` (`Val(false)`); `P ≤ 8` favors legacy. These motivate `019`
  (tuning), `019b` (small-`P` fallback), and `022` (GPU) and are already recorded
  there — flagged here so the milestone does not imply realized speedup.
- The `O(P^4)` `S_pos`/`S_neg` cache (`013`) is flagged for `015`/`019` revisit.
- `FactoredRotationM2M` / `FactoredRotationL2L` (`016`) intentionally delegate to
  the materialized y-stage; dedicated factored M2M/L2L modes are deferred to a
  later performance task, consistent with the near-term-factored-M2L-only roadmap.

### Conclusion

Implementation tasks `013`–`016` are consistent with the background design, the
Hard Phase Gate, and the task ordering. Work is correct (parity verified),
appropriately scoped (side-by-side/parity-only), minimally invasive, and well
tested. Recommending this Milestone Review for clear-context approval. Downstream
rows (`017` onward) should not start until that approval is recorded by a separate
agent.

## Approval Notes

Clear-context approval (different agent), 2026-06-24: **APPROVED.**

Reviewed `START_HERE.md`, `MATRIX_OPERATOR_REFACTOR.md`, this milestone review,
the completed task files `013`, `013b`, `013c`, `014`, `015`, and `016`, and the
listed production/test/benchmark surfaces. No coordination-document disagreement
was found. The Plain-H `013c` amendment is consistently reflected in the roadmap
and implementation: the factored M2L path uses cached fixed `U_n`/`V_n` mode
matrices with an explicit `e^{i nu theta}` middle stage, while the materialized
path remains the `Ts(theta)` rebuild variant. The reviewed work remains
side-by-side/parity-only and does not replace the production translation hot
paths.

Verification reproduced:

```text
julia --project=. -e 'using Test; using FastMultipole; using FastMultipole.StaticArrays; using Random; include("test/rotate_batched_test.jl"); include("test/m2l_operator_test.jl"); include("test/m2m_l2l_operator_test.jl")'
  fixed y-swap primitives (013b): 548 passed
  factored rotation alignment (013c): 389 passed
  z-rotation operators (batched): 25684 passed
  axis-swap y-rotation operators (batched): 6819 passed
  M2L operator pipeline (task 014): 9908 passed
  M2M/L2L operator pipelines (task 016): 23260 passed

julia --project=. -e 'using Test; using FastMultipole; using FastMultipole.StaticArrays; include("test/operator_cache_types_test.jl")'
  operator cache support types: 212 passed
  operator cache construction is side-effect-free: 5 passed
  cache and scratch drive translations identically to legacy workspace: 4 passed
```

Non-blocking carry-forward items remain as recorded in the review notes:
small-`P` fallback policy, performance tuning / real batched-GEMM realization,
GPU implementation, and dedicated factored M2M/L2L modes are later-task concerns
and do not block this milestone.
