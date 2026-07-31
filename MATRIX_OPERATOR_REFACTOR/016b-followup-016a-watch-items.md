# 016b Follow-up Resolution Of 016a Watch Items

## Objective

Resolve (or formally defer with a recorded rationale) the non-blocking watch
items raised by the `016a` Milestone Review of Implementation tasks `013`–`016`.
These items do not change the already-approved `013`–`016` deliverables; they
harden documentation, correctness invariants, and test-protection so later
Implementation tasks (`017`+, especially `023` integration and `019a` final
review) inherit a clean baseline.

This task **must not** edit the approved `013`–`016` task files or their shipped
operator code except where an item below explicitly calls for an additive
invariant/assertion or a documentation pointer.

## Dependencies

- `016a-milestone-review-impl-013-016.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- `016a-milestone-review-impl-013-016.md` (the watch-item source)

## Watch Items And Required Resolution

### 1. Batched-GEMM speedup unrealized on the benchmark host — DEFERRED

`015` found the current CPU operators behave as scalar per-column kernels
(`blas1 ≈ blas8`, per-expansion time batch-independent, no offset-class angle
reuse), so the batching/GPU performance thesis is not yet realized, and on this
host the operators beat the legacy recurrence only at `P ≳ 12`.

**Resolution: defer, do not investigate here.** Per user direction
(`2026-06-24`), this is likely an artifact of the macOS host's BLAS
configuration (Apple M2 / OpenBLAS regime used in `015`); a different machine
with a different BLAS is expected to show the batched speedup. The required
follow-through is therefore to **re-run the `015` harness on at least one
non-macOS / different-BLAS host** as part of `019`/`022`/`024` and record whether
the batched speedup appears there. No code change in `016b`.

- [x] Recorded the deferral and the cross-machine BLAS hypothesis as an explicit
  check item in `024` (definitive end-to-end benchmark) and `019a` (final
  roadmap go/no-go), so the performance thesis gets a real verdict rather than an
  assumption. **Done:** added a "Cross-machine BLAS check" deliverable to
  `024-impl-operator-ab-benchmark.md` and a "Batched-GEMM speedup verdict"
  go/no-go deliverable to `019a-milestone-review-final-roadmap.md`. No code change.

### 2. Factored path is correct only on the physical subspace

`014` notes that `FactoredRotationM2L`'s rank-1 mode decomposition reproduces
production **only for physical inputs** (m=0 imaginary part = 0), whereas
`MaterializedYRotationM2L` is exact for any input. Every real expansion is
physical, so this is correct today, but it is a silent correctness boundary: a
future intermediate buffer carrying a nonzero `m=0` imaginary component would
make the factored path diverge from production with no error.

- [x] Document the physical-subspace invariant explicitly at the factored
  operator entry points (`*_factored_*_alignment_batch!` and the
  `FactoredRotationM2L`/`M2M`/`L2L` dispatch) as an additive docstring/comment.
  **Done:** added a comment block above the four public
  `*_factored_*_alignment_batch!` entry points in `src/rotate_batched.jl`, and a
  comment at the `FactoredRotationM2L`/`M2M`/`L2L` struct defs in
  `src/containers.jl`.
- [x] Add a cheap, debug/assert-level guard (e.g. behind an existing assertion
  toggle, not in the inner loop) verifying the `m=0` imaginary rows are zero on
  factored-path inputs, to be wired in at the `023` integration boundary. Record
  the requirement in `023` so it is not lost if `016b` only documents it.
  **Done:** added `_factored_input_is_physical` and the `DEBUG[]`-gated
  `_assert_factored_input_physical` helpers in `src/rotate_batched.jl` (whole-buffer
  scan, never in the inner loop; OFF by default). Recorded the requirement to wire
  the guard at the integration boundary as a deliverable in
  `023-impl-production-integration.md`.

### 3. The `013c` anti-collapse structural test is load-bearing

The factored path twice regressed into the materialized `Σ_ν S·trig(νθ)`
per-entry arithmetic and was only genuinely fixed on the third pass. The guard
that catches this is the structural test that reconstructs the production
y-operator from the constant cached `U_n`/`V_n` modes — not the name/string
checks.

- [x] Add a comment at that test in `test/rotate_batched_test.jl` marking it as a
  protected anti-collapse invariant (do not weaken to a string/name check), and
  note in any future-refactor entry (e.g. the deferred "shared-plain-swap +
  dressing" variant mentioned in `013c`) that it must keep this test intact.
  **Done:** added a "PROTECTED ANTI-COLLAPSE INVARIANT (016b)" comment at the
  structural reconstruction `let` block, distinguishing it from the weaker
  string/name checks above and calling out the deferred variant.

### 4. Background design doc drift

`../MATRIX_OPERATOR_REFACTOR.md` still describes the operator cache as absorbing
`Hs_π2`, `ζs_mag`, `ηs_mag`, `M̃`, `L̃`, with no mention of the shipped
`S_pos`/`S_neg`, the Plain-H `U_n`/`V_n` mode matrices, the two-variant
materialized/factored strategy, or the `Val(true)` `P_chi = P_phi + 1` policy. It
is explicitly "background only" and `START_HERE.md` is the operational source of
truth, so this is not a contradiction — but a reader starting from the background
doc would be misled.

- [x] Add a short pointer paragraph to `../MATRIX_OPERATOR_REFACTOR.md` (Operator
  Layer / Operator Cache sections) directing readers to the `START_HERE.md`
  roadmap amendments (`2026-06-18`/`-19`/`-23`) for the as-built operator set and
  the two-variant strategy. Do not rewrite the background derivations.
  **Done:** added an "As-built note (016b)" blockquote to both the Operator Layer
  and Operator Cache sections of `../MATRIX_OPERATOR_REFACTOR.md`; derivations
  unchanged.

### 5. Minor items

- [x] `O(P^4)` `S_pos`/`S_neg` cache: already flagged for `019`; confirm the
  flag is present in `019` and leave as-is. **Done:** the flag was implicit under
  "over-retained operator/cache data" only, so made it explicit as a sub-bullet in
  `019-impl-operator-performance-tuning.md`.
- [x] `015` `env.md` overwrite: the second BLAS-regime run overwrites `env.md`
  while CSVs are preserved per-regime. Either tag `env.md` per regime or note in
  `015-results.md` that `env.md` reflects the last regime run. Low priority.
  **Done:** added a note in the `015-results.md` body (Artifacts) that `env.md`
  reflects only the last (8-thread) regime while CSVs preserve both.
- [x] Float32 apply kernels inherit allocations from the legacy y-rotation
  kernels — irrelevant on CPU now, but record as a watch item for `022` (GPU /
  batched), where Float32 + per-call allocation matters. **Done:** added a
  "Float32 allocation watch item" deliverable to
  `022-impl-gpu-device-resident-m2l.md`.

## Deliverables

- Resolution or recorded deferral for each watch item above, with the deferred
  performance item (1) carried into `024`/`019a` as an explicit cross-machine
  BLAS check.
- Any additive invariants/assertions and documentation pointers described above.
- No change to the approved `013`–`016` deliverables or their operator semantics.

## Verification

- Re-run the batched operator and cache tests to confirm no regression from any
  additive comment/assertion:
  `julia --project=. -e '...; include("test/rotate_batched_test.jl");
  include("test/m2l_operator_test.jl"); include("test/m2m_l2l_operator_test.jl");
  include("test/operator_cache_types_test.jl")'`
- Confirm the `019`/`019a`/`022`/`023`/`024` cross-references for the deferred and
  forwarded items are actually present in those task files.
- Record commands and result summaries.

## Resolution Summary (2026-06-24)

All five watch items resolved (1 deferred with recorded rationale and forwarded
cross-references; 2–5 resolved). Changes are additive only — no approved
`013`–`016` operator semantics or task files were altered beyond the additive
comments/guard/pointers this task explicitly authorizes.

Production-surface changes:

- `src/rotate_batched.jl`: physical-subspace comment block above the four public
  `*_factored_*_alignment_batch!` entry points; new `_factored_input_is_physical`
  (whole-buffer scan) and `DEBUG[]`-gated `_assert_factored_input_physical`
  helpers (OFF by default, not in any inner loop).
- `src/containers.jl`: physical-subspace comment at `FactoredRotationM2L/M2M/L2L`.
- `test/rotate_batched_test.jl`: protected anti-collapse invariant comment.

Documentation / coordination changes: `../MATRIX_OPERATOR_REFACTOR.md` (Operator
Layer + Operator Cache as-built pointers); cross-references added to `024`,
`019a`, `023`, `019`, `022`; note added to `015-results.md`.

### Verification

Re-ran the batched operator and cache tests — all pass, counts match the `016a`
baseline exactly (no regression from the additive comments/guard):

```text
julia --project=. -e '... include("test/rotate_batched_test.jl");
  include("test/m2l_operator_test.jl"); include("test/m2m_l2l_operator_test.jl");
  include("test/operator_cache_types_test.jl")'
  fixed y-swap primitives (013b):           548/548
  factored rotation alignment (013c):       389/389
  z-rotation operators (batched):         25684/25684
  axis-swap y-rotation operators (batched):6819/6819
  M2L operator pipeline (task 014):        9908/9908
  M2M/L2L operator pipelines (task 016):  23260/23260
  operator cache support types:             212/212
  cache construction side-effect-free:          5/5
  cache/scratch == legacy workspace:            4/4
```

Guard sanity check: `_factored_input_is_physical` returns `true` on an all-zero
(physical) batch and `false` once an `m=0` imaginary row is set; with
`FastMultipole.DEBUG[] = false` the assertion is a no-op on non-physical input,
and with `DEBUG[] = true` it raises a "non-physical" error. Cross-references
confirmed present in `019`/`019a`/`022`/`023`/`024` and the `015-results.md` note.

## Approval Notes

Clear-context approval (different agent), 2026-06-24: **APPROVED.**

Reviewed `START_HERE.md`, `MATRIX_OPERATOR_REFACTOR.md`, this task file, the
`016a` watch-item source, and the 016b production/test/documentation surfaces:
`src/rotate_batched.jl`, `src/containers.jl`, `test/rotate_batched_test.jl`,
`MATRIX_OPERATOR_REFACTOR.md`, `015-results.md`, and the forwarded task files
`019`, `019a`, `022`, `023`, and `024`.

All required watch items are resolved or explicitly deferred with recorded
follow-through. The factored-path physical-subspace boundary is documented at
the public staged entry points and operator tags; the debug-gated whole-buffer
guard exists and is carried into `023` for integration-boundary wiring. The
`013c` structural anti-collapse test is protected by an explicit comment, the
background design doc now points readers to the as-built roadmap amendments, and
the minor cache/env/Float32 items are carried into the appropriate later tasks.
The batched-GEMM performance item is deferred to the cross-machine `024`/`019a`
verdict as directed.

Verification reproduced:

```text
julia --project=. -e 'using Test; using FastMultipole; using FastMultipole.StaticArrays; using Random; include("test/rotate_batched_test.jl"); include("test/m2l_operator_test.jl"); include("test/m2m_l2l_operator_test.jl"); include("test/operator_cache_types_test.jl")'
  fixed y-swap primitives (013b): 548 passed
  factored rotation alignment (013c): 389 passed
  z-rotation operators (batched): 25684 passed
  axis-swap y-rotation operators (batched): 6819 passed
  M2L operator pipeline (task 014): 9908 passed
  M2M/L2L operator pipelines (task 016): 23260 passed
  operator cache support types: 212 passed
  operator cache construction is side-effect-free: 5 passed
  cache and scratch drive translations identically to legacy workspace: 4 passed
```

Additional guard sanity check: `_factored_input_is_physical` returns `true` for
an all-zero physical batch and `false` after setting an `m=0` imaginary slot.
