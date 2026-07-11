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

## Approval Notes

To be filled by a different agent after the decision is recorded, implementation, and
verification are complete.
