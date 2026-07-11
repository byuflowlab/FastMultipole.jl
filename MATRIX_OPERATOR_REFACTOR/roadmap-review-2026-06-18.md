# Roadmap Review 2026-06-18 — Implementation Phase

This record documents the `2026-06-18` review of the unimplemented Implementation
Phase rows (`010`-`019a`) and the resulting coordination-document changes. It is a
coordination artifact; like every roadmap change it requires clear-context approval
by a different agent before downstream task work relies on it.

## Scope Confirmed With User

- The roadmap must deliver **realized production speedup**, not parity-only parallel
  code.
- A **GPU operator path is in scope** for this roadmap.
- The **radix clustering + constant-`P` interaction-list driver** gets its own
  implementation row(s).

## Structural Gaps Found And Closed

1. **No driver for the new operators.** Rows `010`-`019` built operator kernels but
   nothing implemented the radix grid (`008f`), constant-`P` stencil (`008d`), or
   radix interaction list (`008g`) the operators are meant to run on. Added rows
   `020` and `021`.
2. **No realized speedup.** `014`/`016` are parity-only and hot-path replacement was
   deferred out of the roadmap, so the refactor would have ended as dead parallel
   code. Added row `023` (production integration behind basis-type dispatch, legacy
   path default).
3. **GPU had recommendations but no implementation home.** `008c` requires device
   residency; added row `022` behind a CUDA extension/flag.

## Sequencing And Policy Changes

- **`013a` spike** added so the M2L batching target and dynamic-`P` porting
  feasibility are known before `014` composition and `017` storage, instead of
  surfacing at `019a`. `014` must expose M2L as separable stages so the `015`
  batching decision is a composition choice, not a rewrite.
- **`019b`** added to stage the small-`P`/tiny-batch fallback and the
  padded-vs-ragged `chi` layout as an exploratory benchmark plus a required
  user-discussion decision, then implementation. No fallback is baked into `011`/`013`
  beforehand.

## Order Rule Note (`P_chi = P_phi + 1`)

The `008h` order rule is a proven accuracy floor (verified across
`delta in {-1, 0, +1, +2}`; below `+1` corrupts the top retained `chi` row, `+2` is
wasteful) and is **not** revisited. Only the *layout* that carries it (padded vs
ragged) is open, and is handled in `019b`.

## Existing Rows Sharpened

- `011`: size blocks only through `009` accessors; no baked-in fallback.
- `013`: precompute angle-independent `S_n` blocks to kill the dominant per-call
  Wigner `Ts` rebuild (`008c`).
- `014`: compose M2L as separable, individually-callable stages.
- `015`: decide batching over the stable `014` stage API.
- `017`: prune the dead `chi` channel for `Val(false)`; keep `chi` layout swappable.
- `018`: native real-basis execution behind an experimental flag unless benchmarks
  justify it.
- `019`: tune on the integrated end-to-end radix path.
- `019a`: spans `017`-`023`; dynamic-`P` porting becomes a go/no-go informed by `013a`.

## Full Rationale

The complete reviewed analysis (per-priority improvements, alternatives considered,
and verification approach) is preserved in the approved plan file:
`~/.claude/plans/let-s-review-the-final-zesty-rossum.md`.
