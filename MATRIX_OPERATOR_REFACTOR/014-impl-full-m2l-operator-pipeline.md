# 014 Implementation Full M2L Operator Pipeline

## Objective

Compose the full M2L operator pipeline and test it against current production
behavior.

## Dependencies

- `005-theory-full-m2l-composition.md`
- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `008d-theory-dynamic-p-error-m2l-integration.md`
- `008h-theory-lamb-helmholtz-accuracy-order.md`
- `010-impl-z-rotation-operators.md`
- `011-impl-m2l-z-translation-blocks.md`
- `012-impl-lamb-helmholtz-operators.md`
- `013-impl-axis-swap-operators.md`
- `013b-impl-fixed-y-swap-primitives.md`
- `013c-impl-factored-rotation-alignment.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Approved `theory/full-m2l-composition.md`
- Approved `theory/dynamic-p-error-m2l-integration.md`
- Current production M2L call sites and tests

## Artifacts or Production Surface

- Production full M2L operator pipeline code
- Tests comparing full operator output with current M2L behavior

## Deliverables

- End-to-end explicit M2L operator composition.
- Swappable whole-M2L operator interfaces:
  - `MaterializedYRotationM2L`: uses `Ts(theta)` materialized by `013`.
  - `FactoredRotationM2L`: uses explicit `Z/S/Z/S` rotation alignment stages from
    `013c`, built on the fixed y-swap primitives from `013b`.
- The factored path must preserve the 013c requirement to reuse the same y-swap
  apply operator for forward and return y stages when possible; any separate
  inverse-y operator must be justified by production-parity evidence.
- Cache and scratch usage integrated with earlier implementation tasks
- Both M2L variants share the same `011` z-axis M2L blocks, `012`
  Lamb-Helmholtz coupling, common cache/scratch conventions, and common parity
  tests against production M2L.
- Stage API that can compose either near-term variant without rewriting the
  batching layer. The folded no-`Ts` y-rotation path is not a `014` deliverable
  except as temporary debug scaffolding for validating explicit stages.
- Z-translation reuse should distinguish shared direction from shared distance:
  pre/post scaling is reusable only when the physical `r` / offset norm / level
  key matches.
- `Val(true)` M2L policy uses `P_phi` as the requested physical order and
  carries `chi` at `P_chi = P_phi + 1` through the M2L/evaluation pipeline.
  The constant-`P` stencil should use `B_phi(P_phi)` and
  `B_chi(P_phi + 1)` per `theory/lamb-helmholtz-accuracy-order.md`.
- Compatibility path that preserves current production behavior. Per the `008b`
  re-plan, this first pass is **side-by-side, parity-only**: the explicit M2L
  operator pipeline is validated against production but does **not** replace the
  production `multipole_to_local!` / `multipole_to_local_II!` internals.
  Production hot-path replacement is a later, explicitly scoped step.

## Verification

Run parity tests across representative source-target offsets and expansion
orders. Record commands and result summaries.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
