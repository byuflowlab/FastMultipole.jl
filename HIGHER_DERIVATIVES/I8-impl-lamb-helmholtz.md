# I8 — Lamb–Helmholtz Implementation

## Objective

Add packed-18 complex and real L2B plus analytic point-vortex direct formulas and capability opt-in. Remove temporary rejection only when all agree.

## Dependencies

T5, I7

## Deliverables and acceptance

Follow the corresponding detailed task and acceptance clauses in REVIEW_PLAN.md. Record artifacts, focused verification, and review notes here. Preserve unrelated work and inspect connected interfaces when the codebase map may be stale.

## Clear-context review (2026-09-07)

Reviewed with fresh context against the I8 clauses in `REVIEW_PLAN.md` and the T5 theory
note. Accepted; no changes required. Evidence:

- **Theory/direct agreement.** The `theory/lamb-helmholtz.md` formula
  `T_ijk = c[15 A_i x_j x_k/r⁷ − 3(A_i δ_jk + B_ij x_k + B_ik x_j)/r⁵]` was re-derived by
  hand from `v = c(Γ×x)/r³` during this review and matches. The vorton `direct!` TS block
  (`test/vortex.jl:126-143`) implements it exactly: `B` entries were checked against
  `∂(Γ×x)_i/∂x_j`, the packed loop emits the canonical component-major
  `(xx,xy,xz,yy,yz,zz)` slots, and the convention is consistent with the existing VG
  block (`H[i,j] = ∂v_i/∂x_j`). `scripts/verify_direct_formulas.jl` re-run this session:
  PASS (scalar and vortex formulas vs nested ForwardDiff at scales 1e-3, 1, 1e3).
- **L2B recurrence.** `_third_derivative_from_gradient_coefficients!`
  (`src/evaluate_expansions.jl:634-655`) applies one further solid-harmonic
  differentiation pass to the same `gradient_n_m` velocity coefficients the verified
  Hessian path contracts, then gradient-contracts columns 4–12; the slot mapping was
  checked index-by-index for both the LH branch (18 independent values, `(j,k)` symmetry
  used to skip redundant contractions) and the scalar branch (full-symmetry shortcut,
  5 contractions). Aliasing in the in-place differentiate (reads cols 1–3, writes 4–12)
  is safe; contraction degree bounds (`P−1` after one differentiation) are correct.
- **Scratch and guards.** The 12-column scratch is allocated only when some target
  requests TS (`src/tree.jl:2152`, `src/fmm.jl:775-776`); complex and real L2B gate all
  TS work behind `GS || HS || TS` / `HS || TS` / `if TS` so TS is independent of lower
  switches.
- **Capability opt-in and guard removal.** `supports_third_derivative` opt-ins exist for
  vorton sources (`test/vortex.jl:167`) and the temporary LH+TS rejection is fully
  removed (repository grep finds no residual guard); preflight remains in both
  `direct!` entry points.
- **All-three agreement.** Complex↔real L2B TS parity at LH=true passes
  (`test/real_solid_harmonic_basis_test.jl:152-177`); vorton direct vs ForwardDiff
  passes (`test/third_derivative_test.jl:92-108`); and FMM (complex L2B) vs vorton
  direct agreement is quantitative, not just smoke: this session's P-sweep (n=2000
  vortons, leaf 30, MAC 0.5, 537 M2L pairs) gives relative TS error 7.2e-9 (P=2) →
  3.0e-10 (P=6) → 4.0e-13 (P=12) → 8.4e-16 (P=20), clean geometric convergence to
  machine precision (recorded permanently by the I9 testset).
- **Focused verification.** `julia --project=test --threads=4 --check-bounds=yes` over
  `metadata_extra_test.jl` + `third_derivative_test.jl` +
  `real_solid_harmonic_basis_test.jl`: 1451 pass / 0 fail / 0 error.

## Status

See the authoritative per-phase table in [`START_HERE.md`](START_HERE.md); update its
1-sentence summary cell in place whenever this item's state changes.
