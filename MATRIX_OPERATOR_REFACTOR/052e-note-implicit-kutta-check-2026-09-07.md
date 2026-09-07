# 052e one-time code check (2026-09-07): implicit Kutta strength difference in HybridWakePotential

Claim checked (from `052e-accuracy-plan-v2-draft-2026-09-05.md` §2):
"`HybridWakePotential` does not call `_apply_kutta_map!` explicitly, but the
ordinary body operator and downstream shedding apply the equivalent strength
difference implicitly." **VERIFIED** against source on FLOWPanel
`fastmultipole` @ b9c24e0 (plus the 052e.1 init-error edit, which does not
touch these paths):

1. **Operator side** — `_induced_wake`
   (`src/FLOWPanel_elements_fmm.jl:1457-1500`): each shedding body panel's
   influence includes its own attached-wake strip using that panel's doublet
   strength; paired upper/lower strips combine so the net attached
   circulation is $\gamma=\mu_u-\mu_l-c$ (comment at the
   `wake_strength_shift` block). The hybrid runs with $c=0$
   (`clear_wake_correction!` in `solve_formulation!`,
   `src/FLOWPanel_formulation.jl:1077`). The Kutta coupling is therefore
   inside the assembled operator G; `suppress_attached_wake` removes it only
   for Green B-matrix assembly (body-only operator), by design.
2. **Downstream shedding** — `_get_wakestrength_Gamma`
   (`src/FLOWPanel_liftingbody.jl:1619-1627`) sheds
   `strength1 - strength2` (upper-minus-lower); the particle conversion uses
   the same expression and sign (`src/FLOWPanel_wake.jl:2745-2753`).
3. **Hybrid path unmodified** — the wake trace enters only through
   `body.potential` (Dirichlet RHS, `solver.rhs .= -self.potential`) before
   the ordinary tuple `solve!`; no separate Kutta application exists or is
   needed.

Corroborating hazard now closed: for an unpaired edge,
`_get_wakestrength_mu` silently substitutes `strength2 = 0`
(`src/FLOWPanel_liftingbody.jl:1608-1618`) — exactly the failure mode the
new 052e.1 hard initialization error blocks.

Consequence for Tier 1: the fixture may rely on the ordinary solve to carry
the Kutta strength difference; no explicit Kutta-map plumbing is required in
the Tier 1 harness.
