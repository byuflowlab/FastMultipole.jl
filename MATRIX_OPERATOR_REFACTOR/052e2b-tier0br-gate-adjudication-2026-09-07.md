# 052e.2b Tier 0B-R gate adjudication — R3 ruled artifact under ADOPT (2026-09-07)

**Ruling: ADOPT (Ryan 2026-09-07, "that's good enough"),** on the results
in `052e2b-tier0br-results-2026-09-07.md` (run 1, PID 66786,
script_sha256=d86e666e43d0). The pre-registration
`052e2b-tier0br-preregistration-2026-09-07.md` stays LOCKED and unedited;
this note records the adjudication of its one formal FAIL, following the
Tier 0B gate-supersession precedent
(`052e2a-tier0b-gate-supersession-2026-09-07.md`).

## The formal FAIL

**R3/C2/L2** (deliberately incompatible RHS, $F'=1.00\times10^{-2}$):
trace parity $\Pi_{q'}=2.702\times10^{-11}$ vs gate
$\tau(3816)=1.37\times10^{-11}$ — factor 2.0 over. Multiplier parity in
the same variant: $\Pi_{\lambda'}=6.2\times10^{-16}$ on
$\lambda'=-7.94\times10^{-2}$.

## Adjudication: tolerance-model artifact, not a reduction defect

- The preregistered tolerance $\tau(N)=10^3\sqrt N\,\varepsilon$ carries
  no condition-number or RHS-scale factor. The incompatible variant
  inflates $\|b'\|$ with a near-constant component carried almost entirely
  by $\lambda'$ (~450× the compatible case's), so LU roundoff through two
  different factorizations scales with the inflated RHS while the metric
  denominator (aligned trace RMS) does not.
- The quantity R3 exists to test — preservation of the bordered treatment
  of discrete incompatibility via $\lambda$ — agreed at machine precision.
- Both routes were individually consistent (full-coordinate residuals
  ~1e-15), and the compatible-data parity at the same level/case was
  2.9e-13; the passing R1/C2/L4 landed at 0.61× of its gate, showing the
  $K=10^3$ constant sat close to the true roundoff scale rather than
  conservative.
- Lesson for future parity tiers (extends the Tier 0B B5 N-aware lesson):
  roundoff-parity tolerances should carry a condition/RHS-scale factor,
  e.g. $\tau \propto \kappa_1\,\varepsilon\,\|b\|_2/\mathrm{scale}$, not
  a purely $N$-based constant.

## Effect of ADOPT

The implicit-Householder $(N-1)\times(N-1)$ reduction becomes the
production representation of the area-gauge Green solve; the bordered
`:area_mean` route is retained as the debug/reference path (per the
ruling structure of the locked prereg). Production implementation in
FLOWPanel (`_build_green_solve_state`-level integration + tests) is the
follow-on task; until it lands, the validated reference implementation is
`scripts/tier0br_052e2b_householder_parity.jl`
(`make_reflector` / `reduced_setup!` / `reduced_solve`).
