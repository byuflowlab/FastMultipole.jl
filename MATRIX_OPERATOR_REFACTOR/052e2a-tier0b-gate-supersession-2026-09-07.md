# 052e.2a Tier 0B — gate supersession record — 2026-09-07

Companion to `052e2a-tier0b-results-2026-09-07.md` (run 2). The
preregistration `052e2a-tier0b-preregistration-2026-09-07.md` is LOCKED
and remains unedited; per the Tier 0A precedent, ill-posed gate clauses
are superseded here with reasons, after results existed and under Ryan's
explicit ruling.

**Ruling: PASS on 052e.2a stage 1 (bordered :area_mean trace
formulation), accepted by Ryan 2026-09-07.** B9's strict all-gates
reading is superseded by this adjudication of its two blocking failures:

1. **B3/C1 (flux ≤ 1e-3 AND monotone decreasing) — monotonicity clause
   ill-posed for the symmetric case.** At AOA 0° the flux metric is
   machine-zero by symmetry at every level
   ([3.514e-17, 2.761e-18, 0.000e+00, 6.940e-19] — ~14 orders below the
   1e-3 gate); the FAIL arose solely from comparing 6.9e-19 against an
   exact 0.0. A monotone-decrease requirement on roundoff noise around
   zero cannot be satisfied in principle. The clause is superseded for
   cases whose flux is at machine zero (≤ ~1e-14): there the magnitude
   bound alone governs. C2, where flux is physically meaningful, passed
   both clauses with margin.
2. **B5/C2 gauge-defect component (≤ 1e-12) — threshold did not scale
   with N.** Measured 1.436e-12 at N=19,384 while the residual component
   passed at 1.17e-13 (gate 1e-10). The gauge-defect series
   (3.5e-16 → 1.3e-14 → 2.7e-14 → 1.4e-12) grows with system size as
   roundoff accumulation in the area-weighted constraint sum, not as a
   constraint-enforcement defect (C1 finest: 7.4e-17). Superseded as a
   1.4× miss of a machine-precision threshold at the largest system;
   future tiers should gate the gauge defect with an N-aware bound
   (e.g. C·N·eps or relative to ‖a‖‖q‖·sqrt(N)·eps).

All physically substantive gates (B1, B2, B6, B7, B8; B3/C2, B5/C1)
passed both cases; the locked kill rule was not triggered (E_q finest
0.17% / 0.57% vs 20%, monotone refinement). The max-error locus was
verified post-hoc to be the tip-cap sliver panels at y=±b/2 (elliptic-
loading wake-edge singularity meeting the worst-conditioned panels),
converging at order ~1.3 — recorded in the results doc discussion.

**Consequence:** 052e.2b (implicit-Householder reduction, Tier 0B-R
parity gates) is unlocked per accuracy plan §5 / theory note §3.1;
stages 2–4 run as a .2a continuation addendum; stage 5 folds into
052e.3.
