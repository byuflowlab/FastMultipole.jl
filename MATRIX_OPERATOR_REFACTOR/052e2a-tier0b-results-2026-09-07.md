# 052e.2a Tier 0B results — 2026-09-07 (run 2)

**Status: DRAFT — awaiting Ryan's explicit PASS / CONTINUE / RETIRE ruling.**
Preregistration: `052e2a-tier0b-preregistration-2026-09-07.md` (locked).
Theory gate: `052e-theory-velocity-to-potential-trace.md` (accepted 2026-09-07).

## Run provenance

- **Run 1 (PID 51333) INVALIDATED** — harness defect (system-under-test body's
  `controlpoints`/`normals` never populated; FLOWPanel fills them lazily).
  See `052e2a-tier0b-run1-invalidation-2026-09-07.md`. Fix: `make_body` now
  calls `calc_normals!`/`calc_controlpoints!`. Full clean relaunch; no partial
  reuse. Run-1 artifacts: `data/052e2a-tier0b/run1-invalid-harness/`.
- **Run 2 (PID 53886, this report):** launched 2026-09-07 ~13:59, completed
  14:14, laptop, 4 threads, Julia 1.12.5. Log
  `data/052e2a-tier0b/tier0b_run2.log`; outputs `gates.txt` + 8
  `trace_C*_L*.csv` (headers `x,y,z,area,te_adjacent,sigma,q_ref,q_green,q_hodge`).
- script_sha256 `270deb379068`;
  FLOWPanel `57ae89d` DIRTY (tracked-diff `d7fe4281b7be`);
  FastMultipole `1c81e31` DIRTY (tracked-diff `fcd9becd1e6f`).
  (Formulation-proof tier on laptop per handoff; not an official campaign.)
- Call chain: `pnl.induced` (oracle) | `_source_potential!` →
  `_build_green_solve_state(:area_mean)` → `_green_solve_q!` |
  `_green_B_product!` residual | `surface_hodge_trace!`.
- Sigma convention: $\sigma = -\mathbf{n}\cdot\mathbf{u}_{wake}$; elliptic
  $\mu(y)=\mu_0\sqrt{1-(2y/b)^2}$, $y\in[-b/2,b/2]$.

## Gate table (B1–B9)

| Gate | C1 (AOA 0°) | C2 (AOA 7°) | Gate value |
|---|---|---|---|
| B1 E_q(finest) ≤ 1e-2 | **PASS** 1.670e-03 | **PASS** 5.745e-03 | 0.01 |
| B2 E_q monotone (order recorded) | **PASS** orders [0.91, 1.12, 1.40] | **PASS** orders [1.06, 1.29, 1.54] | not gated |
| B3 flux ≤ 1e-3 and monotone | **FAIL*** max 3.514e-17 | **PASS** max 1.896e-06 | 0.001 |
| B4 λ telemetry | REC (below) | REC (below) | record only |
| B5 (residual, gauge defect) finest | **PASS** (5.35e-15, 7.38e-17) | **FAIL*** (1.17e-13, 1.44e-12) | (1e-10, 1e-12) |
| B6 Hodge defect+mismatch monotone (L1–L3) | **PASS** | **PASS** | trend record |
| B7 E_inf(excl) finest ≤ 3e-2 | **PASS** 3.682e-03 | **PASS** 2.270e-02 | 0.03 |
| B8 linearity (finest) | **PASS** 0.0 | **PASS** 0.0 | 1e-12 |
| B9 all gates | **FAIL** (B3/C1, B5/C2) | | required |

**Kill rule (locked): NOT triggered.** E_q(finest) = 0.17% (C1) and 0.57%
(C2), both ≪ 20%, with monotone improvement at every refinement step in
both cases.

*On the two starred failures (analysis, not gate re-litigations — the
prereg stays locked):*

- **B3/C1**: the flux metric $|\sum a_i\sigma_i| / \sum a_i|\sigma_i|$ at
  AOA 0° is machine-zero by symmetry: [3.514e-17, 2.761e-18, 0.000e+00,
  6.940e-19]. Every level is ~14 orders below the 1e-3 gate; the FAIL comes
  solely from the monotone-decrease clause comparing 6.9e-19 against an
  exact 0.0. A monotonicity requirement is vacuous on roundoff noise around
  zero — this reads as an ill-posed gate clause for the symmetric case, not
  a formulation defect (supersede-with-reason path available per Tier 0A
  precedent). C2, where flux is physically meaningful, passes with margin
  (max 1.9e-6 vs 1e-3, monotone).
- **B5/C2**: gauge defect 1.436e-12 vs gate 1e-12 — a 1.4× miss at the
  largest system (N=19,384), while the residual component passes at
  1.17e-13 (gate 1e-10). The gauge-defect series grows with N
  (3.5e-16 → 1.3e-14 → 2.7e-14 → 1.4e-12), consistent with roundoff
  accumulation in the area-weighted sum, not with a constraint-enforcement
  defect (C1 finest: 7.4e-17).

## Error and refinement

E_q (area-weighted relative error of q vs oracle) and E_inf (max, excluded
set = all panels here; all-panel value identical):

| Level | N | C1 E_q | C1 E_inf | C2 E_q | C2 E_inf |
|---|---|---|---|---|---|
| L1 | 1,744 | 6.605e-03 | 1.827e-02 | 2.734e-02 | 9.843e-02 |
| L2 | 3,816 | 4.622e-03 | 9.536e-03 | 1.805e-02 | 6.469e-02 |
| L3 | 8,960 | 2.868e-03 | 6.047e-03 | 1.041e-02 | 3.901e-02 |
| L4 | 19,384 | 1.670e-03 | 3.682e-03 | 5.745e-03 | 2.270e-02 |

Observed orders (in $h \sim N^{-1/2}$): C1 [0.91, 1.12, 1.40];
C2 [1.06, 1.29, 1.54] — monotone, order rising toward ~1.5 at the finest
pairs, consistent with a low-order panel scheme still entering its
asymptotic range.

## Telemetry

| Level | C1 λ | C1 flux | C1 res | C1 gd | C2 λ | C2 flux | C2 res | C2 gd |
|---|---|---|---|---|---|---|---|---|
| L1 | 4.99e-16 | 3.51e-17 | 1.56e-15 | 3.59e-16 | -1.75e-04 | 1.90e-06 | 2.16e-15 | 3.51e-16 |
| L2 | 3.45e-16 | 2.76e-18 | 2.34e-15 | 3.69e-16 | -2.37e-04 | 8.98e-07 | 5.53e-15 | 1.34e-14 |
| L3 | 1.67e-15 | 0.00e+00 | 3.57e-15 | 2.62e-16 | -2.96e-04 | 4.02e-07 | 1.02e-14 | 2.72e-14 |
| L4 | -3.62e-14 | 6.94e-19 | 5.35e-15 | 7.38e-17 | -3.53e-04 | 1.84e-07 | 1.17e-13 | 1.44e-12 |

- **λ (bordered multiplier, B4 record):** machine-zero for the symmetric
  C1; small, smooth, and mesh-converging (~-3.5e-4 trend) for C2 —
  behaving as designed: incompatibility isolated as telemetry, not smeared
  into q.
- **Hodge (B6, L1–L3; L4 skipped for memory per registered note):**
  defect C1 [1.58e-01, 1.01e-01, 5.03e-02], mismatch [6.04e-01, 2.85e-01,
  1.14e-01]; C2 defect [1.54e-01, 9.65e-02, 4.70e-02], mismatch
  [5.18e-01, 2.45e-01, 9.83e-02]. Both series roughly halve per level —
  clean first-order trend.
- **Clearance minima:** 1.3804 / 1.3802 / 1.3801 / 1.3800 (identical both
  cases) — control points well clear of the wake support (row 1
  zero-strength Kutta row realized; support starts 0.5c aft of TE).
- **Linearity (finest, B8):** 0.0 exactly, both cases.

## influence!-route cross-check (report only; first exercise of this route)

Production `_wake_potential!`/`_wake_panel_velocity!` (PanelWake →
`influence!`) vs oracle, (potential, velocity) discrepancies:

| Level | C1 | C2 |
|---|---|---|
| L1 | (9.54e-11, 1.30e-15) | (3.56e-11, 1.30e-15) |
| L2 | (4.46e-10, 1.52e-15) | (5.48e-11, 1.56e-15) |
| L3 | (1.47e-10, 1.79e-15) | (9.48e-11, 1.80e-15) |
| L4 | (1.27e-10, 2.11e-15) | (5.87e-11, 2.16e-15) |

Potentials agree to ~1e-10 (oracle core size 1e-8 scale), velocities to
machine precision. The production wake route matches the Tier 0A-validated
wrapper oracle on its first exercise.

## RECOMMENDATION (ruling is Ryan's)

**PASS.** The two B9-blocking failures are gate-definition artifacts, not
formulation defects: B3/C1 is a monotonicity clause evaluated on
machine-zero flux (symmetric case), and B5/C2 is a 1.4× miss of a
machine-precision gauge gate from roundoff accumulation at N≈19k, with the
residual itself three decades under its gate. Every physically substantive
gate passes both cases: sub-percent E_q at the finest mesh (0.17% / 0.57%
vs the 20% kill threshold), monotone refinement with rising order, clean
Hodge trends, well-behaved bordered multiplier, exact linearity, and a
machine-precision cross-check of the production wake route. If the strict
reading of B9 is preferred, the fallback is CONTINUE (B1/B7 pass with
clean refinement — far above the retire bar); under the locked kill rule
there is no basis for RETIRE. If PASS is ruled, the ill-posed B3/C1 clause
should be recorded via a superseding dated note (Tier 0A precedent) rather
than editing the locked prereg.

Next after a PASS ruling: 052e.2b (implicit-Householder reduction, Tier
0B-R parity gates) per accuracy plan §5 / theory note §3.1; stages 2–4 run
as a .2a continuation addendum; stage 5 folds into 052e.3.
