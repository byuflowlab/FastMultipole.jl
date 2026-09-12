# 052e.2b Tier 0B-R results — implicit-Householder reduction parity (2026-09-07)

**Status: RULED — ADOPT (Ryan 2026-09-07, "that's good enough").** The
single formal gate FAIL (R3, factor 2.0 over its preregistered tolerance;
details below) is adjudicated a tolerance-model artifact under the ruling
— see `052e2b-tier0br-gate-adjudication-2026-09-07.md`. The
pre-registration stays LOCKED and unedited (Tier 0B B3/B5 precedent).
Adoption makes the implicit-Householder reduction the production
representation; the bordered route is retained as the debug/reference
path.

Pre-registration: `052e2b-tier0br-preregistration-2026-09-07.md`
(LOCKED 2026-09-07). Run 1, PID 66786, launched nohup-detached,
completed 2026-09-07; provenance and registered values in
`data/052e2b-tier0br/gates.txt` (script_sha256=d86e666e43d0,
julia 1.12.5, 4 threads; FLOWPanel abb65073 DIRTY
tracked-diff=d7fe4281b7be, FastMultipole 8c250067 DIRTY
tracked-diff=921614b73f12). Smoke validation passed before launch
(TIER0BR_SMOKE=1, exit 0).

## R0 implementation audit (required pre-run) — PASS

Recorded in the harness header
(`scripts/tier0br_052e2b_householder_parity.jl`) and confirmed by
inspection of `make_reflector` / `reduced_setup!` / `reduced_solve`:

- no dense $Z$ or explicit basis formed or stored (reflector vector $v$
  only; transforms via two rank-one BLAS updates);
- no $N\times N$ projection $P=I-aa^T/(a^Ta)$ formed or factored;
- two-sided transform $\widetilde A = HAH^T$ (trial **and** equation
  spaces), LU of the leading $(N-1)\times(N-1)$ block in place;
- cancellation-avoiding sign $s=-\mathrm{sign}(\widehat a_N)$ with the
  normalized area vector; $\lambda$ recovered from the omitted last
  equation.

The reduced route consumed $A=I-B$ from a second call of the same
deterministic `_assemble_B!` the production route uses (the production LU
destroys its copy); identity of the assemblies is confirmed by gate R2's
full-coordinate residual, which evaluates the reduced solution against
this assembly at 1e-15 scale.

## Environment-drift precondition — PASS (all four anchors)

Recomputed bordered $E_q$ vs run-2 registered anchors (applied at the
anchors' 7-significant-digit quantization, ≤1e-6 relative; registered
implementation note in the harness header — the prereg's 1e-10 clause is
unapplicable as written against 7-digit anchors):

| Case/Level | recomputed $E_q$ | run-2 anchor | drift |
|---|---|---|---|
| C1/L2 | 4.622344743e-3 | 4.622345e-3 | 5.6e-8 |
| C1/L4 | 1.669866537e-3 | 1.669867e-3 | 2.8e-7 |
| C2/L2 | 1.804733863e-2 | 1.804734e-2 | 7.6e-8 |
| C2/L4 | 5.745107858e-3 | 5.745108e-3 | 2.5e-8 |

## Gate table

$\tau(N)=10^3\sqrt N\,\varepsilon$: 1.37e-11 (L2, N=3,816), 3.09e-11
(L4, N=19,384). $\tau_g(N)=10^2\sqrt N\,\varepsilon$: 1.37e-12 / 3.09e-12.

| ID | Values | Gate | Result |
|---|---|---|---|
| R0 | audit above | required | **PASS** |
| R1/C1/L2 | $\Pi_q$=2.53e-14, $G$=2.00e-18 | (1.37e-11, 1.37e-12) | **PASS** |
| R1/C1/L4 | $\Pi_q$=6.24e-14, $G$=3.81e-18 | (3.09e-11, 3.09e-12) | **PASS** |
| R1/C2/L2 | $\Pi_q$=2.94e-13, $G$=3.72e-17 | (1.37e-11, 1.37e-12) | **PASS** |
| R1/C2/L4 | $\Pi_q$=1.89e-11, $G$=1.67e-16 | (3.09e-11, 3.09e-12) | **PASS** (0.61× gate) |
| R2/C1/L2 | $\Pi_\lambda$=6.80e-18, res=2.34e-15 | (1.37e-11, 1e-10) | **PASS** |
| R2/C1/L4 | $\Pi_\lambda$=8.82e-18, res=5.39e-15 | (3.09e-11, 1e-10) | **PASS** |
| R2/C2/L2 | $\Pi_\lambda$=4.77e-17, res=3.12e-15 | (1.37e-11, 1e-10) | **PASS** |
| R2/C2/L4 | $\Pi_\lambda$=3.99e-19, res=7.29e-15 | (3.09e-11, 1e-10) | **PASS** |
| R3/C2/L2 | $\Pi_{q'}$=**2.70e-11**, $\Pi_{\lambda'}$=6.21e-16 (F′=1.00e-2, $\lambda_b'$=−7.94e-2) | ≤ 1.37e-11 | **FAIL** (2.0×) |
| R4 | — | all pass | **FAIL** (via R3) |

## The R3 failure, raw and in context

The incompatible-RHS variant ($\sigma'=\sigma+c\mathbf 1$, achieved
$F'=1.00\times10^{-2}$) produced trace parity $\Pi_{q'}=2.70\times10^{-11}$
against gate $\tau(3816)=1.37\times10^{-11}$ — a factor-2.0 formal miss.
Observations (analysis, not adjudication):

- The **multiplier parity** — the quantity R3 exists to test, per the
  theory-note claim that the reduction preserves the bordered treatment of
  discrete incompatibility — is $\Pi_{\lambda'}=6.2\times10^{-16}$:
  machine-precision agreement on $\lambda'=-7.94\times10^{-2}$, eleven
  orders inside the gate.
- Both routes' solutions are individually consistent: full-coordinate
  residuals stayed at 1e-15 throughout, and the compatible-data parity at
  the same level/case is 2.9e-13.
- The incompatible RHS adds a large near-constant component to $b$ whose
  image is carried almost entirely by $\lambda'$ (|λ'| is ~450× the
  compatible case's), so intermediate quantities scale with the inflated
  $\|b'\|$ while the metric denominator (aligned trace RMS) does not.
  LU roundoff on that scale mix, through two different factorizations,
  plausibly accounts for the ~30× inflation over compatible-case parity.
  The preregistered $\tau(N)$ carries no condition-number or RHS-scale
  factor ($1/\mathrm{rcond}\approx3.8\times10^3$ at L2 alone predicts
  $\kappa\varepsilon\approx8.5\times10^{-13}$ before growth factors).
- Note also R1/C2/L4 passed at 0.61× of its gate: the $K=10^3$ constant
  sits genuinely close to the true roundoff scale of this system; it was
  not conservative.

## Telemetry (recorded, not gated)

| | L2 | L4 |
|---|---|---|
| setup bordered / reduced [s] | 1.67–2.03 / 1.73–1.83 | 58.7–63.0 / 61.1–61.2 |
| per-RHS solve bordered / reduced [s] | 0.0044–0.0046 / 0.0043 | 0.111–0.114 / 0.110–0.111 |
| storage bordered / reduced [B] | 1.1656e8 / 1.1659e8 | 3.00623e9 / 3.00638e9 |
| rcond bordered / reduced (1-norm est.) | 2.62e-4 / 3.16e-4 | 4.99e-5 / 8.20e-5 |

Setup within ±4%, per-RHS solve at parity, storage at parity (the reduced
figure counts the full retained transform buffer including its last
row/column; the LU proper is $(N-1)^2$). The reduced block is **better**
conditioned than the bordered matrix at both sizes (ratio 0.61–0.83) — no
conditioning finding; the >10× reporting trigger was never approached.
Peak maxrss 7.39 GB.

## Ruling recommendation

All compatible-data parity gates (R1, R2) pass at every level and case,
most by 2–3 orders of magnitude; conditioning and cost telemetry favor or
match the bordered route. The single formal FAIL is a factor-2.0 miss on
the incompatible-RHS trace-parity tolerance while that variant's
multiplier — its actual subject — matches at machine precision. Options
under the locked decision rule:

1. **Strict reading:** parity failed ⇒ RETAIN-BORDERED (the rule as
   written).
2. **Artifact adjudication** (Tier 0B B3/B5 precedent): rule the R3 miss
   a tolerance-model artifact (no κ/RHS-scale factor in $\tau$) in a
   supersession note, prereg stays locked/unedited, and rule ADOPT.
3. **Supersede and rerun:** new dated prereg with a condition-aware
   tolerance, rerun the single variant.

The evidence is consistent with an exact algebraic equivalence
implemented correctly, with the miss located in the tolerance model
rather than the reduction. Decision is Ryan's.
