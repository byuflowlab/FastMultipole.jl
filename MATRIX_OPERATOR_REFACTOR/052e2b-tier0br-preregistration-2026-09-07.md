# 052e.2b Tier 0B-R pre-registration — implicit-Householder reduction parity (2026-09-07)

**Status: LOCKED 2026-09-07 (Ryan: "lock it and do it").** Tolerance
constants and the descope paragraph ratified with the lock (open items 1–2
resolved). Authorized by the Tier 0B PASS ruling (Ryan 2026-09-07,
`052e2a-tier0b-results-2026-09-07.md`). No value below may be edited after
the first results are viewed; if a fixture or gate proves ill-posed, this
file is superseded by a new dated pre-registration with the reason
recorded (Tier 0A/0B precedent).

**Scope:** the Tier 0B-R reduced-system parity gate of
`052e-accuracy-plan-v2-draft-2026-09-05.md` §5. The theory note
(`052e-theory-velocity-to-potential-trace.md` §3.1, ACCEPTED 2026-09-07)
proves the implicit-Householder solve is an *exact* $(N-1)\times(N-1)$
coordinate reduction of the bordered system, so this tier only verifies
(1) the implementation follows the theory and (2) the reduced solution
matches the bordered one at roundoff scale. Representation test only —
the formulation was validated in Tier 0B. Host-only, ≤ 4 threads.

**Descope (ratified at lock, Ryan 2026-09-07):** plan §5 additionally
lists disconnected-body, distorted-area-mesh, and
circulation/exterior-velocity/loads parity items, and performance gating.
These are descoped from the .2b ruling: trace parity at roundoff scale
implies parity of every derived gauge-invariant output, and the fixture
variants add harness work without testing new theory. Timings are
recorded, not gated. If the reduction is adopted and later exercised on
multi-body or strongly graded meshes in production, per-body reduction
correctness is covered there (052e.3 hybrid fixture and beyond).

## System under test

The implicit-Householder reduction of theory note §3.1. With $A=I-B$,
$b=S\sigma$, $a$ the panel-area vector, $\widehat a=a/\|a\|_2$: choose the
cancellation-avoiding Householder reflector $H=I-2vv^T$ with
$H\widehat a=s\,e_N$; implicitly transform $\widetilde A=HAH^T$ via the
rank-one form, LU-factor the leading $(N-1)\times(N-1)$ block, transform
$\widetilde b=Hb$, solve for $y$, recover $q=H^T[y;0]$ and

$$
\lambda=\frac{s}{\|a\|_2}\left(\widetilde b_N-\widetilde A_{N,1:N-1}\,y\right).
$$

## Fixture

Inherited unchanged from the LOCKED Tier 0B pre-registration
(`052e2a-tier0b-preregistration-2026-09-07.md`): capped NACA0012 wing via
`pitching_wing_mesh`, frozen elliptic doublet-panel wake oracle, cases
**C1** (AOA 0°) and **C2** (AOA 7°). Levels **L2** (N=3,816) and **L4**
(N=19,384) only — two sizes suffice to exercise the N-dependence of
roundoff parity. The oracle is unchanged; this tier compares two solve
routes on identical assembled $A$, $b$, $a$.

**Incompatible-RHS variant (C2/L2 only):** $\sigma'=\sigma+c\,\mathbf 1$
with $c = 10^{-2}\left(\sum_i A_i|\sigma_i|\right)/\left(\sum_i A_i\right)$;
achieved flux measure recorded. Tests the theory claim that the reduction
preserves the bordered treatment of a discretely incompatible right-hand
side (nonzero $\lambda$ recovered identically).

## Reference

The bordered `:area_mean` solution **recomputed in the same process from
the same assembled $A$, $b$, $a$** — roundoff-scale parity is only
meaningful same-process. **Environment-drift precondition:** the
recomputed bordered $E_q$ (vs oracle) must match the Tier 0B run-2
registered values (`data/052e2a-tier0b/gates.txt`,
script_sha256=270deb379068) to $10^{-10}$ relative — L2: 4.622345e-3
(C1), 1.804734e-2 (C2); L4: 1.669867e-3 (C1), 5.745108e-3 (C2) — else the
run is INVALID (drift, not evidence about the reduction).

## Metric definitions (locked with the gates)

$\varepsilon=2^{-52}$; N-aware roundoff-scaled tolerance (B5-supersession
lesson: no fixed thresholds at large $N$):

$$
\tau(N)=10^3\sqrt N\,\varepsilon,\qquad
\tau_g(N)=10^2\sqrt N\,\varepsilon.
$$

($\tau\approx1.4\times10^{-11}$ at L2, $2.9\times10^{-11}$ at L4.)

- **Trace parity:** $\Pi_q=\|q_{\rm red}-q_{\rm bord}\|_\infty/\mathrm{rms}_A(\tilde q_{\rm bord})$
  (both traces area-mean aligned as in Tier 0B).
- **Multiplier parity:** $\Pi_\lambda=|\lambda_{\rm red}-\lambda_{\rm bord}|/\max\!\left(|\lambda_{\rm bord}|,\ \|b\|_2/\|a\|_2\right)$
  (floor keeps the metric meaningful when $\lambda$ is machine zero, C1).
- **Gauge defect (reduced):** $G=|a^Tq_{\rm red}|/(\|a\|_2\|q_{\rm red}\|_2)$.
- **Residual (reduced, full coordinates):** $\|Aq_{\rm red}+\lambda_{\rm red}a-b\|_2/\|b\|_2$.

## Checks and gates (numbers ratified at lock, 2026-09-07)

| ID | Check | Gate |
|---|---|---|
| R0 | Implementation audit (code inspection, recorded in results file): no dense $Z$ or explicit basis; no full $N\times N$ projection $P=I-aa^T/(a^Ta)$ formed or factored; two-sided transform (trial *and* equation spaces); cancellation-avoiding reflector sign with normalized $\widehat a$ | required before the registered run |
| R1 | Trace parity $\Pi_q$, L2 & L4, C1 & C2 | $\le \tau(N)$ |
| R2 | Multiplier parity $\Pi_\lambda$, gauge defect $G$, residual — same cases | $\Pi_\lambda\le\tau(N)$; $G\le\tau_g(N)$; residual $\le10^{-10}$ |
| R3 | Incompatible-RHS variant (C2/L2): $\Pi_{q'}$ and $\Pi_{\lambda'}$ | $\le \tau(N)$ |
| R4 | All of R0–R3, all cases | required for ADOPT |

**Recorded, not gated:** setup time, per-RHS solve time, factor storage,
peak construction memory (both routes, L4/C2, ≤4 threads, median of 5
after warmup); 1-norm condition estimates of both matrices. A reduced
condition estimate exceeding the bordered one by >10× is reported
prominently as a conditioning finding under the decision rule.

**Decision rule (plan §5, verbatim in effect):** adoption requires parity
without regression. **If parity or conditioning fails, retain the bordered
route; do not automatically substitute least squares, rank-one completion,
row replacement, or an iterative method.** No formulation kill rule — a
failure here retires only the reduced representation.

**Ruling structure:** ends with an explicit **ADOPT / RETAIN-BORDERED**
ruling by Ryan.

**Failure handling (Tier 0A/0B clause):** any gate failure stops the tier
and is reported with raw numbers before any code or fixture change; no
retuning in place. An INVALID drift precondition stops the run without
prejudice.

## Harness and run protocol

- Extend `scripts/tier0b_052e2a_bordered_formulation.jl` (run-2 version,
  includes the `calc_normals!`/`calc_controlpoints!` fix); both routes
  consume the identical assembled $A$, $b$, $a$ in one process.
- Smoke-validate mechanically first (`TIER0B_SMOKE=1` pattern, exit
  status only); registered run nohup-detached,
  `julia --project=../FLOWPanel.jl --threads=4`, logs/CSVs under
  `data/052e2b-tier0br/`. Never read `data/**` CSVs into agent context.

## Open items blocking LOCK

None — locked 2026-09-07 (see Status header). Tolerance constants and the
descope paragraph ratified as proposed; the descope supersedes the
plan-§5 item list for the .2b ruling; the plan file itself is left
unedited.

## Outputs and provenance

- `052e2b-tier0br-results-<date>.md` — per-gate table, parity values,
  audit record, timings/conditioning telemetry, ruling recommendation.
- `data/052e2b-tier0br/gates.txt` snapshot: FastMultipole + FLOWPanel SHAs
  with dirty state and tracked-diff hash, thread count, script SHA-256
  (Tier 0A/0B provenance standard). Threads ≤ 4.
