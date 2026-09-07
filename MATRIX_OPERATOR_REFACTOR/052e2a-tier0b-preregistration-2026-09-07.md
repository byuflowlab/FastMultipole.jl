# 052e.2a Tier 0B pre-registration — bordered formulation proof (2026-09-07)

**Status: LOCKED 2026-09-07 (Ryan: "lock it").** Theory gate closed (note
accepted 2026-09-07); gates, scoping, manufactured wake, truncated wake,
mesh family, and constructor details (`pitching_wing_mesh` route,
`Union{ConstantSource,ConstantDoublet}` kernel, triangular mesh,
`thickness=0.12`) all ratified. No value below may be edited after the
first results are viewed; if a fixture or gate proves ill-posed, this file
is superseded by a new dated pre-registration with the reason recorded
(Tier 0A precedent).

**Revision (same date, pre-lock):** fixture changed from unit sphere to the
capped-wing configuration of `FLOWPanel.jl/examples/simple_wing_capped.jl`
per Ryan's direction, so the fixture sheds a proper trailing-edge wake and
exercises the fixed Kutta panels. Ryan ratified the proposed gate numbers and
the stage-1-only scoping (2026-09-07).

**Scope:** the direct doublet-panel-wake formulation proof of
`052e-accuracy-plan-v2-draft-2026-09-05.md` §5 — stage 1 of its error
decomposition only (directly evaluated doublet-panel potential + reference
normal velocity). It exercises the full production bordered `:area_mean`
reconstruction $(I-B)q=S\sigma$ and nothing else: no reduced solver, no
least squares, no filament/particle stages. Stages 2–5 (finely integrated
filament, discretized filament, regularized particles,
total-minus-retained extraction) and their core-ratio / azimuthal-offset
sweeps remain registered in the accuracy plan §5 and are deliberately **out
of scope for the .2a ruling**; they run only after the formulation proof,
under their own addendum. Host-only, laptop-scale, ≤ 4 threads.

## Fixture

- **Body:** capped NACA0012 rectangular wing, span $b=2.7$ m, chord
  $c=0.76$ m, built parametrically by
  `pitching_wing_mesh` / `build_pitching_wing_body` from
  `FLOWPanel.jl/examples/pitching_wing.jl` (Ryan's direction 2026-09-07;
  replaces the fixed Gmsh mesh of `simple_wing_capped.jl` so a controlled
  refinement family exists without external tooling). Parameters:
  `thickness=0.12` (NACA0012), `endcap=:round`, `caps=true`,
  `semiinfinite_wake=false`, constructor defaults otherwise
  (`core_size=1e-6c`, `kernelcutoff=1e-12c`, `watertight=true`). Body type
  `RigidWakeBody{Union{ConstantSource,ConstantDoublet},2,Float64,true}` —
  the ConstantDoublet kernel family Tier 0A validated. Shedding via
  `calc_pitching_wing_shedding` (seed-grown along the TE at $x=c$). The
  constructor enforces watertightness and rejects degenerate triangles.
  Control points at the production collocation points; exact construction
  call chain recorded in the results file.
- **Body refinement sweep (independent of the wake), enumerated 2026-09-07
  by construction (geometry query only, no oracle run;
  script `scratchpad/tier0b_mesh_family.jl`):**

  | Level | `n_airfoil` | `n_span` | `n_endcap` | panels | shedding edges | dense $(I{-}B)$ mem |
  |---|---|---|---|---|---|---|
  | L1 | 81 | 7 | 5 | 1,744 | 7 | 0.02 GB |
  | L2 | 121 | 10 | 7 | 3,816 | 10 | 0.12 GB |
  | L3 | 181 | 15 | 11 | 8,960 | 15 | 0.64 GB |
  | L4 | 271 | 22 | 15 | 19,384 | 22 | 3.01 GB |

  ≈1.5× linear refinement per level with the spanwise/chordwise ratio held
  roughly constant; all four levels constructed watertight with shedding
  detected (one edge per spanwise station). L4 is the "finest mesh" for
  gates B1/B7; its dense bordered solve needs ≈3 GB, laptop-feasible.
- **Wake (the oracle object):** a prescribed flat doublet-panel wake shed
  from the trailing edge along the freestream direction, length $20c$,
  built of finite constant-doublet panels (Tier 0A-validated kernel family),
  paired to the trailing-edge/Kutta panels exactly as the production shedding
  machinery pairs them. The spanwise doublet distribution is **manufactured
  and frozen** (elliptic: $\mu(y)=\mu_0\sqrt{1-(2(y-b/2)/b)^2}$,
  $\mu_0=1$; secondary $\mu_0=2$ for linearity), constant streamwise — no
  body solve feeds the oracle, so solver correctness cannot leak into the
  reference. The discrete wake panel set **is** the reference object: oracle
  and production route both see the same panels, so wake discretization error
  cancels and only body-side formulation/discretization error is measured.
- **Cases (both must be run):**
  - **C1:** wake inclined at AOA $=0°$ (planar, in the chord plane).
  - **C2:** wake inclined at AOA $=7°$ (the example's operating condition),
    breaking the symmetry of C1.
- **Support/branch condition (plan §5's strict clearance HOLDS — no
  deviation):** the TE-attached Kutta (live) panel row is implicitly folded
  into the body-side solve and is excluded from the wake-potential source
  set: `PanelWake`'s old-wake source views skip the `live_rows[]` newest
  rows owned by the body-side attachment operator
  (`FLOWPanel_wake.jl` — `_n_wake_source_rows = nwakes[] - live_rows[]`;
  live-block reservation, BRAINSTORM 015 Route B / TEAnchoredAttachment).
  The oracle therefore comprises only the wake panels **beyond** the Kutta
  row, whose singular support is strictly outside the closed body, with
  minimum clearance of one Kutta-row streamwise length. Registered checks:
  (a) the harness verifies every body control point has strictly positive
  distance to every oracle wake panel and records the minimum; (b) the tail
  statistic B7 is additionally reported with the TE-adjacent body-panel row
  excluded, as a diagnostic of near-wake error concentration (not a
  deviation). The oracle sheet is the only jump surface of its potential and
  does not intersect the body, so one continuous branch covers the whole
  body surface (theory note §5).

## Reference (oracle)

At every body control point $x_i$, directly evaluate from the frozen wake
panels:

- $q_{\mathrm{ref},i}$ — induced scalar potential, via the Tier 0A-validated
  constant-doublet kernel through the public wrapper route (A10-validated);
- $\sigma_i = n_i \cdot v_{\rm wake}(x_i)$ — induced normal velocity from the
  same panels' velocity kernel (the "analytic/reference normal velocity" of
  decomposition stage 1; the structural kill rule attaches to this case).

## System under test

The production `PanelWake` → `influence!` → bordered `:area_mean` route:
build $A=I-B$ and $b=S\sigma$ from $\sigma$, solve

$$
\begin{bmatrix}A&a\\a^T&0\end{bmatrix}
\begin{bmatrix}q\\\lambda\end{bmatrix}
=
\begin{bmatrix}b\\0\end{bmatrix},
$$

with $a$ the panel-area vector. Full call chain recorded in the results file
(Tier 0A exercised only the low-level kernel; Tier 0B must exercise the
complete route, including buffer packing and any quad splitting).

## Metric definitions (locked with the gates)

- **Gauge alignment:** on the body, subtract the area-weighted mean from both
  traces before comparison: $\tilde q = q - \frac{a^Tq}{a^T\mathbf 1}$,
  likewise $\tilde q_{\rm ref}$.
- **Trace error:**
  $E_q = \dfrac{\left(\sum_i A_i(\tilde q_i-\tilde q_{\mathrm{ref},i})^2\right)^{1/2}}
  {\left(\sum_i A_i\,\tilde q_{\mathrm{ref},i}^2\right)^{1/2}}$.
- **Tail statistic (registered):**
  $E_\infty = \max_i|\tilde q_i-\tilde q_{\mathrm{ref},i}| \,/\,
  \mathrm{rms}_A(\tilde q_{\rm ref})$ with
  $\mathrm{rms}_A(\cdot)$ the area-weighted RMS; reported over all panels and
  excluding the TE-adjacent row (see support condition).
- **Flux compatibility:** $F = |\sum_i A_i\sigma_i| / \sum_i A_i|\sigma_i|$
  (declared scale: area-weighted mean absolute normal velocity).
- **Green residual:** $\|Aq+\lambda a-b\|_2/\|b\|_2$. **Gauge defect:**
  $|a^Tq|/(\|a\|_2\|q\|_2)$. Both are solver telemetry, not accuracy
  measures.
- **Hodge diagnostics:** tangential-projection defect and gauge-aligned
  Green/Hodge mismatch as already implemented in the existing telemetry
  (definitions cited from source in the results file).

## Checks and gates (numbers ratified 2026-09-07; LOCK pending mesh-family panel counts only)

| ID | Check | Gate (finest body mesh unless noted) |
|---|---|---|
| B1 | Gauge-aligned trace error $E_q$, both cases | ≤ 1e-2 |
| B2 | Body-mesh refinement of $E_q$: monotone decrease over the sweep; observed order reported | monotone; order recorded, not gated |
| B3 | Flux-compatibility precondition $F$ at every level | ≤ 1e-3 and decreasing under refinement; violation ⇒ case INVALID (not evidence against the formulation) |
| B4 | Bordered multiplier $\|\lambda\|$ recorded at every level; scales with the flux defect and → 0 under refinement | recorded; consistency noted, not hard-gated |
| B5 | Green residual and gauge defect | residual ≤ 1e-10; gauge defect ≤ 1e-12 |
| B6 | Hodge tangential defect and Green/Hodge mismatch decrease under refinement | monotone trend; values recorded |
| B7 | Tail statistic $E_\infty$ (TE-adjacent row excluded; all-panel value recorded alongside) | ≤ 3e-2 |
| B8 | Linearity: $\mu_0=2$ reconstruction equals $2\times$ ($\mu_0=1$) | rel. err ≤ 1e-12 |
| B9 | Both cases C1, C2 pass B1–B8 | required |

**Structural kill rule (from the accuracy plan, preserved verbatim in
effect):** recommend retirement of the formulation only if this
analytic-normal-velocity reconstruction exceeds 20% on the finest body mesh
**or** fails to improve under body-mesh refinement. Per the theory note, such
a failure falsifies the implemented Green trace map or its conventions;
errors in later stages (filament/particle/extraction — out of scope here)
would instead diagnose those approximations.

**Ruling structure:** 052e.2a ends with an explicit **PASS / CONTINUE /
RETIRE** ruling by Ryan. Only a PASS authorizes beginning 052e.2b
(implicit-Householder reduction + Tier 0B-R parity gate). A B1/B7 miss that
still shows clean refinement (B2) and passes the kill rule is a CONTINUE
finding (tighten discretization or revisit targets by supersession), not a
retirement.

**Failure handling (Tier 0A clause carried over):** any gate failure stops
the tier and is reported with raw numbers before any code or fixture change;
no retuning of gates in place.

## Open items blocking LOCK

None — locked 2026-09-07 (see Status header).

Resolved 2026-09-07: theory note accepted; manufactured elliptic $\mu(y)$
and finite $20c$ truncated wake ratified; the TE-attachment concern was a
misunderstanding — the Kutta row is folded into the solve and excluded from
the oracle source set (see support/branch condition), so no deviation from
plan §5 exists.

## Outputs and provenance

- `052e2a-tier0b-results-<date>.md` — per-gate PASS/FAIL table, error and
  refinement tables per case, call chain, ruling recommendation.
- CSVs of all sampled values under `data/052e2a-tier0b/`; `gates.txt`
  snapshot recording FastMultipole and FLOWPanel SHAs with dirty state and
  tracked-diff hash, thread count, and script SHA-256 (Tier 0A provenance
  standard).
- Threads ≤ 4; logs preserved under `data/052e2a-tier0b/`.
