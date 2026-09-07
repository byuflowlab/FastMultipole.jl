# 052e.0 Tier 0A pre-registration — doublet kernel convention test (2026-09-07)

**Status:** LOCKED before first run per
`052e-accuracy-plan-v2-draft-2026-09-05.md` §4 and §11 (gates ratified per
subitem immediately before it runs). Ryan directed work to proceed on
2026-09-07 ("get to work") following presentation of the theory note; formal
per-gate ratification may be recorded post-hoc against this file. No value
below may be edited after the first results are viewed; if a fixture or gate
proves ill-posed, this file is superseded by a new dated pre-registration and
the reason recorded.

**Scope:** standalone kernel convention check only. Does NOT gate the hybrid
reconstruction (that is Tier 0B / 052e.2). Host-only, laptop-scale,
single-threaded or ≤4 threads.

## Fixture

- Flat circular disk, radius $R=1$, centered at origin, normal $+\hat z$,
  constant doublet strength $\mu=1$ (secondary value $\mu=2$ for linearity).
- Discretization: structured polar grid of quadrilateral panels (triangles at
  the hub if required by the mesh helper), $n_\theta$ azimuthal ×
  $n_r$ radial panels with $n_r = n_\theta/4$.
- Refinement sweep: $n_\theta \in \{32, 64, 128, 256\}$.
- Kernel under test: the SAME constant-doublet panel potential routine that
  `HybridWakePotential` uses to evaluate retained-panel potential at body
  control points (call chain recorded in the results file).

## Reference

On-axis zero-at-infinity branch, $z \ne 0$:

$$
\Omega(z)=2\pi\left(\operatorname{sgn}(z)-\frac{z}{\sqrt{z^2+R^2}}\right),
\qquad \varphi_a(z) = -\frac{\mu\,\Omega(z)}{4\pi}.
$$

Upper-minus-lower convention: $[\Omega]=4\pi$, $[\varphi]=-\mu$. The
implementation's branch is identified (not assumed): it must match
$\varphi_a$ either directly or after adding a constant $\pm\mu$
(= $4\pi$ solid-angle ambiguity) uniformly on one side; the branch found is
recorded. Off-axis reference points use the polygon-exact solid angle of the
discretized disk computed by an independent per-triangle Van Oosterom–Strackee
implementation written for this test (independent of FLOWPanel source).

## Point sets (locked)

- On-axis: $z/R \in \pm\{0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20\}$.
- Off-axis (exercises non-axisymmetric evaluation): $\rho = 0.5R$ at
  azimuth $\{0°, 45°\}$, $z/R \in \pm\{0.25, 1\}$; compared against the
  independent polygon solid-angle reference. Exclusion: no point on the sheet
  plane $z=0$ or within $0.05R$ of the rim circle.
- Far-field decay fit: on-axis $z/R \in \{5, 10, 20\}$, both signs.
- Circulation loop: rectangular loop in the $x$–$z$ plane through the disk
  interior: $(0.5R,\pm Z)$ and $(X,\pm Z)$ corners with $X=25R$, $Z=25R$,
  composite Gauss–Legendre quadrature with total error estimated by doubling
  nodes (quadrature-refinement delta must be < 10% of the gate margin).

## Checks and gates (PROPOSED → LOCKED)

| ID | Check | Gate (finest mesh $n_\theta=256$ unless noted) |
|---|---|---|
| A1 | Sign/jump convention: $\varphi(z{=}+0.05R)-\varphi(z{=}-0.05R)$ vs analytic $-\mu\,[\Omega(+0.05R)-\Omega(-0.05R)]/4\pi$ | rel. err ≤ 1e-3 |
| A2 | On-axis branch match both sides, all on-axis points, after branch identification | max abs err ≤ 1e-3·\|μ\| |
| A3 | Linearity: $\varphi(\mu{=}2) = 2\varphi(\mu{=}1)$ at all points | rel. err ≤ 1e-12 |
| A4 | Orientation reversal (flip node ordering/normal) negates $\varphi$ | rel. err ≤ 1e-12 |
| A5 | Far-field decay: log–log slope of \|φ\| on $z/R\in[5,20]$, both sides | slope $=-2.0 \pm 0.05$ |
| A6 | Circulation normalization: loop integral of kernel velocity $\Gamma$ vs $\mu$ | rel. err ≤ 1e-3 |
| A7 | Branch ambiguity: identified branch constant ∈ $\{0, \pm\mu\}$ per side, uniform across that side's points | consistency to 1e-6·\|μ\| |
| A8 | Refinement: A2 error vs $n_\theta$ decreases monotonically; observed order over the sweep | order ≥ 1.5 |

Notes locked with the gates:

- If the implemented panel potential is analytically exact per flat panel
  (solid-angle formula), A2/A8 measure only polygon-vs-circle geometry error
  and A8's expected order is 2 in $n_\theta$; if it is quadrature-based, A8
  additionally reflects quadrature order. Either is acceptable; the observed
  mechanism is recorded, and the polygon-exact off-axis comparison must then
  agree to ≤ 1e-6·\|μ\| (kernel-vs-independent-implementation on identical
  geometry) — recorded as **A9, gate 1e-6** when the kernel is exact, waived
  with reason if quadrature-based.
- Failure handling: any gate failure stops Tier 0A and is reported with the
  raw numbers before any code or fixture change; no retuning of gates.
- Threads ≤ 4; results and logs go to
  `MATRIX_OPERATOR_REFACTOR/data/052e0-tier0a/` and a results md in
  `MATRIX_OPERATOR_REFACTOR/`.

## Addendum (2026-09-07, recorded BEFORE any run; no results viewed)

Source reading (not a run) indicates the implementation may realize
$\varphi = +\mu\Omega/4\pi$ rather than the plan's $-\mu\Omega/4\pi$,
depending on winding convention. Amendment to A2/A7: the test identifies an
overall sign $s\in\{+1,-1\}$ and per-side branch constants
$c_\pm\in\{0,\pm\mu\}$ such that $\varphi_{\rm impl} = s\,\varphi_a + c_\pm$;
gates A2/A7 apply after applying the identified $(s, c_\pm)$, and the
identified convention is reported verbatim. A1 (jump magnitude and its sign
relative to the panel winding/normal) and A4 are unaffected: the jump must
be $\mp s\,\mu$ consistently and orientation reversal must negate $\varphi$.

## Outputs

- `052e0-tier0a-results-<date>.md` — per-gate PASS/FAIL table, error tables,
  identified branch and call chain, environment/revision metadata.
- CSVs of all sampled values under `data/052e0-tier0a/`.
