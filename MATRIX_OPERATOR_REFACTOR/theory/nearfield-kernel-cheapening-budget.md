# Nearfield kernel-cheapening error budget (task 037f)

Derivation of the pointwise accuracy budgets that gate the cheapened
`gaussianerf` $g/h$ pair-kernel modes (`CUDA_NEARFIELD_GH_MODE`:
`:reduced`, `:fp32`, `:reduced_fp32`, `:lut`), and the mapping from those
pointwise budgets to delivered sampled-velocity RMS error.  Companion
script: `scripts/fm037f_error_budget.jl` (stdlib-only, local-safe); data
of record: `data/kernel_splitting/fm037f_budget.csv`.  All numbers below
are from the 2026-08-14 run at the campaign operating cutoff
$\rho_t = 3.668$.

## 1. Delivered-error allowance

The accuracy gate is $10^{-3}$ sampled relative velocity RMS against the
exact regularized field $R$.  For a cheapened FMM field $F'$ and the
shipped field $F$, the triangle inequality gives, with no modeling
assumptions,

$$
\|F' - R\| \le \|F - R\| + \|F' - F\|,
$$

so the new mechanism may add at most the recorded headroom.  From the
shipped-default anchor rows of
`data/flowvpm_gpu_campaign/fm037b_error_decomposition.csv`
(`u_total_rel` = $\|F-R\|/\|R\|$), with a $1.1\times$ repeatability
margin:

| case | $n$ | tf | $u_{\rm total}$ | headroom | budget |
|---|---:|---|---:|---:|---:|
| cube | 1e5 | F32/F64 | 6.806e-4 | 3.194e-4 | 2.904e-4 |
| **cube** | **1e6** | **F32/F64** | **7.078e-4** | **2.922e-4** | **2.656e-4** |
| wake | 1e5 | F32/F64 | 3.299e-4 | 6.701e-4 | 6.092e-4 |
| wake | 1e6 | F32/F64 | 2.989e-4 | 7.011e-4 | 6.373e-4 |
| rotor | 1e5 (l7q6) | F32 | 6.417e-4 | 3.583e-4 | 3.258e-4 |
| rotor | 1e6 (l8q6) | F32 | 6.966e-4 | 3.034e-4 | 2.758e-4 |

The binding case is cube $n=10^6$:

$$
B \;=\; \frac{10^{-3} - 7.078\times10^{-4}}{1.1} \;=\; 2.656\times10^{-4}
\quad\text{(delivered relative U RMS).}
$$

## 2. Pointwise $\to$ delivered mapping

The modes perturb only the direct-pair regularization factors.  Writing a
pair's singular velocity kernel as $C_j$ (the Biot–Savart cross product
including $1/4\pi r^3$), a perturbation $\delta g_j$ on the pairs of a
target $k$ perturbs its velocity by

$$
\delta U_k \;=\; \sum_{j \in \mathcal{S}(k)} \delta g_j\, C_j ,
$$

which yields two bounds:

**Coherent (rigorous, worst case — every pair error aligned):**

$$
\|\delta U\|_{\rm RMS} \;\le\; \varepsilon \cdot
\underbrace{\frac{\mathrm{RMS}_k\!\left(\sum_{j\in\mathcal S(k)} w_j\,\lvert C_j\rvert\right)}
{\mathrm{RMS}_k\,\lvert U_R\rvert}}_{\displaystyle \kappa_{\mathcal S}} ,
$$

with $w_j = g_j$ for a relative budget ($\lvert\delta g\rvert \le
\varepsilon\, g$) and $w_j = 1$ for an absolute one.

**Incoherent (the theory §6.4/§7 no-coherent-cancellation model that
already underlies the shipped RMS $\rho_t$):** replace the inner sum by
its root-sum-square, giving $\kappa^{\rm rss}_{\mathcal S}$.

Both are measured on an overlap-matched synthetic cube ($n=10^4$,
$S=200$ sample targets, worst case over overlaps $1.0/1.5/2.0$; the
cluster oracle re-validates on the real cases at $10^5$–$10^6$):

| pair set $\mathcal S$ | $\kappa$ (coherent) | $\kappa^{\rm rss}$ |
|---|---:|---:|
| series branch $\rho\le 2$ ($w=g$) | 7.15 | 0.531 |
| outer branch $2<\rho\le\rho_t$ ($w=1$) | 18.6 | 0.717 |
| all direct pairs ($w=g_{\rm used}$) | 56.8 | 0.993 |

Splitting $B$ evenly between the two active branches, the **coherent-tier
pointwise budgets** are

$$
\left.\frac{\lvert\delta g\rvert}{g}\right|_{\rho\le2}
\le \frac{B}{2\kappa_{\rm series}} = 1.86\times10^{-5},
\qquad
\lvert\delta g\rvert\big|_{2<\rho\le\rho_t}
\le \frac{B}{2\kappa_{\rm outer}} = 7.15\times10^{-6},
$$

and the whole-stream relative budget (for `:fp32`, which touches every
pair the regularized-family functor evaluates) is $B/\kappa_{\rm all} =
4.68\times10^{-6}$.  The incoherent-tier budgets are $2.50\times10^{-4}$
(series, rel) and $1.85\times10^{-4}$ (outer, abs).

$\varepsilon$ is the pointwise **delta versus the shipped evaluator**,
not versus the analytic reference: the anchors' $u_{\rm total}$ already
contains the shipped kernel's own evaluation error, so only the change is
new delivered error.

**Sizing rule.**  A mechanism ships when its coherent prediction passes
$B$ with $\ge 2\times$ margin.  A mechanism passing only the incoherent
tier ($\ge 10\times$) would be admissible solely with cluster-oracle
confirmation (the §6.4 precedent); none of the shipped choices below
needs that concession.

**J mapping (diagnostic — no gate).**  A pair's Jacobian perturbation is
bounded by $\lvert\delta h\rvert\,\lvert{\rm crss}\rvert/r$ (the
$a$-term) plus $\sqrt2\,\lvert\delta g\rvert\,\lvert\Gamma\rvert/4\pi
r^3$ (the $b$-term).  Measured coherent amplifications:
$\kappa^{J,h}_{\rm series}=7.0$, $\kappa^{J,g}_{\rm series}=12.2$,
$\kappa^{J,h}_{\rm outer}=8.9$, $\kappa^{J,g}_{\rm outer}=16.0$.  The
shipped choices below deliver $\lesssim 5\times10^{-5}$ relative J RMS —
negligible against the cube/wake $j_{\rm total}\sim 3$–$11\times10^{-3}$,
but comparable to the rotor's $j_{\rm total}\sim5\times10^{-5}$; J on the
rotor is the one metric the cluster job must watch.

## 3. Mechanisms against the budget

Pointwise deltas vs the shipped evaluator (dense sweeps vs a 256-bit
reference; `fm037f_budget.csv`):

| mechanism | param | $\delta$ series (rel) | $\delta$ outer (abs) | pred. coherent | pred. incoherent | verdict |
|---|---|---:|---:|---:|---:|---|
| `:reduced` | F64, 12 terms | 4.6e-6 | 0 | 3.3e-5 | 2.4e-6 | **PASS (coh, 8x)** |
| `:reduced` | F32, 12 terms | 4.9e-6 | 1.0e-7 | 3.7e-5 | 2.7e-6 | **PASS (coh, 7x)** |
| `:reduced` | F64/F32, 11 terms | 2.9e-5 | — | 2.1e-4 | 1.6e-5 | rss-only (not shipped) |
| `:reduced` | 10 terms | 1.7e-4 | — | 1.2e-3 | 9.2e-5 | FAIL |
| outer deg-2 | recorded fit | — | 7.4e-4 | 1.4e-2 | 5.3e-4 | **FAIL — outer stays deg-3** |
| `:fp32` | shipped 13-term F32 | 6.9e-7 | 1.0e-7 | 6.8e-6 (+6.8e-6 asm) | 4.4e-7 | **PASS (coh, 20x)** |
| `:reduced_fp32` | 12-term F32 | 4.9e-6 | 1.0e-7 | 3.7e-5 | 2.7e-6 | **PASS (coh, 7x)** |
| `:lut` | N=1024 | 3.0e-6 | 1.1e-6 | 4.2e-5 | 2.4e-6 | **PASS (coh, 6x)** |
| `:lut` | N=512 | 7.0e-5 | 4.5e-6 | 5.8e-4 | 4.0e-5 | FAIL |

Decisions:

- **`:reduced`**: truncate the exact alternating series to **12 terms in
  both precisions** (from 19/13); the outer $s(u)$ polynomial **stays
  degree 3** (the recorded degree-2 fit fails even the incoherent tier).
- **`:fp32`** (Float64 configurations only): the shipped 13-term/deg-3
  math evaluated in Float32, U/J assembled in Float32, accumulated in
  Float64.  The extra Float32 assembly rounding is budgeted coherently as
  $1.2\times10^{-7}\cdot\kappa_{\rm all} = 6.8\times10^{-6}$.  On
  Float32 configurations `:fp32` is the shipped path (documented no-op).
  In the ballot/queue mechanism (`:ballot`, mixed `:classsplit_ballot`
  bucket) only the $g/h$ transcendental is narrowed — the accumulator-side
  assembly stays in the configuration precision; both variants sit inside
  the same budget line.
- **`:lut`**: $N=1024$ entries, linear interpolation in $x=\rho^2$ over
  $[0, \rho_t^2]$, Float32 storage ($2\times1024\times4$ B $=$ 8 KB
  shared memory per block).  The table stores the **normalized**
  functions $G(x) = g/\rho^3$ and $H(x) = h/\rho^5$ — both analytic in
  $x$ with $G(0)=A/3\neq0$ — so interpolation preserves *relative*
  accuracy down to $\rho\to0$ (a direct $(g,h)$ table diverges in
  relative error near the origin where $\lvert C\rvert$ is largest).
  Reconstruction: $g=\rho\,x\,G$, $h=\rho\,x^2 H$ (uses the already
  computed $\rho$; no `exp`, no series).  Values sample the shipped
  Float64 evaluator, so the LUT inherits rather than adds the shipped
  outer-fit error.  Beyond $x\ge\rho_t^2$ the mode returns the singular
  $(1,-3)$ — identical to the partitioned cutoff semantics.

## 4. Empirical validation of the mapping

Measured delivered $\|F'-F\|/\|R\|$ on the smoke case (exact CPU pair
sums, $n=10^4$) against the predictions, which must be upper bounds:

| candidate | pred. coherent | pred. incoherent | actual | bound |
|---|---:|---:|---:|---|
| `:fp32` | 6.8e-6 | 4.4e-7 | 9.4e-8 | OK (both) |
| `:reduced` F64/12 | 3.3e-5 | 2.4e-6 | 2.7e-7 | OK (both) |
| `:lut` N=1024 | 4.2e-5 | 2.4e-6 | 4.2e-6 | OK (coherent only) |

The LUT's interpolation error is curvature-correlated within cells (one
sign per cell), so it is *partially coherent*: the incoherent tier
under-predicts it by $1.7\times$.  This is exactly why the shipped
choices are sized on the **coherent** tier; the incoherent tier is
recorded for context only.  The same validation runs at $n=10^5/10^6$ on
the real cases via `fm037f_cutoff_configs.txt` through the
error-decomposition oracle (each mode's $F'$ vs the exact $P/R$ fields)
in the cluster job.

## 5. Caveats and scope

- **Branch-boundary sliver.**  Float32 rounding of $\rho$ (and the LUT
  cell containing $x=4$) can move a pair across the $\rho=2$ branch
  boundary, where the shipped evaluator itself is discontinuous by its
  outer-fit error ($2.09\times10^{-4}$).  The affected set has measure
  $\sim\rho\,\varepsilon_{32}$ (one LUT cell, width $0.0033$ in $\rho$,
  respectively); measured window maxima are recorded as `boundary_*`
  rows.  The delivered effect is far below every budget (the `:fp32`
  smoke actual is $9\times10^{-8}$ total).
- **Cutoff-boundary half-open set.**  `:lut` evaluates the singular pair
  at exactly $\rho=\rho_t$ where shipped partitioned evaluates
  regularized ($\ge$ vs $>$) — measure zero, and the two differ there by
  $\bar g(\rho_t)\approx3.8\times10^{-3}$ on that set only.
- **Scope.**  The modes cover the regularized-family pair functors
  (`RegularizedVortex`, `PartitionedVortex`, `TwoPassVortex` pass 1) on
  host and device.  The TwoPass pass-2 deficit sweep (non-default
  kernel) stays on the shipped $\bar g$ evaluation in every mode.  Host
  and device agree per mode, except `:lut`, which is a device
  shared-memory mechanism: the host reference path falls back to
  `:shipped` under `:lut` (documented; device-vs-host parity for `:lut`
  is gated at its budgeted pointwise error, not bitwise).
- **Amplification provenance.**  $\kappa$ is measured on a synthetic
  overlap-matched cube, worst case over overlap $\in\{1,1.5,2\}$
  ($\kappa$ grows with overlap; the real cases sit inside this range).
  The cluster oracle validation is the authoritative per-case check.
