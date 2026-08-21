# Sigma-Adaptive Multilevel Smooth Nearfield: Decomposition, Cost Model, and Verdict (041f)

Status: COMPLETE (`2026-08-18`). Theory/measurement artifact of row `041f`;
no production code is touched. Verdict: **NO-GO** (§6.1), unified-solver
verdict **one mesh family — 037d global VIC** (§6.2), banded hybrid
**rejected by strict additivity** (§6.3).

Standing contracts: sampled relative velocity RMS error `<= 1e-3` (gate), all
nine `J` entries logged as a diagnostic, source-directed regularization,
free-space boundary conditions, Float32 and Float64 both considered.

---

## 1. Exact multilevel Gaussian decomposition

### 1.1 The regularized kernel as a Gaussian convolution

The `gaussianerf` regularized Green function with source smoothing radius
$\sigma$ is

$$
G_\sigma(r) \;=\; \frac{\operatorname{erf}\!\big(r/(\sqrt{2}\,\sigma)\big)}{4\pi r},
\qquad
G_0(r) \;=\; \frac{1}{4\pi r}.
$$

The single identity everything below rests on is the *Gaussian filtering
identity*: $G_\sigma$ is exactly the free-space Newtonian potential of a unit
isotropic Gaussian density of standard deviation $\sigma$,

$$
G_\sigma \;=\; G_0 \ast \rho_\sigma,
\qquad
\rho_\sigma(\mathbf{x}) \;=\; \frac{1}{(2\pi\sigma^2)^{3/2}}
   \exp\!\big(-|\mathbf{x}|^2/(2\sigma^2)\big),
$$

which is elementary from Poisson's equation ($-\nabla^2 G_\sigma = \rho_\sigma$
with decay at infinity). Because Gaussians form a convolution semigroup,

$$
\rho_{\sigma}\;=\;\rho_{s}\ast\rho_{\tau},
\qquad
\tau^2 \;=\; \sigma^2 - s^2
\quad (0 \le s \le \sigma),
$$

we obtain the **exact width-transfer identity**

$$
G_{\sigma} \;=\; G_{s} \ast \rho_{\tau},
\qquad \tau = \sqrt{\sigma^2 - s^2}.
\tag{1}
$$

Identity (1) is what makes a *sigma-heterogeneous* particle set servable by a
*finite* set of level widths: a source with actual radius $\sigma_i$ can be
carried exactly on a level of nominal width $s \le \sigma_i$ by spreading it
with the compensating Gaussian $\rho_{\tau_i}$, $\tau_i=\sqrt{\sigma_i^2-s^2}$,
instead of as a point. No approximation has been made yet.

### 1.2 Telescoping band kernels

Choose a geometric ladder of level widths

$$
s_0 \;<\; s_1 \;<\; \dots \;<\; s_M,
\qquad s_{\ell+1} = \gamma\, s_\ell,
\qquad \gamma \in \{\sqrt{2},\, 2,\, 4\},
$$

with $s_0 \le \sigma_{\min}$ of the snapshot (so every particle has an
admissible level) and $s_M$ the coarsest width. Define the **band kernels**

$$
D_\ell(r) \;=\; G_{s_\ell}(r) - G_{s_{\ell+1}}(r),
\qquad \ell = 0,\dots,M-1,
$$

so that for any starting level $\ell$ the telescoping identity

$$
G_{s_\ell} \;=\; \sum_{k=\ell}^{M-1} D_k \;+\; G_{s_M}
\tag{2}
$$

holds **exactly** — it is a finite telescoping sum, not a limit. Combining (1)
and (2): a source $i$ assigned to level $\ell_i$ (the finest level with
$s_{\ell_i} \le \sigma_i$; the deterministic assignment rule is §2.1)
contributes

$$
G_{\sigma_i}
\;=\;
\rho_{\tau_i} \ast \Big[\; \sum_{k=\ell_i}^{M-1} D_k \;+\; G_{s_M} \Big],
\qquad
\tau_i = \sqrt{\sigma_i^2 - s_{\ell_i}^2}.
\tag{3}
$$

Equation (3) is the complete decomposition: *every* pairwise interaction is a
sum of band terms plus one globally smooth top term, each term to be computed
on a mesh whose resolution matches that term's smoothness. The numerical
approximations enter only afterwards and are enumerated exhaustively in §1.5.

Two structural facts drive the cost model:

**(a) Band kernels are short-ranged.** For $a < b$,

$$
0 \;<\; G_a(r) - G_b(r)
\;=\; \frac{\operatorname{erf}(r/(\sqrt2 a)) - \operatorname{erf}(r/(\sqrt2 b))}{4\pi r}
\;\le\; \frac{\operatorname{erfc}\!\big(r/(\sqrt2\, b)\big)}{4\pi r},
$$

so $D_\ell$ (and its first two derivatives, needed for $U$ and $J$) decays
like $e^{-r^2/(2 s_{\ell+1}^2)}$ beyond a few $s_{\ell+1}$. Truncating
$D_\ell$ at

$$
R_\ell \;=\; \rho_b\, s_{\ell+1}
$$

incurs a relative tail bounded by $\operatorname{erfc}(\rho_b/\sqrt2)$-type
terms; $\rho_b$ is chosen per precision from the standing budget (§1.5). The
band is therefore a *compact-support convolution* — patch-local FFTs or MSM
stencils apply, with halo thickness $R_\ell$ plus the spreading support.

**(b) Band kernels are smooth at scale $s_\ell$.** The Fourier transform of
$D_\ell$ is

$$
\widehat{D_\ell}(k) \;=\; \frac{e^{-s_\ell^2 k^2/2} - e^{-s_{\ell+1}^2 k^2/2}}{k^2},
$$

whose content above $k \sim c/s_\ell$ is Gaussian-suppressed. A mesh with
spacing $h_\ell \propto s_\ell$ resolves it; the validated `037d` resolution
rule $h \le 0.55\,\sigma$ (at the accuracy-passing spread/interpolation
configurations) transfers directly with $\sigma \to s_\ell$, because the
narrowest Gaussian factor present on level $\ell$ is exactly $s_\ell$: sources
are never spread with anything narrower than their compensated width, and the
kernel factor contributes $s_\ell$ itself. The top term $G_{s_M}$ lives on the
coarsest mesh at $h_M \le 0.55\, s_M$.

This pair of facts is the entire point of the construction: the *fine* levels
have small support (cheap, local), the *coarse* levels have large support but
coarse meshes (cheap, few points), and the ladder converts the rotor's
$18\times$ sigma spread from a global $\sigma_{\min}$ mesh floor (the `037d`
failure) into per-level meshes each sized to its own population.

### 1.3 U and J from the level fields

Write the discrete vorticity source field carried on level $\ell$ as the
spread field

$$
\omega_\ell(\mathbf{x}) \;=\; \sum_{i:\,\ell_i \le \ell \text{ owns band } \ell}
   \big(\rho_{\tau_i} \ast \delta_{\mathbf{x}_i}\big)(\mathbf{x})\,\boldsymbol{\Gamma}_i ,
$$

(the precise band-ownership bookkeeping is §1.4). The band-$\ell$ vector
potential, velocity, and velocity gradient are

$$
\boldsymbol{\psi}_\ell = D_\ell \ast \omega_\ell,
\qquad
\mathbf{u}_\ell = \nabla \times \boldsymbol{\psi}_\ell,
\qquad
J_\ell = \nabla \mathbf{u}_\ell ,
$$

and the totals are the sums over bands plus the top term. Two admissible
evaluation routes for the derivatives:

1. **Spectral differentiation** (AMR-FFT path): multiply by $i\mathbf{k}$
   inside the patch transform. $U$ needs the curl (3 output components from 3
   input components; realizable as 3 forward + 3 inverse real-to-complex
   transforms per patch with the curl applied in $k$-space), $J$ needs all
   nine $\partial_a u_b$: nine additional inverse transforms, or six using
   $\nabla\times$ / symmetry splittings — the census prices the
   straightforward $3$ forward $+\,3+9$ inverse count and notes the shared-
   forward saving, exactly as `037d` did for the global mesh.
2. **Analytic window derivatives** (interpolation-side): interpolate
   $\boldsymbol{\psi}$ (or $\mathbf{u}$) with a window whose analytic gradient
   supplies the missing derivative order. This trades transforms for wider
   interpolation error bars; only the `037d` accuracy-passing configurations
   are admitted into the design space.

For MSM the same fields are produced by real-space compact stencils
(discretized $D_\ell$ tables) instead of $k$-space multiplication; derivative
fields use differentiated kernel tables $\partial_a D_\ell$, which have the
same support radius and smoothness class, so $U+J$ costs $\le 12/3 = 4\times$
the scalar-component stencil work unless intermediate fields are shared
(§3.2).

Source-directedness is automatic: the spreading width $\tau_i$ and the level
assignment depend only on the *source* $\sigma_i$; targets are pure
interpolation points and carry no smoothing. Distinct-target-system evaluation
(FLOWVPM probe points) is therefore free — targets never enter the spreading
pass.

### 1.4 Band ownership and the exact-once composition

Let $\ell_i$ be the assigned level of source $i$. Expanding (3) over all
sources, the total field at target $\mathbf{y}$ is

$$
\mathbf{u}(\mathbf{y})
=\sum_{\ell=0}^{M-1}
   \underbrace{\Big[\nabla\times\big(D_\ell \ast \omega_\ell\big)\Big](\mathbf{y})}_{\text{band }\ell\ \text{term}}
\;+\;
   \underbrace{\Big[\nabla\times\big(G_{s_M} \ast \omega_M\big)\Big](\mathbf{y})}_{\text{top term}},
\qquad
\omega_\ell \;=\; \sum_{i:\; \ell_i \le \ell} \rho_{\tau_i}\!\ast\delta_{\mathbf{x}_i}\boldsymbol{\Gamma}_i .
$$

**Ownership rule:** the ordered pair $(i \to j)$ receives exactly the bands
$k \in \{\ell_i, \dots, M-1\}$ plus the top term, each band computed on level
$k$'s mesh and nowhere else. This is a partition of the exact kernel
$G_{\sigma_i}$ by construction of the telescoping identity — no pair term is
ever split across levels or duplicated, provided:

- **(O1)** source $i$ is spread onto level $\ell_i$ once, and its coarser-level
  images are produced by *restriction of the already-spread field* (or
  re-spreading at the coarser width $\tau_i^{(k)} = \sqrt{\sigma_i^2 - s_k^2}$
  — the two are analytically identical by the semigroup property; numerically
  they differ, see E4 in §1.5), and
- **(O2)** each band field is interpolated to each target exactly once, from
  exactly one patch owner when patches overlap (§2.4).

The census's exact-once oracle paints (source, band, target) contribution
ownership and asserts zero omissions/duplicates on the §5 configuration list;
the analytic identity above is what it verifies the *implementation* of.

**Banded-hybrid variant (review amendment #2).** The mandatory FMM-retained
configuration replaces the top term and the coarse tail of the ladder with the
*shipped singular FMM far field*. Choosing a cut level $c$, identities
(1)+(2) give exactly

$$
G_{\sigma_i}
\;=\; \underbrace{\rho_{\tau_i}\ast\sum_{k=\ell_i}^{c-1} D_k}_{\text{meshed bands}}
\;+\; G_{\sigma_i^{(c)}},
\qquad
\sigma_i^{(c)} \;=\; \sqrt{\sigma_i^2 - s_{\ell_i}^2 + s_c^2} \;\ge\; s_c .
$$

The tail $G_{\sigma_i^{(c)}}$ differs from the singular kernel $G_0$ by
$\operatorname{erfc}\big(r/(\sqrt2\,\sigma_i^{(c)})\big)/(4\pi r)$, which is
below the delivered-error floor only for pairs with
$r \ge \rho_t\,\sigma_i^{(c)}$ — the same kernel-difference bound as `031a`
§4, but at the **inflated** effective radius $\sigma_i^{(c)} > \sigma_i$. The
banded hybrid therefore keeps the shipped singular far field and U/V routing
unchanged, meshes the bands $k<c$, and must evaluate the tail kernel
$G_{\sigma_i^{(c)}}$ directly for every pair with
$r < \rho_t\,\sigma_i^{(c)}$.

**Strict-additivity consequence.** The shipped baseline evaluates the full
regularized kernel directly on pairs with $r < \rho_t\,\sigma_i$ and the
singular FMM beyond. In the hybrid, the direct set *inflates* to
$r < \rho_t\,\sigma_i^{(c)} \supseteq r < \rho_t\,\sigma_i$ (per-source pair
count scales like $(\sigma_i^{(c)}/\sigma_i)^3$, which is enormous for fine
sources: $\sigma_i \ll s_c \Rightarrow (s_c/\sigma_i)^3$), the per-pair tail
kernel costs the same erf-class arithmetic as the shipped kernel, and the
meshed bands are pure additional work. Hence, under this decomposition,
**every** FMM-retained banded hybrid costs strictly more than the shipped
baseline: keeping the singular far field can never *shrink* the direct ball,
because compensating the removed fine bands inflates the tail width, never
deflates it. The only way a mesh reduces direct work is to own the tail's
near range on a mesh as well — i.e. full replacement, where the "tail at
level $c$" is itself the meshed top term. This is exactly the "double
coverage of the mid band" failure mode the review amendment anticipated; the
census quantifies the inflation factor and the added band cost per case in
`hybrid.csv` rather than leaving the conclusion qualitative.

### 1.5 Complete error taxonomy

Every numerical approximation applied after the exact identities (1)–(3),
with its controlling parameter and its budget line:

- **E1 — band truncation.** $D_\ell$ (and $\partial D_\ell$,
  $\partial^2 D_\ell$) truncated at $R_\ell = \rho_b s_{\ell+1}$. Bounded by
  the erfc tail; set per precision. Absent when the band is evaluated by FFT
  over a patch whose padded region covers the full support (then E1 merges
  into E5 patch truncation).
- **E2 — mesh resolution / spreading-interpolation.** Level mesh spacing
  $h_\ell \le 0.55\, s_\ell$ (validated `037d` rule; sensitivity at the other
  passing resolutions is a census axis). Includes window order for spreading
  $\rho_{\tau_i}$ (truncated Gaussian or `037d`-passing window) and for
  target interpolation; $J$ costs one extra derivative order of window
  accuracy when route 2 of §1.3 is used.
- **E3 — spreading-window truncation.** $\rho_{\tau_i}$ truncated at
  $\rho_w \tau_i$; Gaussian tail bound. Note $\tau_i \le \sigma_i$ always, and
  $\tau_i = 0$ for particles sitting exactly at their level width (spread
  degenerates to the bare mesh window — the uniform-sigma limit, which is how
  the unified-solver check of §6.2 connects to `037d` global VIC).
- **E4 — restriction/prolongation.** Coarse-level images produced by
  restriction of fine spread fields rather than exact re-spreading; error is
  the restriction operator's aliasing at the coarse Nyquist, standard
  multigrid-order bound; the census logs transferred point counts and the
  chosen operator order.
- **E5 — patch truncation / free-space padding.** Patch FFTs need free-space
  (Hockney–Eastwood zero-padded) convolution; a band contribution whose
  support crosses a patch boundary must be covered by the guard region
  ($R_\ell$ + spreading support), else it is an *omission*, not an error term
  — the exact-once oracle treats guard sufficiency as a hard pass/fail.
- **E6 — FFT / stencil arithmetic and accumulation.** Float32 vs Float64:
  band fields are bounded and short-ranged, so Float32 accumulation is benign
  on fine levels; the top term aggregates the whole domain and is the one
  place Float64 (or compensated) accumulation may be forced — priced in both
  precisions.
- **E7 — MSM kernel tabulation.** Discrete kernel tables sampled at $h_\ell$;
  interpolation order of the table lookup; same budget line as E2 in the MSM
  branch.

Budget allocation: the standing gate is sampled relative velocity RMS
$\le 10^{-3}$; the census allocates $\le 3\times10^{-4}$ each to E1+E3+E5
(tails), E2+E4+E7 (resolution/transfer), holding E6 below $10^{-4}$ in the
priced precision, mirroring the `037d` allocation discipline; $J$ is logged
as a diagnostic under the same decomposition without gating.

### 1.6 The sigma-compensation cost dilemma

The width-transfer identity (1) is exact, but *realizing* $\rho_{\tau_i}$
numerically is the central cost tension of the whole construction, and it is
what 041d's $2\times$–$5\times$ "AMR coupling" placeholder silently absorbed.
The options, with their exact cost/error laws:

1. **Real-space Gaussian spreading.** Spread particle $i$ with
   $\rho_{\tau_i}$ (times the mesh window). Support per axis is
   $p + 2\rho_w\tau_i/h_\ell$ cells with $\tau_i \le s_\ell\sqrt{\gamma^2-1}$,
   i.e. up to $p + 2\rho_w\sqrt{\gamma^2-1}/(h/s)$ — at $\gamma=2$,
   $\rho_w=5$, $h=0.55\,s$: up to $\sim 22$ cells, $10^4$ samples per
   particle. Exact, but the spreading pass alone can exceed the entire
   nearfield budget.
2. **Binned k-space compensation with point spreading.** Spread with the bare
   window and multiply by $e^{-\tau_b^2 k^2/2}$ per $\sigma$-bin in $k$-space.
   Cheap ($p^3$ per particle), but the within-bin width error is *linear* in
   the bin width: the induced relative field error is
   $\sim \Delta(\tau^2)\,\langle k^2\rangle/2$ with
   $\langle k^2\rangle \sim 2/s_\ell^2$ under the band spectrum — meeting a
   $3\times10^{-4}$ budget line would need $O(10^3)$ bins per level.
   Two-bin interpolation in $\tau^2$ improves this to quadratic,
   $\sim \Delta(\tau^2)^2\langle k^4\rangle/8$ with
   $\langle k^4\rangle = 15/s_\ell^4$, which still needs $O(10^2)$ bins.
   Both are inadmissible: forward-transform count scales with the bin count.
3. **The δ-split (registered architecture).** Split
   $\tau_i^2 = \tau_b^2 + \delta_i^2$ with $\tau_b$ the bin lower edge of a
   $b$-bin geometric subdivision of $[s_\ell, \gamma s_\ell)$: apply
   $\rho_{\tau_b}$ exactly in $k$-space per bin, and spread the residual
   $\rho_{\delta_i}$ in real space. Per-particle **exact** (no bin error
   line), with support $p + 2\rho_w\delta_i/h_\ell$ and
   $\delta_i \le \sigma_i\sqrt{1-r_b^{-2}}$, $r_b = \gamma^{1/b}$. Cost moves
   smoothly between option 1 ($b=1$) and per-bin transform inflation
   ($b$ large): forward transforms scale with the number of *occupied*
   $(\ell,\text{bin})$ patch sets, which the census measures — and on fields
   where $\sigma$ correlates with position (the rotor: $\sigma$ grows with
   wake age, hence with distance from the rotor plane), the per-bin
   occupancies are spatially clustered slabs, not copies of the whole level.

The census therefore records, per level: the exact per-particle spreading
sample sums $\sum_i (p + 2\rho_w\delta_i/h)^3$ for every $b$ in the design
space, and the measured per-bin occupancy dilation (`padded8_ratio`) that
prices the forward-transform inflation. The optimizer picks $b$ per
configuration; there is no free parameter left in the compensation.
At coarser bands $k > \ell_i$ the compensation width shrinks relative to
$h_k$ like $\gamma^{-(k-\ell_i)}$, so the effective bin count decays
$\max(1, b\,\gamma^{-2(k-\ell_i)})$ and coarse-band images come from
restriction of the fine spread density with window deconvolution folded into
the level's $k$-space multiplier (E4).

---

## 2. Patch and level geometry

*(Deterministic rules; all inputs are refresh-time statistics. Filled against
the `038`–`041` hierarchy; census constants inserted after the runs.)*

### 2.1 Level assignment

$\ell_i = \max\{\ell : s_\ell \le \sigma_i\}$, with $s_0$ chosen from the
snapshot's recorded $\sigma_{\min}$ (floored, per the ladder ratio, to a
finite ladder also containing $\sigma_{\max}$). Deterministic, per-particle,
$O(1)$ from the recorded sigma field. CoreSpreading growth moves particles
monotonically up the ladder; a stable-epoch policy re-bins only when a
particle's $\sigma_i$ crosses its level boundary with hysteresis margin
(census sensitivity axis).

### 2.2 Level meshes and occupied patches

Level $\ell$ mesh spacing $h_\ell = 0.55\, s_\ell$ aligned to the existing
Morton cell lattice (cells are dyadic; $h_\ell$ snaps to the nearest dyadic
refinement of the level's owning tree depth so restriction/prolongation are
lattice-aligned). The **field region** at level $\ell$ is the set of
occupancy tiles containing band-$\ell$ sources, dilated by *one tile* (the
spreading/interpolation window support only — for coincident source/target
systems, which is the FLOWVPM case, targets live where sources live). The
band-kernel reach $R_\ell$ is **not** an occupancy dilation; it is carried by
the per-patch halo gather and transform padding (2.3). Patches = occupied
tiles grouped by a fixed block lattice of edge $E_{\max}$ mesh cells (census
axis), each patch trimmed to the tight bounding box of its occupied tiles.
The block lattice partitions space, so patch ownership of any target position
is unique by construction; grouping is deterministic from refresh-time
occupancy and has explicit capacity bounds (patch count $\le$ occupied-tile
count; bytes bounded by padded-point capacity per level).

### 2.3 Halo gather and padding

A patch computes its band field from all sources within the kernel reach of
its interior: its transform region is the interior bounding box grown by
$g_\ell = \lceil R_\ell/h_\ell\rceil + p$ cells per side ($R_\ell$ band
reach, $p$ window support), with the linear-convolution size rounded to the
next 5-smooth integer per axis. This one padding serves both the free-space
(Hockney) requirement and cross-patch coverage: a source within $R_\ell$ of a
neighboring patch's interior is gathered into that patch's halo, so its
contribution to that patch's targets is computed there — and only there,
because targets are interpolated exclusively from their unique owner patch
(O2). Contributions omitted because a source is beyond the gather reach are
bounded by the E1 tail; the oracle audits exactly this omission. MSM patches
need the same halo but no padding rounding.

### 2.4 Overlap ownership

When two patches' guarded regions overlap, interior cells have a unique owner
(the patch whose un-guarded interior contains them — guaranteed disjoint by
the grouping sweep); targets are interpolated from their owner patch only
(O2). Coarse/fine composition is by the band decomposition itself: levels do
not exchange fields except through restriction of spread sources (E4), so
there is no coarse/fine "correction step" to double-count.

### 2.5 Ladder extension and the top term

The top term $G_{s_M}$ is long-range: it must be evaluated as a *global*
free-space convolution over the whole occupied bounding box at
$h_M = 0.55\,s_M$. Anchoring $s_M$ at $\sigma_{\max}$ is untenable on the
rotor: $\sigma_{\max}/L_z \approx 2.6\times10^{-4}$, so a global mesh at
$0.55\,\sigma_{\max}$ has $O(10^9)$ points — this is the multilevel
construction's own miniature version of the `037d` $\sigma_{\min}$ floor,
and it is why 041d's occupancy-only point count ($N_{\rm mesh}\approx 0.75n$)
was incomplete. The remedy is **ladder extension**: append bands above
$\sigma_{\max}$ (they carry *all* sources, remain compact with reach
$\rho_b s_{k+1}$, and live on occupied-dilated patches whose meshes coarsen
like $\gamma^{-3}$ per level) until the residual top width $s_M$ makes the
global coarse mesh affordable. The census optimizer chooses the extension
depth $M_{\rm ext} \in [0, 10]$ per configuration by total cost; the top term
is then priced exactly as a small `037d` global VIC at width $s_M$. In the
uniform-sigma limit ($\sigma_{\max}=\sigma_{\min}$, $M=0$, no bands) the
whole construction collapses to precisely `037d`'s funded global VIC, which
is the mechanism behind the unified-solver check of §6.

---

## 3. AMR-FFT and MSM formulations

### 3.1 AMR-FFT

Per patch on level $\ell$ with interior $n_1{\times}n_2{\times}n_3$ and
padded extent $\bar n_a = 2(n_a + 2g_\ell/h_\ell)$:

- transforms: 3 forward R2C (spread $\omega$ components), then per output
  field inverse C2R: 3 for $U$ (curl in $k$-space), up to 9 for $J$
  (spectral differentiation; shared forwards). Kernel multiplier
  $\widehat{D_\ell}$ is analytic (§1.2) — evaluated on the fly or tabulated
  per patch shape class; no kernel transform needed (Gaussian-split kernels
  have closed-form $\hat D$, unlike general Hockney kernels).
- forward transforms are per $(\ell,\text{bin})$ occupied patch set (§1.6);
  inverse transforms per level patch after $k$-space accumulation of bin
  contributions sharing a patch grid.
- batching: patches of equal padded shape are batched into one cuFFT plan
  execution; the census records the shape-class histogram, which is what
  decides whether small-FFT efficiency or launch fragmentation dominates —
  the calibration explicitly penalizes sub-`037d` transform sizes
  (bracketed $\times 1/2/4$ bandwidth penalty below $2^{18}$ padded points)
  and prices steady-state launches as captured-graph node replays at the
  `028`/`029` measured floors ($2/5/15\,\mu$s per node), not as raw
  $\sim50\,\mu$s kernel launches.
- restriction/prolongation: lattice-aligned (2.2), bandwidth-priced.
- no global coarse solve: the top term $G_{s_M}$ is one (typically single-
  patch) coarse free-space convolution over the whole occupied domain at
  $h_M$ — this is exactly a small `037d` global VIC at width $s_M$, and is
  priced with the `037d` model directly.

### 3.2 MSM

Level kernels: tabulated $D_\ell$ and its first derivatives on the level
lattice, support $R_\ell/h_\ell = \rho_b \gamma / 0.55$ points per axis —
support point count $(2\rho_b\gamma/0.55+1)^3$ is the census's stencil size;
separability does not hold exactly ($D_\ell$ is radial, not a tensor
product), but rank-compressed (truncated-SVD) separable approximations are a
recorded option with their rank counted against E7. $U$ and $J$ share the
convolved $\boldsymbol\psi_\ell$ intermediate when derivatives are taken on
the interpolation side; otherwise each derivative field is its own stencil
pass. No FFTs, no padding; guard-only halos; cost scales with
(interior points) × (stencil points) × (fields), and the census prices it
from measured stencil-throughput anchors rather than a generic $O(N)$ claim.

### 3.3 Coexistence with the shipped path

Full replacement: mesh owns *all* smooth work; the shipped direct U list and
singular far field are switched off for the meshed system; the only surviving
direct work is nothing — configurations must therefore carry the *entire*
budget on E1–E7. Banded hybrid (§1.4): shipped singular FMM + U/V routing
retained; meshed bands own $k<c$; the inflated-cutoff complement is an
explicit direct list priced by the census. Both variants appear in the
registered design space with every ordered contribution owned exactly once.

---

## 4. Deterministic census results (2026-08-18)

Script: `scripts/sigma_adaptive_nearfield_census.jl` (4 threads, run under the
FLOWVPM project for the fm033 rotor generator only). Data:
`data/sigma_adaptive_nearfield/` (checksummed). Cases: cube / wake /
sigma\_multiscale proxies (041c/041d constructors and seeds) and the real
DJI-9443 rotor reconstruction at $n = 10^5, 10^6$ (measured $\sigma$ spread
$17.9\times$).

### 4.1 Accuracy / exact-once oracle

All 28 rows pass (`oracle.csv`): U converges at the expected $\sim 6$th order
of the instrument (ratios 47–193 per $h$-halving), J converges once the
finite coincident self-term $J_{\rm self} = -G''(0)\,[\Gamma]_\times$,
$G''(0) = -1/(3(2\pi)^{3/2}\sigma^3)$, is included in the reference; the
multilevel-vs-single-mesh burden ratio is 1.00 on both uniform-$\sigma$ cases
(the decomposition adds *no* resolution burden over a global mesh at matched
$h$); the largest omitted-tail magnitude across all cases and ladders is
$10^{-17}$ relative (band truncation at $\rho=5$ is vastly inside the
$3\times10^{-4}$ budget line); block-partition ownership is unique by
construction and audited. Boundary, extreme-ladder ($17.9\times$ in two
deltas), island-with-distinct-targets, and filament cases included.

### 4.2 Level/patch geometry — where the points actually are

Rotor $n=10^6$, $\gamma=2$, nominal resolution (`level_tables.csv`): the
ladder has 5 assigned levels plus extension; the optimizer picks
$M_{\rm ext}=3$. Interior mesh points by level peak at the $\sigma$-mass
levels — 5.9×10⁸ interior / 7.4×10⁹ padded points at level 3 alone — and
total ≈ 2.7×10¹⁰ padded transform points across the ladder. Against 041d's
occupancy-count law ($N_{\rm mesh}\approx 0.75n = 7.5\times10^5$) that is a
measured inflation of $\sim3\times10^4$, decomposed as:

1. **Cumulative band participation**: every source feeds every band at and
   above its level, so mid-ladder bands carry 0.55–1.0 of all $n$ sources
   (`cum_sources`) on meshes only $\gamma\times$ coarser per step.
2. **Codimension deficit**: the wake is a jittered sheet/filament complex —
   fine-band occupancy is locally 2-D. One-tile-thick occupancy dilates
   $\sim 9\times$ under the window dilation, and the band-kernel halo
   ($\lceil\rho_b\gamma/0.55\rceil + p = 22$ cells per side) is much thicker
   than the sheet, inflating each thin patch transform a further
   $\sim 10$–$30\times$ (measured `padded/interior` ≈ 11–12).
3. Hockney free-space padding and 5-smooth rounding (≈ 2–4×), already
   included in the halo figure above.

The sigma\_multiscale proxy fails differently: its fine-$\sigma$ population
fills the unit-cube volume, so level 0 is a near-global mesh at
$0.55\,\sigma_{\min}$ — exactly the `037d` $\sigma_{\min}$-floor mechanism,
now measured at 5.3 s nominal.

### 4.3 Cost model and optimizer

`cost_model.csv` / `optimizer.csv`, all three brackets, best configuration per
case (F32, epoch refresh):

| case, n | best config | zero-coupling LB (opt) | total opt/nom/pess (ms) | shipped near / complete (ms) |
| --- | --- | --- | --- | --- |
| cube 1e6 | γ=2, b=1, B=128, AMR-FFT, M_ext=0 | 2.53 | 3.28 / 6.66 / 15.5 | 12.3 / 30.7 |
| wake 1e6 | γ=√2, b=1, B=128, AMR-FFT, M_ext=0 | 1.96 | 2.70 / 5.85 / 14.2 | 9.2 / 62.8 |
| sigma_multiscale 1e6 | γ=2, b=8, B=128, AMR-FFT | 3567 | 3709 / 5297 / 9068 | 12.8 / 59.4 |
| rotor 1e5 | γ=√2, b=8, B=128, AMR-FFT, M_ext=1 | 133.5 | 144.7 / 213.5 / 380.5 | 4.9 / 7.0 |
| rotor 1e6 | γ=2, b=8, B=128, AMR-FFT, M_ext=3 | 2557 | 2667 / 3826 / 6589 | 23.3 / 33.3 |

Rotor nominal stage split (b=8): FFT 2836 ms + k-multiply 709 ms dominate;
spreading 37 ms (the §1.6 δ-split at b=8; 303 ms at b=1 — the compensation
tax is real and the optimizer trades it against forward-transform inflation
exactly as derived); interpolation 15 ms; restriction 155 ms; top term 5.4 ms;
graph-replay launches 4 ms. MSM is worse everywhere (rotor ≈ 22.5 s: the
$39^3$-point band stencils at the required reach dwarf the FFT bytes, and
U/J field sharing cannot save two orders). The σ-bin sweep behaves as
§1.6 predicts: spreading falls 303→37→11 ms as $b$ goes 1→8→32 while forward
FFT rises past $b=8$; $b=8$ is the measured optimum.

The rotor age-scale sweep (×0.5 / ×2 on the CoreSpreading growth term) moves
19–26% of particles across one level boundary (`refresh_epoch.csv`), so
stable-epoch refresh with hysteresis is viable in principle — immaterial to
the verdict.

## 5. Gates

**Count gate** (zero-coupling lower bound at optimistic anchors vs ≥10%
nearfield / ≥5% complete-solve): **FAIL** on every $\sigma$-heterogeneous
case — rotor 1e6: 2557 ms LB vs a 21.0 ms bar (≈120× over); rotor 1e5:
133 ms vs 4.4 ms; sigma\_multiscale: 3567 ms vs 11.5 ms. Uniform-σ cube/wake
pass (see §6.2). Per the registered procedure, the σ-adaptive direction stops
at the count gate; the accuracy/ownership gate had already passed (§4.1), and
the full modeled gate is reported for completeness in `optimizer.csv`.

**Robustness of the kill.** The census architecture (8-cell tiles, block
bounding boxes) is conservative, so we also computed an *idealized floor*
that no implementation of this decomposition can beat: cell-granular sheet
occupancy (`tiles_core`×64 lateral cells), minimal window thickness, and the
irreducible per-patch transform thickness (interior + $2R_\ell/h$): ≈
6.3×10⁸ transform points across the material levels, giving ≈54 ms of FFT at
*optimistic* bandwidth, plus ≥15 ms spreading (opt, b=8), ≥7 ms
interpolation, ≥13 ms k-multiply ⇒ ≈**90 ms optimistic floor vs the 21 ms
bar** — the verdict survives a further 30× of census conservatism and any
calibration bracket. No pre-registered microbenchmark could flip a count
gate failed on geometry alone, so none is requested.

## 6. Verdicts

### 6.1 Sigma-adaptive multilevel smooth nearfield: **NO-GO**

Closed, with the failure attributed (in decreasing order):

1. **Codimension deficit** — the regime that motivated this row
   ($\sigma$-heterogeneous rotor/wake fields) has sheet/filament fine-band
   occupancy; compact band kernels force a halo ≈22 mesh cells thick around
   patches whose payload is ~1–3 cells thick. Patch-local convolution wastes
   volume precisely where the field is thin.
2. **Cumulative band participation** — the telescoping decomposition puts
   most of $n$ on 5–8 levels simultaneously; the per-level mesh area decays
   only $\gamma^{-2}$ for surface occupancy while cumulative sources grow.
3. **Sigma-compensation tax** (§1.6) — 37 ms nominal at the optimal $b=8$
   for the rotor: even alone, and at optimistic anchors, it is comparable to
   the entire 21 ms bar.
4. The sigma\_multiscale volumetric-fine-σ variant reproduces the `037d`
   global $\sigma_{\min}$-floor kill instead.

041d's OPEN part-2 estimate is thereby closed: its $N_{\rm mesh}\approx0.75n$
law measured $\sim3\times10^4$-fold optimistic on the real rotor because it
counted occupancy volume, not band participation × codimension × halo.

### 6.2 Unified-solver verdict (review amendment #1)

In the uniform-$\sigma$ limit the machinery collapses exactly to `037d`
global VIC ($M=0$, $b=1$, top term only), and the census's independent
pricing lands inside/near the `037d` bands: cube 1e6 6.66 vs 5.81 ms
(ratio 1.15), wake 1e6 5.85 vs 6.16 ms (0.95), cube 1e5 1.45×, wake 1e5
1.43× (`unified_check.csv`) — within the stated calibration uncertainty.
**One mesh-solver family suffices**: the already-funded `037d` global VIC.
`042` should stage at most that one mesh implementation, and no σ-adaptive
mesh infrastructure. The measured regime boundary: a mesh solver is viable
iff (i) the σ spread is ≈1 (no ladder, no compensation), and (ii) the
fine-σ population is *volumetric* at its own scale (cube/wake yes; rotor
sheets no; volumetric-fine-σ multiscale only via the σ_min floor, i.e. no).
Everything outside that boundary stays on the shipped FMM/partitioned-direct
path.

### 6.3 FMM-retained banded hybrid (review amendment #2): rejected, with numbers

The §1.4 strict-additivity result makes every FMM-retained banded hybrid
cost more than the shipped baseline: the direct complement inflates
(measured mean pair inflation on the rotor: 1.25× at cut level 1 growing to
189× at cut 4 and 1494× at cut 5; sigma\_multiscale: 4.5×–1.6×10⁴×,
`hybrid.csv`) while the meshed bands *add* 234–2943 ms (rotor) on top. The
hybrid loses to full replacement for the anticipated reason — double
coverage of the mid band (the inflated singular tail must still be evaluated
directly wherever the shipped kernel was) — and full replacement itself is
NO-GO per §6.1. No configuration of the registered space escapes this: it is
an identity-level property of the decomposition, not a tuning failure.

### 6.4 Selector

`selector.csv`: `direct_fallback` (shipped path) on every σ-heterogeneous
row; the cube/wake 1e6 "adopt" rows are the global-VIC collapse and defer to
the existing funded `037d` verdict rather than staging anything new from this
row. No production implementation successor is proposed.
