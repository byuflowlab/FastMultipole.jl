# Partitioned Nearfield for the Regularized Biot-Savart Kernel (031a)

Scope: the FLOWVPM default `gaussianerf` kernel only (user decision
2026-08-05; sole `CoreSpreading`-compatible kernel). The SFS (`ζ`/`Estr`)
kernel derivation is deferred. All FLOWVPM references are to branch
`gpu-full`; formulas are transcribed from `src/FLOWVPM_kernel.jl:51-57` and
`src/FLOWVPM_fmm.jl:102-168`.

§§1–6 derive the partitioned-replacement nearfield. §6.1 (added by user
direction 2026-08-05) derives a third `032a` candidate — a two-pass additive
correction over an unmodified singular FMM — together with the conditioning
analysis that bounds when its subtraction is safe. §§6.2–6.4 (same date) cover
three levers that outrank the choice among the three candidates: removing the
`erf` from the expensive branch, the warp divergence that makes a naive split
kernel slower than no split at all, and tightening the cutoff radius itself.

## 1. Setup and exact regularized formulas

For source strength $\Gamma$, smoothing radius $\sigma$, separation
$\Delta x=x_t-x_s$, $r=|\Delta x|$, and $\rho=r/\sigma$, define

$$
C_i=-\frac{1}{4\pi}\frac{(\Delta x\times\Gamma)_i}{r^3}.
$$

The gaussianerf factor and derivative are

$$
g(\rho)=\operatorname{erf}\!\left(\frac{\rho}{\sqrt2}\right)
-\sqrt{\frac{2}{\pi}}\rho e^{-\rho^2/2},
\qquad
g'(\rho)=\sqrt{\frac{2}{\pi}}\rho^2e^{-\rho^2/2}.
$$

The regularized velocity and velocity gradient are

$$
U_i^{\rm reg}=g(\rho)C_i,
\qquad
J_{ij}^{\rm reg}=a(\rho)C_i\Delta x_j
+b(\rho)\varepsilon_{ijk}\Gamma_k,
$$

$$
a(\rho)=\frac{g'(\rho)}{\sigma r}-\frac{3g(\rho)}{r^2},
\qquad
b(\rho)=-\frac{g(\rho)}{4\pi r^3}.
$$

In the exact component order used by FLOWVPM:

$$
\begin{aligned}
J_{11} &= aC_1\Delta x_1, &
J_{21} &= aC_2\Delta x_1-b\Gamma_3, &
J_{31} &= aC_3\Delta x_1+b\Gamma_2, \\
J_{12} &= aC_1\Delta x_2+b\Gamma_3, &
J_{22} &= aC_2\Delta x_2, &
J_{32} &= aC_3\Delta x_2-b\Gamma_1, \\
J_{13} &= aC_1\Delta x_3-b\Gamma_2, &
J_{23} &= aC_2\Delta x_3+b\Gamma_1, &
J_{33} &= aC_3\Delta x_3.
\end{aligned}
$$

Setting $g\to1$ and $g'\to0$ gives the singular pair:
$U_i^{\rm sing}=C_i$, $a_\infty=-3/r^2$, and
$b_\infty=-1/(4\pi r^3)$.

## 2. Partitioned replacement operator

Choose a conservative dimensionless cutoff $\rho_t$ from §4. Interactions
are partitioned exactly once:

$$
(U,J)_{ts}=
\begin{cases}
(U,J)_{\rm reg}, & r/\sigma_s\le\rho_t,\\
(U,J)_{\rm sing}, & r/\sigma_s>\rho_t.
\end{cases}
$$

The FMM far field remains singular. The direct nearfield evaluates the stable
regularized formula inside the cutoff and the cheaper singular formula for the
rest of its pair list. *Within this strategy* there is no correction pass and no
subtraction of a singular contribution from a separately accumulated
correction; §6.1 derives a separate additive-correction strategy and bounds the
conditions under which its subtraction is safe.

This replacement is valid only if the direct interaction geometry contains
every pair satisfying $r/\sigma_s\le\rho_t$. For each source cell, its direct
neighborhood must cover a ball of radius
$\rho_t\max_{s\in\mathrm{cell}}\sigma_s$. If a cutoff pair remains represented
by M2L, direct replacement would double count it; the implementation must
enlarge the direct geometry or reject that configuration.

**The binding quantity is the smallest AABB gap the stencil leaves to M2L, not
its outer radius.** A radix stencil is a fixed offset set, so the pair that
constrains it is the *closest* cell pair routed to M2L. For a cell offset
$o\in\mathbb Z^3$ the gap is

$$
\mathrm{gap}(o)=h\sqrt{\sum_{q=1}^3\max(0,|o_q|-1)^2},
$$

and adequacy requires
$\min_{o\notin\mathcal D}\mathrm{gap}(o)>\rho_t\sigma_{\max}$ over the direct
set $\mathcal D$. Writing that minimum as $g_{\min}h$, for the $\theta=0.5$
stencil ($|o|^2\le12$, 179 classes) it is attained at $o=(3,2,0)$ with
$g_{\min}=\sqrt5\approx2.236$; for the classic stencil
($|o|_\infty\le1$, 27 classes) $g_{\min}=1$. The stencil's outer
centre-to-centre radius ($\sqrt{12}\,h$ for $\theta=0.5$) is **not** the
relevant comparison: bodies sit anywhere in their cells, so a target near the
corner of its cell and a source in the M2L cell at offset $(3,2,0)$ are only
$\sqrt5\,h$ apart.

At the shipped operating point ($n=10^6$, $\ell=5$, overlap $\beta=2$,
$\varepsilon=10^{-3}$) this gives $\rho_t\sigma\approx9.58d=3.065h$ against
$g_{\min}h=2.236h$: the shipped $|o|^2\le12$ leaf stencil is **not adequate**.
Worst-case M2L pairs then sit at
$\rho=\sqrt5\,h/\sigma=6.99d/2d\approx3.49$, where the
§4 bounds give $E_U=6.8\times10^{-3}$ and $E_J=4.5\times10^{-2}$ — well above
the $10^{-3}$ phase tolerance. §5.1 derives the general adequacy rule and the
two remedies; §6 costs them.

Beyond $\rho_t$, replacing gaussianerf by the singular kernel introduces only
the bounded Gaussian tail derived in §4. Thus this operator retains the
classical long-range FMM and confines expensive regularization math to the
physical smoothing neighborhood.

**This adequacy requirement is not specific to partitioning.** The FMM far
field evaluates the singular kernel under *either* nearfield strategy, so a
regularized-everywhere direct kernel incurs exactly the same $\bar g$ tail
error on any cutoff pair its near set fails to capture. §5.1 is therefore a
correctness condition on the vortex FMM coupling as a whole, and both
candidates in §6 must be measured on the same adequate geometry.

## 3. Cancellation-safe small-$\rho$ evaluation

The ordinary formula for $g$ subtracts equal $O(\rho)$ terms near the origin.
The Jacobian coefficient also cancels because
$\rho g'(\rho)$ and $3g(\rho)$ share the same leading $O(\rho^3)$ term. With
$A=\sqrt{2/\pi}$, use fixed Horner series for small $\rho$:

$$
g(\rho)=A\rho^3\sum_{k=0}^{\infty}
\frac{(-1)^k\rho^{2k}}{(2k+3)2^k k!}
=A\rho^3\left(\frac13-\frac{\rho^2}{10}
+\frac{\rho^4}{56}-\frac{\rho^6}{432}
+\frac{\rho^8}{4224}-\cdots\right).
$$

Define the combined Jacobian numerator directly:

$$
h(\rho)\equiv\rho g'(\rho)-3g(\rho)
=A\rho^5\sum_{k=0}^{\infty}
\frac{(-1)^{k+1}\rho^{2k}}{(2k+5)2^k k!}
=A\rho^5\left(-\frac15+\frac{\rho^2}{14}
-\frac{\rho^4}{72}+\frac{\rho^6}{528}
-\frac{\rho^8}{4992}+\cdots\right).
$$

Then evaluate

$$
a(\rho)=\frac{h(\rho)}{r^2},
\qquad
b(\rho)=-\frac{g(\rho)}{4\pi r^3}.
$$

These expressions never subtract equal leading terms. Although
$C=O(r^{-2})$, the products $gC$, $aC\Delta x$, and $b\Gamma$ have their
correct finite or vanishing limits. For $\rho\le0.5$, successive terms in
both alternating series decrease monotonically, so the first omitted term
bounds the absolute truncation error. Use six Horner terms (`k=0:5`) in
Float32 and ten (`k=0:9`) in Float64. At the worst-case switch $\rho=0.5$,
the resulting series truncation is below working-precision rounding for the
regularized U/J evaluation; §7 verifies the complete formulas. Use FLOWVPM's
existing device-safe `custom_erf` plus the exponential form above for
$\rho>0.5$. No runtime `erfc` implementation is required.

## 4. Cutoff error and truncation radius

The complement is used only to bound the omitted tail offline:

$$
\bar g(\rho)=1-g(\rho)
=\operatorname{erfc}\!\left(\frac{\rho}{\sqrt2}\right)
+\sqrt{\frac{2}{\pi}}\rho e^{-\rho^2/2}.
$$

For velocity, the relative singular-kernel tail is
$E_U(\rho)=\bar g(\rho)$. For the Jacobian, rotational invariance permits
aligning $\Delta x$ with the z axis. The largest transverse singular-J entries
are $2k\Gamma_{1,2}$, $k=1/(4\pi r^3)$, while the omitted regularization tail
is $(2\bar g+\rho g')k\Gamma_{1,2}$. Hence the conservative componentwise
bound is

$$
E_J(\rho)=\bar g(\rho)+\frac{\rho}{2}g'(\rho).
$$

The bounds above are relative to the singular result. If $E$ is either tail
bound, the error relative to the retained regularized result is bounded by

$$
\frac{E}{1-E}.
$$

The radii below solve $E_U,E_J\le\varepsilon/2$. For
$0<\varepsilon\le1$ this implies

$$
\frac{E}{1-E}\le
\frac{\varepsilon/2}{1-\varepsilon/2}
=\frac{\varepsilon}{2-\varepsilon}\le\varepsilon,
$$

so the half-budget rule rigorously converts the singular-relative tail bound
into the required regularized-relative tolerance while leaving rounding and
accumulation margin:

| ε target | ρ_t (U) | ρ_t (J) | context |
|---|---:|---:|---|
| 1e-3 | 4.211 | 4.789 | fixed Integration Phase tolerance |
| 1e-4 | 4.749 | 5.303 | one decade below the gate |
| 1e-6 | 5.665 | 6.182 | comfortable Float32 margin |
| 1e-7 | 6.069 | 6.571 | Float32-epsilon neighborhood |
| 1e-12 | 7.767 | 8.216 | near Float64 working precision |
| 1e-15 | 8.622 | 9.049 | Float64-epsilon neighborhood |

Use the J radius whenever U and J are evaluated together. With per-source
$\sigma$, source-cell geometry uses the maximum source $\sigma$ in that cell;
the pair branch itself uses $r/\sigma_s$.

## 5. Exact-once geometry contract

For every target-source pair, exactly one route must own the interaction:

1. M2L owns well-separated cell pairs, all of whose body pairs satisfy
   $r/\sigma_s>\rho_t$; it evaluates the singular kernel.
2. Direct routing owns the complement. Its pair kernel evaluates regularized
   U/J when $r/\sigma_s\le\rho_t$ and singular U/J otherwise.
3. Self pairs remain excluded by the existing `r2 > 0` rule.

The required source-directed cell predicate is explicit. For target-cell AABB
$B_t$, source-cell AABB $B_s$, and
$\sigma_{s,\max}=\max_{j\in B_s}\sigma_j$, define

$$
d_{\min}^2(B_t,B_s)=\sum_{q=1}^3
\left[\max\!\left(0,\ell_{s,q}-u_{t,q},
\ell_{t,q}-u_{s,q}\right)\right]^2,
$$

where $\ell$ and $u$ are AABB lower and upper coordinates. The cell pair may
be assigned to M2L only if

$$
d_{\min}(B_t,B_s)>\rho_t\sigma_{s,\max};
$$

otherwise it is direct. Equality is direct. This predicate is directional
because regularization uses the source's $\sigma$. Since the AABB minimum is
no larger than any contained body-pair distance and
$\sigma_{s,\max}\ge\sigma_s$, every pair with
$r/\sigma_s\le\rho_t$ is necessarily direct. Conservative false positives
only add direct work.

`scripts/validate_031a_kernel_split.jl` enumerates all ordered, non-self body
pairs on a deterministic test grid, constructs direct and M2L sets from this
predicate, and asserts: no missing pair, no duplicate pair, and no cutoff pair
assigned to M2L. Two regimes are covered in `geometry_coverage.csv`: a
sub-cell cutoff ($\rho_t\sigma_{\max}=0.955h$) and the overlap-2 leaf regime
($\rho_t\sigma_{\max}=3.065h$, where cutoff pairs span several cells). Both
report zero missing, duplicate, and cutoff-to-M2L pairs. `032a` must repeat the
same assertions against its production route construction.

### 5.1 Adequacy of a fixed translation-invariant stencil

The predicate above is a per-cell-pair test, but the radix path ships one fixed
offset set $\mathcal D$ used at every level. Combining the §2 gap formula with
the uniform-cube relations $\sigma=\beta d$, $d=L/n^{1/3}$, and $h=L/2^\ell$,
the stencil is adequate at level $\ell$ iff

$$
\rho_t\,\beta\,\frac{2^\ell}{n^{1/3}}<g_{\min},
\qquad
g_{\min}=\min_{o\notin\mathcal D}
\sqrt{\sum_{q=1}^3\max(0,|o_q|-1)^2},
$$

which rearranges into a **level-invariant bodies-per-cell floor**

$$
\boxed{\;\frac{n}{8^\ell}>\left(\frac{\rho_t\beta}{g_{\min}}\right)^3\;}
$$

with $g_{\min}=\sqrt5$ for the $\theta=0.5$ stencil and $g_{\min}=1$ for the
classic stencil. At $\beta=2$, $\varepsilon=10^{-3}$ ($\rho_t=4.789$) the floors
are **79 bodies per cell** ($\theta=0.5$) and **879 bodies per cell**
(classic).

The count in that floor is bodies per **occupied** cell. Writing it as
$n/8^\ell$ — an average over *all* cells — additionally assumes the bodies fill
the box uniformly and isotropically, which the uniform-cube case satisfies by
construction and a clustered field does not; §5.2 gives the general form and
works both `033` cases. Both forms are asserted equivalent in
`near_set_adequacy_rows`; the full grid ($n=10^3$–$10^6$,
$\beta\in\{1.5,2,2.5\}$, $\varepsilon\in\{10^{-3},10^{-4},10^{-6}\}$,
$\ell=0$–$6$) is tabulated in `data/kernel_splitting/near_set_adequacy.csv`.

Because $\rho_t\sigma/h$ halves with each level up, the constraint **binds only
at the leaf level**: at $n=10^6$, $\ell=5$ the leaf ratio is $3.065$ but the
$\ell=4$ ancestors see $1.533<\sqrt5$. A per-level near set is therefore
sufficient — only the deepest level needs enlarging.

Two remedies exist at fixed accuracy, and their costs differ substantially
(§6):

1. **Enlarge the leaf near set at fixed $\ell$.** Take
   $\mathcal D=\{o:\mathrm{gap}(o)\le\rho_t\sigma\}$. At $n=10^6$, $\ell=5$,
   $\beta=2$, $\varepsilon=10^{-3}$ this is 389 classes, versus the shipped 179.
2. **Reduce $\ell$** until the shipped stencil complies. At the same operating
   point $\ell\le4$ suffices, where the 179-class stencil in fact over-covers
   (117 classes would do).

Remedy 1 is much cheaper. Reducing $\ell$ by one multiplies bodies per cell by
8 while the required class count grows only about $3\times$, so compliant
direct work *decreases* with depth: 52,734 pairs per target at $\ell=3$,
28,564 at $\ell=4$, 11,871 at $\ell=5$, approaching the
$N_{\rm reg}\approx3{,}681$ physics floor. **The correct response to
inadequacy is to enlarge the deepest-level near set, not to make the tree
shallower.**

### 5.2 Applying the rule through the experimenter's overlap parameter

Overlap is the knob an experimenter actually sets, and both Integration Phase
test cases (`scripts/benchmark_033_common.jl`) set the *same* one: $\beta=2$
against the local mean spacing $s=(V_{\rm occupied}/n)^{1/3}$ — the unit-box
cube via $\sigma=2(1/n)^{1/3}$, the helical wake cylinder via
$\sigma=2(V_{\rm cyl}/n)^{1/3}$.

What differs is therefore not $\beta$ but the occupancy. With
$\sigma=\beta s$ and cell size $h$, the §5.1 criterion is

$$
\left(\frac{h}{s}\right)^3>\left(\frac{\rho_t\beta}{g_{\min}}\right)^3,
$$

i.e. the floor counts bodies per *occupied* cell; $n/8^\ell$ substitutes for that
only when the field fills its box. Better still, the criterion needs no
occupancy estimate at all — with $h=L_{\rm box}/2^\ell$ it is a pure depth
ceiling,

$$
\boxed{\;2^{\ell}<\frac{g_{\min}L_{\rm box}}{\rho_t\,\sigma_{\max}}\;}
$$

evaluable from the cache's own box and a max-reduction over $\sigma$.

The two forms agree on the cube at every resolution (asserted in the script) and
diverge on the wake, a solid cylinder of diameter $D$ and length $5D$ whose
bounding **cube** has side $L_{\rm box}=5D$: it fills only
$\tfrac{\pi}{4}D^2\cdot5D/(5D)^3=3.14\%$ of that cube, so its occupied cells
hold $\approx32\times$ the all-cell average
(`data/kernel_splitting/case_adequacy.csv`):

| case | $n$ | $\sigma$ | $\ell$ ceiling (geometric) | $\ell$ ceiling ($n/8^\ell$ rule) | occupied cells | bodies each |
|---|---:|---:|---:|---:|---:|---:|
| cube | $10^6$ | 0.0200 | 4 | 4 | 4,096 | 244 |
| wake | $10^3$ | 0.3155 | 2 | 1 | 2 | 497 |
| wake | $10^4$ | 0.1465 | 3 | 2 | 16 | 622 |
| wake | $10^5$ | 0.0680 | 5 | 3 | 1,029 | 97 |
| wake | $10^6$ | 0.0316 | 6 | 4 | 8,235 | 121 |

Because a clustered field concentrates bodies into the occupied cells, the
all-cell average can only *under*-report the admissible depth — for the wake by
one to two levels, and by more as the aspect ratio grows. §5.1 showed compliant
direct work falls steeply with depth (52,734 pairs per target at $\ell=3$
against 11,871 at $\ell=5$), so applying the box-filling form to an elongated
domain imposes a real and unnecessary cost. The error runs the other way too:
with $n$ and the nominal $\beta$ unchanged, the $n/8^\ell$ form does not see a
$\sigma$ that has grown under `CoreSpreading`. Production must therefore test
the boxed ceiling against the measured $\sigma_{\max}$, and use the
bodies-per-cell floor only as a design-time sizing heuristic for a uniform
field.

Note what the ceiling does *not* say: the finest admissible cell
$h_{\min}=\rho_t\sigma_{\max}/g_{\min}\approx4.28s$ depends on particle spacing
alone, so the count of *occupied* cells at $h_{\min}$ is independent of box
shape. An oversized box costs extra tree levels and inflates anything sized by
total rather than occupied cells; it does not inflate far-field work directly.
Quantifying that penalty on the wake is staged as a `035` lever.

## 6. Cost model and expected regime

For approximately uniform density $n/V$, the number of regularized pairs per
target is the physics floor

$$
N_{\rm reg}\approx\frac{4\pi}{3}(\rho_t\sigma)^3\frac{n}{V},
$$

independent of tree depth. For the `033` overlap-2 cube,
$\sigma=2(V/n)^{1/3}$, giving about 3,681 U+J regularized pairs per target at
$\varepsilon=10^{-3}$.

The regularized fraction is that floor divided by the direct pairs the
**production** near set delivers. That near set is
`max(stencil classes, adequate classes)`: it can never be *smaller* than the
stencil's own class count, because the near/far split is fixed by far-field
multipole accuracy (179 classes at `θ=0.5`, 27 classic), and §5.1 may force it
larger. All candidate strategies run on that same geometry, so the comparison
is like-for-like:

$$
f=\frac{N_{\rm reg}}{|\mathcal D|\,n/8^\ell},
\qquad
\frac{c_{\rm partition}}{c_{\rm reg}}=f+\frac{1-f}{q},
\qquad
q=\frac{c_{\rm reg}}{c_{\rm sing}}\approx1.5\text{-}2.5 .
$$

At $n=10^6$, $\beta=2$, $\varepsilon=10^{-3}$
(`data/kernel_splitting/near_set_adequacy.csv`):

| $\ell$ | adequate $\|\mathcal D\|$ | production $\|\mathcal D\|$ | direct pairs/target | $f$ | pair-kernel speedup |
|---:|---:|---:|---:|---:|---|
| 3 | 27 | 179 | 349,609 | 0.011 | 1.49–2.46x |
| 4 | 117 | 179 | 43,701 | 0.084 | 1.44–2.22x |
| 5 | 389 | 389 | 11,871 | 0.310 | 1.30–1.71x |
| 6 | 1,839 | 1,839 | 7,015 | 0.525 | 1.19–1.40x |

So partitioning helps most where the production near set most over-covers the
physical cutoff — shallow trees, where the `θ=0.5` stencil alone is far wider
than the smoothing cutoff — and its advantage decays as the tree deepens and
the direct volume converges onto the cutoff ball. At the shipped $\ell=5$
operating point the prediction is a **1.30–1.71x pair-kernel speedup** before
branch divergence and routing overhead.

Note these two levers pull in opposite directions: deepening the tree cuts
total direct work (11,871 pairs/target at $\ell=5$ against 28,564 at $\ell=4$)
while shrinking partitioning's relative advantage. `032a` must therefore
compare the two strategies at a fixed, adequate geometry rather than at each
strategy's own optimum. A finer grid reduces conservative cell overcoverage,
but it cannot shrink the physical cutoff or excuse a violation of §5.1.

`032a` therefore compares three candidates:

1. single-pass `RegularizedVortex`, which evaluates regularized math for every
   direct pair;
2. partitioned replacement, which evaluates stable regularized math only
   inside $\rho_t$ and singular math for the remaining direct pairs; and
3. the two-pass additive correction of §6.1.

Measurement on H200 decides whether the saved transcendental work outweighs
branch divergence and any geometry overhead.

## 6.1 Third candidate: two-pass additive correction

### Operator

Leave the FMM completely unmodified — singular far field *and* singular direct
nearfield — and add a second pass carrying only the regularization deficit:

$$
\Delta U_i=-\bar g(\rho)C_i,
\qquad
\Delta J_{ij}=\Delta a\,C_i\Delta x_j+\Delta b\,\varepsilon_{ijk}\Gamma_k,
$$

$$
\Delta a=\frac{\rho g'(\rho)+3\bar g(\rho)}{r^2},
\qquad
\Delta b=\frac{\bar g(\rho)}{4\pi r^3},
$$

obtained by subtracting the singular coefficients
($a_\infty=-3/r^2$, $b_\infty=-1/(4\pi r^3)$) from the regularized ones of §1.
In exact arithmetic $(U,J)_{\rm sing}+(\Delta U,\Delta J)=(U,J)_{\rm reg}$
identically.

Neither pass needs `erfc`: $\bar g=1-g$ loses digits only where $\bar g$ is
already a negligible fraction of the retained singular value (at $\rho_t$,
$\bar g=4.2\times10^{-5}$, so even a 1% error in the correction is $4\times
10^{-7}$ of the pair's field contribution).

### What this changes about §5.1, and what it does not

The exact-once *routing* constraint disappears. Pass 2 rides on top of a
**complete** pass 1 and never asks which route owned a pair: a pair at
$\rho=3.5$ receives its singular value from M2L (at FMM truncation accuracy)
and its exact deficit from pass 2, and they add. Nothing in the `025`
hierarchical construction is touched, and there is no double-count bookkeeping
to get wrong.

The §5.1 *reach* requirement does not disappear — it moves. Pass 2 must visit
every pair with $r/\sigma_s\le\rho_t$, so its traversal needs the same
$\rho_t\sigma_{\max}$ reach the single-pass strategies demand of the near set
(389 offset classes at $n=10^6,\ell=5,\beta=2,\varepsilon=10^{-3}$). If pass 2
simply reuses the shipped 179-class direct route list, it inherits the identical
inadequacy. Its one geometric relief is that **pass 1 needs no enlargement at
all** beyond what far-field accuracy already requires.

### Conditioning: why this needs Float64 or a $\rho_c$ floor

Pass 1 contributes $-2k\Gamma$ to the largest transverse $J$ component and pass 2
contributes $(\rho g'+2\bar g)k\Gamma$; both approach magnitude $2k\Gamma$ as
$\rho\to0$ while their sum is $(\rho g'-2g)k\Gamma\approx A\rho^3k\Gamma/3$. The
subtraction happens **in the target's accumulator, across two kernels**, so —
unlike §3 — no series reformulation can reach it. A unit rounding error is
amplified by

$$
\text{amp}_U=\frac{1}{g(\rho)},
\qquad
\text{amp}_J=\frac{2}{|\rho g'(\rho)-2g(\rho)|},
$$

both $\sim\rho^{-3}$. Measured against the 256-bit reference
(`data/kernel_splitting/two_pass_conditioning.csv`):

| $\rho$ | amp$_J$ | rel. err $J$, F64 | rel. err $J$, F32 | rel. err $J$, F32 hybrid |
|---:|---:|---:|---:|---:|
| 0.01 | 7.5e6 | 5.2e-10 | 1.6e-1 | 1.1e-7 |
| 0.02 | 9.4e5 | 5.2e-11 | 2.4e-2 | 1.3e-7 |
| 0.05 | 6.0e4 | 5.1e-12 | 1.1e-3 | 1.5e-7 |
| 0.1 | 7.6e3 | 1.1e-12 | 4.5e-4 | 4.9e-8 |
| 0.5 | 76 | 4.1e-15 | 2.2e-6 | 3.3e-8 |
| 2.0 | 3.3 | 2.4e-16 | 7.3e-8 | 7.3e-8 |

**Float64 is sufficient by itself.** The per-pair $J$ error stays below
$10^{-3}$ anywhere in the scanned range $\rho\in[10^{-4},1]$, and the expected
number of pairs closer than that scan floor in a uniform $n=10^6$, $\beta=2$
field is $1.7\times10^{-5}$ — it never occurs.
Only the *singular direct term and its correction for the same pair* need F64;
the far field may remain FP16-WMMA/Float32, since its contribution is not part
of the cancelling pair.

**Float32 alone is not sufficient**, but the failure is a tail effect, not a
gate effect. Per-pair $J$ error crosses $10^{-3}$ below $\rho\approx0.0765$, and
$\sim7.5\times10^3$ of the $\sim5\times10^{11}$ ordered pairs at $n=10^6$ sit
there. Because the phase gate is a sampled RMS over targets, and a close pair's
*regularized* contribution is finite and comparable to any $\rho\sim1$ neighbor,
the RMS impact is of order $\sqrt{7.5\times10^3/10^6}\times$(per-target error)
— well inside $10^{-3}$. The exposure is a few thousand particles per step with
0.1–10% $J$ error, which matters for the VPM stretching term and dynamical
stability rather than for the accuracy gate.

**The $\rho_c$ hybrid removes it entirely in Float32.** Evaluate pairs with
$\rho\le\rho_c$ with the §3 stable regularized formula inside pass 1 and exclude
them from pass 2; only the shell $\rho_c<\rho\le\rho_t$ is corrected. The final
column above confirms working-precision accuracy at every $\rho$. Two
constraints fix $\rho_c$:

- $\rho g'-2g$ **changes sign at $\rho=1.3688$**, where the transverse $J$
  component vanishes and any relative measure against it is meaningless, so
  $\rho_c$ must sit well beyond the crossing;
- pass 1 must be adequate for $\rho_c$, i.e. $n/8^\ell>(\rho_c\beta/g_{\min})^3$.

$\rho_c=2$ satisfies both: amp$_J=3.3$, and the pass-1 floor becomes
$(4/\sqrt5)^3=5.7$ bodies per cell against the 30.5 available at
$n=10^6,\ell=5$ — so the shipped 179-class stencil remains adequate for pass 1,
which is the whole point.

### Cost model

Pass 1 visits $P_1$ pairs, pass 2 visits $P_t$, the single-pass baselines visit
$P_d$; $N_t$ is the physics floor and $N_c$ the count inside $\rho_c$. With
$\lambda$ the per-visit load/index cost and $q$ the regularized-math cost, both
in units of one singular U+J evaluation:

$$
\begin{aligned}
c_{\rm reg-everywhere}&=P_d(\lambda+q), \\
c_{\rm partition}&=P_d\lambda+N_tq+(P_d-N_t), \\
c_{\rm two-pass}&=(P_1+P_t)\lambda+N_tq+(P_1-N_c).
\end{aligned}
$$

The transcendental count is $N_t$ in both split strategies — the physics floor
is invariant, so two-pass never buys transcendental work. It trades
$(P_d-N_t)-(P_1-N_c)$ singular evaluations for $P_1+P_t-P_d$ extra visits,
and beats partitioning iff $\lambda$ is below the crossover
(`data/kernel_splitting/two_pass_cost_model.csv`, $n=10^6$, $\beta=2$,
$\varepsilon=10^{-3}$, $\rho_c=2$):

| $\ell$ | pass 1 | pass 2 | partitioned | extra visits | saved singular | $\lambda^\ast$ |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 179 cls / 349,609 | 27 cls / 52,734 | 179 cls / 349,609 | 52,734 | −3,413 | −0.065 |
| 4 | 179 cls / 43,701 | 117 cls / 28,564 | 179 cls / 43,701 | 28,564 | −3,413 | −0.119 |
| 5 | 179 cls / 5,463 | 389 cls / 11,871 | 389 cls / 11,871 | 5,463 | 2,996 | 0.548 |
| 6 | 275 cls / 1,049 | 1,839 cls / 7,015 | 1,839 cls / 7,015 | 1,049 | 2,553 | 2.434 |

**Two-pass and partitioning have opposite depth trends.** At $\ell\le4$ the
stencil's own near set already contains the whole cutoff ball, so pass 2 is pure
overhead and two-pass can never win ($\lambda^\ast<0$). At $\ell=5$ the two are
within a factor of $\lambda\approx0.55$ of each other — genuinely a measurement
question on H200, where a coalesced, tiled source load is cheap relative to a
full singular U+J evaluation. At $\ell=6$ two-pass wins for any plausible
$\lambda$, because pass 1 stays at 275 classes while only the correction pass
pays the 1,839-class traversal. Since §5.1 showed compliant direct work *falls*
with depth, the deep-tree regime is the one production is moving toward, and
this is where two-pass is strongest.

### Summary for `032a`

| | routing change | precision floor | best regime |
|---|---|---|---|
| regularized-everywhere | near set must reach $\rho_t$ | F32 ok | the divergence-proof fallback (§6.3) |
| partitioned replacement | near set must reach $\rho_t$ | F32 ok | shallow trees ($\ell\le5$), **binned stream required** |
| two-pass, plain | none; pass 2 needs its own $\rho_t$ reach | **F64 accumulate** | deep trees ($\ell\ge6$) |
| two-pass, $\rho_c=2$ hybrid | none; pass 1 must reach $\rho_c$ only | F32 ok | deep trees ($\ell\ge6$) |

**Neither split strategy reduces the number of expensive kernel evaluations
below the other.** The set needing $g$ is physical — the pairs with
$\rho\le\rho_t$ — so both hit the same floor $N_{\rm reg}=3{,}681$ per target at
the shipped operating point, against $11{,}871$ for regularized-everywhere.
Splitting buys a 3.2x reduction in transcendental work; the choice *between*
split strategies trades only cheap singular evaluations against extra visits.
The three subsections below are therefore worth more than that choice: §6.2
removes the `erf` from the expensive branch outright, §6.3 shows that a naive
implementation of either split strategy is *slower* than regularized-everywhere,
and §6.4 attacks the floor $N_{\rm reg}$ itself.

## 6.2 The expensive branch needs one `exp` and no `erf`

Above $\rho_c$ the retained result is $O(1)$, so $\bar g$ is needed only to
**absolute** tolerance $(\varepsilon/2)g$ — not relative. Write

$$
\bar g(\rho)=e^{-\rho^2/2}\bigl(A\rho+s(\rho)\bigr),
\qquad
s(\rho)=\operatorname{erfc}\!\left(\frac{\rho}{\sqrt2}\right)e^{\rho^2/2},
$$

which moves the entire error function into $s$. Over the shell
$[2,\,4.789]$, $s$ falls smoothly and monotonically from $0.3362$ to $0.1601$ —
a nearly flat function of $u=1/\rho^2$ — and $\rho g'=A\rho^3e^{-\rho^2/2}$
reuses the same exponential. The binding accuracy point is $\rho_c$, where
$|\delta\bar g|\propto e^{-\rho^2/2}$ is largest while the budget
$(\varepsilon/2)g$ is smallest; at $\rho_c=2$, $\varepsilon=10^{-3}$ that budget
is $3.69\times10^{-4}$. Least-squares fits of $s$ in $u$
(`data/kernel_splitting/cheap_gbar_fit.csv`):

| degree in $u=1/\rho^2$ | max $|\delta s|$ | max $|\delta\bar g|$ | meets $3.69\times10^{-4}$ |
|---:|---:|---:|---|
| 1 | 2.2e-2 | 3.0e-3 | no |
| 2 | 5.5e-3 | 7.4e-4 | no |
| 3 | 1.5e-3 | 2.1e-4 | **yes** |
| 4 | 4.7e-4 | 6.4e-5 | yes |

So the outer branch is **one hardware exponential plus four FMAs**: no FDLIBM
`custom_erf` port, no `erfc`. This applies to the partitioned strategy as well —
above $\rho\approx2$ it may form $g=1-\bar g$, where the cancellation costs
under one digit. Full relative accuracy in $g$ is still required below
$\rho\approx2$, and that range is already served by the §3 series, which is
itself only a polynomial. **The `erf` can therefore be removed from the
production nearfield entirely**, in every candidate strategy.

## 6.3 Warp divergence: the unbinned split kernels are slower than no split

The §6/§6.1 cost models assume a branch-free pair stream. A GPU warp pays any
branch taken by *any* of its 32 lanes, so with a regularized fraction $f$ the
probability that a warp is branch-homogeneous is $f^{32}+(1-f)^{32}$. At the
shipped $\ell=5$ operating point $f=0.310$ and that probability is
$6.9\times10^{-6}$: essentially **every** warp executes both paths. The
unbinned partitioned kernel then costs $\lambda+q+1$ per pair instead of
$\lambda+fq+(1-f)$, which is *worse* than regularized-everywhere's
$\lambda+q$ — **1.56x** at $\ell=5$, $q=1.5$, $\lambda=0.3$
(`data/kernel_splitting/divergence_model.csv`). Weighting the two homogeneous
outcomes by their probabilities, the honest per-pair cost is

$$
\lambda+f^{32}q+(1-f)^{32}+\bigl(1-f^{32}-(1-f)^{32}\bigr)(q+1),
$$

which is what the table below reports.

Cell-level classification does not rescue it. A class $o$ is divergence-free
only if its *farthest* corner is still inside the cutoff,
$h\sqrt{\sum_q(|o_q|+1)^2}\le\rho_t\sigma$. At $\ell=5$ only **19 of the 389**
direct classes qualify; 370 are mixed. (At $\ell=6$ it improves to 389 of 1,839,
still leaving 1,450 mixed.)

| $\ell$ | $f$ | $P(\text{homogeneous warp})$ | classes | entirely inside | mixed | unbinned / reg-everywhere |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.011 | 7.1e-1 | 179 | 0 | 179 | 0.96x |
| 4 | 0.084 | 6.0e-2 | 179 | 0 | 179 | 1.51x |
| 5 | 0.310 | 6.9e-6 | 389 | 19 | 370 | 1.56x |
| 6 | 0.525 | 1.1e-9 | 1,839 | 389 | 1,450 | 1.56x |

The penalty saturates at 1.56x once warps actually diverge, which is why it is
$f$-independent from $\ell=4$ up. It relents only at $\ell=3$, where $f$ is tiny
enough that 71% of warps are all-singular and the unbinned kernel is *not*
slower (0.96x) — but $\ell=3$ is also the regime where the binned kernel would
have saved the most (1.49–2.46x by §6), so the divergence loss relative to a
correct implementation is largest exactly there.

**Consequence for `032a`:** either split strategy must present the pair kernel
with a distance-**binned or sorted** stream so warps are homogeneous, or it will
lose to the `032` baseline it is trying to beat. Two-pass is partially
insulated — its pass 1 is uniformly singular and branch-free — but its pass 2
carries the same requirement. This is a first-order implementation constraint,
not a tuning detail, and it should be settled before the A/B is run.

## 6.4 Tightening $\rho_t$ against the accumulated tail

$N_{\rm reg}\propto\rho_t^3$, so the cutoff radius is the single largest lever
on the expensive-pair count, and it is set by the §4 **per-pair** bound. The
phase gate is a sampled RMS over targets. For a uniform field with random
$\Gamma$ orientations the omitted tail adds incoherently, so modelling shell
pair counts as $4\pi r^2(n/V)\,dr$ and per-pair magnitudes as $|C|\sim r^{-2}$
(U) and $r^{-3}$ (J) gives the relative RMS error
$\sqrt{I_{\rm tail}(\rho_t)/I_{\rm tot}}$ with

$$
I^U_{\rm tail}=\int_{\rho_t}^{\infty}\frac{\bar g^2}{\rho^2}d\rho,
\quad
I^U_{\rm tot}=\int_{0}^{\infty}\frac{g^2}{\rho^2}d\rho,
\quad
I^J_{\rm tail}=\int_{\rho_t}^{\infty}\frac{(\rho g'+2\bar g)^2}{\rho^4}d\rho,
\quad
I^J_{\rm tot}=\int_{0}^{\infty}\frac{(\rho g'-2g)^2}{\rho^4}d\rho.
$$

Both denominators converge; $I^J_{\rm tot}=0.0967$ is domain-independent
(the $r^{-3}$ weight kills the far field), while $I^U_{\rm tot}$ carries a weak
$1/R$ domain tail ($0.514$ at $R=20$, $0.548$ at $R=60$, $0.554$ at $R=100$) —
a 2% spread that moves $\rho_t$ negligibly. Solving
$\sqrt{I_{\rm tail}/I_{\rm tot}}\le\varepsilon/2$
(`data/kernel_splitting/rt_accumulated.csv`):

| $\varepsilon$ | $\rho_t^U$ per-pair | $\rho_t^U$ RMS | $\rho_t^J$ per-pair | $\rho_t^J$ RMS | expensive pairs | $\ell=5$ classes |
|---|---:|---:|---:|---:|---:|---:|
| 1e-3 | 4.211 | 3.668 | 4.789 | 4.252 | ×0.700 | 389 → 275 |
| 1e-4 | 4.749 | 4.230 | 5.303 | 4.770 | ×0.728 | 485 → 389 |
| 1e-6 | 5.665 | 5.185 | 6.182 | 5.664 | ×0.769 | 613 → 565 |

At the phase tolerance this cuts expensive pairs by **30%** and shrinks the
required leaf near set from 389 to 275 classes — a saving that applies to every
candidate strategy at once, and that also relaxes the §5.1 adequacy floor from
79 to 55 bodies per cell. It is a *statistical* bound, not the rigorous
worst-case guarantee of §4, and it inherits the §7 no-coherent-cancellation
assumption. The recommendation is therefore: keep the §4 per-pair radii as the
conservative default, and let `032a` adopt the RMS radii only after its
sampled-direct measurement confirms them on both test cases.

## 7. Validation

`scripts/validate_031a_kernel_split.jl` is stdlib-only and validates:

1. the stable $g$ and $h$ series against a 256-bit regularized reference for
   $\rho\in[10^{-6},0.5]$;
2. ordinary-form parity above the series switch;
3. partitioned evaluation versus the full regularized reference through
   $\rho=10$, with error normalized to regularized U/J; and
4. the U/J cutoff table and its regularized-relative half-budget proof;
5. exhaustive exact-once/cutoff coverage on deterministic uniform-grid body
   sets using the source-directed AABB predicate, in both the sub-cell and
   overlap-2 leaf regimes; and
6. §5.1 stencil adequacy: the enumerated $g_{\min}$ of each shipped stencil,
   agreement between the gap test and the closed-form bodies-per-cell floor
   over the full $(n,\ell,\beta,\varepsilon)$ grid, the minimal adequate and
   production class counts, and the §6 cost model
   (`data/kernel_splitting/near_set_adequacy.csv`);
6b. §5.2 per-case adequacy: the depth ceiling
   $2^\ell<g_{\min}L_{\rm box}/(\rho_t\sigma_{\max})$ for both `033` cases at
   every resolution, asserting that it agrees exactly with the $n/8^\ell$ form
   on the box-filling cube and is never exceeded by it on the wake
   (`data/kernel_splitting/case_adequacy.csv`);
7. §6.1 two-pass conditioning: measured Float64, Float32, and $\rho_c$-hybrid
   errors against the 256-bit reference, asserted to track the predicted
   amplification and to hold the hybrid at working precision; the
   $\rho g'-2g$ sign crossing located and required to sit clear of $\rho_c$;
   and the per-pair Float32 breakdown radius
   (`data/kernel_splitting/two_pass_conditioning.csv`); and
8. §6.1 three-strategy cost model and its $\lambda^\ast$ crossovers
   (`data/kernel_splitting/two_pass_cost_model.csv`);
9. §6.2 the `erf`-free outer-shell form: least-squares fits of $s(\rho)$ in
   $1/\rho^2$ with the achieved $|\delta\bar g|$ against the
   $(\varepsilon/2)g(\rho_c)$ budget, asserting that some degree $\le4$ meets
   it (`data/kernel_splitting/cheap_gbar_fit.csv`);
10. §6.3 warp-divergence model: regularized fraction, homogeneous-warp
    probability, entirely-inside versus mixed class counts, and binned,
    fully-diverged, and warp-weighted unbinned cost ratios
    (`data/kernel_splitting/divergence_model.csv`); and
11. §6.4 accumulated-tail radii: the four convergent integrals, the RMS-solved
    $\rho_t$, the resulting expensive-pair and class-count reductions, and the
    $I^U_{\rm tot}$ domain sensitivity
    (`data/kernel_splitting/rt_accumulated.csv`).

Results in `data/kernel_splitting/partitioned_replacement.csv`:

| precision | max relative U, close series | max relative J, close series |
|---|---:|---:|
| Float64 | 8.12e-16 | 6.15e-16 |
| Float32 | 3.35e-7 | 3.08e-7 |

The implementation validation also confirms the singular branch beyond the
$\varepsilon=10^{-6}$ J cutoff stays below the requested tail error in both
precisions. `data/kernel_splitting/geometry_coverage.csv` reports zero missing,
duplicate, or cutoff-to-M2L pairs in both regimes.

**Assumption on error accumulation.** The §4 bounds are per-pair and relative
to that pair's singular contribution. Converting them to the phase's field-level
RMS gate assumes no systematic cancellation across the omitted tail
contributions; with sign-varying vortex contributions the accumulated relative
error tracks the per-pair bound, but a configuration with strong coherent
cancellation in U or J could exceed it. `032a` measures sampled-direct error
directly and is the binding check.

## 8. Deferred: SFS kernel

The SFS/`Estr` path evaluates the Gaussian blob basis directly and is already
numerically compact. Its truncation and pipeline are deferred to a later row,
per user direction.
