# Fourier Nearfield Cost Model (037d, paper study)

Decides whether a device-resident **full particle-mesh (VIC-style)
evaluation** of the `gaussianerf`-regularized U/J field — spread vorticity to
a rectangular mesh, convolve with the mollified Biot–Savart kernel in
k-space, interpolate U and J back; **no direct pass, no kernel split** —
could plausibly beat the shipped partitioned nearfield (and partitioned plus
the anticipated `038` adaptive octree) on the `037b` cases at the campaign
gate (sampled velocity relative RMS $\le 10^{-3}$; J diagnostic).

Pricing basis: the `037b` same-job anchors (jobs 13170509/13170230/13170520).
Supporting artifacts: `scripts/fm037d_error_spotcheck.jl` →
`data/fm037d_error_spotcheck.csv` (measured error model, stdlib-only, local)
and `scripts/fm037d_cost_tables.jl` → `data/fm037d_cost_tables.csv`,
`data/fm037d_ewald_rotor.csv` (priced tables; every assumption A1–A12 is
recorded in that script next to its use and summarized in §5).

## 1. Why the idea is admissible at all

The shipped reference field is already regularized: with
$\rho = r/\sigma$ and

$$
g(\rho)=\operatorname{erf}\!\left(\frac{\rho}{\sqrt2}\right)
-\sqrt{\frac{2}{\pi}}\,\rho\,e^{-\rho^2/2},
$$

the velocity kernel is $K_\sigma(\vec r) = -\,g(\rho)\,\vec r/(4\pi r^3)$,
i.e. the singular Biot–Savart kernel convolved with the Gaussian core
$\zeta_\sigma(\vec r) = (2\pi)^{-3/2}\sigma^{-3} e^{-r^2/2\sigma^2}$, whose
spectrum is

$$
\hat\zeta_\sigma(k) = e^{-\sigma^2 k^2/2}.
$$

The **exact answer the gate compares against is a globally smooth field**
with spectral content cut off at $k\sim 1/\sigma$: representable on a mesh of
spacing $h = O(\sigma)$ with no singular residue. That is the structural
difference from the falsified `037c` premise (meshing a *deficit* that
needed compact support the accuracy gate refused to grant): here nothing is
truncated in real space — the mesh evaluates *all* pairs, near field
included.

Pipeline modeled (per U/J solve):

1. spread $\vec\Gamma_i$ (3 components) onto the mesh with an order-$p$
   B-spline window;
2. 3 forward FFTs; multiply by the precomputed free-space kernel spectrum
   $\hat K_\sigma$ with $\mathrm{sinc}^{2p}$ (spread+interp) deconvolution and
   form $\hat u_a = \varepsilon_{abc}\hat K_b\hat\omega_c$;
3. 3 inverse FFTs;
4. interpolate U with the same order-$p$ window; obtain J from the
   **analytic B-spline derivative of the interpolant of the U meshes**
   (no extra transforms; J is a campaign diagnostic, not a gate — this is
   the 6-transform scheme; the 12-transform $ik\otimes$ alternative would
   roughly double the FFT line only).

Free space is handled by Hockney doubling (×2 per axis) with the kernel
tabulated in real space and transformed once at construction — exact
free-space convolution, no images. This factor (8× points) is priced in
every band; trimming alternatives (Vico–Greengard, kernel-support-aware
padding) are upside not taken.

## 2. Error model — measured, not asserted

`scripts/fm037d_error_spotcheck.jl` implements the exact pipeline above in
miniature (interior mesh $32^3$, Hockney-doubled $64^3$, hand-rolled FFT,
stdlib only): 1650 particles including 150 injected near pairs at
$r \in [0.5, 1.5]\,\sigma$ (the near-pair regime where mesh error peaks),
sampled at 250 particle positions against the exact regularized direct sum
(erf by Abramowitz–Stegun 7.1.26; small-$\rho$ evaluation via the stable
series $g(\rho)/\rho^3 = \sqrt{2/\pi}\,(1/3 - \rho^2/10 + \rho^4/56 -
\rho^6/432 + \rho^8/3840)$, and J referenced analytically through
$q(r)=g/4\pi r^3$, $q'(r)$ — a finite-difference reference is unusable
because the erf approximation's $1.5\times10^{-7}$ absolute error is
catastrophically amplified at $r\ll\sigma$).

Measured sampled velocity relative RMS (`data/fm037d_error_spotcheck.csv`):

| $h/\sigma$ | $p=2$ | $p=4$ | $p=6$ |
|---:|---:|---:|---:|
| 1.4 | 1.18e-1 | 4.71e-2 | 3.61e-2 |
| 1.0 | 4.93e-2 | 7.94e-3 | 3.72e-3 |
| 0.85 | 3.35e-2 | 3.08e-3 | 1.00e-3 |
| 0.70 | 2.17e-2 | 1.12e-3 | **5.75e-4** |
| 0.55 | 1.33e-2 | **5.90e-4** | 6.99e-4 |
| 0.475 | — | 5.19e-4 | 6.98e-4 |
| 0.40 | — | **4.57e-4** | 6.46e-4 |

J diagnostic at the passing configs: $4.2\times10^{-3}$ ($p=4$,
$h/\sigma=0.55$) and $3.0\times10^{-3}$ ($p=6$, $h/\sigma=0.7$) — the same
order as the shipped anchors' logged J RMS (e.g. cube 1e5 F32 anchor:
$3.9\times10^{-3}$), so the 6-transform J scheme is adequate as a
diagnostic-grade output.

Reading of the table:

- **The gate is met with margin** at $p=4,\ h/\sigma=0.55$ and
  $p=6,\ h/\sigma=0.7$ (both $\approx 5.8\times10^{-4} \le 10^{-3}$), with
  aliasing/deconvolution and the near-pair regime included in the number.
- **Floor caveat:** below $h/\sigma\approx0.55$ the error flattens near
  $5\times10^{-4}$ instead of falling as $h^p$. The suspected cause is a
  script conservatism — deconvolution by the *continuous* $\mathrm{sinc}^p$
  transform rather than the exact discrete PME (Euler-spline) factor;
  production PME/NUFFT implementations (exact $b(m)$ factors, Kaiser–Bessel
  windows) sit below this floor in the literature. The measured numbers are
  therefore an **achievable upper bound** on error, the safe direction for a
  go/no-go model. Verifying the exact-factor variant is delegated to the
  implementation row (kept off the local machine per the 2026-08-14 compute
  directive). The consequence today: gate margin is $\approx1.7\times$, not
  $10\times$, and the pessimistic cost band uses the tightest measured
  point ($h/\sigma = 0.40$) rather than an unmeasured smaller $h$.

Band picks used by the cost tables (A8): optimistic $p=6$ or $p=4$ at their
$\le 10^{-3}$ points, nominal $p=4$ at $h/\sigma=0.55$ ($\le 7\times10^{-4}$),
pessimistic $p=4$ at $h/\sigma=0.40$ (tightest measured; per-band $p$ chosen
to minimize total time — $p=4$ wins throughout).

## 3. Mesh sizing from the real case geometries

Tight rectangular bounds (the `037` machinery), uniform $\sigma$ from
`benchmark_033_common.jl`, rotor from `rotor_case_stats.csv`:

| case | box [m or unitless] | $\sigma$ (1e5) | $\sigma$ (1e6) | tight-box fill |
|---|---|---|---|---|
| cube | $1\times1\times1$ | 0.0431 (uniform) | 0.0200 (uniform) | 100% |
| wake | $1\times1\times5$ | 0.0680 (uniform) | 0.0316 (uniform) | 78.5% |
| rotor | $0.232\times0.201\times1.203$ | $[1.73\text{e-}4,\,3.11\text{e-}3]$, 18× spread | $[1.73\text{e-}5,\,3.11\text{e-}4]$, 18× spread | ≪1% (filaments) |

With $h = (h/\sigma)^\ast\,\sigma_{\min}$, interior dims
$\lceil L_d/h\rceil + p$, Hockney ×2 per axis, rounded to 5-smooth FFT sizes
(A9): the padded meshes are **0.9–26 M points for cube and wake**
(0.04–2.5 GB for the six resident complex fields) — comfortably inside the
capacity contract — and **$5\times10^{11}$–$1.3\times10^{15}$ points for the
rotor** (tens of TB to PB): *full VIC on the rotor is structurally
infeasible*, because the mesh must resolve $\sigma_{\min}$ over the whole
mostly-empty bounding box while Fourier cost scales with box volume, not
occupied volume. Since particle spacing ties to $\sigma$ (overlap 2), the
uniform-σ mesh point count is
$N_{\mathrm{pad}} \approx 8\,n\,(2\,(h/\sigma)^\ast)^{-3}/\text{fill}$ —
i.e. $O(n)$, the classic VIC regime.

## 4. σ-heterogeneity — the honest boundary of the method

The k-space multiply applies **one** $e^{-\sigma^2k^2/2}$; per-particle
$\sigma_i$ must be handled elsewhere. The viable mechanisms and their costs:

- **σ-binned multi-mesh:** bin particles by $\sigma$; each bin's field is
  long-range, so each bin needs a *full-domain* solve at $h\sim0.78\,
  \sigma_{\mathrm{bin,min}}$. The domain-volume mesh cost of the smallest-σ
  bin alone equals full-VIC-at-$\sigma_{\min}$ — **multi-mesh does not
  rescue the rotor** (it rescues only fields whose smallest bin is still
  coarse).
- **NUFFT-style factorization:** spread particle $i$ with a real-space
  Gaussian of width $\sqrt{\sigma_i^2-\alpha^2}$ on a mesh carrying
  $e^{-\alpha^2k^2/2}$, $\alpha\le\sigma_{\min}$. The mesh resolution floor
  is again $\sigma_{\min}$, and the spread support for large-σ particles
  grows to $O(\sigma_{\max}/h)^3$ cells. Same wall.

**Conclusion: full VIC is a near-uniform-σ method.** Its mesh cost carries a
multiplier $(\bar\sigma/\sigma_{\min})^3$ relative to a uniform field at
$\bar\sigma$: a spread of 1.5× costs ~3.4× mesh (nominal cube/wake wins
survive it); the rotor's 18× spread costs ~6000× (dead). Workloads this
leaves eligible: the cube/wake campaign classes — well-filled fields with a
narrow σ (age) distribution. Continuously-emitting wakes with wide age
spans (the rotor, and mature FLOWVPM rotor simulations generally) are
excluded; for those the `038` adaptive octree remains the funded path.
(Core spreading itself is additive in $\sigma^2$ and *shrinks* relative
spread over time; heterogeneity comes from birth-time differences.)

## 5. Cost model and priced assumptions

Full definitions live as A1–A12 at the top of
`scripts/fm037d_cost_tables.jl`; summary: H200 HBM 4.8 TB/s with effective
fraction 0.70/0.50/0.30 (opt/nom/pess) for FFT and sweep kernels (A1); FFT
priced bandwidth-bound at 48 B/point/transform F32 (3 passes × R+W × 8 B,
R2C savings ignored) (A2); 6 transforms per step, +3 in pess for kernel
re-tabulation on geometry drift (A3); one 72 B/point k-multiply+deconvolve
sweep (A4); spread 2e9/8e8/3e8 particles/s at $p=4$ scaled by $(4/p)^3$,
calibrated against cuFINUFFT-class spreaders and this project's
Morton-sorted layout (A5); interpolate U+J at half the spread rate (A6);
0.5/1.0/2.0 ms fixed launch/orchestration overhead from the measured
~50 µs/launch H200 latency (A7); h/σ per §2 (A8); Hockney ×2 + 5-smooth
rounding (A9); F64 = 2× bandwidth terms, 0.5× rates (A10); Ewald real-space
wall $r_c = 3.5\alpha$ with measured rotor neighbor scaling (A11); post-038
bar = anchors for cube/wake, measured pinned-depth partitioned winners for
the rotor (A12).

**Critical-path pricing:** the VIC pipeline is a serial dependency chain
(spread → FFT → k-mult → iFFT → interpolate) with no overlap credit — its
stages are *summed*. The anchors are the measured **overlapped** U/J wall
times from the same H200 jobs. This is the conservative orientation of the
standing lesson (no stage-sum credit is taken on the baseline side).

## 6. Priced results — full VIC vs anchors (`data/fm037d_cost_tables.csv`)

U/J solve, modeled total (ms) and speedup vs the same-job anchor
(= post-038 bar for cube/wake, A12); F32 primary:

| case/n | anchor F32 | VIC opt | VIC nom | VIC pess | speedup (opt/nom/pess) |
|---|---:|---:|---:|---:|---|
| cube 1e5 | 11.34 | 0.75 | 1.51 | 3.73 | 15.2× / 7.5× / **3.0×** |
| cube 1e6 | 102.28 | 2.76 | 5.81 | 18.89 | 37.1× / 17.6× / **5.4×** |
| wake 1e5 | 7.85 | 0.78 | 1.55 | 4.09 | 10.1× / 5.1× / **1.9×** |
| wake 1e6 | 83.60 | 3.00 | 6.16 | 21.19 | 27.8× / 13.6× / **4.0×** |
| rotor (both n) | 12.33 / 240.2 | — | — | — | **INFEASIBLE** (mesh $5{\times}10^{11}$–$1.3{\times}10^{15}$ pts) |

F64 rows (where anchors recorded): cube 1e5 21.2×/10.4×/3.9×; wake 1e5
13.7×/6.9×/2.3×; wake 1e6 32.5×/15.8×/4.4×. Memory: ≤2.5 GB resident mesh
fields in every feasible configuration (capacity contract compatible;
allocation is construction-time, refresh is a re-zero + re-tabulation on
box drift).

**The cube/wake verdict does not flip anywhere in the stated band** — the
pessimistic corner (tightest measured mesh, 30% bandwidth efficiency, 3e8/s
spread, 2 ms launch overhead, kernel re-tabulation every step) still shows
1.9–5.4×. The dominant residual uncertainties (spread/interp throughput,
achieved cuFFT fraction, the error floor) move the *magnitude*, not the
sign.

## 7. Ewald-split secondary bound (`data/fm037d_ewald_rotor.csv`)

Mesh far field at split width $\alpha$ + compact real-space near field with
the `037b`-measured wall $r_c = 3.5\alpha$ (the deficit-truncation floor
$\rho_x \ge 3.2$–$3.668$ rounded to 3.5), real-space neighbor count anchored
to the measured rotor pair counts (density exponent $d \approx 2.0$ from the
ℓ6/ℓ8 pair-count ratio), pair rate from the measured fused-NF throughput
with a 1.3× regularized-pair penalty:

- **Cube/wake:** strictly dominated by full VIC — the mesh at $\alpha =
  \sigma$ is already the whole (cheap) cost; adding a real-space pass at
  $3.5\sigma$ re-creates precisely the pair work the shipped partitioned
  path does (the `037b` wall), so Ewald interpolates between VIC and the
  shipped baseline and never beats VIC. Not tabulated further.
- **Rotor** (only σ-heterogeneity-compatible Fourier variant, α free,
  $\alpha \ge \sigma_{\max}$), speedup vs the post-038 bar (pinned-depth
  partitioned): 1e5: 3.0× / 1.7× / 0.55×; 1e6: 1.07× / 0.74× / 0.25×
  (opt/nom/pess). **The sign flips inside the band at 1e5 and the best case
  at 1e6 is a tie** — and the bar itself is a *stand-in lower bound* on
  `038`, which attacks exactly the fat-cell nearfield concentration the
  Ewald real-space pass would also pay. A direction whose best case is a
  tie against a bar that `038` is expected to raise is not fundable on this
  evidence. What would settle it: a measured spread/interp+cuFFT
  microbenchmark at the rotor's $\alpha$-optimal mesh (~1.2e8 points) plus
  a measured real-space pair count at $r_c = 3.5\alpha$ — an afternoon H200
  job, worth running only if `038` under-delivers on the rotor.

The Ewald bound also brackets the whole design family: any split-width
choice lands between full VIC ($\alpha\to\sigma$, wins only for
near-uniform σ) and the shipped partitioned path ($\alpha\to$ large, the
`037b` wall makes the real-space pass identical to today's nearfield).

## 8. Verdict

**Fund a scoped implementation row for full VIC on near-uniform-σ
workloads; close the Fourier direction for σ-heterogeneous fields (rotor
class), where the `038` adaptive octree remains the path.** The drafted row
and the full sensitivity statement are recorded in
`037d-theory-fourier-nearfield-cost-model.md` (task file, Verdict section).

Residual risks the implementation row must retire, in order: (i) measured
spread/interpolate throughput on H200 (the largest cost share in nom/pess);
(ii) gate accuracy on the real cases with exact-PME deconvolution (the §2
floor); (iii) achieved cuFFT throughput at the actual 5-smooth sizes;
(iv) J-diagnostic quality feeding FLOWVPM vortex stretching.
