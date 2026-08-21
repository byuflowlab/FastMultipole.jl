# 041f Theory and Census: Sigma-Adaptive Smooth Nearfield

## Status and entry gate

**DONE `2026-08-18` — verdict NO-GO (sigma-adaptive smooth nearfield closed);
unified-solver verdict: one mesh family (037d global VIC); banded hybrid
rejected by strict additivity. Result section below. Clear-context approval
pending.** Reviewed `2026-08-17` (Review Amendment below is binding).

Entry gate: `037d`, `038`, `041a`, and `041d` complete and independently
approved. This is a theory/measurement row only: artifacts may be added under
`theory/`, `scripts/`, and `data/`; no production `src/` or FLOWVPM changes
and no large hardware campaign. Local work uses at most four threads.

This row blocks `042`. Its purpose is to turn 041d's promising but
placeholder-priced sigma-adaptive estimate into a concrete fund/close verdict
before the adaptive-octree milestone chooses the next nearfield architecture.

## Objective

Determine whether a locally sigma-resolved smooth solver can replace the
expensive `gaussianerf` U/J nearfield on sigma-heterogeneous, sparse rotor/wake
geometries without inheriting the domain-wide `sigma_min` mesh floor that
killed global VIC in 037d.

Evaluate two concrete solver families on the same locally refined hierarchy:

1. **patch-local AMR free-space convolution**, using bounded per-level or
   per-patch FFTs, guard regions, and explicit restriction/prolongation; and
2. **multilevel summation (MSM)**, using compact level-split real-space
   convolution stencils and avoiding FFTs.

Both must evaluate the full smooth regularized field, not truncate a Gaussian
deficit or silently leave an unpriced direct correction. The result must
preserve velocity U and all nine J entries under the standing delivered-error
contract.

## Prior evidence and non-duplication

- 037b closed compact real-space deficit splitting at the required cutoff.
- 037d found full uniform-sigma VIC strongly favorable for cube/wake but
  globally infeasible for the 18x-sigma rotor because every global mesh must
  resolve `sigma_min` over the entire mostly empty bounding box.
- 038--041 provide a 2:1-balanced occupied Morton hierarchy, local sigma
  extrema, device-resident refresh semantics, and the geometry needed for an
  offline adaptive census. Do not implement a second tree.
- 041c closed harmonic refinement inside the residual U list.
- 041d closed regularized-basis P2M/M2P substitution, but found the locally
  resolved mesh-count law

  ```text
  N_mesh ~= n / (0.55*beta)^3 ~= 0.75n       at overlap beta = 2,
  ```

  independent of sigma spread, with a preliminary 2--20 ms band versus the
  roughly 33 ms million-particle rotor evaluation. Its 2x--5x AMR coupling
  factor is a placeholder; replacing it is the central obligation here.

This row does not re-open global VIC, P2M/M2P bases, strategic targets,
fine-bin pruning, or direct-kernel organization. 041e owns the last item.

## Review Amendment (`2026-08-17`, user-directed review)

Two scope additions, binding on execution:

1. **Unified-solver verdict (new deliverable).** In the uniform-sigma limit
   the locally refined hierarchy collapses toward a single-level mesh, i.e.
   toward 037d's funded global VIC. The census must therefore also run the
   cube and wake constructors at their uniform sigma and report whether the
   sigma-adaptive machinery matches the 037d global-VIC prediction band
   there (within calibration uncertainty). The verdict section must state
   explicitly whether ONE mesh-solver family can serve all cases — so `042`
   stages one implementation, not two parallel mesh infrastructures — or
   whether global VIC and the adaptive solver remain separate rows with a
   regime boundary, and where that boundary lies (sigma spread, fill,
   patch-count threshold).
2. **Mandatory FMM-retained banded hybrid.** The registered design space
   must include at least one fully priced configuration that KEEPS the
   shipped singular far-field FMM and existing U/V routing, and meshes only
   the short/mid sigma bands (the regularization-affected region plus a
   bounded margin), with the exact-once complement explicit. This is the
   lowest-risk migration path — it preserves the already-optimized far
   field, the 025 operator tables, and the resident lifecycle — and the
   full-replacement configurations must be compared against it, not only
   against the shipped baseline. If the banded hybrid loses to full
   replacement, the verdict must say why (e.g. double coverage of the mid
   band, coarse-solve cost) with numbers.

## Required mathematical derivation

### 1. Exact multilevel Gaussian decomposition

Starting from the source-dependent regularized Green function

```text
G_sigma(r) = erf(r/(sqrt(2)*sigma)) / (4*pi*r),
```

derive a telescoping family over widths `sigma_ell`, with each particle
assigned to a level from its actual `sigma_i`. State precisely which level
carries each long- and short-band contribution and how a source with
`sigma_i != sigma_ell` is spread. The sum over levels plus any explicitly
bounded local remainder must equal the requested regularized field within the
standing budget.

The derivation must cover:

- U from spatial derivatives/curl of the smoothed potential;
- all nine J entries using either spectral differentiation or analytic
  derivatives of the interpolation basis;
- unequal source sigma and target sigma (regularization is source-directed);
- free-space rather than periodic boundary conditions;
- error allocation across kernel splitting, spread/interpolation, patch
  truncation, restriction/prolongation, FFT/stencil evaluation, and floating
  point accumulation;
- Float32 and Float64 under the existing `1e-3` velocity gate and J diagnostic
  convention.

Do not assert that Gaussian telescoping is exact without writing the actual
identity and identifying every numerical approximation subsequently applied.

### 2. Patch and level geometry

Reuse the occupied, balanced Morton hierarchy from 038. Derive deterministic
rules for:

- level selection from `sigma_i` and the validated 037d resolution
  `h_ell <= 0.55*sigma_i` (plus sensitivity at the other passing resolutions);
- grouping occupied cells into bounded rectangular patches;
- guard/halo thickness required by each level kernel;
- coarse/fine overlap ownership and exact-once field composition;
- free-space padding for patch FFTs, including interactions whose support
  crosses patch or level boundaries;
- stable-epoch versus per-step rebuild under CoreSpreading and particle
  motion.

Patch rules must depend only on refresh-time statistics and must have explicit
capacity bounds. Arbitrary case-tuned patch layouts are not admissible.

### 3. AMR-FFT and MSM formulations

For AMR-FFT, specify transform counts, real/complex layouts, padding, kernel
tables, batched small-transform shapes, halo exchange, restriction,
prolongation, and whether any global coarse solve is required. Treat poor
small-FFT efficiency and fragmented patch launches explicitly.

For MSM, derive the level kernels and their support radii at the allocated
error. Record stencil point counts, separability opportunities, boundary and
coarse/fine coupling, and whether U/J can share intermediate convolution
fields. Do not price MSM as a generic `O(N)` constant.

Both paths must state how they coexist with or replace the current singular
FMM far field and direct U list. Every ordered source-target contribution must
have exactly one owner. A hybrid may be considered only when its direct or
FMM complement is explicit and priced.

## Deterministic real-snapshot census

Add a local script that consumes the existing deterministic cube, wake, and
DJI-9443 rotor/multiscale constructors and the recorded sigma fields without
evaluating a million-particle direct field. At minimum include `n=1e5` and
`n=1e6` count reconstructions, the rotor's full 18x sigma spread, and a
CoreSpreading/time-age sensitivity sweep.

For every case and candidate level ratio/patch policy, record:

- particle count and sigma histogram per level;
- occupied cells, patches, patch dimensions, fill, and imbalance per level;
- interior, guard, padded, and transformed points per patch and in aggregate;
- restriction/prolongation interfaces and transferred point counts;
- spread/interpolate samples and support counts;
- FFT transform sizes/counts or MSM stencil points and convolution work;
- persistent and peak bytes, kernel-table bytes, graph nodes/launches, and
  refresh work;
- predicted stage times with calibration source and uncertainty;
- complete overlapped critical-path prediction against the current shipped
  baseline, not a comparison with an isolated stage sum.

Store only aggregates and small diagnostic patch examples. Do not commit
particle-scale meshes, assignments, or interaction lists.

## Calibration and cost model

Reuse 037d's measured error/resolution points and H200 bandwidth/spread/
interpolate anchors, but do not inherit its global-FFT efficiency for small
patches without a penalty. Reuse 041a's current adaptive lifecycle, memory,
refresh, and overlap anchors. If 041e finishes before final analysis, include
its selected nearfield time as an additional control without making 041f
depend on a 041e win.

Calibrate or conservatively bracket:

- batched FFT efficiency over the actual census transform shapes;
- launch and graph-node overhead for fragmented patch batches;
- halo pack/unpack or direct strided access;
- restriction/prolongation bandwidth;
- MSM stencil throughput at observed supports;
- spread/interpolate throughput at each support and precision;
- kernel-table refresh and level reassignment;
- serial tails introduced by sparse large/small patch imbalance.

No million-particle field evaluation is required in this row. Small targeted
CPU/GPU microbenchmarks may be proposed only after the census identifies the
few constants capable of flipping the verdict; any hardware run must be
pre-registered and separately authorized.

## Accuracy and exact-once oracle

Implement small deterministic numerical checks that compare the complete
multilevel construction against exact regularized U/J sums on:

- uniform sigma and 18x heterogeneous sigma;
- particles on coarse/fine and patch boundaries;
- isolated fine islands, adjacent fine patches, and sparse filaments;
- coincident source/target systems and distinct static source/target systems;
- extreme level differences allowed by the policy.

Separately paint contribution ownership by ordered body-pair ID or an
equivalent independently checkable kernel-band partition. Assert zero
omissions and duplicates. Reuse existing painters and case constructors where
applicable instead of introducing divergent tree/oracle machinery.

## Bounded optimizer and selector

Enumerate a finite registered design space containing:

- AMR-FFT versus MSM;
- geometric sigma-level ratios sufficient to cover `{sqrt(2), 2, 4}` unless
  an analytic argument removes one;
- all 037d accuracy-passing spread/interpolation configurations;
- bounded patch edge lengths selected from actual census shapes;
- Float32 and Float64;
- per-step rebuild and stable-epoch refresh;
- any explicitly derived hybrid cutoff.

For every census configuration report the exhaustive or bounded-reference
optimum and a deterministic selector using only refresh-time statistics. The
selector must default to the shipped FMM/direct path unless its predicted win
exceeds calibration uncertainty and capacity is available.

## Phase gates and verdict

Run three gates:

1. **Count gate:** actual patch+halo+padded point inflation must preserve a
   plausible >=10% nearfield and >=5% complete-solve win under a zero-coupling
   lower bound. Otherwise stop with NO-GO.
2. **Accuracy/ownership gate:** the complete U/J construction and exact-once
   oracle must pass at practical level/patch parameters. Otherwise stop.
3. **Full modeled gate:** after all coupling, refresh, launch, memory, and
   critical-path costs, require:
   - >=10% nearfield reduction and >=5% complete overlapped solve reduction on
     a material sigma-heterogeneous case;
   - no >3% regression on supported cases under automatic fallback;
   - bounded capacity, device residency, graph compatibility, and zero
     recurring allocation.

Return one verdict:

- **GO:** stage a narrowly scoped implementation successor after 042;
- **REGIME-ONLY:** stage only a measurable sigma-spread/fill/patch regime with
  automatic fallback;
- **NO-GO:** close sigma-adaptive smooth nearfield and attribute the failure
  to patch inflation, halos/padding, level coupling, accuracy, fragmentation,
  refresh, capacity, or loss of overlap.

Do not implement production kernels in this row.

## Required artifacts

1. `theory/sigma-adaptive-multilevel-nearfield.md`: complete decomposition,
   error/ownership derivation, cost model, selector, and verdict.
2. `scripts/sigma_adaptive_nearfield_census.jl`: deterministic real-snapshot
   level/patch census, small accuracy/ownership oracle, and bounded optimizer.
3. `data/sigma_adaptive_nearfield/`: manifest, counts, patch/level tables,
   calibration, accuracy/oracle results, selector/oracle gaps, report, and
   checksums.
4. Result section here and Done checkbox update. Independent clear-context
   review applies the Approved checkbox.

## Result (`2026-08-18`)

Artifacts: `theory/sigma-adaptive-multilevel-nearfield.md` (complete
derivation, cost model, gates, verdicts — note the artifact name uses
`multilevel` rather than the `smooth` of the original spec, chosen to avoid
colliding with 041d's `theory/sigma-adaptive-smooth-nearfield.md`),
`scripts/sigma_adaptive_nearfield_census.jl`, and
`data/sigma_adaptive_nearfield/` (level/patch tables, cost model, optimizer,
selector, oracle, hybrid, unified check, refresh sweep, manifest, checksums).
Census run locally on 4 threads under the FLOWVPM project (needed only for
the fm033 DJI-9443 rotor generator); no production `src/` or FLOWVPM changes;
no hardware runs were requested or needed.

**Derivation.** The exact multilevel Gaussian decomposition is written in
full: the width-transfer identity `G_sigma = G_s * rho_tau`
(`tau^2 = sigma^2 - s^2`), the telescoping band family with erfc-bounded
compact band kernels, U/J via curl and analytic interpolant derivatives,
free-space handling, complete error taxonomy (E1–E7) under the 1e-3 gate,
and exact-once band ownership. Three structural results fell out that 041d's
placeholder could not see: (i) the **sigma-compensation cost dilemma** — the
compensation Gaussian must be realized per-particle-exactly (binned k-space
compensation needs O(10^2–10^3) bins/level to meet budget); the registered
delta-split architecture moves the cost smoothly between real-space
spreading and per-bin transform inflation, with the optimizer choosing
`b=8`; (ii) **ladder extension** — the top term is long-range, and anchoring
it at `sigma_max` costs O(10^9) global mesh points on the rotor, so bands
must extend above `sigma_max` until the coarse global solve is bounded;
(iii) the **banded-hybrid strict-additivity theorem** — keeping the singular
FMM far field forces the direct complement to the inflated cutoff
`rho_t*sigma_i^(c) >= rho_t*sigma_i` at unchanged per-pair cost, so every
FMM-retained hybrid costs strictly more than the shipped baseline
(measured rotor pair inflation 1.25x–1494x by cut level, plus 234–2943 ms
of added band mesh).

**Oracle.** 28/28 rows pass: ~6th-order U convergence (47–193x per
h-halving), J convergent once the finite coincident self-term
`J_self = -G''(0) [Gamma]_x` is included in the reference,
multilevel-vs-single-mesh burden ratio 1.00 on uniform cases, worst omitted
tail 1e-17 relative, unique block ownership; boundary, 17.9x extreme-ladder,
distinct-target, and filament adversarial cases included.

**Census and gates.** On the real rotor (17.9x sigma spread) the count gate
fails by two orders of magnitude: zero-coupling lower bound at optimistic
anchors 2557 ms (n=1e6) and 133 ms (n=1e5) versus bars of 21.0 / 4.4 ms;
nominal totals 3826 / 213 ms. Attribution: cumulative band participation
(most of n on 5–8 levels), the codimension deficit (sheet/filament fine-band
occupancy vs a ~22-cell band-kernel halo, measured padded/interior ~11–12x),
and the 37 ms sigma-compensation floor. An idealized cell-granular floor
(~90 ms optimistic) shows the kill survives ~30x of census conservatism, so
no microbenchmark could flip it and none is requested. The
sigma_multiscale proxy dies by the 037d sigma_min-floor mechanism (5.3 s).
MSM is uniformly worse (rotor ~22.5 s). 041d part 2 is thereby closed: its
`N_mesh ~ 0.75n` law measured ~3e4-fold optimistic on the rotor.

**Verdicts.** (1) **NO-GO** — sigma-adaptive smooth nearfield closed;
selector: `direct_fallback` on every sigma-heterogeneous row; no successor
proposed. (2) **Unified solver** — in the uniform-sigma limit the machinery
collapses exactly to 037d global VIC and independently reprices it inside
the 037d bands (cube 1e6 ratio 1.15, wake 1e6 0.95): ONE mesh family
suffices; `042` should stage at most the already-funded 037d global VIC and
no sigma-adaptive mesh infrastructure; regime boundary = uniform sigma AND
volumetric fine-sigma occupancy. (3) **Banded hybrid** rejected by the
strict-additivity theorem plus measured inflation — the anticipated
"double coverage of the mid band," now quantified.

## Clear-context approval (`2026-08-18`)

Approved. Checked: derivation soundness by hand (Poisson + Gaussian-semigroup
width transfer, exact telescoping, erfc band bound,
`G''(0) = -1/(3(2pi)^{3/2} sigma^3)` self-term, strict additivity via
`sigma_i^(c) >= sigma_i`); script-vs-theory fidelity (registered design space
executed as declared, delta-split spreading sums, halo-gather padding, ladder
extension, graph-replay launch model, 037d/041a/037b anchors, `<=4`-thread
guard); all 10 checksums verify; no `src/` or FLOWVPM changes. Spot-checked
arithmetic reproduces every headline number (rotor level-3 7.39e9 padded
points, nominal stage split and 3826.29 ms total, count-gate bars
21.0/4.4/11.5 ms = 0.9x near baselines, unified ratios 1.146/0.949, hybrid
inflations 1.25x-1494x and 234-2943 ms band mesh, b-sweep optimum b=8 with
spread 303/37/11.5 ms at b=1/8/32). Gates applied as registered; count-gate
stop rule honored with the full modeled gate reported for completeness; the
~90 ms idealized floor makes the NO-GO robust to ~30x census conservatism.
All three verdicts follow from the evidence. One wording note beyond the
documented warts (non-blocking): "28/28 oracle rows pass" means all six
per-case verdicts (convergence, abs_min, burden) pass — the per-row
`pass_abs` column is false on the deliberately naive instrument's
coarsest-h rows, as the theory doc itself explains.
