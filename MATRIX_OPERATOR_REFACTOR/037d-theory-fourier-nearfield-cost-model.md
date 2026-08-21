# 037d Fourier Nearfield Cost Model (Paper Study)

## Status and Entry Gate

**DONE 2026-08-14.** Verdict: fund a scoped implementation row for full
particle-mesh (VIC) evaluation on near-uniform-σ workloads (cube/wake
class; no flip anywhere in the sensitivity band); the Fourier direction is
closed for σ-heterogeneous fields (rotor class), where full VIC is
structurally infeasible and the Ewald variant flips sign inside the band
against a bar `038` is expected to raise. Full record below; awaiting
clear-context approval.

Entry gate: `037b` Done (its anchors, cases, and error-decomposition evidence
are the pricing basis). Does not gate `038`; may run in parallel with it.
Gating note: `037b` is Done but its clear-context approval is still pending;
the user explicitly authorized starting `037d` on 2026-08-14 notwithstanding
(user instructions are the first source of truth per `START_HERE.md`).

## Motivation

`037b` falsified two-pass deficit splitting: accuracy pins any deficit/
correction shell at `rho >= 3.2–3.7 sigma`, at which point the correction
duplicates the partitioned nearfield's pair work, and the smaller-primary/
deeper-tree trade loses on far-field growth. `037c` (meshing the deficit) was
recommended NO-GO on the same compact-support grounds.

The remaining Fourier-space idea is different in kind: the
`gaussianerf`-regularized U/J field is globally smooth with spectral content
cut off at `k ~ 1/sigma`, so a particle-mesh (VIC-style) evaluation —
spread vorticity to a mesh, convolve with the singular kernel spectrum in
k-space, interpolate U and J back — can in principle evaluate ALL pair
interactions, near field included, with **no direct pass and no kernel
split**. Whether that wins at our accuracy gate, domain shapes, and σ
distributions is a cost-model question answerable on paper. This row answers
it before any implementation is considered.

## Objective

Produce a defensible cost/error model and a clear verdict: is a device-
resident particle-mesh nearfield replacement plausibly faster than the
shipped partitioned path (and than partitioned + the expected `038` adaptive
octree) on any workload we care about? If yes, specify the implementation row
to stage; if no, close the Fourier direction by recorded verdict.

## Required Model Content

1. **Error model** (sets the mesh): interpolation/spreading order `p`
   (B-spline/window options) and mesh spacing `h` vs `sigma` required for
   sampled velocity RMS `<= 1e-3` (Jacobian logged as diagnostic; note its
   extra derivative cost in error terms). Include the aliasing/deconvolution
   term and the near-pair regime (`r ~ sigma`) explicitly — this is where
   mesh error peaks. Validate the error model with a cheap 1D/3D numerical
   spot check in a stdlib-only Julia script (no hardware runs; local, <= 4
   threads) rather than by assertion.
2. **Mesh sizing**: from the `037b` case geometries (cube, AR-5 wake, rotor
   wake) — rectangular tight bounds (the `037` machinery), fill fraction,
   free-space boundary handling (Hockney doubling or alternatives; justify
   the chosen factor), and resulting mesh point counts at `n = 1e5` and
   `1e6` per case, at the `h(sigma_min, p)` the error model demands.
3. **Cost model**: transform count and sizing for U+J (forward transforms of
   the vorticity components; inverse transforms for 3 U + 9 J components, or
   fewer via k-space differentiation trades — state the chosen scheme),
   cuFFT throughput priced from published/typical H200 rates with the
   assumption recorded; spread and interpolate cost per particle at order
   `p` (atomics/gather assumptions stated); per-step residency/allocation
   implications under the capacity contract (qualitative — no code).
4. **σ-heterogeneity**: the rotor case's 18x σ spread. Model at least one
   viable handling (σ-binned multi-mesh with per-bin `h`, or
   spread-with-`sqrt(sigma_i^2 - alpha^2)` NUFFT-style factorization) and
   its cost multiplier; state clearly if the mechanism only works for
   near-uniform σ.
5. **Ewald-split secondary bound**: mesh far field + compact real-space near
   field with free split width; show where its real-space floor
   (`~3.5 sigma`, the `037b` wall) leaves it relative to (a) full VIC and
   (b) the shipped partitioned path. This bounds the whole design family.
6. **Pricing against anchors**: compare modeled totals against the `037b`
   same-job anchors (partitioned baselines: cube/wake/rotor at `n = 1e5`,
   `1e6`, F32 and F64 where recorded) using critical-path pricing —
   overlapped critical path, not isolated stage sums. Include a sensitivity
   band (optimistic/nominal/pessimistic on FFT throughput, `h`, and padding)
   so the verdict is robust to the paper-estimate uncertainty.

## Deliverables

- `theory/fourier-nearfield-cost-model.md` — the derivation, model, tables,
  and verdict (math per the standing `$`/`$$` conventions).
- `scripts/fm037d_error_spotcheck.jl` — stdlib-only validation of the error
  model (records its numbers to `data/`).
- `scripts/fm037d_cost_tables.jl` — generates the cost tables (CSV to
  `data/`) from recorded assumptions + the `037b` anchor CSVs.
- A recorded verdict in this file: stage an implementation row (with a
  drafted row summary) or close the Fourier direction.

## Constraints

- Paper study only: `theory/`, `scripts/`, `data/` artifacts. No production
  `src/` changes, no cluster/H200 jobs. Local scripts stay `<= 4` threads.
- Read the `037b` anchor data from
  `MATRIX_OPERATOR_REFACTOR/data/flowvpm_gpu_campaign/` and the `037b` task
  file's recorded numbers; do not re-run benchmarks.
- Every priced assumption (FFT rate, memory bandwidth, spread cost/point)
  must be written down next to its use.
- Clear-context approval required, as for every row.

## Work Record

### 2026-08-14 — model, spot check, tables, verdict (single session)

**Deliverables produced (all under `theory/`, `scripts/`, `data/`; no
`src/` changes, no cluster/GPU jobs):**

- `theory/fourier-nearfield-cost-model.md` — derivation, error model, mesh
  sizing, cost model (assumptions A1–A12), σ-heterogeneity analysis,
  Ewald-split bound, priced tables, verdict.
- `scripts/fm037d_error_spotcheck.jl` → `data/fm037d_error_spotcheck.csv` —
  stdlib-only 3D validation of the error model: the full pipeline in
  miniature (32³ interior/64³ Hockney-doubled hand-rolled FFT, order-p
  B-spline spread/interp with sinc^{2p} deconvolution, tabulated free-space
  gaussianerf kernel), 1650 particles incl. 150 injected near pairs at
  r ∈ [0.5,1.5]σ, 250 sampled targets vs the exact regularized sum
  (stable small-ρ series; analytic J reference — an FD reference is
  unusable because the erf approximation's 1.5e-7 absolute error is
  amplified catastrophically at r ≪ σ, the one bug found and fixed during
  the session). Runtime ~2 min single-threaded; the 0.40/0.475 rows were
  appended from an identical-code probe run (local compute kept light per
  the 2026-08-14 user directive).
- `scripts/fm037d_cost_tables.jl` → `data/fm037d_cost_tables.csv`,
  `data/fm037d_ewald_rotor.csv` — cost tables from the spot-check picks +
  the 037b anchor CSVs, opt/nom/pess bands.

**Measured error model:** gate met with ~1.7x margin at p=4, h/σ=0.55
(u_rel_rms 5.90e-4) and p=6, h/σ=0.7 (5.75e-4), near pairs and
aliasing/deconvolution included; J diagnostic ~3–4e-3 (same order as the
shipped anchors' logged J). Known conservatism: continuous-sinc
deconvolution gives an error floor ~5e-4 below h/σ≈0.55; exact-PME
factors/Kaiser–Bessel windows sit below it in the literature — verifying
that is delegated to the implementation row. Pessimistic band therefore
uses the tightest measured point (h/σ=0.40, 4.57e-4).

**Priced results (F32, U/J solve, vs same-job 037b anchors; VIC priced as a
serial chain — no overlap credit — against measured overlapped anchor
walls):** cube 1e5 15.2/7.5/3.0x; cube 1e6 37.1/17.6/5.4x; wake 1e5
10.1/5.1/1.9x; wake 1e6 27.8/13.6/4.0x (opt/nom/pess; F64 similar where
anchors exist). Mesh memory ≤ 2.5 GB in all feasible configs. **No flip
anywhere in the band on cube/wake.** Rotor full VIC: INFEASIBLE — mesh
must resolve σ_min over the whole mostly-empty box: 5e11–1.3e15 points
(tens of TB–PB). σ-binned multi-mesh and NUFFT-style factorization both
retain the σ_min domain-wide resolution floor and do not rescue it; full
VIC carries a (σ̄/σ_min)³ mesh multiplier and is a near-uniform-σ method
(≤ ~1.5x spread tolerable, 18x fatal).

**Ewald-split secondary bound** (α free, r_c = 3.5α from the 037b wall,
neighbor scaling anchored to measured rotor pair counts, d ≈ 2.0):
cube/wake — strictly dominated by full VIC; rotor vs the pinned-depth
post-038 bar — 1e5: 3.0/1.7/0.55x, 1e6: 1.07/0.74/0.25x: **flips sign
inside the band** at 1e5, best-case tie at 1e6, against a stand-in bar
`038` should raise. Not fundable on this evidence; the settling
measurement (H200 spread/cuFFT microbenchmark at the α-optimal ~1.2e8-pt
mesh + measured pair count at 3.5α) is recorded in the theory doc and is
worth running only if `038` under-delivers on the rotor.

## Verdict

**FUND (scoped) + CLOSE (scoped).** Stage an implementation row for full
particle-mesh evaluation restricted to near-uniform-σ workloads; close the
Fourier direction for σ-heterogeneous fields (rotor class) — there the
`038` adaptive octree remains the funded path, and the Ewald fallback is
bounded by measurement-anchored modeling to at best a tie. The cube/wake
verdict is robust: it does not flip in the stated sensitivity band
(pessimistic corner still 1.9–5.4x). The residual uncertainties are
magnitude-only and are exactly what the implementation row's benchmark
ladder retires first (measured spread/interp throughput, exact-PME
deconvolution accuracy on the real cases, achieved cuFFT rates, J quality).

Drafted implementation-row summary (for the user to stage, suggested id
`037e`, gated on user approval; does not gate `038`):

> `037e-impl-particle-mesh-uniform-sigma.md` — Device-resident full
> particle-mesh (VIC-style) U/J evaluation for near-uniform-σ fields
> (`gaussianerf` only): order-4/6 B-spline spread of Γ onto the `037`
> rectangular tight-bound mesh (h from the `037d` error model, exact-PME
> deconvolution factors), Hockney free-space convolution with the
> precomputed mollified-kernel spectrum via cuFFT (6 transforms: 3 forward,
> 3 inverse U; J via derivative interpolant), device-resident under the
> capacity/zero-allocation/counter contracts with construction-time
> spectrum tabulation. Entry benchmark gates its own staging: an H200
> microbenchmark ladder (spread/interp throughput, cuFFT at the case mesh
> sizes, end-to-end accuracy vs checksummed references at P-independent
> gate 1e-3) must land within the `037d` nominal-to-pessimistic band
> before production integration proceeds. A/B against the shipped
> partitioned anchors on cube and wake at n=1e5/1e6, both precisions;
> eligibility guard σ_max/σ_min ≤ 1.5 with automatic fallback to the
> partitioned path; off-by-default, default change only on explicit user
> approval. Modeled expectation from `037d`: 4–17x (nominal) on cube/wake,
> pessimistic floor 1.9–5.4x.
