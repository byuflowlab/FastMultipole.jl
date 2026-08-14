# DJI 9443 rotor-wake case — data provenance and derived numbers (task 037b)

## Circulation table

`dji9443_fixed_bin_circulation.csv` is a verbatim copy (2026-08-13, sha256
`a1383796d25c6f6b34a873e05de144a87cdf359b636b2bdc78c9f89593f0be65`) of

    /Users/ryan/Dropbox/research/projects/FLOWPanel.jl/data/dji_convergence_20260722/phase_02c_dji_mesh_convergence/fixed_bin_circulation.csv

(FLOWPanel DJI 9443 mesh-convergence study, phase 02c). It is vendored here so
cluster runs never need FLOWPanel. Columns: `case, n_airfoil, topology,
formulation, rpm, abs_r_over_R, gamma_blade_1, gamma_blade_2, gamma_mean,
gamma_direct_mean, gamma_slice_mean`. Cases present: dji81c/u, dji97c/u,
dji121c/u. The benchmark uses **dji121c** (finest capped watertight mesh,
n_airfoil=121, Dirichlet formulation, RPM 5400): 35 stations at
r/R = 0.125:0.025:0.975, `gamma_mean` in m^2/s, peak 0.24015 at r/R = 0.65,
tip station 0.11122, root station 0.05085.

## Case definition

Implemented in `MATRIX_OPERATOR_REFACTOR/scripts/benchmark_033_common.jl`
(case name "rotor"); all decisions fixed by the 037b task spec.

Parameters: R = 0.119 m, B = 2 blades, RPM = 5400 (Omega = 2*pi*5400/60 =
565.4867 rad/s, T_rev = 1/90 s), rho_air = 1.225 kg/m^3 (used only for the
thrust/convection derivation below). Deterministic RNG:
`MersenneTwister(33025 + 104729 + n_target)`, used only for position jitter,
drawn in fixed emission order; a build produces exactly n_target particles and
is bitwise reproducible (verified at n = 10^4).

### Trailed filaments (per blade)

From Gamma(r/R) = gamma_mean(dji121c):

* root filament: +Gamma_1 = +0.0508512 m^2/s at r_1 = 0.014875 m (r/R = 0.125)
* inboard sheet: the 34 bin-boundary filaments of strength
  Gamma_{i+1} - Gamma_i, thinned by lumping groups of k = 34 into m = 1
  filament per blade of strength Gamma_35 - Gamma_1 = +0.0603700 m^2/s at the
  |dGamma|-weighted mean boundary radius 0.0663588 m (r/R = 0.55764)
* tip filament: -Gamma_35 = -0.1112211 m^2/s at r_35 = 0.116025 m (r/R = 0.975)

Total trailed circulation sums to zero (asserted in code); the tip filament is
the strongest single structure (|G| = 0.111 vs 0.060 inboard, 0.051 root).

Thinning k: every filament shares one azimuthal step (below), so each of the
2+m filaments per blade carries the same particle count and the inboard share
is m/(m+2). m minimizes |m/(m+2) - 0.35| over 1..34, giving **m = 1, k = 34,
inboard fraction 1/3 ~ 33.3%**. Note this collapses the sheet to a single
lumped filament — the arithmetically forced consequence of the equal-step
rule plus the 35% target (m = 2 would give 50%). Placement deviation: the
literal "every 34th boundary" position (r/R = 0.9625) would sit adjacent to
the tip vortex and cancel roughly half of it, so the lumped filament sits at
the |dGamma|-weighted mean boundary radius instead (r/R = 0.55764), keeping
the tip vortex coherent.

### Kinematics

* Kutta-Joukowski thrust of the measured distribution (trapezoid over the
  dimensional bins): T = B*rho*Omega*trapz(Gamma*r dr) = **1.74987 N**.
* Momentum-theory induced velocity v_i = sqrt(T/(2*rho*pi*R^2)) =
  **4.00680 m/s**; far-wake doubling blended as
  v(age) = v_i*(2 - exp(-age/1 rev)).
* Axial displacement Z(a revs) = v_i*T_rev*(2a - 1 + e^{-a}); wake convects
  in -z, rotor plane z = 0, blade b at azimuth 2*pi*b/B at age 0.
* Radial contraction r(age) = r0*(0.78 + 0.22*exp(-age/0.25 rev)), applied to
  all filaments proportionally.
* Wake age: base 12 rev gives Z = 1.0240 m, AR = Z/(2R) = 4.30 < 5, so the
  age is extended in 0.5-rev steps: **final age 14.0 rev**, Z = **1.20204 m**,
  AR = **5.0506** (measured particle bbox at 10^5: extents 0.232 x 0.201 x
  1.203 m, AR 5.19).

### Discretization

One azimuthal step dpsi for all 6 filaments; per-filament count
n_per = ceil(n_target/6); dpsi = 2*pi*14.0/n_per. Emission stops at exactly
n_target (order: blade 0 then 1; within a blade root, inboard, tip; within a
filament young to old), trimming at most 5 particles off the oldest end of
blade 1's tip filament.

| n_target | n_per filament | dpsi [rad] |
|---|---|---|
| 10^5 | 16667 | 5.27777e-3 |
| 10^6 | 166667 | 5.27787e-4 |

Each particle: segment midpoint age a = (j-1/2)*da; Gamma_p = filament
strength * ds * unit tangent of the contracting helix (analytic dX/da,
ds = |dX/da|*da midpoint rule); position = midpoint + 0.15*sigma_p*randn(3).

### Sigma (per particle)

sigma0 = beta*ds with beta = 2 (ds = local inter-particle spacing along the
filament); core-spreading growth sigma(age) = sigma0*sqrt(1 + (2/3)*age_rev),
i.e. c_growth = 2/3 per rev so sigma(12 rev) = 3*sigma0 (at the final 14-rev
age the oldest particles reach ~3.2*sigma0). No global floor or cap: measured
range at n = 10^5 is 1.73e-4 .. 3.11e-3 m (18x spread, p05/p95 =
2.89e-4/2.89e-3), scaling as 1/n (at 10^6: 1.73e-5 .. 3.11e-4). This is the
intended heterogeneity: tight young tip vortices near the rotor plane,
diffuse old inboard wake. `fm033_sigma("rotor", n)` returns **sigma_max**
(the quantity the FastMultipole geometry gate consumes), computed
deterministically without RNG.

## Files

* `dji9443_fixed_bin_circulation.csv` — vendored circulation table (above).
* `rotor_case_stats.csv` — output of
  `MATRIX_OPERATOR_REFACTOR/scripts/rotor_case_stats.jl`: sigma/bbox/|Gamma|
  and uniform-grid occupancy (ell = 4..7) for rotor and the existing wake
  case at n = 10^5, 10^6.

Direct references (512 sampled targets, Float64, single-thread direct) live
with the other 033 references at
`MATRIX_OPERATOR_REFACTOR/data/flowvpm_baseline/references/`
(`direct_reference_rotor_n100000.csv`, `direct_reference_rotor_n1000000.csv`,
manifest rows appended additively to `direct_reference_checksums.sha256`).

Rebuild commands (from the FastMultipole repo root; the project env must
provide FLOWVPM with FastMultipole 2.0.4, e.g. the sibling
`--project=../FLOWVPM.jl`):

    julia --project=../FLOWVPM.jl MATRIX_OPERATOR_REFACTOR/scripts/rotor_case_stats.jl
    FM033_CASES=rotor FM033_NS=100000,1000000 \
      julia -t 1 --project=../FLOWVPM.jl MATRIX_OPERATOR_REFACTOR/scripts/prepare_033_references.jl
