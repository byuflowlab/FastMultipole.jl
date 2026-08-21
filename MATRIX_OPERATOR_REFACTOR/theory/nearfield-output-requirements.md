# Nearfield Output Requirements and Feasibility Verdict (041b)

## Scope and sources

This audit covers every `U_INDEX`, `J_INDEX`, `VORTICITY_INDEX`, and
`SFS_INDEX` use under the available `../FLOWVPM.jl/{src,ext,test}` tree and
traces Euler and all three RK3 stages. No FLOWUnsteady or VortexLattice
checkout is present in this workspace, so those external surfaces remain
unknown consumers and are handled by retaining the full-J public mode.

The current row layout is U `10:12`, vorticity `13:15`, J `16:24`, and SFS
`40:42` (`FLOWVPM_particlefield.jl:287-293`). That layout is storage, not a
statement of the minimum mathematics.

## Consumer audit

| Consumer / configuration | Read or write | Mathematical requirement | Persistence |
| --- | --- | --- | --- |
| U/J direct, legacy FMM, radix FMM; CPU/CUDA | writes U and nine J rows; reset preserves static rows | U plus selected output policy | through integration/output |
| ClassicVPM Euler and each of three RK3 stages; either `transposed` value | reads U, J, Gamma, SFS | U and `T(Gamma)` | only through that stage |
| ReformulatedVPM Euler and each RK3 stage; either `transposed` value | reads U, J, Gamma, SFS | U, `T(Gamma)`, and its dot with Gamma | only through that stage |
| NoSFS | no SFS read beyond a zero term | no SFS quantity | none |
| ConstantSFS | U/J call produces Estr; integrator reads SFS | `E` and `T(Gamma)` | through the stage |
| Dynamic two-/pseudo-three-level | repeats U/J/Estr at test and domain widths, stores their differences in M, then forms numerator/denominator/C | `T(Gamma)` and E at both widths | M/C persist across before/after-UJ procedure |
| Dynamic sensor | repeats U/J at test/domain widths and evaluates `|curl U|^2`; domain call also requests Estr | curl at both widths plus E at domain width | C rows retain the two sensor values |
| Pedrizzetti and corrected Pedrizzetti relaxation | J rows 2,3,4,6,7,8 | curl `(J6-J8,J7-J3,J2-J4)` | only at relaxation call |
| Inviscid | no additional U/J use | none | none |
| CoreSpreading | `zeta` overwrites J[1:3], copies it to M[7:9], RBF reuses J[1:3] as scratch | no velocity-gradient quantity; requires a distinct three-row scratch alias | until RBF completes |
| ParticleStrengthExchange | reads PSE rows, not J | no additional U/J quantity | none |
| Vorticity rows | reset/access/output only in audited tree | stored vorticity is separate from `get_W`, which computes curl from J | public/output compatibility |
| HDF5 output and monitors | writes full J; monitor computes enstrophy from curl(J) | full J for legacy files, curl for monitor | output boundary |
| FastMultipole adapter / CUDA extension / tests | pack, accumulate, and compare full J | full J compatibility oracle | API boundary |
| external/unknown consumers | unavailable in workspace | conservatively full J | legacy mode remains |

Static particles are skipped as targets by the scalar paths and retained by
reset masking; sources still participate where the current routines include
them. Both reduced identities preserve that distinction. Relaxation after RK3
can issue an additional final U/J evaluation, so a specialization cannot assume
exactly three calls per time step.

## Exact reduced identities

With `T(v)=Jv` when `transposed=true` and `T(v)=J'v` otherwise, both VPM
formulations consume `T(Gamma)` rather than J. Relaxation and the dynamic
sensor additionally consume curl. The direct SFS implementation is

`E_p = sum_q zeta_pq [T_p(Gamma_q) - T_q(Gamma_q)]`.

Linearity gives

`Omega_p = sum_q zeta_pq Gamma_q`,

`Q_p = sum_q zeta_pq T_q(Gamma_q)`, and

`E_p = T_p(Omega_p) - Q_p`.

Omega and Q are six outputs of one shared zeta pair pass. The corrected
deterministic verifier covers both transpose conventions in Float32, Float64,
and BigFloat on an elongated/clustered snapshot with unequal sigma, a static
target, and coincident self pairs. SFS reassociation relative errors are
`1.75-2.57e-7`, `4.49-7.64e-16`, and `2.92-5.14e-77`; the independently
formed per-pair contraction sums have maximum absolute residuals `1.97e-6`,
`2.98e-15`, and `3.99e-76`.
It evaluates analytic singular and `gaussianerf` pair Jacobians plus a
distant-source singular sum: maximum trace residuals are `1.14e-5`,
`7.11e-15`, and `1.24e-75` by precision. It also forms two filter widths and
compares the pseudo-three-level dynamic numerator, denominator, and coefficient;
maximum relative differences are `2.65e-6`, `1.84e-13`, and `1.85e-74`.

The analytic velocity gradient is trace-free but not symmetric. Eight stored
components can reconstruct the ninth, but this saves only one output row and
does not remove the arithmetic accumulated in the current register-resident
pair kernel.

## A0 performance pre-kill

The stronger pre-existing experiment is the 028 same-kernel attribution at
1M bodies, Float32, 128 threads, 16,384 blocks:

| terminal mode | median ms |
| --- | ---: |
| 12 atomic outputs | 11.4521 |
| plain store | 11.4704 |
| no store | 11.4375 |

Removing every terminal write improved the median by only 0.0146 ms (0.13%);
plain stores were slower. This brackets *all* trace-free/stretch output-width
policies, since they retain the same inner loop and cannot outperform a
no-store kernel. Even granting the whole isolated 0.13% to the complete solve
is far below the registered 5% gate. Thus A3/A4 kernel work is correctly
skipped and the performance verdict is **Output NO-GO**. The legacy full-J
representation remains the minimum economical common representation.

This does not close the independent physics-enablement result: a successor
may add radix/GPU SFS with the fused Omega/Q zeta pass, `T(Gamma)`,
`T(Omega)`, optional curl, and a full-J fallback. Its value is restored model
coverage, not a pre-claimed speedup.

## Strategic-target screen

Stage 0 used the approved equal-cell relationship
`rho_t sigma/h ~= sqrt(q)` and the measured 037d `h_sample/sigma=0.55` result.
It used `ceil(rho_t/(0.55sqrt(q)))^3` velocity samples and an explicit 2x
per-coordinate derivative penalty for U/J. The optimistic adaptive bracket
uses q=12 because the 041 CSV has no per-leaf sigma/h histogram.

Velocity alone survives in 34/36 geometry rows. U/J survives only for the two
precision copies of the same rotor-1M uniform geometry (216 samples versus
mean occupancy 499). The adaptive winners have mean occupancies 13-67 and
fail the U/J half-occupancy rule even under the optimistic 64-sample estimate.

The corrected offline probe follows the Stage-0 gate: it tests only the actual
surviving rotor-1M uniform geometry. A deterministic extractor runs the
approved DJI-9443 constructor at `ell=6` and retains compact, checksummed
self/face/edge/corner/smoother-shell blocks. The selected fat cells contain
2459--3035 particles. Actual snapshot positions and sigma values are combined
with low-discrepancy training and held-out sources, all three independent
strength directions, 216 candidate strategic targets, and 64 actual validation
target positions held out from the practical interpolant fits. Three separately
conditioned sigma bins span the rotor's recorded 18x property range. U and U/J
are weighted and gated separately.

The scalar-row QDEIM ceiling passes 12/30 cases: corner and smoother-shell U
and U/J at all three sigma bins. Self, face, and edge fail. The result is not
promoted on that optimistic ceiling. Total-degree and tensor Chebyshev
interpolation fail accuracy through 216 points. Gaussian RBF interpolation
passes the smoother shell at 120 points for U (`2.46e-4`) and 165 points for
U/J (`9.51e-5` U, `3.63e-4` J), but those counts are 2.67x and 2.39x the
corresponding QDEIM ranks. Both exceed the pre-registered 1.25x practical-rank
gate, so 0/30 cases pass the joined gate and no H200 prototype is permitted.

## Final decisions and positioning

- **Output NO-GO (performance):** retain full J as the universal economical
  representation. No output specialization cleared the gate.
- **SFS enablement remains justified:** stage separately after the adaptive
  review if restoring SFS on radix/GPU is desired; do not market it as an
  output-width optimization.
- **Strategic-target NO-GO:** some corner/shell QDEIM ceilings pass, and a
  shell RBF passes accuracy, but no case passes the registered practical-rank
  gate; no production task or default change is justified.
- **Positioning:** uniform-sigma cube/wake work is already the funded 037d VIC
  niche; adaptive cells make occupancy smaller and strategic reconstruction
  less favorable; the only Stage-0 U/J survivor is the 18x-sigma-spread rotor
  regime, where only the already-smooth distant shell admits a practical
  interpolant and it misses the rank gate by 2.39x. That shell is the existing
  expansion/demotion rung rather than a new strategic-target niche. Strategic
  targets therefore beat neither VIC nor the adaptive U-list in a supported
  class.

Raw arithmetic, singular-value curves, QDEIM results, seeds, set sizes, and
SHA-256 checksums are under `data/strategic_target_feasibility/`. No dense
matrices are committed.
