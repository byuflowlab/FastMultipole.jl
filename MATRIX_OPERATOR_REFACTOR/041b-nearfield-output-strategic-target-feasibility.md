# 041b Nearfield Output Requirements and Strategic-Target Feasibility

## Status and roadmap relationship

Numbered measurement/theory task authorized and executed by user direction on
`2026-08-15`. No production source changes are permitted in this row. The row
is marked Done in `START_HERE.md`; clear-context approval remains intentionally
unset for a different agent.

Execution sequencing: run A0 (§2) and Stage 0 (§3.2) first —
both are pure arithmetic over existing CSVs and prior measurements, need no
new scripts or cluster time, and are expected to close the Workstream A
performance track and most Workstream B cell classes. Only the workstreams
surviving those two screens justify further rank work; the SFS-enablement
track of Workstream A (§2) survives A0 by construction and can be proposed
independently. Any production work remains a new successor row after `042`.

The plan has two joined goals:

1. determine the minimum mathematical output of the FLOWVPM velocity-gradient
   solve, including every SFS, formulation, relaxation, and viscous consumer,
   and determine whether producing less than the full velocity-gradient tensor
   can reduce end-to-end cost without changing the equations; and
2. decide, as cheaply and confidently as possible, whether evaluating the
   induced field at strategic targets and interpolating it to particles can
   replace a material fraction of direct nearfield evaluations. The screen
   includes bases conditioned on source-target cell displacement and source
   properties.

No candidate is promoted on a one-shot velocity result alone. SFS intermediate
quantities and short trajectory behavior are mandatory because the current
radix/GPU coupling rejects `sfs=true`, and restoring that physics is part of the
objective rather than an optional compatibility check.

## 1. Established facts and hypotheses

### 1.1 Current output and cost

The resident vortex solve produces velocity `U` and all nine entries of the
velocity gradient `J`. The production kernel
(`_cuda_direct_pairs_vortex_kernel!`, `src/translate_batched_cuda.jl:1729`)
is warp-per-cell-pair: it accumulates 3 velocity and 9 Jacobian values in
per-thread **registers** across the entire source cell and flushes them as 12
`CUDA.@atomic` adds once per (pair, target instant). Output width therefore
affects only the flush and downstream L2B/storage, not the per-pair inner
loop — this structural fact bounds Workstream A and is made explicit in A0
below.

The post-cycle-3D 035 measurements show the fused L2B+nearfield remains the
critical path at 47–88% of solve kernel time on cube/wake (92–96% on the
rotor), driven primarily by interaction count. The 035 bound-ness analysis
(jobs 13157887/13157931) classifies the nearfield as **compute-bound at
39–60% of the H200 vector-op ceiling** with DRAM traffic at only 2–12% of
peak; the 12-atomic flush is one of three named residual inefficiencies
(with predicated dual-path retirement in mixed instants and ragged
`n_t mod 32` tails), not the binding constraint. The mixed
(`PartitionedVortex`) bucket is 82–90% of nearfield kernel time. Caveat: NCU
hardware counters are blocked on unprivileged H200 jobs
(`ERR_NVGPUCTRPERM`), so all "exact" operation counts here and in A3 are
op-count models cross-checked against nsys node traces, not measured
counters. Any output reduction must be priced against the complete
overlapped solve rather than against output traffic alone.

The tensor is not symmetric. It is trace-free analytically because the induced
velocity is divergence-free, so a generic representation has eight independent
components, not six. Merely replacing nine stored components with eight is a
valid candidate but has a low expected ceiling; it must not be confused with a
symmetric-Hessian representation.

### 1.2 Consumer identities that must be verified and pinned

Use the FLOWVPM row-major convention visible in `FLOWVPM_timeintegration.jl`.
For particle `p`, write the selected stretching operator as

$$
T_p(v)=\begin{cases}
J_p v, & \texttt{transposed=true},\\
J_p^T v, & \texttt{transposed=false}.
\end{cases}
$$

The source audit must confirm the following current dependencies and find any
additional downstream or extension consumer before an interface is proposed:

- convection needs `U_p`;
- ClassicVPM and ReformulatedVPM time integration need only
  `T_p(\Gamma_p)`, not `J_p` separately;
- both dynamic SFS procedures need the same stretching vector at the test and
  domain filter widths;
- Pedrizzetti relaxation needs `curl(U_p)`, represented by
  `(J6-J8, J7-J3, J2-J4)` in the current layout;
- the `zeta`/RBF viscous routines use `J[1:3]` as temporary storage, not as a
  velocity-gradient requirement; this storage alias must be separated from
  the mathematical output decision;
- the general FastMultipole/FLOWVPM interface and unknown external consumers
  may request the full tensor, so the legacy full-J mode remains available.

The SFS model is the binding case. Its current direct expression is, exactly,

$$
E_p=\sum_q \zeta_{pq}\left[T_p(\Gamma_q)-T_q(\Gamma_q)\right],
$$

where `T` consistently denotes `J` or `J^T` according to the scheme. By
linearity this can be reordered without changing the mathematics:

$$
\Omega_p=\sum_q\zeta_{pq}\Gamma_q,\qquad
Q_p=\sum_q\zeta_{pq}T_q(\Gamma_q),\qquad
E_p=T_p(\Omega_p)-Q_p.
$$

This identity is the only currently identified route to an exact SFS-capable
solve that does not retain full `J`. It changes evaluation order and therefore
floating-point roundoff, but not the modeled operator. Note the pass-count
accounting: `Omega_p` and `Q_p` are both `zeta_{pq}`-weighted sums over the
**same** pair list (`T_q(Gamma_q)` is per-source data computable before the
pass), so they fuse into a single zeta convolution with 6 vector outputs.
The reduced sequence is then one U/T pass plus one fused zeta pass — the
same two pair passes as the current U/J pass plus `Estr` pass, with fewer
output atomics. The crossover question is therefore not "two extra
convolutions vs one" but the cheaper comparison of per-pass output width and
any extra dynamic-SFS filter-width calls; it remains a required measurement,
not an assumption, but the prior is now favorable.

### 1.3 Relationship to prior and concurrent nearfield levers

This plan must be positioned against four completed rows it previously did
not cite; together they set the empirical scale for the 10%/5% gate and
occupy much of the target space.

- **029 (nearfield ILP):** falsified at +1.9%.
- **037e (fine-bin/AABB pruning of the direct list):** failed the identical
  10%-nearfield/5%-end-to-end promotion gate — best material row +2.41%,
  worst regression −17.3%; ships opt-in only.
- **037f (`g`/`h` cheapening):** only the `:fp32` mode passed, at +7.4–7.8%
  on the wake with ~1e-8 delivered-error deltas; `:reduced` and `:lut`
  failed the 5% bar. A whole-kernel precision change barely cleared the same
  gate this plan reuses — output-row reductions should expect less.
- **037d (Fourier/VIC cost model, Done 2026-08-14):** a scoped full-VIC
  implementation row is funded for near-uniform-`sigma` workloads with
  modeled F32 speedups of 3.0–15.2x (cube 1e5) up to 5.4–37.1x (cube 1e6)
  and 1.9–27.8x (wake); the error model meets the 1e-3 gate at `p=4`,
  `h/sigma=0.55`. The rotor (18x `sigma` spread) is structurally infeasible
  for mesh methods, and the Ewald split flips sign there.
- **041 (adaptive octree):** attacks the sigma-heterogeneous pair count
  structurally (multiscale 1.87x step / 3.02x lifecycle; U-list body pairs
  cut 3–30x on wake/multiscale), opt-in.

Consequences: Workstream B's target-cell interpolation competes with 037d's
VIC on exactly the near-uniform-`sigma` cube/wake classes, and with the
adaptive octree on the `sigma`-heterogeneous classes where interpolation
bases are hardest to condition. The final verdict in §5 must state which
niche remains for strategic targets if the funded VIC row and the adaptive
octree both ship, and 037d's measured `h/sigma=0.55` resolution requirement
supplies a free sample-density lower bound for any grid-like basis (used in
Stage 0 of §3.2).

## 2. Workstream A: determine and cost the true output

Workstream A carries two separable values that must not be conflated:

1. a **performance** track (reduced output width), whose ceiling is bounded
   analytically in A0 and is expected to close below the gate; and
2. an **SFS-enablement** track: the radix/GPU path currently hard-rejects
   `sfs=true` (`FLOWVPM_fmm_radix.jl:499`), SFS device kernels were
   deliberately deferred by rows 031a/032, and all 035/037 measurements ran
   `noSFS`. The factorized identity of §1.2 means SFS support needs only the
   fused zeta pass with 6 vector outputs (`Omega`, `Q`) plus per-target
   contractions — a far smaller device surface than a full-J `Estr` port.
   Restoring this physics is part of the stated objective and survives even
   if A0 kills the performance track.

### A0. Analytic pre-kill of the performance track

Before any audit or script: because the kernel accumulates in registers and
contracts like `T_p(v)` involve only per-target constants, every reduced
policy leaves the per-pair inner loop unchanged and can at best shrink the
12-atomic flush (to 6–7) and L2B/storage width. Using the existing 035
counters (flush frequency per target instant, atomic share of the modeled
39–60%-of-ceiling residual), bound each policy's end-to-end ceiling on one
page of arithmetic. If no policy's ceiling reaches 5% end-to-end on a
material case — the expected outcome given 037e/037f scale — record
**Output NO-GO (performance)** immediately, skip A3/A4 script work, and
retain A1/A2 only at the depth needed for the SFS-enablement track.

### A1. Complete consumer and configuration audit

Create a table covering every read/write of `U_INDEX`, `J_INDEX`,
`VORTICITY_INDEX`, and `SFS_INDEX` in FLOWVPM source, extensions, tests, and
the FLOWUnsteady/VortexLattice coupling surfaces available in the workspace.
For each consumer record:

- ClassicVPM or ReformulatedVPM;
- `transposed` setting;
- NoSFS, ConstantSFS, dynamic two-level, dynamic three-level, and sensor SFS;
- no relaxation, Pedrizzetti, or corrected Pedrizzetti;
- Inviscid, CoreSpreading, and ParticleStrengthExchange;
- the exact required quantity (`U`, `J`, `T(Gamma)`, `T(Omega)`, curl, or
  temporary RBF storage), when it is required, and whether it must persist
  after the U/J call.

The audit deliverable must distinguish mathematical requirements from the
current 46-row storage layout. Search results alone are insufficient: trace the
call sequence for Euler and all three RK3 substeps, including dynamic-SFS test
filter evaluations and relaxation cadence.

### A2. Prove and numerically pin reduced identities

Add a small, production-independent script that evaluates random and realistic
particle snapshots in Float64 and BigFloat and verifies:

1. direct `J*Gamma` / `J'*Gamma` against contractions formed during each pair;
2. `tr(J)=0` for the analytic singular and `gaussianerf` pair formulas and for
   far-field evaluation, while recording floating-point residuals;
3. the reordered SFS identity above against `Estr_direct` for both transposed
   conventions, unequal `sigma`, static particles, and self interactions;
4. all dynamic-SFS numerator, denominator, and coefficient values when full J
   is replaced by the reduced intermediates.

This stage establishes mathematical equivalence. It does not use the existing
`U <= 1e-3` FMM gate as permission to alter the SFS equations.

### A3. Compare concrete output policies

Cost these policies using exact scalar-operation, register, output-row, atomic,
and extra-pass counts from the current kernels:

| Policy | Produced quantities | Eligible configurations |
|---|---|---|
| `full_j` | `U` + 9-component `J` | universal control/fallback |
| `tracefree_j` | `U` + 8 independent J components; reconstruct the ninth | universal if parity passes |
| `stretch` | `U` + `T(Gamma)` | no SFS and no relaxation/full-J consumer |
| `stretch_curl` | `U` + `T(Gamma)` + curl | no SFS, relaxation enabled |
| `factorized_sfs` | `Omega`; then `U`, `T(Gamma)`, `T(Omega)`, optional curl; then `Q` and `E` | SFS configurations if two Gaussian passes win |

Derive the pair formulas for `T(v)` directly from
`J = a C tensor_product r + b cross(Gamma_source)` rather than forming nine
entries and contracting afterward — but note the A0 structural fact: since
`Gamma_p` is constant per target, deferring the contraction to the register
flush costs no per-pair work at all, so the per-pair-contraction form must
beat the accumulate-then-contract form, not merely the naive form-nine-then-
contract strawman. Count the transposed and non-transposed forms separately.
Include far-field L2B evaluation and final FLOWVPM storage; nearfield-only
savings are not sufficient.

Price `factorized_sfs` against the actual current SFS sequence:

$$
C_{\rm current}=C_{U+J}+C_{Estr},\qquad
C_{\rm reduced}=C_{Omega}+C_{U+T(Gamma)+T(Omega)}+C_Q.
$$

Use measured direct-route/body-pair counts from cube, wake, and rotor cases and
the 035/041 kernel timings. Include dynamic SFS's extra filter-width calls.

### A4. Promotion gate for an output microbenchmark

A4 runs only if A0 did not close the performance track. Implement no
production code unless the static model predicts at least 10% nearfield
reduction and 5% end-to-end U/J-or-U/J/SFS reduction on a material case. If
it does, make one isolated H200 kernel benchmark with the same body pair
lists and output modes. Promotion requires:

- same mathematical quantities within the rounding envelope established in
  A2, with full-J direct evaluation as oracle;
- every SFS model completing both transposed modes on the CPU reference
  path; GPU/radix SFS parity is gated on the SFS-enablement track actually
  building a device zeta kernel (today `sfs=true` is a hard error on the
  radix path, so this criterion cannot be tested as originally written —
  stage CPU-reference parity first and make the device kernel an explicit
  deliverable of the enablement track, not a precondition of this gate);
- identical static-particle and reset/accumulation semantics;
- at least 5% same-job end-to-end improvement, not merely fewer atomics; and
- no more than 3% regression in any supported configuration.

If no policy passes, record that full `J` is the minimum economical common
representation when SFS is enabled. A separate `stretch` fast path may still
be retained as a NoSFS specialization if it independently passes the gate and
does not complicate the public interface.

## 3. Workstream B: strategic-target interpolation

### 3.1 Candidate algorithm shapes

Test two shapes before considering implementation.

**Target-cell aggregate interpolation (preferred low-overhead form).** For a
target cell `T`, fix its direct source set and define the smooth regularized
nearfield

$$
F_T(x)=\sum_{q\in near(T)} K_{\sigma_q}(x-x_q)\Gamma_q.
$$

Evaluate `F_T` at `r_T` strategic points, accumulate all source-cell
contributions there, and interpolate once to the `n_T` particles in `T`.
This changes the leading direct work from approximately `n_T*m_T` pair
evaluations to `r_T*m_T` sample evaluations plus reconstruction, where `m_T`
is the number of bodies in the direct source set. Use direct evaluation for
cells below the measured crossover.

**Displacement/source-conditioned interpolation.** For a source-target cell
pair, condition a precomputed basis on:

- normalized integer displacement and, for the adaptive tree, source/target
  level ratio and touching relation;
- `sigma_source/h_target` bin and the within-cell min/max ratio;
- regularized, singular, or mixed branch class; and
- source occupancy/intra-cell spread where it changes rank.

Strength magnitude is not a basis parameter because the operator is linear in
`Gamma`; all three independent strength directions must be included in the
operator test. Use cubic rotations/reflections to canonicalize displacement
classes. Pair-specific coefficients must ultimately be accumulated into a
common target-cell representation; a method requiring a separate
reconstruction at every source-target pair is presumed too expensive unless
the cost model proves otherwise.

Both shapes must reconstruct the complete SFS-required quantities selected by
Workstream A. Testing velocity alone is an informative first rung but cannot
produce a positive final verdict.

### 3.2 Cheapest high-confidence feasibility screen: sizing arithmetic, then operator SVD plus QDEIM

**Stage 0 — arithmetic sizing screen (run first; hours, no new code).**
The win ratio is approximately `n_T / r_T` when `c_sample ≈ c_pair`, and the
measured winner geometry pins `n_T` today: cube 36–46, wake 112–165, rotor
153–499 bodies per leaf, with `m_T` = 5.4k–27k direct sources per target
(from `data/flowvpm_gpu_campaign/fm037e_screen.csv` and the 035 tables; also
compute the same statistics for the 041 adaptive-`K_max` winners from
`data/fm041_cuda_cost.csv`, since the adaptive tree changes `n_T`
materially). For each cell class, estimate the required `r_T` from 037d's
measured resolution result — `p=4` interpolation at sample spacing
`h/sigma = 0.55` meets the 1e-3 velocity gate — scaled by the class's
`sigma`-to-cell-size ratio, with an explicit derivative-accuracy penalty for
the J-bearing quantities (differentiation amplifies interpolation error by
~`1/sigma`). Kill any class whose predicted `r_T >= 0.5 * n_T` before
writing any operator-assembly code, and prioritize the fat rotor/wake-1e6
classes (`n_T` 165–499) as the only plausibly-open niche — while flagging
that their `sigma` spread (18x on the rotor) is precisely the regime that
closed 037d's mesh methods, so basis conditioning is expected hardest where
the headroom is largest. Only classes surviving Stage 0 proceed to the SVD
screen below.

The next experiment is an offline numerical-rank ceiling test, not a CUDA
prototype. It answers whether *any* linear sample/interpolation basis could
win before investing in a particular basis.

For each representative cell class, construct a nondimensional discrete
operator mapping arbitrary three-component source strengths at candidate
source locations to `U` and the required derivative/SFS quantities on a dense
target validation set. Candidate locations combine actual snapshot particles
with low-discrepancy points covering both cells. Include `sigma/h` as a
stratified parameter. Then:

1. compute the weighted SVD to obtain the best possible rank-`r`
   approximation and its spectral- and RMS-error curves;
2. apply QDEIM/pivoted QR to the left singular vectors to select actual target
   samples and build the interpolation operator;
3. validate those samples on held-out particle positions, strengths, sigma
   values, seeds, and time snapshots; and
4. compare achieved rank and cost with simple analytic bases.

This is high confidence for low cost: failure of the optimal empirical
subspace at an economical rank closes every less capable linear basis for that
class. Success supplies both a constructive sample set and a lower bound that
analytic/runtime-friendly bases should approach. The screen is small dense
linear algebra over sampled cell blocks; it needs no million-particle solve
and no GPU.

### 3.3 Basis ladder

Evaluate candidates in this order, stopping when a cheaper candidate meets the
rank/error target:

0. the existing solid-harmonic local expansion at lowered `rho_t` and/or
   raised `P` — i.e. demoting the smoother part of the direct list to the
   already-implemented M2L/L2B machinery. This is itself a linear
   sample/interpolation basis and costs zero new code; the QDEIM ceiling is
   only interesting for a class where it materially beats this rung, and
   any custom basis must beat it on the measured cost model, not just on
   rank. (035 cycle 3D's `rho_t` retune already harvested 29–56% of the
   nearfield this way; the question is how much remains.)
1. tensor Chebyshev and total-degree polynomial interpolation in normalized
   target-cell coordinates, differentiating the interpolant analytically for
   J;
2. displacement-aligned anisotropic polynomials, with one coordinate parallel
   to the cell-center displacement and sample density biased toward the
   closest face/edge/corner;
3. kernel-scaled RBF interpolation with length scales tied to the relevant
   `sigma/h` bin and centers chosen by pivoted QR;
4. precomputed POD/QDEIM bases per canonical displacement/property class;
5. a common target-cell POD basis receiving projected coefficients from
   displacement-specific source-cell bases, only if pair-conditioned ranks are
   materially smaller.

Also compare two derivative strategies: interpolate sampled U and
differentiate the basis, versus sample U+J and interpolate both. The former
uses cheaper samples but amplifies interpolation error; the latter has a
higher pair cost but may need substantially fewer points. Include the
factorized SFS fields `Omega` and `Q` if Workstream A selects that route.

Additionally test a **hybrid radius split** orthogonal to the ladder: keep
true near-neighbor sources (within a distance threshold of the target cell)
on the direct path and sample/interpolate only the smoother distant-direct
shell. The nearfield operator's rank is dominated by the closest sources, so
splitting can lower the required `r_T` dramatically and replaces the
all-or-nothing per-cell fallback of §3.6 with a graded one. The split
threshold is a per-class tuning parameter of the cost model.

Do not explore additional named bases after the QDEIM ceiling says the needed
rank cannot win. Conversely, do not reject interpolation merely because an
untuned tensor grid fails if the optimal-rank/QDEIM result passes.

### 3.4 Training and validation matrix

Use existing construction code and checksummed references for:

- cube, helical wake, and realistic rotor/multiscale fields;
- `n=1e5` and `n=1e6` geometry statistics without evaluating all blocks;
- uniform and adaptive trees at each case's measured winner;
- Float32 and Float64 source states;
- initial, displaced RK-stage, and later-time snapshots where available.

Stratify sampled target cells by contribution to total body-pair work rather
than sampling cells uniformly. Mandatory hard classes are self cells,
face/edge/corner neighbors, the most distant direct offsets, mixed
regularized/singular buckets, adaptive unequal-level U interactions, minimum
sigma, maximum sigma spread, nearly empty cells, and fat cells.

Hold out entire seeds and time snapshots for validation. Add adversarial
off-grid tests with particles on cell faces/corners, a source arbitrarily
close to a target, minimum allowed sigma, aligned circulation, cancelling
circulation, and arbitrary independent strength directions. Training and
validation membership and random seeds are written to the CSV metadata.

### 3.5 Error measures

Report all of the following, both per class and weighted by production work:

- incremental velocity RMS and maximum error;
- full-J, stretching, curl, and SFS-vector RMS errors normalized by their
  direct-reference RMS scales (not unstable pointwise relative errors near
  zeros);
- dynamic-SFS numerator, denominator, and coefficient error;
- errors separated into interpolation and existing FMM components; and
- short RK3 trajectory errors in position, Gamma, sigma, SFS coefficient,
  and any existing conserved/monitored quantities.

Default feasibility budgets are incremental velocity RMS `<=2.5e-4` (safe in
RMS quadrature against the delivered 3.0–7.1e-4 far-field baseline, keeping
the total under the 1e-3 gate) and total velocity RMS `<=1e-3`. For the
J-derived quantities, budget **relative to the delivered baseline**, not an
absolute figure: the shipped solve's sampled-J diagnostic already runs
2.4–4.2e-3 relative RMS, so holding interpolation to an absolute 1e-3 on
stretching/curl/SFS would demand it outperform the far-field it augments.
Default: incremental stretching/curl/SFS RMS no more than 0.3x the measured
same-case far-field J diagnostic (so the interpolation contribution is
subdominant in quadrature). Tighten a budget if the A2 sensitivity
test shows that dynamic SFS or trajectory behavior requires it. Full-J output
from the legacy API must still be available; when strategic interpolation
claims to reproduce it, report all nine components and the trace residual.

### 3.6 Cost and crossover model

For every target cell compare the measured forms

$$
C_{direct}=n_T m_T c_{pair},
$$

$$
C_{sample}=r_T m_T c_{sample}+n_T r_T c_{reconstruct}+C_{class/setup}.
$$

For a displacement-conditioned method, include coefficient projection and
accumulation for every source cell. Measure `c_pair`, `c_sample`, and
`c_reconstruct` with the exact output policy selected in Workstream A; do not
assume they are equal. Measure `c_pair` on the mixed (`PartitionedVortex`)
bucket, which is 82–90% of nearfield kernel time — a singular-bucket
`c_pair` understates the baseline. Price the crossover on the overlapped
critical path per the standing lever rule, not on isolated stage sums.
Apply a per-cell direct fallback whenever interpolation does not win. Include basis lookup, sample generation, source-property
classification, capacity, graph-capture, and occupancy-epoch rebuild costs.

The accuracy-only experiment advances to an H200 microbenchmark only if the
ideal QDEIM result and the best practical basis both predict:

- at least 10% reduction of the nearfield critical path and 5% of the complete
  overlapped solve on a material case;
- a practical rank no more than 1.25 times the QDEIM rank;
- positive savings after direct fallback on low-occupancy cells; and
- bounded precomputed storage by canonical displacement/property class.

If QDEIM needs as many samples as the cost-weighted target occupancy, or if
source-property binning makes setup/storage erase the modeled gain, close the
idea without GPU work. This is the primary cheap kill rule.

## 4. Deliverables if the plan is executed

Keep the initial artifact surface small:

1. `theory/nearfield-output-requirements.md`: complete consumer table,
   reduced-output derivations, operation/pass model, and output-policy verdict;
2. `scripts/extract_strategic_target_rotor_cells.jl` and
   `scripts/strategic_target_rank_probe.jl`: deterministic extraction from the
   actual DJI-9443 rotor snapshot, operator assembly, SVD/QDEIM,
   analytic-basis comparison, cross-validation, and cost model;
3. `data/strategic_target_feasibility/`: compact CSVs containing class
   metadata, singular-value/rank curves, errors, and predicted crossover; no
   dense matrices committed; and
4. a result section appended to this file with a go/no-go verdict and, only on
   GO, a separately reviewed production implementation task proposed for
   `START_HERE.md`.

The rank probe must run locally on small extracted blocks. H200 time is used
only after the registered rank/error/cost gates pass. No production source in
FastMultipole or FLOWVPM changes during this feasibility task.

## 5. Final decision rules

The result must choose one of these conclusions explicitly:

- **Output GO:** a reduced policy preserves every enabled mathematical model
  and clears the end-to-end performance gate; stage an interface/kernel task
  with `full_j` fallback.
- **Output specialization only:** SFS economically requires full J, but a
  NoSFS `stretch` policy clears the gate; stage it without weakening SFS.
- **Output NO-GO:** eight/nine J components are the cheapest shared
  representation once SFS and relaxation are included.
- **Strategic-target GO:** held-out and adversarial accuracy passes at a rank
  with modeled and then measured end-to-end savings; stage a device-resident,
  opt-in implementation with direct fallback.
- **Strategic-target regime-only:** only fat-cell, uniform-sigma, or another
  measurable class passes; expose an automatic cost/geometry selector and do
  not change the general default.
- **Strategic-target NO-GO:** the optimal QDEIM ceiling cannot beat direct
  evaluation after reconstruction/setup, closing further basis searches for
  the tested accuracy and geometry.

Whatever the verdict, the result section must additionally answer the
positioning question of §1.3: for each class where strategic targets win on
paper, state whether the funded 037d VIC row or the 041 adaptive octree
would capture the same savings, and only classes where strategic targets
beat both justify a production task. "Ship 037d VIC + adaptive octree
instead" is an acceptable and explicit closing recommendation.

Any eventual default change retains the existing sampled-direct velocity gate,
adds SFS/trajectory gates from this plan, preserves zero recurring allocation
and device residency, and requires explicit user approval after same-job H200
evidence.

## 6. Execution result (`2026-08-15`)

**Done; awaiting independent clear-context approval.** The consumer audit,
identity proof, A0 bound, Stage-0 sizing, and corrected SVD/QDEIM screen are recorded in
`theory/nearfield-output-requirements.md`. Reproduction scripts are
`scripts/verify_nearfield_output_identities.jl` and
`scripts/extract_strategic_target_rotor_cells.jl` plus
`scripts/strategic_target_rank_probe.jl`; compact CSV results are in
`data/strategic_target_feasibility/`.

A0 closed the performance track: the existing same-kernel 1M-body attribution
measured 11.4521 ms with atomics and 11.4375 ms with all stores removed, only a
0.13% isolated ceiling. The exact factorized-SFS identity passed Float32,
Float64, and BigFloat for both transpose conventions. The corrected verifier
also covers unequal sigma, a static target, coincident self pairs, analytic
singular/`gaussianerf` and distant-source trace residuals, and the dynamic-SFS
numerator/denominator/coefficient. The identity remains a separately valuable
route to restore SFS physics on radix/GPU.

Stage 0 left only the rotor-1M uniform U/J geometry alive; every adaptive U/J
mean-occupancy class failed the optimistic half-occupancy rule. The corrected
screen therefore uses compact self/face/edge/corner/shell blocks extracted
from the actual deterministic 1M-particle DJI-9443 snapshot at `ell=6`, not
the unrelated synthetic multiscale case. It spans three property-conditioned
sigma bins over the recorded 18x range and tests U and U/J separately.

The optimistic scalar-row QDEIM ceiling passed 12/30 cases: all corner and
smoother-shell sigma-bin/output combinations. Self, face, and edge failed.
The registered practical-basis gate then closed the apparent niche. Tensor
Chebyshev and total-degree polynomial interpolation failed accuracy through
216 points. Gaussian RBF interpolation passed shell accuracy at 120 points
for U and 165 for U/J, but those counts are respectively 2.67x and 2.39x the
QDEIM ranks, exceeding the registered 1.25x ceiling. Thus 0/30 cases pass the
joined QDEIM-plus-practical-rank gate. Per the stop rule, no H200 prototype was
run.

Final verdicts remain **Output NO-GO (performance)** and
**Strategic-target NO-GO**, now for the corrected evidence above.
Retain full J and ship no strategic-target path. If desired, stage factorized
SFS enablement as a new production row after `042`; it is physics restoration,
not a claimed performance optimization. The funded VIC direction owns the
uniform-sigma niche and the adaptive U-list owns the heterogeneous geometry;
no residual strategic-target niche passed.

## 7. Clear-context approval log

**Approved `2026-08-17` by user direction after independent review.** The
output-width/SFS result was reproduced: committed checksums pass, the 0.13%
no-store attribution supports the performance pre-kill, and the identity
verifier passes Float32, Float64, and BigFloat.

Non-blocking suggestions retained for any future reopening of strategic-target
interpolation:

- select Gaussian-RBF centers with pivoted QR/QDEIM rather than taking the
  first Halton points;
- compare interpolation of sampled U followed by analytic differentiation
  against sampling and interpolating U+J;
- include the displacement-aligned basis, hybrid-radius split, and existing
  M2L/demotion rung in the practical-basis comparison; and
- extend validation to held-out seeds/time snapshots and the registered
  adaptive/adversarial classes.

These suggestions do not alter this row's recorded Strategic-target NO-GO or
authorize production work.
