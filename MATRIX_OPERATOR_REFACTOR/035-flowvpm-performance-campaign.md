# 035 FLOWVPM Performance Campaign

## Status and Entry Gate

**Added by user request on `2026-08-04`; consolidated by roadmap review on
`2026-08-05`.** Not started.

Entry gate: `034` (working GPU coupling), `032a` (measured nearfield default),
and `030` (matched-`n` reference measurements) must all be Done and
clear-context approved.

This row owns the phase's tuning, profile-driven optimization decisions, and
sole definitive speedup report. Benchmark scripts live in `scripts/`, data in
`data/`, and figures follow the `024a` conventions. FLOWVPM changes go on
`gpu-full`; FastMultipole changes go in this repository. Structural radix-grid
work selected by this campaign is specified and implemented by follow-on item
`037`, rather than being folded into this benchmark/report row.

## Objective

Tune the GPU-coupled FLOWVPM separately for the cube and wake cases, then
identify and rank significant profiled speedup levers. Implement low-risk
approved levers within the campaign, and hand structural radix-grid changes to
their dedicated follow-on items. In particular, determine whether the
equal-physical-cell rectangular radix grid in `037` clears the 5% expected
end-to-end U/J-solve gain bar. Document the final performance once, against
eligible `033` baselines and the matched-`n` `030` record.

## Campaign

### 1. Initial tuning and profile

1. Pre-register the parameter grid before running. Sweep depth `ell`,
   precision policy, M2L strategy, near radius / level-radius geometry, and
   relevant `032` interface knobs separately for each case at the
   representative `033` particle count.
2. Winner eligibility requires sampled relative velocity RMS error `≤1e-3`.
   Log sampled Jacobian RMS error for every configuration as a diagnostic; it
   does not select or disqualify a winner.
3. Benchmark the wake at the cube-optimal parameters and report timing,
   velocity error, and Jacobian error deltas from the wake-optimal result.
4. Profile the winning U/J solve by stage. Rank remaining levers by expected
   end-to-end U/J-solve gain, risk, affected repository, and verification
   burden. Only levers with a defensible expected gain of at least 5% proceed
   to an optimization cycle.

### 1a. Pre-registered lever: the cube-box penalty on elongated domains

Staged by user direction `2026-08-05` when the vortex ring was replaced by the
AR=5 wake cylinder. It enters the ranked list of step 4 like any other lever and
must clear the same 5% bar; it is named here because it is known in advance and
must be *measured*, not assumed.

The radix box is a **cube** (`_assert_radix_positions_in_box`:
`x_max = x_min .+ 2*h0`), so a domain whose longest axis dominates wastes the
grid. The wake at AR=5 fills 3.14% of its bounding cube
(`V_cyl = 3.927` against `L³ = 125`); real rotor wakes are longer still, and the
penalty grows as the aspect ratio cubed in nominal cell count.

What is and is not at stake — establish this by measurement before proposing an
implementation:

- The finest admissible cell is `h_min ≈ ρ_t σ / g_min ≈ 4.28·s` (`031a` §5.1),
  a function of particle spacing only and therefore **independent of box
  shape**. The count of *occupied* cells at `h_min` is likewise box-independent,
  so far-field work is not directly inflated. This is why the lever is not
  self-evidently worth 5%.
- What elongation does cost: extra tree levels to reach the same `h`
  (`ℓ ≤ 6` for the wake at `n=1e6` against `ℓ ≤ 4` for the cube — measured
  ~121 bodies per occupied leaf versus an all-cell average of 3.8), more
  sparse M2M/L2L levels, and inflation of anything sized by *total* rather than
  occupied cells. `024b` already failed to construct `ℓ=6/7` grids, so the
  practical `ℓ` cap is a real constraint, not a hypothetical one.
- Small `n` degenerates: at `n=1e3` the wake's adequacy ceiling is `ℓ ≤ 2`,
  leaving ~2 occupied cells and ~500 bodies each — effectively all-direct. Check
  whether the low end of the sweep is measuring the FMM at all.

Candidate remedies, in increasing invasiveness: a non-cubic (anisotropic) radix
box; tiling the domain as an `a×b×c` arrangement of cube grids; or per-axis
level counts. For this campaign, the preferred structural candidate is a
rectangular logical grid with approximately equal *physical* cell widths, so
the cells remain cubic for the error model while the axis cell counts differ.
Profile first — report where the time actually goes on the wake versus the cube
at matched `n` — then specify the rectangular-grid acceptance case for `037`.

`035` owns the measurement, parameter selection, and go/no-go recommendation.
`037` owns the FastMultipole production implementation: generalized
quantization/keying, rectangular occupancy metadata, CUDA refresh, routing, and
the hierarchy needed by the resident lifecycle. The first implementation scope
should prefer a fixed-resolution rectangular leaf grid; a generalized
anisotropic hierarchy should be added only if the measured design requires it.

### 2. Optimization cycles

Repeat while an eligible lever remains:

1. Present the current ranked lever list and proposed next lever(s) to the
   user. Production implementation requires explicit user approval; ordinary
   sweep execution and measurement do not.
2. Implement only approved low-risk levers owned by this campaign. For the
   rectangular radix lever, record the measured recommendation and handoff to
   `037`; commits for `037` remain separate from the campaign record.
3. Run the correctness, accuracy, allocation/residency, CPU-compatibility, and
   scalar-path regression gates below.
4. Re-measure the U/J solve and stage profile, recording expected versus
   realized gain. Re-rank the remaining levers using the new profile.

Stop when no credible untried lever has at least 5% expected end-to-end
U/J-solve gain. Task `030` is a comparison reference, not a numeric stop
threshold.

### 3. Final report

Produce the phase's definitive tables and figures only after the final cycle:

- Per-case final U/J-solve and full RK3-step timings, with warmup, repeat,
  median, and variability policy stated.
- Speedups versus the fixed `033` single-thread and 64-thread baselines only
  where that baseline's sampled velocity RMS error passes `≤1e-3`. Preserve
  failing historical timings and errors in the report, but do not compute or
  headline a speedup from them; do not run a replacement tuned CPU campaign.
- Per-stage profiles for eligible CPU baselines and the final GPU result,
  showing bottleneck movement and realized optimization gains.
- One-U/J-solve time divided by the matched-`n` `030` resident-evaluation time,
  with vector strength, Lamb-Helmholtz, hessian, regularized-nearfield, and
  other workload differences itemized. Report the full three-solve RK3 step
  separately; do not label the ratio as passing or failing a numeric target.
- A cycle ledger and closing list of implemented, rejected, and remaining
  sub-5% levers.

## Dependencies and Reading

- `034-flowvpm-gpu-integration.md`, Done and clear-context approved.
- `032a-impl-split-nearfield-comparison.md`, Done and clear-context approved.
- `030-benchmark-cost-vs-n-fixed-ell.md`, Done and clear-context approved.
- Transitively `033` for baselines, cases, and sampled-direct references.
- Read `START_HERE.md`, `../FLOWVPM.jl/CLAUDE.md`, the completed `030`, `032a`,
  `033`, and `034` reports, and the optimization-cycle sections of `028`.

## Work Record

### 2026-08-11 — reading gate, tuning surface, pre-registered grid (session 1)

**Reading gate completed** by the executing agent (Claude Fable 5):
`START_HERE.md` (protocol + Integration Phase preamble incl. the 1e-3
velocity gate, J-diagnostic convention, per-case tuning rules), this task
file, `../FLOWVPM.jl/CLAUDE.md` in full, `033` (results + approval record;
CPU baselines and the gate audit — every FMM-active default-parameter CPU row
fails 1e-3, so speedup headlines vs 033 are restricted to n≤3162 cube /
n=1e3 wake), `034` (H200 coupling results; warm-solve order-of-magnitude
context), `030` (matched-`n` reference: per-`n` retuned verdict costs, e.g.
3.097 ms at n=1e5 FP16), and `032a` (nearfield defaults: PartitionedVortex +
classsplit/sub-Morton, `rho_t=4.252`, shipped at `f829e9b`; Stage D step-level
1.16–1.77x on these exact case constructions).

**Key finding from the reading gate:** FLOWVPM's 034 coupling still hardwires
`RegularizedVortex` + `ConcatenatedFixedZM2L` and exposes no geometry/strategy
overrides beyond `ell`/`near_radius2` — the shipped 032a winner
(`PartitionedVortex`) is not reachable from FLOWVPM. Also, FP16-WMMA is
structurally unavailable to this workload (`DENSE_CUDA_TENSOR_FORMAT` engages
only for `!LH && D == 16`; FLOWVPM requires Lamb-Helmholtz), so the precision
grid is Float64/Float32 only, and the 030 FP16 numbers are not reachable
targets for the ratio comparison.

**FLOWVPM tuning surface (gpu-full commit `4d188fa`):** benchmark-enabling
extension of the internal `RadixFMMSettings` — `direct_kernel`
(:regularized/:partitioned/:twopass + `rho_t` override, resolved through the
`fmm.direct_kernel` trait), `m2l_strategy` (:concat/:dense/:precomputed_y with
the paired operator), `level_radii2` passthrough, and the auto-`ell` adequacy
inequality now uses the selected kernel's `rho_t`. Defaults unchanged
(shipped behavior identical); new Part A testset (p=4) covers the selections
against `UJ_direct` plus loud invalid-symbol errors. Local Part A suite green
(30 assertions across the coupling testsets, 4 threads).

**Pre-registered grid** (`scripts/fm035_cases_initial.txt`, 75 configs,
committed before any cluster run): per case at the representative n=1e5 —
status-quo derived baselines (RegularizedVortex, both precisions), a
partitioned-kernel (ell, leaf-q) geometry scan over the locally pre-computed
adequacy-admissible set (`g_min(q)·h_leaf > rho_t·sigma_max`; cube ell≤4,
wake ell≤5 at n=1e5), Float64 subsets, regularized-at-winner-geometry kernel
attribution rows, M2L strategy A/B (:concat/:dense/:precomputed_y) at
candidate winner geometries, window-classes {64, 256, 4096}, `rho_t` 4.789
spot, boosted-coarse level schedules; two wake-at-cube-optimal candidate rows;
and a 13-config n=1e6 spot grid (both cases, deepest-admissible depths).
Stage profiling (`profile=1`: refresh/b2m/m2m/m2l/l2l/l2b + hierarchical
refresh telemetry + nodes/routes per level) on baselines and candidate
winners; RK3 full-step timing (`rk3=1`, relax and U_prev bookkeeping off) on
per-case winners. Winner eligibility: sampled velocity RMS ≤ 1e-3 vs the
sha256-checksummed 033 references; J logged as diagnostic on every row; 023
counter flatness and steady-state `CUDA.@allocated` asserted per row.

**Harness:** `scripts/benchmark_035_gpu.jl` (driver; FM035_DRYRUN validated
all 75 configs locally), `scripts/cuda_035_{submit,run,fetch}.sh` (034
cluster pattern: task-034-owned trees `~/FLOWVPM-034` + `~/FastMultipole-034`
refreshed to current tips — the FastMultipole copy predated the 032a
defaults — env `~/fm034env`, julia 1.11.7 pinned). Job preflights: 032
CUDA interface tests, host `device_system_interface_test.jl` (first
on-hardware run of the 032a `shipped nearfield defaults` assertions, per the
032a approval note), `cuda_radix_nearfield_binning_test.jl` (CUDA mechanism
Ref defaults), and the FLOWVPM coupling suite (Part A + Part B). Data lands
in `data/flowvpm_gpu_campaign/`.

## Verification Gates

- Every timed configuration records sampled velocity and Jacobian RMS errors;
  only velocity `≤1e-3` is a winner/speedup gate.
- The FastMultipole adapter/cache lifecycle retains `body_uploads = 0`,
  `expansion_host_copies = 0`, construction-only route/operator uploads, and
  zero recurring allocation.
- FLOWVPM CPU tests remain unchanged; exported names and keyword defaults do
  not change without separate user approval.
- If FastMultipole is touched, rerun the shipped scalar `028`/`030` harness
  configuration and reject a statistically meaningful regression.
- GPU measurements run on H200. CPU comparison values come from `033` and are
  not rerun except to establish environment parity.
