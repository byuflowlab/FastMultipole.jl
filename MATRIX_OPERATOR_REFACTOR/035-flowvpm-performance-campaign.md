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

### 2026-08-11 — initial sweep complete (job 13134906): tuned configs, profile, lever ranking (session 1 cont.)

**Job 13134906** (H200, m13h-1-1, 3h58m, exit 0): all preflights green —
032 CUDA interface tests, host `device_system_interface_test.jl` (35k
assertions incl. the 032a `shipped nearfield defaults` testset — **first
on-hardware run of the shipped-default assertions**, closing the 032a
approval note), `cuda_radix_nearfield_binning_test.jl` **304/304** (CUDA
mechanism Ref defaults on hardware), FLOWVPM coupling suite Part A+B. All
14 reference checksums OK. **75/75 pre-registered configs measured, zero
failures**; `counters_flat=true` on every row; steady-state allocation ≤
251 KB (the known framework scatter-dict metadata; 023 counters prove zero
body traffic). Data: `data/flowvpm_gpu_campaign/fm035_sweep.csv` + job log.

**Root cause found (reading gate, confirmed by measurement): the 034
coupling's explicit `ConcatenatedFixedZM2L` override discards
FastMultipole's measured auto default.** `_default_radix_m2l_strategy`
selects `DenseTranslationM2L` at `P ≤ 4` (the 024/028 measured rule); the
coupling pinned concat. Measured penalty at matched geometry: cube (ℓ=4,
q=17, F32) 33.8 → 12.2 ms (2.8x), wake (ℓ=5, q=12, F32) 26.3 → 16.2 ms
(1.6x). Similarly the 032a winner `PartitionedVortex` was unreachable
pre-035: regularized→partitioned at matched geometry is 1.22x (cube F32),
1.88x (cube F64), 1.19x (wake F32).

**Deliverable 1 — per-case tuned GPU configurations (n=1e5, all
gate-passing, U-gate 1e-3, J logged as diagnostic):**

| case | TF | config (kernel/ℓ/q/strategy/K) | U/J solve (ms) | u_rel_rms | j_rel_rms | vs 034 status quo |
|---|---|---|---:|---|---|---|
| cube | F32 | part / 4 / 17 / dense / 256 | **12.23** | 9.35e-4 | 5.2e-3 | 51.4 → 4.2x |
| cube | F32 robust-margin alt | part / 3 / 16 / dense | 29.13 | 5.66e-4 | 1.7e-3 | 1.77x |
| cube | F64 | part / 3 / 16 / dense | **42.81** | 5.66e-4 | 1.7e-3 | 94.6 → 2.2x |
| wake | F32 | part / 5 / 12 / dense / 256 | **16.24** | 6.94e-4 | 6.0e-3 | 36.6 → 2.25x |
| wake | F32 margin alt | part / 5 / 16 / dense | 16.72 | 4.79e-4 | 4.4e-3 | 2.19x |
| wake | F64 | part / 5 / 16 / dense | **23.82** | 4.79e-4 | 4.4e-3 | 57.1 → 2.4x |

Status quo = the shipped 034 coupling (auto-derived geometry, regularized,
concat): cube 51.4/94.6 ms (F32/F64), wake 36.6/57.1 ms. The cube F32
winner's error is 0.93x the gate (passes the 030-style 0.95x robust rule,
barely; the ℓ=3 alternative has 1.8x margin at 2.4x the cost). Full RK3
step ≈ 3 × U/J + ~0 integrator overhead (measured at the concat
geometry rows: cube (3,16) 126.3/150.8 ms F32/F64 vs 3×42.2/49.9; wake
(5,16) 90.2/105.5 vs 3×30.0/35.1). Depth ceilings at n=1e5 are
σ-adequacy-bound: cube ℓ≤4, wake ℓ≤5 (ℓ+1 needs g_min > 3, unsupported).
FP16-WMMA is structurally unavailable (LH on), so F32 is the fast lane.

n=1e6 spot grid (concat only — dense at 1e6 is a cycle-1 measurement gap):
cube best gate-passing (ℓ=4, q=16, F32) 299.4 ms u=7.2e-4 (ℓ=5 FAILS the
gate at 1.27e-3 — depth at 1e6 cube is accuracy-limited, not
adequacy-limited); wake best (ℓ=6, q=12, F32) 168.1 ms u=7.0e-4.
Status-quo baselines: cube 427.1, wake 282.8 ms (F32).

**Deliverable 2 — wake at cube-optimal parameters** (ℓ=4, q=17, part,
F32; concat form): **101.0 ms vs 16.2 ms wake-optimal (6.2x slower)**;
u=1.39e-4 (5.0x lower error than needed), J=6.8e-4. The cube-optimal
depth leaves the wake with 168 occupied cells (~600 bodies/leaf) — the
solve degenerates toward all-direct with the one-thread-per-cell B2M
serialization on top. Cube-optimal parameters do not transfer; per-case
depth selection is mandatory.

**Deliverable 3 — matched-n 030 ratio (n=1e5).** 030 best admissible
verdict-boundary: 3.097 ms (FP16-WMMA/F32) / 3.389 ms (F64). One tuned
FLOWVPM U/J solve: 12.23 ms F32 → **3.9x the 030 FP16 row, 3.6x the 030
F64 row** (F64: 42.8/3.389 = 12.6x). Itemized workload differences:
Lamb-Helmholtz χ carried end-to-end (double expansion channel; disables
the FP16-WMMA tensor path entirely); vector strength (3 B2M source
channels); 13-row output incl. the 9-component hessian (vs 4-row);
σ-carrying regularized nearfield whose ρ_t·σ coverage forces q=16-17 near
sets at shallow σ-bound depths (030 runs q²=4-6 at ℓ=4-5); plus
FLOWVPM-side reset/dispatch (~0.1 ms). Boundary composition is comparable
(030 verdict = refresh+eval+finalize+Euler; ours = reset+refresh+eval+
finalize). Full RK3 step is reported separately above, not in the ratio.

**Deliverable 4 — per-stage profile.** The U/J solve is FastMultipole
eval-bound everywhere: FLOWVPM-side overhead (reset 0.05 + finalize 0.06 +
dispatch ~0) ≤ 0.5 ms ≈ 1-3%; refresh 0.33-1.3 ms. Stage medians
(CUDA-event, ms; l2b is fused with the nearfield):

| row | b2m | m2m | m2l | l2l | l2b+nearfield | eval |
|---|---:|---:|---:|---:|---:|---:|
| cube (3,16) F32 concat | **21.8** | 0.9 | 7.5 | 0.9 | 19.1 | 41.8 |
| cube (4,17) F32 concat | 3.6 | 1.2 | 23.8 | 1.1 | 7.7 | 33.7 (dense: 11.8) |
| wake (5,12) F32 concat | **11.4** | 1.4 | 8.5 | 1.4 | 8.3 | 25.9 (dense: 15.8) |
| cube 1e6 (4,16) F32 | 27.5 | 1.2 | 21.9 | 1.2 | **277.7** | 297.6 |
| wake 1e6 (6,16) F32 | 14.7 | 1.6 | 42.6 | 1.6 | **166.5** | 211.5 |

Key attribution: **B2M is one-thread-per-cell** (`_launch_cuda_b2m!`:
`threads=128, blocks=cld(ncells,128)`, serial over the cell's bodies) —
at the σ-forced shallow depths this leaves the H200 nearly idle (cube ℓ=3:
512 threads total, 195 bodies each → 21.8 ms = 52% of eval; wake winner
(5,12) dense: b2m 11.4 of 15.8 ms eval ≈ 72%). At 1e6 the fused
nearfield dominates (86% cube, 79% wake) because σ-coverage plus the
gate-driven depth cap (cube ℓ=5 fails accuracy) pin bodies/leaf high.

**Deliverable 5 — ranked lever list (≥5% end-to-end U/J bar):**

1. **Dense M2L + PartitionedVortex + per-case geometry as the coupling
   defaults** (FLOWVPM `gpu-full`; measured, not modeled): 4.2x/2.2x
   (cube F32/F64), 2.25x/2.4x (wake) vs the shipped coupling. Risk: low —
   every ingredient is a production FastMultipole surface already measured
   passing all gates in this sweep. This is proposed cycle 1 (below).
2. **B2M within-cell parallelization** (FastMultipole `src/`,
   `translate_batched_cuda.jl`): block-per-cell with parallel bodies +
   coefficient reduction. Expected (modeled from occupancy): wake winner
   11.4 → ~1-2 ms ⇒ up to ~1.6x end-to-end wake, ~10-15% cube (4,17);
   larger wherever depth is σ/accuracy-capped. Risk: medium (new kernel,
   φ+χ parity tests at P=4/P=8 both precisions). Candidate cycle 2.
3. **n=1e6 fixed-error geometry** (measurement + possibly richer coarse
   schedules): cube ℓ=5 fails the gate at 1.27e-3 while ℓ=4 pays 277 ms of
   nearfield; a schedule/radius combination that passes at ℓ=5 (plus dense
   M2L, unmeasured at 1e6) is modeled ≥ 1.5-2x at 1e6. Risk: low
   (measurement-first). Folded into cycle 1's measurement rider.
4. **window_classes for concat** (K=4096: cube (3,16) 42.2→36.2, wake
   (5,16) 30.0→21.1) — superseded by dense as default; dense×K sensitivity
   is a cycle-1 measurement rider. K=64 is a measured loss (1.4-1.9x).
5. Below the bar / rejected by measurement: `rho_t` 4.252 vs 4.789 (≤0.3%
   here — nearfield is compute-bound at these fractions, keep the shipped
   4.252); boosted-coarse schedules (measured 4-14% *worse*: extra
   transition offsets outweigh accuracy headroom at fixed leaf q);
   `precomputed_y` (loses to dense everywhere measured, cube (4,17) by
   2.3x); FLOWVPM-side overhead (≤1-3%, nothing to win).

**Deliverable 5a — the 037 rectangular-grid verdict (measured estimate):
does NOT clear the 5% bar at the tuned operating points; defer.** The
occupancy-compacted radix path already absorbs the wake's 96.9%-empty
bounding cube: occupied node counts per level are tiny (1/8/16/32/168/896
at ℓ=5), B2M and nearfield scale with occupied cells only, and ℓ=6
constructs fine (the 024b-era ℓ-cap no longer binds). What elongation
actually costs at fixed n is the extra sparse coarse levels: ~2 extra
M2M+L2L levels ≈ 0.9-1.8 ms of launch floor plus a small route-gen term —
≈5.5% of the 16.2 ms wake winner (borderline), <2% at 1e6 (168 ms). The
wake is *faster* than the cube at 1e6 (168 vs 299 ms) despite the box
penalty. Caveat recorded: if cycle 1 + a B2M fix shrink the wake solve to
~8 ms, the same ~0.9-1.8 ms becomes 11-22% and 037 would clear the bar —
re-evaluate on the post-cycle profile before starting 037.

**Deliverable 6 — proposed optimization cycle 1 (awaiting user
approval):** flip the FLOWVPM coupling defaults (`gpu-full`,
`src/FLOWVPM_fmm_radix.jl` only) to the measured winners:
`m2l_strategy=:dense` (restores FastMultipole's own measured auto rule),
`direct_kernel=:partitioned` (the user-approved 032a default for
σ-carrying vortex systems), and a joint deepest-admissible (ℓ, q)
auto-geometry rule using the kernel's ρ_t with an accuracy-margin guard
(cube-1e6-style depth overreach must be rejected; exact rule to be
validated against this sweep's error data before shipping). Expected
realized gain vs the shipped coupling: ≥4x cube F32 / ≥2.2x elsewhere at
n=1e5 (measured). Risk: low. Verification plan: Part A/B suites + the
sha256-gated 033 refcheck (both cases × both precisions) + a small H200
confirmation grid that also fills the measurement gaps in the same job —
dense at (4,17) F64 cube, dense at n=1e6 (both cases, incl. an ℓ=5 cube
schedule probe), dense×K sensitivity, RK3 at the dense winners. No
FastMultipole `src/` change in cycle 1 (B2M is cycle 2, separately
approved).

STOPPED here per the run contract: no optimization cycle is implemented
without user approval.

### 2026-08-12 — cycle 1 approved, implemented, confirmed (job 13148807)

**User approval (2026-08-12, via coordinator):** cycle 1 as proposed.
Implemented on FLOWVPM `gpu-full` commit `bc7df09` (no FastMultipole `src/`
change, per scope): `RadixFMMSettings` defaults flipped to
`m2l_strategy=:dense` + `direct_kernel=:partitioned`, and `_radix_auto_ell`
replaced by the joint `_radix_auto_geometry(L, σ_max, np, q_floor, ρ_t,
margin)` rule — deepest `ell` (occupancy-capped as before) admitting a
supported leaf `q ≥ near_radius2` under `g_min(q)·h_leaf ≥
accuracy_margin·ρ_t·σ_max`, smallest such `q`; `accuracy_margin=1.15` from
the sweep's error data (x=4.26 fails the 1e-3 gate, x=4.92 passes). Tests
assert the new defaults and that the rule reproduces the measured n=1e5
winners (cube (4,17), wake (5,16)); Part A green locally. Harness commits
`58cbfb6`/`4b6…` added the 22-config confirmation grid, a `leaf_q` column,
and the 033-refcheck stage (job 13148748 was cancelled pre-start to add the
refcheck; 13148807 is the job of record).

**Job 13148807** (H200 m13h-1-1, exit 0): all preflights green (Part A
asserts the new defaults; Part B device suite; 032/032a FastMultipole
preflights), **033 refcheck at the shipped defaults PASSED** (cube/wake ×
1e4/1e5 × F64/F32; worst Float64 u_rel_rms 9.35e-4 at cube 1e5 — the
tuned geometry trades accuracy margin for 4x speed, still inside the
gate), 22/22 sweep rows ok, counters flat, alloc ≤ 10 KB.
Data: `data/flowvpm_gpu_campaign/fm035_cycle1.csv`.

**Realized vs expected (U/J solve, shipped coupling before → after cycle 1,
all rows gate-passing):**

| case | TF | before (034 defaults) | after (auto = shipped rule) | realized | expected |
|---|---|---:|---:|---|---|
| cube 1e5 | F32 | 51.4 | **12.09** (ℓ4 q17) | **4.25x** | ≥4x ✓ |
| cube 1e5 | F64 | 94.6 | **20.57** (ℓ4 q17) | **4.60x** | ~2.2x (beaten: the (4,17) F64 dense gap fill wins over ℓ3) |
| wake 1e5 | F32 | 36.6 | **16.64** (ℓ5 q16) | **2.20x** | 2.25x ✓ |
| wake 1e5 | F64 | 57.1 | **23.76** (ℓ5 q16) | **2.40x** | 2.4x ✓ |
| cube 1e6 | F32 | 427.1 | **149.3** (ℓ5 q17) | **2.86x** | modeled ≥1.5-2x ✓ |
| wake 1e6 | F32 | 282.8 | **177.1** (ℓ6 q16) | 1.60x (2.03x at (6,12)=139.5) | — |

**Margin-boundary probes (auto-rule validation at 1e6):** auto picks cube
(5,17), which measures u=9.83e-4 — PASSES the gate (0.98x; fails the 0.95x
robust convention; the q18 neighbour is 8.8e-4 at +6% cost, a documented
robust alternative). The guard did its job: bare adequacy would have
accepted (5,16), which the initial sweep measured FAILING at 1.27e-3.
Auto's q-floor conservatism costs the wake at 1e6: (6,12) passes at
7.0e-4 and is 21% faster than auto's (6,16) — recorded as a per-case
tuning note (q floor 12 is cube-inadmissible at P=4, so the default floor
stays 16).

**Gap fills:** dense F64 cube (4,17) = 20.55 ms; dense at 1e6 (table
above; cube (4,16) F64 656.2, wake (6,16) F64 403.5); dense × K — flat to
<1% (K ∈ {256, 1024, 4096}), K=256 stands; RK3 full steps at winners:
cube 37.2/62.3 ms (F32/F64), wake 50.8/71.8 ms (≈ 3×U/J + ~1 ms).
Updated matched-n 030 ratio (n=1e5): F32 12.09/3.097 = **3.9x**, F64
20.57/3.389 = 6.1x.

**Post-cycle-1 profile (stage medians, ms):** cube (4,17) F32: b2m 3.6 /
m2m 1.05 / m2l 3.3 / l2l 1.04 / l2b+nearfield 7.7 (eval 11.7). wake
(5,16) F32: **b2m 11.4 (70% of eval 16.2)** / m2l 0.69 / l2b+nf 9.8;
wake F64: b2m 18.7 (80%). At 1e6 the fused nearfield dominates (cube 77%,
wake 95%); b2m is 3-8%. Dense-M2L stage vs concat: wake 11.1 → 0.69 ms
(16x), cube (4,17) 23.8 → 3.3 ms (7x).

**Proposed cycle 2 (awaiting user approval): B2M within-cell
parallelization** (FastMultipole `src/translate_batched_cuda.jl`).
Mechanism: `_launch_cuda_b2m!` currently assigns one thread per occupied
cell, serial over that cell's bodies — at σ-adequacy-forced shallow depths
(112-195 bodies/cell) the H200 runs a few hundred threads. Replace with a
block-per-cell scheme (bodies parallel across the block, per-coefficient
shared-memory reduction, φ+χ), or a warp-per-body-chunk variant, for both
`Point{Source}` and `Point{Vortex}` kernels. Expected gain (modeled):
wake 1e5 b2m 11.4 → ~1-2 ms ⇒ **~1.6-1.9x end-to-end wake U/J** (F64
~2.2x); cube 1e5 ~10-25%; 1e6 cases 3-8%. Risk: medium — new reduction
kernel; verified by existing B2M/device parity tests (P=4 and P=8, both
precisions, φ+χ) plus lifecycle accuracy gates and the scalar 028/030
no-regression harness (B2M is shared with the scalar path). Verification:
local + H200 before/after stage timings at the four winner configs, full
preflight suites, 033 refcheck.

STOPPED again per the cycle contract: cycle 2 requires explicit user
approval before any implementation.

### 2026-08-12 — cycle 2 approved, implemented, confirmed (job 13150961)

**User approval (2026-08-12, via coordinator):** cycle 2 as proposed, with
the graph-capture and 023-contract preservation requirements and the
mandatory scalar 028/030 no-regression gate. Implemented in FastMultipole
`src/translate_batched_cuda.jl` (commit `2938192`-series on `matrix-ops`):
`_cuda_b2m_leaf_nodes_kernel!` and `_cuda_b2m_vortex_leaf_nodes_kernel!`
rewritten block-per-cell — threads stride the cell's bodies inside the
(n, m) loop, a `CUDA_B2M_BLOCK = 128` shared-memory tree reduction
(`_cuda_b2m_block_reduce`) collapses the partials, thread 1 writes the
coefficient. Per-thread state stays tiny (unlike the task-028 rejected
(n, m)-striding variant, whose note is updated in place). Launch config
remains epoch-constant host data (`n_cells`), so graph-capture eligibility
and the 023 zero-allocation/counter contracts are unchanged; the one-shot
non-grid `_cuda_b2m_kernel!` path is untouched. Local: parse/load clean,
host suites unaffected (host B2M untouched), FLOWVPM Part A green.

**Job 13150961** (H200, exit 0): preflights green including the newly
added `cuda_radix_lifecycle_test.jl` (scalar device B2M parity) and the
032 interface tests (vortex device B2M parity, P=4/P=8, both precisions);
**033 refcheck PASSED at shipped defaults**; 8/8 after-rows ok, counters
flat, alloc unchanged. Data: `fm035_cycle2.csv`, `fm035_nr_*.csv`.

**Scalar 028/030 no-regression (mandatory gate) — PASSED:**

| config (n=1e6, ℓ=5, sched6-5-5-5) | verdict before → after | b2m stage | err_gradient_rel_rms |
|---|---|---|---|
| F32 + fp16 | 9.591 (030 rec.) / 6.58 (032a) → **6.898 ms** | 0.586 → 0.611 (+4%) | 1.05924e-3 → 1.05924e-3 (identical to 6 digits) |
| F64 | 20.740 (030) / 17.83 (032a) → **18.135 ms** | 1.020 → 1.184 (+0.16 ms) | 1.04972e-3 → 1.04972e-3 (identical) |

Verdicts are inside the established cross-job spread and better than the
rows of record; the small b2m stage cost at the deep scalar config (30
bodies/cell — reduction overhead ≈ serial work) is documented and far
below any end-to-end significance. Errors unchanged despite the reduction
reordering.

**Realized vs expected (U/J solve, cycle-1 → cycle-2 rows at identical
configs, all gate-passing, counters flat):**

| config | B2M stage | end-to-end | expected | realized |
|---|---|---|---|---|
| wake 1e5 (5,16) F32 auto | 11.39 → **0.27 ms (42x)** | 16.64 → **11.28 ms** | 1.6-1.9x | **1.48x** |
| wake 1e5 (5,12) F32 tuned | 11.4 → 0.27 | 16.19 → **9.77 ms** | — | **1.66x** ✓ |
| wake 1e5 F64 auto | 18.75 → 0.51 (37x) | 23.76 → 21.56 | ~2.2x | 1.10x |
| cube 1e5 (4,17) F32 auto | 3.63 → 0.40 (9x) | 12.09 → 12.15 | 1.1-1.25x | ~1.00x |
| cube 1e5 F64 auto | 6.21 → 0.70 | 20.57 → 20.63 | — | ~1.00x |
| cube 1e6 auto F32 | 4.94 → 3.03 | 149.3 → 151.2 | 1.03x | ~1.00x |
| wake 1e6 (6,12) F32 | 14.76 → 1.84 (8x) | 139.5 → 139.6 | 1.08x | ~1.00x |

**Why the shortfall where it fell short (root cause, recorded for the
method):** the per-stage medians are *isolated* launches, while the
production lifecycle overlaps the fused nearfield with the far-field chain
(the known `eval < Σ stages` gain). B2M was therefore on the critical path
only where its isolated time exceeded the concurrent nearfield chain —
the wake F32 rows. On the cube and at 1e6 the nearfield chain already
covered B2M entirely, and on the wake F64 the 19.8 ms nearfield became
the new wall the moment B2M dropped. The stage collapse is real
(9-42x) and future-proofs the σ-forced shallow-tree regime; the
end-to-end model over-credited it by ignoring overlap.

**RK3 full steps after cycle 2:** wake F32 50.8 → **34.4 ms**; wake F64
71.8 → 66.9; cube F32 37.3, F64 62.4 (unchanged, as expected).

**Post-cycle-2 winners (n=1e5):** cube 12.15/20.63 ms (F32/F64, auto);
wake **9.77 ms** (5,12) tuned / 11.28 auto F32, 21.56 F64. vs the shipped
034 coupling: cube **4.2x/4.6x**, wake **3.7x tuned (3.2x auto) / 2.6x**.
030 matched-n ratio (n=1e5 F32): 12.15/3.097 = 3.9x cube, 9.77/3.097 =
3.2x wake. n=1e6: cube 151.2, wake 139.6/177.2 (tuned/auto) — unchanged.

**Post-cycle-2 profile and remaining levers:** the fused nearfield now
dominates every configuration (wake 1e5 F32: 8.2 of 9.3 ms eval = 88%;
cube 66%; 1e6 77-95%), and it was itself just optimized by 032a
(classsplit + sub-Morton shipped; the alternatives measured as losses).
Direct-work reduction through depth is σ-adequacy- and accuracy-capped
(measured, not modeled: cube 1e6 ℓ5 q16 fails the gate; ℓ+1 at 1e5 needs
unsupported q > 20). Remaining candidates against the 5% bar:

- **wake auto q-floor** (q12 vs 16): 13-21% on the wake, 0% on the cube,
  and q≤14 *fails* the gate on the cube — field-dependent accuracy, so
  not shippable as a geometry-only default; recorded as per-case tuning
  guidance (available via `radix_fmm_settings!`).
- **037 rectangular grid — verdict UPDATE:** with the wake winner at
  9.77 ms, the ~2 spare transverse coarse levels (~0.9-1.8 ms of
  M2M/L2L/launch floor) are now **9-18% of the wake U/J solve at n=1e5**
  — the caveat recorded at the initial verdict has materialized, and 037
  now *clears* the 5% bar at the 1e5 scale (still <2% at 1e6, where the
  nearfield dwarfs it). The go/no-go recommendation passes to `036`/`037`
  with this estimate.
- Everything else measured under 5% (K, rho_t, schedules, precomputed_y,
  FLOWVPM overhead, B2M residual).

**Campaign recommendation: conclude the optimization cycles.** No
credible untried lever within 035's scope (coupling defaults, tuning,
low-risk kernels) retains a defensible ≥5% expected end-to-end gain; the
one structural lever above the bar (rectangular radix grid) is explicitly
owned by follow-on row `037`. Next step on user go-ahead: write the
definitive 035 report (§3 of this file: final tables, per-stage profiles,
033-baseline speedup policy — noting every FMM-active 033 CPU row fails
the 1e-3 gate, so headline speedups vs CPU are restricted to n ≤ 3162 —
the 030 ratio, figures per the 024a conventions, and the cycle ledger).

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
