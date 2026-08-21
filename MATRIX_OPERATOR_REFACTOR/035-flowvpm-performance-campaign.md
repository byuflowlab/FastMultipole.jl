# 035 FLOWVPM Performance Campaign

## Status and Entry Gate

**Added by user request on `2026-08-04`; consolidated by roadmap review on
`2026-08-05`.** **DONE `2026-08-12`** (optimization cycles 1, 2, 3A-3D
complete and confirmed; definitive Final Report in this file). Clear-context
approval pending.

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

### 2026-08-12 — cycle 3A approved: measurement-first P/cutoff/stencil co-design

**User direction:** do not conclude before measuring one remaining coupled
lever. Literature `P=4` means FastMultipole `expansion_order=3`; cycle 3A
therefore records both conventions explicitly and tests literature `P=4/5/6`
as `expansion_order=3/4/5`. This is a measurement-only cycle: no production
kernel or default changes are authorized by this approval.

The missed interaction is expansion order × smoothing cutoff × direct-stencil
radius. The fixed-order campaign used the shipped `rho_t=4.252`, derived in
`031a` from the Jacobian RMS target, while this task's winner gate is sampled
velocity RMS ≤1e-3 and Jacobian RMS is diagnostic. The theory's corresponding
velocity-RMS cutoff is `rho_t=3.668`. Reducing the cutoff also relaxes the
exact-coverage gate; increasing `P` may recover FMM truncation accuracy at a
smaller direct shell. At the post-cycle-2 profile this clears the 5% expected
gain bar: cube `q=17 -> 14/12` and wake `q=12 -> 9/8/6` remove material direct
work from a stage carrying 66–95% of the solve, while the wake's current M2L is
only 0.7/4.6 ms at n=1e5/1e6.

**Pre-registered first-pass grid:**
`scripts/fm035_cases_cycle3a.txt`, 33 Float32 rows across cube/wake × n=1e5/1e6.
It repeats four cycle-2 anchors in the same job; tests
`expansion_order=3/4/5`, `rho_t=4.252/3.668`, the geometry-admissible shells
above, and dense/concat/precomputed-y spot comparisons at higher order. Every
row profiles the U/J solve. Float64 and full RK3 are intentionally deferred to
a small confirmation grid around any velocity-gate-passing winner. The harness
now accepts and records `expansion_order` plus `literature_P`; old case files
retain `expansion_order=3` by default, and an output-header check prevents
accidental mixed-schema append.

Local dry-run against the actual sibling FLOWVPM checkout and this
FastMultipole checkout: **33/33 configurations parse and resolve** with the
intended kernel, M2L strategy, cutoff, and order mapping. Geometry arithmetic
was checked before submission: the tight candidates remain strictly above the
production coverage gate (`g_min*h_leaf > rho_t*sigma_max`), notably wake
`q=6, rho_t=3.668` and cube `q=12, rho_t=3.668`. Accuracy is deliberately not
predicted; the checksummed 033 references decide eligibility on H200.

**Cycle 3A result — positive (job 13154216, H200 m13h-1-2, 18m57s,
exit 0):** all preflights green (032 interface 1246/1246, scalar lifecycle
216/216 + concat 37/37, host interface/032a suites including 32367 binned-pair
assertions, CUDA binning 304/304, FLOWVPM Part A+B), reference checksums and
the shipped-default 033 refcheck passed. All **33/33** rows completed, 25 pass
the velocity gate, every row has flat 023 counters and unchanged small
framework allocation. Data: `data/flowvpm_gpu_campaign/fm035_cycle3a.csv` and
`fm035-13154216.out`.

The winner in all four case/scale groups is literature **P=5**
(`expansion_order=4`), `rho_t=3.668`, dense M2L, with `q=12` cube / `q=6`
wake. Same-job anchor → winner:

| case | n | P4 anchor → P5 winner (ms) | speedup | u_rel_rms | J diagnostic |
|---|---:|---:|---:|---:|---:|
| cube | 1e5 | 12.180 → **11.520** | 1.06x (5.4%) | 6.81e-4 | 3.88e-3 |
| cube | 1e6 | 151.097 → **102.414** | 1.48x | 7.08e-4 | 3.85e-3 |
| wake | 1e5 | 9.782 → **7.989** | 1.22x | 3.30e-4 | 2.45e-3 |
| wake | 1e6 | 139.780 → **83.702** | 1.67x | 2.99e-4 | 2.87e-3 |

The mechanism matches the hypothesis: direct pairs fall 37–41%; isolated
nearfield falls from 7.74→5.45, 115.13→65.70, 8.22→5.41, and
132.63→73.83 ms respectively. The added P5 far-field cost is small enough to
retain the gain (cube 1e5 M2L 3.28→4.43 ms; wake 1e5 0.66→1.27 ms), and at
1e6 the smaller route set offsets the higher order. P4 at the smaller shells
fails the velocity gate; P5 recovers it with wide margin. P6 passes but is
slower. Dense remains decisively best at P5: concat is 1.31–2.49x and
precomputed-y 1.36x slower where sampled.

**Pre-registered confirmation (cycle 3B):**
`scripts/fm035_cases_cycle3b.txt`, 12 rows. Confirm P5 winners against same-job
P4 anchors in Float64 at all four points, repeat F32 anchor/winner at n=1e5,
and record full RK3 there. No production default change is included; that
decision follows only if 3B confirms the gain and all gates.

**Cycle 3B result (job 13157488, H200 m13h-1-2, 17m03s, exit 0):** all
preflights and reference checks passed; 12/12 rows completed with flat counters.
The P5 error improvement reproduced in both precisions. P5 retained strong
speedups on the wake and at n=1e6, while cube n=1e5 Float64 was a small
performance loss and is not presented as a winner:

| case | n | TF | P4 anchor (ms) | P5 (ms) | P5/P4 speedup | P5 U RMS | P5 J diagnostic |
|---|---:|---|---:|---:|---:|---:|---:|
| cube | 1e5 | F32 | 12.175 | 11.503 | 1.06x | 6.8061e-4 | 3.8824e-3 |
| cube | 1e5 | F64 | 20.683 | 21.148 | 0.98x | 6.8059e-4 | 3.8824e-3 |
| cube | 1e6 | F64 | 310.203 | 207.779 | 1.49x | 7.0780e-4 | 3.8518e-3 |
| wake | 1e5 | F32 | 9.780 | 7.972 | 1.23x | 3.2988e-4 | 2.4548e-3 |
| wake | 1e5 | F64 | 19.013 | 14.548 | 1.31x | 3.2992e-4 | 2.4548e-3 |
| wake | 1e6 | F64 | 303.131 | 179.081 | 1.69x | 2.9898e-4 | 2.8695e-3 |

At n=1e5, full RK3 was 37.216 -> 35.170 ms (cube F32), 62.841 ->
64.587 ms (cube F64), 29.940 -> 24.542 ms (wake F32), and 60.143 ->
44.202 ms (wake F64). Data: `fm035_cycle3b.csv`, sha256
`a401639e4cc93740afcd600f9faf5a0e1131d99aeadeb3c82395c4c16d2d873b`.

### 2026-08-12 — cycle 3C preregistration: cutoff/FMM decomposition and NCU

The user identified that a higher expansion order cannot restore accuracy lost
by replacing a regularized interaction with singular `1/r` math. Cycle 3C
therefore separates the errors before considering any P5 default. The exact
field to which `PartitionedVortex` converges is the partitioned field `P`, not
the globally singular field: close direct pairs remain regularized and only
pairs beyond `rho_t` use singular math. On the checksummed 033 sample targets,
measure the identity `F-R = (P-R) + (F-P)` using an independent host Float64
erf oracle. The initially preregistered gate assigned `5e-4` separately to
`||P-R||/||R||` and `||F-P||/||R||`; their sum is the worst-case `1e-3`
triangle bound. Report J by the
same decomposition as a diagnostic only. Do not form the globally singular
field. Compare all P4 anchors (`expansion_order=3`, `rho_t=4.252`) with all P5
winners (`expansion_order=4`, `rho_t=3.668`) for cube/wake, n=1e5/1e6, and
Float32/Float64.

After the decomposition job passes, profile the isolated warmed P5 nearfield
launch with Nsight Compute 2025.1.1 on H200 for cube/wake at n=1e6 in both
precisions. Graph capture and stream overlap are disabled only in the profiling
driver. Collect the detailed set for cell-sigma/scalar setup, class binning,
and each singular/regularized/mixed bucket kernel, retaining `.ncu-rep` and raw
CSV exports. Diagnose compute, memory, or latency/occupancy limitation from
SOL/roofline, occupancy, cache/DRAM, instruction, branch, and warp-stall
counters. No production code/default change is part of cycle 3C.

**Cycle 3C result (decomposition job 13157744, H200 m13h-1-1, 3m47s,
exit 0):** the independent host Float64 oracle reproduced every checksummed
033 regularized reference at 1.6e-16--3.0e-15 relative error. The exact
three-field identity residual was zero at reported precision. Velocity results
below are precision-insensitive (Float32 shown; Float64 differs in the last
digits):

| config | case | n | cutoff `||P-R||/||R||` | FMM `||F-P||/||R||` | observed total | triangle bound | governing gate |
|---|---|---:|---:|---:|---:|---:|---|
| P4/rho=4.252 | cube | 1e5 | 4.91e-5 | 9.35e-4 | 9.35e-4 | 9.84e-4 | **pass** |
| P5/rho=3.668 | cube | 1e5 | 5.56e-4 | 4.05e-4 | 6.81e-4 | 9.61e-4 | **pass** |
| P4/rho=4.252 | cube | 1e6 | 4.98e-5 | 9.79e-4 | 9.83e-4 | 1.03e-3 | **fail** |
| P5/rho=3.668 | cube | 1e6 | 5.14e-4 | 4.73e-4 | 7.08e-4 | 9.88e-4 | **pass** |
| P4/rho=4.252 | wake | 1e5 | 1.05e-5 | 6.95e-4 | 6.94e-4 | 7.05e-4 | **pass** |
| P5/rho=3.668 | wake | 1e5 | 1.05e-4 | 2.89e-4 | 3.30e-4 | 3.94e-4 | **pass** |
| P4/rho=4.252 | wake | 1e6 | 3.69e-6 | 7.02e-4 | 7.02e-4 | 7.06e-4 | **pass** |
| P5/rho=3.668 | wake | 1e6 | 3.61e-5 | 2.89e-4 | 2.99e-4 | 3.25e-4 | **pass** |

This confirms the user's concern. Higher P reduces `F-P`; it does not repair
`P-R`. **Post-result policy revision by user direction:** the governing gate is
the conservative sum
`(||P-R|| + ||F-P||)/||R|| < 1e-3`, not a fixed 50/50 allocation between
components. This policy still makes no use of cancellation. Under it, P5 at
`rho_t=3.668` is validated for both cases and scales: the cube bounds are
9.61e-4 at n=1e5 and 9.88e-4 at n=1e6 (the latter has 1.2% margin), while the
wake bounds are 3.94e-4 and 3.25e-4. The P4 cube n=1e6 anchor fails this
stronger bound at 1.03e-3 despite its observed sampled error passing. Jacobian
remains diagnostic: P5 cutoff/FMM/total are 2.74e-3/2.69e-3/3.88e-3 (cube
1e5), 2.73e-3/2.71e-3/3.85e-3 (cube 1e6), 1.24e-3/2.16e-3/2.45e-3 (wake
1e5), and 1.56e-3/2.39e-3/2.87e-3 (wake 1e6). Data:
`fm035_error_decomposition.csv`.

**Nsight Compute attempt (array 13157746, four P5 n=1e6 cases):** all four
drivers built and reached the intended isolated nearfield launch (cube direct
list 3,414,506; wake 401,888), but Nsight Compute rejected metric collection
with `ERR_NVGPUCTRPERM`: unprivileged jobs on this H200 partition cannot access
NVIDIA performance counters. No `.ncu-rep` was produced. Per preregistration,
no memory-/compute-bound verdict is inferred from timing alone. The committed
driver is ready to rerun unchanged when Orc enables counter access; the four
job logs preserve the blocker and exact requested configurations.

### 2026-08-12 — cycle 3D: ship the P5/rho_t=3.668 defaults (session 2)

**User approval (2026-08-12, plan-mode approval in session 2):** adopt the
cycle-3A/3B/3C-validated literature-P5 winner as the shipped FLOWVPM coupling
default; profiling continues by counter-free means only (nsys timeline +
analytic roofline from pair counts — `@time`-style additions add nothing over
the existing CUDA-event stage medians and cannot classify bound-ness; NCU
remains recorded as blocked-external).

**Implemented** (FLOWVPM `gpu-full` commit `5dd0d85`, coupling settings only,
no FastMultipole `src/` change): `RadixFMMSettings` defaults
`expansion_order=4` (literature P5; `nothing` still derives `pfield.fmm.p-1`),
partitioned-kernel coupling default `rho_t=3.668` (031a velocity-RMS cutoff;
the constructor default 4.252 and the :regularized/:twopass kernel defaults
are untouched), `near_radius2` floor `6`, `accuracy_margin=1.03`. The margin
was derived numerically: the joint auto rule reproduces all four cycle-3A P5
winners — cube `(4,12)`/`(5,12)`, wake `(5,6)`/`(6,6)` at n=1e5/1e6 — for any
margin in `[0.886, 1.061]`; margins below 1.0 are excluded (FastMultipole
enforces bare adequacy), so 1.03 is the center of the admissible `[1.0,
1.061]`. Justification is accuracy as well as speed: under the revised
conservative sum gate the old P4/4.252 default *fails* at cube n=1e6
(1.03e-3); the accepted cost is cube 1e5 F64 at 0.98x (3B). Part A extended:
new-default assertions, all four winner reproductions, plus the cycle-1 rule
regression at the old explicit settings; suite green locally (41 assertions,
4 threads, host path).

**Pre-registered confirmation** (`scripts/fm035_cases_cycle3d.txt`, 16 rows,
committed before the cluster run): shipped-auto rows (ell omitted, rho_t
omitted → 3.668, q=6 floor) vs same-job P4 anchors at the previous shipped
geometry, all four case/scale points × Float32/Float64, RK3 at n=1e5,
profile on every row; local dryrun 16/16 with the intended kernel/cutoff/
order resolution. The job's 033 refcheck stage now gates the *new* defaults,
including the previously unmeasured 1e4-scale auto selections (cube `(3,14)`,
wake `(4,9)` — derived locally). Profiling rider (`FM035_NSYS=1`):
`profile_035_nsys.jl` records a production-timeline (graph capture + overlap
ON) five-solve nsys trace per case at n=1e6 in both precisions, plus the
exact nearfield body-pair total (Σ |tgt|·|src| over direct routes) for the
counter-free analytic roofline against H200 peaks.

**Cycle 3D result — CONFIRMED (job 13157887, H200 m13h-1-2, 24m13s, exit
0):** all preflights green on hardware (032 interface, scalar lifecycle, host
032a suites, CUDA binning, FLOWVPM Part A with the new-default assertions +
Part B), **033 refcheck PASSED at the new shipped defaults** (cube/wake ×
1e4/1e5 × F64/F32 — this gates the previously unmeasured 1e4 auto
selections), 16/16 rows ok, counters flat on every row, steady-state
allocation ≤ 11 KB. Every auto row selected exactly the locally derived
geometry (cube (4,12)/(5,12), wake (5,6)/(6,6)) and reproduced the 3A/3B
errors to 4-5 digits. Data: `fm035_cycle3d.csv`, sha256
`69e5790beea06a96ad80ea14635ac2eab93f91abe84808853cafb99a8cb52c1b`.

**Shipped-default U/J solve, same-job P4 anchor (previous shipped geometry) →
P5 auto (all gate-passing):**

| case | n | TF | anchor (ms) | auto (ms) | speedup | auto u_rel_rms |
|---|---:|---|---:|---:|---:|---:|
| cube | 1e5 | F32 | 12.187 | **11.524** | 1.06x | 6.81e-4 |
| cube | 1e5 | F64 | 20.654 | 21.132 | 0.98x (accepted cost) | 6.81e-4 |
| cube | 1e6 | F32 | 151.063 | **102.348** | 1.48x | 7.08e-4 |
| cube | 1e6 | F64 | 309.954 | **207.937** | 1.49x | 7.08e-4 |
| wake | 1e5 | F32 | 11.312 | **7.976** | 1.42x | 3.30e-4 |
| wake | 1e5 | F64 | 21.645 | **14.562** | 1.49x | 3.30e-4 |
| wake | 1e6 | F32 | 177.233 | **83.686** | 2.12x | 2.99e-4 |
| wake | 1e6 | F64 | 398.735 | **178.999** | 2.23x | 2.99e-4 |

RK3 full steps at n=1e5: cube 37.48→35.20 (F32) / 62.73→64.47 (F64); wake
34.54→24.54 (F32) / 67.07→44.30 (F64) ms. The wake speedups exceed the 3B
row-vs-row numbers because the shipped-default anchor is the auto (5,16)/(6,16)
geometry, not the hand-tuned (5,12)/(6,12) rows. Errors improve in every
case/scale. Cumulative vs the shipped 034 coupling at n=1e5 F32: cube
51.4→11.52 (**4.5x**), wake 36.6→7.98 (**4.6x**).

**Profiling rider:** nsys production timelines captured for cube/wake ×
F32/F64 at n=1e6 (5 solves each, graph capture + overlap ON) plus exact
nearfield body-pair totals: cube 3,414,506 direct routes / **8.059e9 body
pairs** (mean occupied cell 30.5); wake 401,888 routes / **13.416e9 body
pairs** (mean cell 3.81, max 261 — pairs concentrate in the large cells).
First capture (13157887) hid graph-internal kernels (nsys default
graph-level trace ⇒ only ~8 ms of non-graph kernels visible); rerun with
`--cuda-graph-trace=node` submitted as job 13157931 (preflights off, sweep
resume no-ops).

### 2026-08-12 — counter-free bound-ness analysis (jobs 13157887/13157931)

Job 13157931 (H200 m13h-1-1, exit 0) repeated the four nsys captures with
`--cuda-graph-trace=node` after the first capture's default graph-level trace
hid every graph-internal kernel. Method: exact per-pair operation counts from
a kernel-source audit (warp-per-cell-pair scheme, no shared-memory tiling,
erf-free Horner-series g/h, 12 atomics per target-instant; singular ≈63/72
ops/pair F32/F64, regularized ≈100/122, div/rsqrt counted 4, FMA counted 1),
exact body-pair totals from the direct route list (cube 1e6: 8.059e9 pairs;
wake 1e6: 13.416e9), and node-level kernel times from the production traces.

**Findings (n=1e6, five production solves per case):**

1. **The GPU is saturated — not launch- or latency-bound.** Total kernel time
   per solve ≈ solve wall time in every case (cube F32 105.1 vs 102.3 ms;
   wake F32 88.6 vs 83.7; cube F64 213.0 vs 207.9; wake F64 187.8 vs 179.0).
   The ~50 us/window launch story of 027 does not reappear at these scales.
2. **The mixed (PartitionedVortex) bucket is the nearfield.** Of nearfield
   kernel time: cube F32 50.3 (mixed) + 10.3 (singular) + 0.7 (regularized)
   ms; wake F32 72.3 ms is 82% mixed with singular/regularized not in the
   top six kernels (at q=6 nearly every route straddles rho_t). The dense
   tiled M2L is the only other material kernel (cube 29.0/53.9 ms F32/F64;
   wake 4.9/8.9 ms).
3. **Compute-bound at 39-60% of the vector-op ceiling.** Pair rate ÷ peak
   op rate (H200 ≈33.5 Top/s F32, ≈17 F64, FMA=1 counting): cube F32
   131 Gpair/s ≈ 39% of ceiling at the regularized op count; wake F32
   186 Gpair/s ≈ 55%; cube F64 ≈ 38%; wake F64 83 Gpair/s ≈ 60%. DRAM
   traffic is 2-12% of the 4.8 TB/s peak (the naive no-reuse estimate would
   exceed peak — warp-broadcast reuse is operative), so bandwidth is not the
   limit. The residual gap to the ceiling is structural to the scheme:
   predicated dual-path retirement inside mixed warp instants, ragged
   n_t mod 32 tails (wake mean occupied cell 3.8, max 261), and the 12-atomic
   flush. NCU counter confirmation remains blocked (ERR_NVGPUCTRPERM);
   this classification rests on the op-count model, not hardware counters.

**Lever consequence: no credible ≥5% end-to-end lever remains in 035 scope.**
Raising nearfield utilization further means a kernel redesign (pair-parallel
with segmented reduction, class-sorted warp instants) — medium-high risk with
uncertain gain against a kernel already at 39-60% of ceiling, in the space
where 029's nearfield-ILP lever was falsified at +1.9%. The M2L strategy
space is measured-exhausted at P5 (dense beats concat 1.31-2.49x and
precomputed-y 1.36x; the FP16-WMMA tensor path is structurally unavailable
under Lamb-Helmholtz). The one structural lever above the bar — the
rectangular radix grid, now 11-23% of the 7.98 ms wake 1e5 solve — is owned
by row `037`. Optimization cycles are concluded.

## Final Report (§3, definitive — 2026-08-12)

Measurement policy: H200 (m13h-1-x), julia 1.11.7, CUDA 12.8; every timing is
the **median of 15 warmed repetitions after 2 warmup solves** (minima also
recorded in the CSVs); stage times are isolated CUDA-event medians (the
production solve overlaps the nearfield with the far-field chain, so eval <
Σ stages); every reported configuration passes sampled velocity RMS ≤ 1e-3
against the sha256-checksummed 033 direct references, with flat 023 counters
and zero recurring body traffic. Jacobian RMS is diagnostic throughout.

### 1. Final shipped configurations and timings

Shipped FLOWVPM coupling defaults (gpu-full `5dd0d85`): literature P5
(`expansion_order=4`), `PartitionedVortex` at `rho_t=3.668`, dense M2L,
K=256, joint auto-geometry (floor q=6, margin 1.03). Auto-selected
geometries: cube (ℓ4,q12)/(ℓ5,q12), wake (ℓ5,q6)/(ℓ6,q6) at n=1e5/1e6.

| case | n | TF | U/J solve (ms) | RK3 step (ms) | u_rel_rms | J diag |
|---|---:|---|---:|---:|---:|---:|
| cube | 1e5 | F32 | **11.52** | 35.20 | 6.81e-4 | 3.88e-3 |
| cube | 1e5 | F64 | **21.13** | 64.47 | 6.81e-4 | 3.88e-3 |
| cube | 1e6 | F32 | **102.35** | — | 7.08e-4 | 3.85e-3 |
| cube | 1e6 | F64 | **207.94** | — | 7.08e-4 | 3.85e-3 |
| wake | 1e5 | F32 | **7.98** | 24.54 | 3.30e-4 | 2.45e-3 |
| wake | 1e5 | F64 | **14.56** | 44.30 | 3.30e-4 | 2.45e-3 |
| wake | 1e6 | F32 | **83.69** | — | 2.99e-4 | 2.87e-3 |
| wake | 1e6 | F64 | **179.00** | — | 2.99e-4 | 2.87e-3 |

RK3 ≈ 3 × U/J + ≤1 ms integrator overhead throughout. Versus the shipped 034
coupling (figure `fig09_035_cycle_ladder`): cube 1e5 51.4→11.52 (**4.5x**),
wake 1e5 36.6→7.98 (**4.6x**), cube 1e6 427.1→102.35 (**4.2x**), wake 1e6
282.8→83.69 (**3.4x**), all F32; F64 1e5: cube 94.6→21.13 (**4.5x**), wake
57.1→14.56 (**3.9x**).

### 2. Speedups vs 033 CPU baselines (eligibility policy)

Every FMM-active default-parameter 033 CPU row **fails** the 1e-3 velocity
gate (cube u=1.07e-2 at n=1e4 rising to 6.75e-2 at 1e6; wake 8.3e-3 at 1e4
to 6.39e-2 at 1e6); the gate-passing CPU rows exist only at cube n ≤ 3162
and wake n = 1e3, all below the campaign's measured GPU range (n ≥ 1e4).
**No eligible matched-n CPU/GPU speedup pairing exists, so no CPU-baseline
speedup is headlined.** The failing historical timings are preserved for
context only (single-thread / 64-thread U/J seconds, with their errors):
cube 1e5 113.7 / 2.51 s (u=4.6e-2), cube 1e6 1776.6 / 56.0 s (6.8e-2), wake
1e5 102.1 / 2.34 s (3.2e-2), wake 1e6 1851.7 / 31.6 s (6.4e-2). The
gate-passing small-n rows (cube 1e3 0.057 s, cube 3162 0.554 s, wake 1e3
0.052 s; identical cpu1/cpu64) are all-direct-regime solves without a
matched GPU measurement; running one would measure launch floor, not the
FMM. Per the campaign contract no replacement tuned CPU campaign was run.

### 3. Per-stage profiles and bottleneck movement

Figure `fig10_035_stage_movement` (isolated CUDA-event medians, F32) and the
node-level traces (13157931) tell one story: the shipped-034 coupling was
B2M- and M2L-bound at 1e5 (cube ℓ3: B2M 21.8 of 41.8 ms eval; wake: B2M
11.4 of 25.9); cycles 1-2 removed both (dense M2L 7-16x stage gain;
block-per-cell B2M 9-42x), leaving the fused nearfield dominant everywhere
(66-95% of eval); cycle 3D shrank that nearfield 29-56% by trading direct
work for far-field work at P5 (cube 1e6 NF+L2B 115.2→65.7 ms, wake 1e6
166.8→74.0 ms, isolated). Post-3D the nearfield remains 47-88% of solve
kernel time; its bound-ness classification (compute-bound at 39-60% of the
op ceiling, GPU saturated, DRAM 2-12%) is in the bound-ness analysis above.
The eligible CPU baselines (033 profiles) are ~99.5% nearfield-bound
(custom_erf alone 36.7%), so the bottleneck on both sides is the
regularized nearfield; the GPU's answer is the classsplit mixed-bucket
kernel plus the P5/3.668 shell reduction.

### 4. Matched-n 030 ratio (n=1e5)

030 best admissible verdict-boundary (scalar workload, per-n retuned):
3.097 ms (FP16-WMMA/F32) / 3.389 ms (F64). Final FLOWVPM U/J solves are
**3.7x / 2.6x** those (cube/wake F32: 11.52/7.98 over 3.097) and **6.2x /
4.3x** (F64: 21.13/14.56 over 3.389). Itemized workload differences (not a
pass/fail target): Lamb-Helmholtz χ carried end-to-end (double expansion
channel; structurally disables the FP16-WMMA tensor path), vector strength
(3 B2M source channels), 13-row output incl. the 9-component hessian (vs
4-row), σ-carrying regularized nearfield (mixed-bucket kernel ≈1.6x the
singular per-pair cost, plus σ coverage forcing larger near sets), and
~0.1 ms FLOWVPM-side reset/dispatch. RK3 full steps are reported in §1 and
deliberately excluded from this ratio.

### 5. Cycle ledger and closing lever list

Implemented (all user-approved, all confirmed on H200 with realized-vs-
expected recorded in this file): **cycle 1** dense M2L + PartitionedVortex +
joint auto-geometry as coupling defaults (4.25x/2.2-2.9x); **cycle 2**
block-per-cell CUDA B2M (stage 9-42x; wake 1.48-1.66x e2e; scalar
no-regression PASSED); **cycle 3A/3B** measurement-only P/cutoff/stencil
co-design (P5 winner found); **cycle 3C** cutoff/FMM error decomposition
(erf-oracle; conservative sum gate adopted; P4 default shown out-of-gate at
cube 1e6); **cycle 3D** P5/3.668 shipped as defaults (1.06-2.23x further;
errors improve everywhere; cube 1e5 F64 0.98x accepted).

Rejected by measurement (sub-5% or losses): window_classes K∈{64,1024,4096}
(dense is K-flat to <1%; K=64 a 1.4-1.9x loss under concat), rho_t=4.789
(≤0.3%), boosted-coarse level schedules (4-14% worse), precomputed_y M2L
(loses to dense at P4 and P5), FLOWVPM-side overhead (≤1-3% total), B2M
residual (≤0.5 ms), nearfield micro-optimization (bound-ness analysis;
029-P3 precedent falsified at +1.9%), wake q-floor 12→6 special-casing
(superseded by cycle 3D, which ships q=6 with P5 accuracy).

Remaining above the bar, explicitly handed off: **037 rectangular radix
grid** — the wake's ~2 spare transverse coarse levels cost ~0.9-1.8 ms of
M2M/L2L/launch floor = **11-23% of the final 7.98 ms wake 1e5 solve** (<2%
at 1e6, where the nearfield dwarfs it); the go recommendation transfers to
`036`/`037` with this estimate. Deferred-external: NCU counter profiling
(ERR_NVGPUCTRPERM on unprivileged H200 jobs; `profile_035_nearfield_ncu.jl`
and the four job logs 13157746_0-3 stand ready unchanged).

Figures: `data/figures/fig09_035_cycle_ladder`, `fig10_035_stage_movement`,
`fig11_035_error_decomposition` (024a conventions; tables regenerated by
`scripts/figures_035_prepare.jl`, compiled with pdflatex). Data of record:
`fm035_sweep.csv`, `fm035_cycle{1,2,3a,3b,3d}.csv`,
`fm035_error_decomposition.csv` (sha256 in the work record), nsys artifacts
`fm035_nsys_*_13157931.*`.

**Task 035 status: DONE.** Optimization cycles concluded; this section is
the phase's sole definitive speedup report. Awaiting clear-context approval.

## Addendum — 037 rectangular-grid verdict (2026-08-13, post-approval)

Recorded here per 037's contract ("the final decision and comparison belong
in 035"); this addendum does not reopen the approved campaign. The
rectangular radix grid (implemented and validated in `037`, five stages)
was benchmarked cubic-vs-rectangular at the shipped defaults with identical
(ℓ, q, leaf width) per arm (job 13160439): wake 1e5 +2.2/+3.4% (slower,
F32/F64), wake 1e6 −1.7/−2.0%, cube neutral — the Deliverable-5a/
post-cycle-2 estimate of 11-23% at wake 1e5 did **not** materialize. The
trimmed coarse-level cost was real in isolated stage timings (M2M+L2L
−0.5 ms) but sat under the production nearfield overlap rather than on the
critical path — the same overlap over-crediting this campaign documented
for its cycle-2 B2M model; additionally the AR=5 wake trims only 1-2
levels (transverse extent needs ℓ−2), and the flat-top root adds ~0.2 ms
of M2L at 1e5. Per the user's pre-authorized criteria (within 10%
everywhere: PASS; >10% on high-aspect: FAIL), the shipped default remains
cubic; rectangular is a validated opt-in
(`radix_fmm_settings!(pfield; rectangular=true)`) expected to pay at
aspect ratios well beyond 5. Data: `fm037_stage5.csv`, full record in
`037-impl-rectangular-isotropic-radix-grid.md`.

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

## Clear-Context Approval

**Date:** 2026-08-12. **Reviewing agent:** clear-context review subagent
(Claude Fable 5), per the `START_HERE.md` clear-context protocol.

**Checked:** `START_HERE.md`, this task file in full, the campaign CSVs and
job logs in `data/flowvpm_gpu_campaign/`, FLOWVPM `gpu-full` commits
`4d188fa`/`bc7df09`/`5dd0d85` (shipped `RadixFMMSettings` defaults and
`_radix_auto_geometry` in `src/FLOWVPM_fmm_radix.jl`, plus the
`test/runtests_gpu_fmm.jl` default/winner assertions), the FastMultipole
cycle-2 B2M rewrite in `src/translate_batched_cuda.jl`, and the three 035
figures with their tables.

**Verified quantitatively:** `fm035_cycle3d.csv` sha256 matches the record
(`69e5790b…`), as does `fm035_cycle3b.csv` (`a401639e…`); every Final-Report
§1 timing/error/RK3/speedup number matches the cycle-3D CSV, with
`gate_pass=true` and `counters_flat=true` on all 16 rows and per-step
allocation ≤ 11 KB; the job-13157887 log shows all preflights green and the
033 refcheck PASSED at the new shipped defaults (checksummed references,
worst F64 u_rel_rms 6.81e-4 at the reviewed scales); the cycle-3C error
decomposition table matches `fm035_error_decomposition.csv` on all 8 rows
(P4 cube 1e6 triangle bound 1.03e-3 fail confirmed); the auto-geometry
margin claim was reproduced independently (winner-reproducing interval
exactly [0.886, 1.061]; shipped 1.03 valid, tests assert all four winners
plus the cycle-1 rule regression); shipped defaults in code match the
claims (`expansion_order=4`, `rho_t=3.668` partitioned coupling default
with constructor 4.252 untouched, `near_radius2=6`, `accuracy_margin=1.03`,
`:dense`, `:partitioned`); the cycle-2 kernel diff is scope-clean (leaf
scalar+vortex kernels only, `threads=CUDA_B2M_BLOCK`/`blocks=ncell`
epoch-constant, one-shot path and defaults untouched, uniform control flow
around the shared-memory reduction) and its scalar no-regression verdicts
(6.898/18.135 ms, errors bit-identical) appear in the 13150961 log;
`figures_035_prepare.jl` regenerates all committed tables with zero diff
and all three `.tex` compile under pdflatex; nsys body-pair counts
(8.059e9 cube / 13.416e9 wake) match the logs. 035 commits touch only the
radix coupling file and GPU tests on the FLOWVPM side; default changes had
recorded user approvals (cycles 1, 2, 3A, 3D).

**Minor non-blocking notes:** (1) the cycle-2 record cites FastMultipole
commit "`2938192`-series" — the actual commit is `9412600` (closeout
`d871a19`); (2) the cycle-3A phrase "concat 1.31–2.49x, precomputed-y
1.36x slower" understates the measured ranges (concat 1.25–2.74x,
precomputed-y 1.36–1.95x across the six sampled rows) — dense remains
decisively best everywhere, so no conclusion changes; (3) figure build
artifacts (`.pdf`/`.aux`/`.log`) in `data/figures/` are untracked but not
gitignored.

**Verdict: APPROVED.** Work is consistent with the stated objectives,
correct on every spot-checked artifact, and minimally invasive; all
verification gates were run and passed on hardware.
