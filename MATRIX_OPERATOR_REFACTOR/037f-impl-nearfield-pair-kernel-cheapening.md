# 037f Nearfield Pair-Kernel Cheapening

## Status and Entry Gate

**Staged by user request 2026-08-14; not started.**

Entry gate: `037b` Done and Approved (met 2026-08-14). Does not gate `038`
and may proceed in parallel with it. Coordinated with `037e` (one effort,
adjacent surfaces); neither blocks the other.

## Motivation

The regularized `gaussianerf` U+J pair kernel is evaluated ~10^3 times per
body (the accuracy-pinned `rho_t sigma` ball at overlap 2), so its per-pair
cost multiplies the dominant stage directly. The shipped `g`/`h` evaluation
(erf-free §3 series + §6.2 one-`exp` outer form, selected by the `032` H200
A/B at 1.5x over a `custom_erf` port) was tuned for accuracy well beyond the
phase gate. The fixed 1e-3 velocity tolerance leaves headroom for cheaper
evaluation at equal *delivered* accuracy.

## Scope

Candidate mechanisms (screen by measured A/B, the `032` methodology —
same-job anchors, equal delivered sampled accuracy):

1. Reduced-order `g`/`h` polynomial/rational approximations sized to the
   1e-3 budget (with the error decomposition's FMM + tail contributions
   accounted, so the pair-kernel error allowance is explicit and
   conservative).
2. Shared-memory or texture lookup tables in `rho^2` with linear/cubic
   interpolation for `g` and `h`.
3. Reduced-precision inner math (distance/`rho` computation, e.g. FP32
   rsqrt fast path or FP16/TF32 where admissible) with FP32 (or better)
   accumulation; note the known `CUDA.rsqrt(Float64)` ~1e-7 accuracy limit
   from the `028` record when pricing F64 variants.

Rules:

- The kernel-error budget must be derived first (from the recorded error
  decomposition of the target configurations), then mechanisms sized to it;
  no mechanism ships on pointwise pair error alone — gate on end-to-end
  sampled velocity RMS.
- Fused U+J structure, the singular branch, and the directed source/target
  path are preserved; the singular kernel may adopt the same tricks only if
  measured safe.
- Off by default behind an option/`Val` flag; shipped evaluation retained
  as control.
- Preserve capacity/zero-allocation, counters, graph capture, residency.

## Benchmark and Test Plan

- Pre-register (in this file, before submission) the H200 A/B ladder: cube,
  AR-5 wake, rotor at `n = 1e5`/`1e6`, Float32 and Float64, warmed U/J
  medians, same-job anchors, per-mechanism rows, sampled velocity RMS and
  Jacobian diagnostic against the checksummed references.
- Tests: `P=4` coverage, both precisions, pointwise kernel-accuracy bounds
  vs the shipped evaluation, graph replay, counter stability, zero
  recurring allocation. Local runs `<= 4` threads.
- Promotion gate (user approval required for any default change): `>= 5%`
  faster end-to-end U/J on a material wake or rotor case, no `> 3%`
  regression on any other measured case, velocity RMS `<= 1e-3` everywhere.
  Otherwise ship opt-in or record as falsified with measured ceilings.

## Theory/Artifact Dependencies

`031a` (regularized kernel forms, series, and cancellation-safety analysis),
`032` work record (erf-free evaluation selection and A/B methodology),
`037b` error-decomposition data (sets the kernel-error allowance).

## Pre-Registered H200 A/B Ladder (registered 2026-08-14, before submission)

Error budget derived FIRST (rule satisfied):
`theory/nearfield-kernel-cheapening-budget.md` + `scripts/fm037f_error_budget.jl`
+ `data/kernel_splitting/fm037f_budget.csv`, signed off by the lead agent.
Binding delivered allowance `B = (1e-3 - 7.078e-4)/1.1 = 2.656e-4` relative
velocity RMS (cube `n=1e6` anchor; per-case table in the note). Pointwise
budgets via the measured coherent amplification (`kappa_series=7.15`,
`kappa_outer=18.6`, `kappa_all=56.8`): series rel `<= 1.86e-5`, outer abs
`<= 7.15e-6`, whole-stream rel `<= 4.68e-6`. Every shipped mode passes the
coherent tier with `>= 2x` margin; the mapping is empirically bounded at
`n=1e4` and re-validated at scale by the oracle stage below. Rotor Jacobian
RMS is the flagged watch metric (diagnostic, not a gate).

Mechanisms under test (`CUDA_NEARFIELD_GH_MODE`, `:shipped` default and
control): `:reduced` (12-term series both precisions; outer stays deg-3 —
deg-2 fails even the incoherent tier), `:fp32` (F64 configs: F32 inner
math/assembly, F64 accumulation; documented no-op on F32 configs),
`:reduced_fp32`, `:lut` (N=1024 shared-memory table of normalized
`G=g/rho^3`, `H=h/rho^5` in `rho^2`, 8 KB/block, construction-built and
uploaded once as an operator upload; host falls back to `:shipped` under
`:lut`, parity gated at the budgeted pointwise error). TwoPass pass-2
deficit stays shipped in all modes.

- Screen grid: `scripts/fm037f_cases_screen.txt` — 48 rows; same six case
  points and shipped geometry as the 037e screen (cube ell4/5 q12, wake
  ell5/6 q6, rotor ell5/6 q6; `kernel=partitioned`, P5, `rho_t=3.668`,
  dense M2L, profile on); per (case, n, tf) block: `*_anchor`
  (`gh_mode=shipped`) plus one row per admissible mode (`:fp32`/
  `:reduced_fp32` skipped on tf=Float32 rows — documented no-ops). Driver:
  `benchmark_035_gpu.jl` with the `gh_mode` key set before cache
  construction; warmed U/J medians, checksummed references, counters,
  allocations, per-stage timings.
- Oracle stage: `scripts/fm037f_cutoff_configs.txt` — 30 configs through
  `benchmark_035_error_decomposition.jl` (new `gh_mode` key): each mode's
  exact delivered `u_total` must stay within the budget-note allowance of
  its same-geometry shipped anchor, at `n=1e5` and `1e6`, both precisions.
- Tests on the same job (preflight): `device_system_interface_test.jl`
  (839-assertion 037f host testset: per-mode pointwise bounds vs BigFloat,
  singular limit, `:shipped` bitwise assertions, LUT domain/boundary) and
  `cuda_radix_nearfield_binning_test.jl` (per-mode device parity at
  budgeted tolerances, P=4/P=8, F32/F64, graph replay with a non-default
  mode, counter and allocation stability incl. `:lut`).
- Gates (promotion; any default change additionally requires explicit user
  approval): velocity RMS `<= 1e-3` on every row; `>= 5%` faster
  end-to-end U/J on a material wake or rotor case; no `> 3%` regression on
  any other measured case; oracle deltas within the budget table.

## Work Record

**Done 2026-08-14; `:fp32` (Float64 configurations) PASSES the full
promotion gate — evidence recorded, default unchanged pending explicit
user approval. All other modes ship opt-in with measured ceilings.**
H200 job `13170769` (stage f; CSVs of record
`data/flowvpm_gpu_campaign/fm037f_screen.csv` and
`fm037f_decomposition.csv`; analyzer `scripts/analyze_037ef_screen.jl`).

Implemented (all committed on `matrix-ops`): `CUDA_NEARFIELD_GH_MODE`
(`:shipped` default, bitwise-identical to pre-037f code) with modes
`:reduced` (12-term series both precisions; outer stays deg-3),
`:fp32` (F64 configs: F32 inner math/assembly, F64 accumulation; no-op on
F32 configs), `:reduced_fp32`, `:lut` (N=1024 shared-memory normalized
`G/H` table in `rho^2`, 8 KB/block, built+uploaded once at construction as
an operator upload; host falls back to `:shipped`). TwoPass pass-2 deficit
stays shipped in all modes. Driver keys `gh_mode` in both the screen and
decomposition drivers. Budget note
`theory/nearfield-kernel-cheapening-budget.md` (signed off; binding
allowance `B = 2.656e-4`, coherent-tier pointwise budgets, measured
amplifications) + `scripts/fm037f_error_budget.jl` +
`data/kernel_splitting/fm037f_budget.csv`.

Verification: host testset 839 assertions (per-mode pointwise bounds vs
BigFloat, `:shipped` bitwise, LUT domain/boundary) and CUDA testset
117 assertions (per-mode device parity at budgeted tolerances, P=4/P=8,
F32/F64, graph replay with a non-default mode, counter/allocation
stability incl. `:lut`) — green on H200 first attempt. Oracle: all 48
mode rows' exact delivered `u_total` deltas vs same-geometry shipped
anchors are `1e-11`–`1e-7` — three to four orders below the allowance
(the coherent-tier sizing was extremely conservative; delivered accuracy
is unchanged at measurement precision). Counters flat, per-step
allocation mode-independent, every row passes the `1e-3` gate.

Screen results (36 candidate rows vs same-job anchors, warmed end-to-end
U/J medians = overlapped critical path):

| mode | cube 1e5/1e6 | wake 1e5/1e6 | rotor 1e5/1e6 | gate |
|---|---|---|---|---|
| `:fp32` (F64) | +6.8% / +10.2% | +7.8% / +7.4% | -2.5% / +1.5% | **PASS** |
| `:reduced_fp32` (F64) | +6.9% / +10.4% | +8.0% / +7.6% | -5.5% / -3.5% | fail (rotor regression) |
| `:lut` (F32) | +3.2% / +7.4% | +3.9% / +4.5% | -2.5% / -2.9% | fail (material best 4.5% < 5%) |
| `:lut` (F64) | +5.3% / +8.6% | +2.9% / +2.9% | -1.9% / -2.9% | fail (material best 2.9% < 5%) |
| `:reduced` | ~0-1.8% | ~0-1.5% | -2.7% / +1.0% | fail (< 5%) |

Promotion recommendation (awaiting explicit user approval per the
stop-and-report rule): make `:fp32` the default g/h evaluation for
Float64 configurations (F32 configurations keep `:shipped`); it passes
every arm of the gate — velocity RMS `<= 1e-3` everywhere (deltas
`~1e-8`), `+7.4-7.8%` on the material wake case, worst other-case
regression `-2.5% < 3%`. `:reduced`, `:reduced_fp32`, `:lut` remain
opt-in with the ceilings above. Rotor J diagnostic: watched, unchanged at
`1e-7` scale.
