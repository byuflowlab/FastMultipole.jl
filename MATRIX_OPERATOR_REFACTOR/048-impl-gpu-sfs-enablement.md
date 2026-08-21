# 048 Impl: GPU SFS Enablement (fused ζ pass)

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `046` complete and approved (all work lands on the unified
branches). Runs in parallel with `047` — both block only on `046`. FLOWVPM
CLAUDE.md constraints apply on that repo.

## Motivation

The 023 profiling of an 018 production step (2026-08-20) found **~75% of a
production step is the Dynamic-SFS estimator `Estr_fmm!` near-field walk** —
the single largest lever in the Production Integration Phase. Today
`sfs=true` is a **hard error** on the GPU radix path
(`FLOWVPM_fmm_radix.jl:499-500`). `041k` built the stack's first fused
direct+SFS kernel (all-pairs) and measured fused SFS at only **+26–31% over
a U/J pass** — evidence that a production U-list ζ pass can be cheap.

## Objective

Device SFS: the fused ζ pass (the `041b` §1.2 factorized identity — Ω/Q
precompute; E = T_p(Ω) − Q) implemented in the radix nearfield lifecycle and
the FLOWVPM adapter, with the `sfs=true` hard error removed, parity gates
passed, and the kernel written **lever-ready** for the `054` optimization
port.

## Method

### Stage 1 — kernel design

- ζ needs completed Jacobians → the SFS pass **cannot merge into the U/J
  pass**; minimal structure = 2 pair passes + an O(N) T_q(Γ_q) precompute
  (041k finding). Self-pair cancels exactly; ζ is skippable beyond the
  saturation cutoff.
- Production shape: Estr over the **U-list** (not all-pairs) with atomic (or
  target-owned, per `041e`) accumulation — the open cost question vs 041k's
  all-pairs number. **Measure it**; do not assume the +26–31% carries over.
- **Lever-ready requirement:** reuse the existing singular/regularized
  partition of the nearfield so the `041k` opt levers (far-field singular
  switch, fast transcendentals, register blocking — row `054`) drop in
  without restructuring.

### Stage 2 — lifecycle + adapter integration

- Wire the ζ pass into the radix nearfield lifecycle (after U/J completes;
  respect graph capture, capacity sizing, counters, zero per-step
  allocation).
- FLOWVPM adapter: remove the `sfs=true` hard error; route SFS output to
  `SFS_INDEX=40:42`; keep the gaussianerf-only check; physics transcribed
  from `FLOWVPM_subfilterscale_models.jl:16-41` (Estr, transposed) with the
  J layout col-major du_i/dx_j at J[(j-1)*3+i].

### Stage 3 — tests + measurement

- Parity vs CPU `Estr_direct!` / `Estr_fmm!` at 1e-3 (F64 tighter), at P=4
  AND P=8, both precisions (standing P=4 rule).
- Counters (`body_uploads==0`, `expansion_host_copies==0`) and
  zero-allocation contracts hold with SFS on.
- Extend FLOWVPM `runtests_gpu_fmm.jl` (Part A host + Part B device) with
  SFS testsets.
- H200 measurement: marginal SFS cost over U/J on the U-list at a realistic
  operating point (feeds the `049` per-pass budget table).

## Gates and verdict

- `sfs=true` works on the GPU path with the parity, counter, and
  zero-allocation gates green.
- Measured U-list SFS marginal cost reported (vs the 041k all-pairs +26–31%
  anchor).
- Verdict states whether the atomics/accumulation strategy on the U-list is
  satisfactory or names the follow-up lever.

## Artifacts

- Source changes on the unified FastMultipole + FLOWVPM branches (kernel,
  lifecycle wiring, adapter, tests).
- Measurement CSV + short results section appended to this doc.

## Verification

- FLOWVPM `runtests_gpu_fmm.jl` green incl. new SFS testsets (device parts
  under `FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1` on the cluster).
- FastMultipole suite green; no regression in non-SFS paths (same-job A/B).

## Recorded context (2026-08-20 staging)

**041k evidence** (`data/direct_bruteforce_ceiling/`): fused direct+SFS via
the 041b identity costs only +26–31% over U/J on all-pairs; SFS needs
completed Jacobians → ζ pass cannot merge into the U/J pass; minimal
structure = 2 pair passes + O(N) T_q(Γ_q) precompute; self-pair cancels
exactly; ζ skippable beyond saturation cutoff. Opt levers (far-field
singular switch ρ²>42.25 F32 / 81 F64, `__nv_fast_expf`/`__nv_erff`,
2-target register blocking) give 1.6–1.7× F32 → 3.3e11 pairs/s ≈ 38% FP32
FMA peak; F64 opt inert below n≈3e4 (9σ cutoff spans domain).

**018 motivation** (`BRAINSTORM/023_018_runtime_cost_profiling.md`):
170–230 s/step on 64 cores; ~75% of a production step is `Estr_fmm!`; wake
FMM velocity itself only ~7 s; per-step cost ~linear in particle count
(~50–100 s per 100k).

**FLOWVPM adapter state (034):** `FLOWVPM_fmm_radix.jl` (520 lines) +
`ext/FLOWVPMCUDAExt.jl` (980 lines: `gpu_direct!:642`,
`gpu_zeta_direct!:725`, `gpu_estr_direct!:835`, `warmup_gpu:864`, device
pack/unpack `source_to_buffer!:920`/`buffer_to_target!:944`). Hard errors:
rbf (`:497`), **sfs (`:499-500`)**, autotune flags. Traits Point{Vortex},
residency from `particles isa Array`, `RegularizedVortex(sigma_row=8)`,
gaussianerf-only hard check (`:62,:260`); one lazily-built `RadixFMMCache`
per pfield (capacity = maxparticles, hessian=true); zero per-step body
H2D/D2H.

**Physics transcription provenance:** `FLOWVPM_fmm.jl:132-198` (U/J pair
math), `FLOWVPM_kernel.jl:51-57` (gaussianerf),
`FLOWVPM_subfilterscale_models.jl:16-41` (Estr, transposed). J layout
col-major du_i/dx_j at J[(j-1)*3+i]; U_INDEX=10:12, J_INDEX=16:24,
SFS_INDEX=40:42 (`FLOWVPM_particlefield.jl:287-344`).

**Contracts (integration-api-spec.md):** counters `body_uploads=0` /
`expansion_host_copies=0`; zero per-step alloc; full 9-component hessian;
out-of-box throws; explicit `recenter!` only.

**Nearfield state P2P builds on:** partitioned singular/regularized +
`037f` `:fp32` gh-mode default + `041e` target-owned CSR (REGIME-ONLY, off
by default).
