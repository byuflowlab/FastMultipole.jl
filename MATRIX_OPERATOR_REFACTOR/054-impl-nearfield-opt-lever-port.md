# 054 Impl: Nearfield Opt-Lever Port (041k levers → production U-list kernels)

## Status and entry gate

**Staged `2026-08-20` (user direction). Not started.**

Entry gate: `053` complete and approved (or pulled forward as a `052`
escape-hatch addendum with user approval — the phase prose allows it).
First row of the Peak Efficiency Phase.

## Motivation

`041k` measured the opt levers on the all-pairs brute-force kernel at
**1.6–1.7× F32** (to 3.3e11 pairs/s ≈ 38% FP32 FMA peak, FMA-bound), but
they live only in the standalone benchmark kernels. The production
partitioned U-list kernels — both UJ and the `048` ζ/SFS pass, which was
deliberately written lever-ready on the same singular/regularized
partition — don't have them. Cost model: T(K) ≈ αKN + βN/K + floors;
cheapening the pair kernel (α) shifts the optimal leaf size upward, so the
port is only harvested after a leaf-size/autotune re-sweep.

## Objective

The `041k` levers ported into the production partitioned U-list kernels
(UJ + ζ/SFS), a leaf-size/autotune re-sweep to move K*, and a
`037f`-style promotion decision per lever.

## Method

### Stage 1 — port the levers

Into the partitioned nearfield kernels, preserving exact-once coverage,
graph capture, capacity, counters, zero-allocation:

- **Far-field singular switch:** beyond ρ² > 42.25 (F32) / 81 (F64), the
  regularized pair is numerically singular — skip g/h entirely.
- **Fast/libdevice transcendentals** (`__nv_fast_expf`, `__nv_erff`) for
  the regularized remainder.
- **2–4-target register blocking** (041k used 2-target; try 4 given the
  U-list's target-owned CSR structure from `041e`).

### Stage 2 — re-tune

Autotune/leaf-size re-sweep on the standard cases + the 018 operating
point to find the new K* (GPU optimum was K=256 at the old α, `027`); the
end-to-end win is measured at the re-tuned point, not the old one.

### Stage 3 — promotion A/B

Same-job A/B on the 018 operating point + standard cases (cube/wake/rotor,
n = 1e5/2.1e5/1e6). `037f`-style promotion gate per lever: ≥5% end-to-end
on a material case, ≤3% regression elsewhere, 1e-3 accuracy, P=4 both
precisions.

## Gates and verdict

- Promotion gate per lever (promote/reject individually, with the measured
  numbers).
- Contracts unbroken (exact-once, counters, zero-alloc, graph capture).
- Verdict records the new K* and the end-to-end delta at the re-tuned
  point, feeding `056`'s scoreboard.

## Artifacts

- Source changes + tests on the unified branch; `scripts/fm054_*` A/B
  drivers; `data/nearfield_opt_lever_port/` CSVs + `report.md`.

## Verification

- Same-job A/B with job IDs; accuracy vs sampled direct references at the
  standing gate; re-sweep results reproducible from committed scripts.

## Recorded context (2026-08-20 staging)

**041k measured ceilings** (`data/direct_bruteforce_ceiling/`): baseline
tiled F32 2.0e11 pairs/s (transcendental-bound); opt variant (far-field
singular switch ρ²>42.25 F32 / 81 F64 + `__nv_fast_expf`/`__nv_erff` +
2-target register blocking) gates-clean at **1.6–1.7× F32 → 3.3e11 pairs/s
≈ 38% FP32 FMA peak**; **F64 opt inert below n≈3e4** (9σ cutoff spans the
domain — the singular switch never fires); fused SFS +26–31% over U/J;
brute-force/FMM crossover ~5.5e3 with opt.

**Production nearfield state:** partitioned singular/regularized kernels +
`037f` `CUDA_NEARFIELD_GH_MODE=:fp32` default + `041e` target-owned CSR
(REGIME-ONLY, off by default) + `048` ζ/SFS pass written lever-ready on the
same partition.

**Cost model / retuning rule:** T(K) ≈ αKN + βN/K + floors; GPU optimum
K=256 (`027`); every kernel cheapening must be followed by leaf-size
retuning to harvest (cheapening α lets the autotuner raise K*). Standing
memory note: price levers against the overlapped critical path, not
isolated stage sums.

**037f promotion-gate precedent:** ≥5% end-to-end on a material case, ≤3%
regression elsewhere, delivered-accuracy deltas negligible at the 1e-3
gate, P=4 + both precisions tested; default flips need explicit user
approval.

**018 operating point for the material-case A/B:** ~181k–342k particles,
36,752 panels, per-step budget ≤3.3 s; `049`/`052` budget tables say which
pass binds.
