# 041h Optimization: Full Step on a Real Simulated Rotor Wake

## Status and entry gate

**Staged by user direction `2026-08-18`; not started.**

Entry gate: `030`, `037f`, `041`, `041a`, and `041e` complete and approved.
This row blocks `042` so the adaptive-octree milestone reviews its outcome.
Benchmark/optimization row: artifacts under `scripts/` and `data/`, plus
opt-in production tuning changes under `src/` ONLY where a measured lever
justifies them — **every production change requires an explicit user
checkpoint before it lands** (the `035` convention). Defaults never change
without user approval. Local work ≤ 4 threads; H200 measurements on the
cluster.

## Objective

Minimize the **complete per-time-step cost** (refresh + resident lifecycle +
finalize; no per-step host/device body transfers) of the U/J evaluation on a
**real simulated rotor wake** — not a synthetic reconstruction — on a single
H200, using everything the 038–041e arc established (adaptive octree, sticky
sigma demotion, resident M2L strategies, graph capture, 037f g/h modes, the
041e fused nearfield and its selector, per-`n` geometry tuning from `030`).
Then give an evidence-based estimate of what 8 H200s could and could not do
for this case.

## The case (real data, fixed at staging)

FLOWPanel rotor-in-hover simulations at
`~/Dropbox/research/projects/FLOWPanel.jl/data/` (path relative to this
repo: `../../FLOWPanel.jl/data/`):

- **Primary**: `rotor_hover/rotor_hover/rotor_hover_wake_particles.719.vtp`
  — final step of a 720-step run, **n = 67,745** particles.
- **Secondary**: `rotor_hover_relax006_full/rotor_hover_relax006_full_wake1_particles/rotor_hover_relax006_full_wake1_particles.921.vtp`
  — final step of a 922-step run, **n = 37,165**.
- Mid-run snapshots from the SAME series (e.g. steps ~180/360/540) are used
  for the refresh/epoch-persistence measurement (see below): consecutive
  real steps give the true particle motion and sigma evolution, which no
  synthetic case provides.

Per-particle fields available in the VTP files: `Points` (3), `gamma` (3),
`sigma` (1), plus simulation `velocity` (3) and `velocity_gradient` (9) for
sanity cross-checks (the accuracy instrument of record remains our own
sampled-direct sums). Extraction: a deterministic script parses the VTP
(zlib-appended format) and writes compact checksummed snapshots
(positions/gamma/sigma + the simulation U/J fields) into
`data/real_rotor_fullstep/`; snapshot sizes are a few MB, so they ARE
committed (unlike the 041e synthetic `.bin`s). Record the file SHA-256 of
each source VTP in the manifest for provenance.

## Regime honesty (record up front)

This case is **n ≈ 4–7 × 10⁴** — an order of magnitude below the synthetic
studies. Expectations, from the record:

- The `041e` fused nearfield is BELOW its measured win envelope (selector
  falls back at `< 400k` bodies; regressions were measured at `n = 1e5`).
  Verify the fallback and record whether a small-n re-tune (thread/block
  shape) changes the sign — do not assume.
- `024b`/`030` measured that optimal uniform depth falls to `ell = 2/3`
  below `n = 1e5`; the adaptive-vs-uniform verdict at small n favored
  uniform on most 041a cases. Both machineries must be swept here.
- At this scale, per-step refresh, graph-replay floors, and launch latency
  are first-order: `028`/`029` established an ~0.87 ms n-independent
  per-GPU control floor and ~50 µs uncaptured launch costs. The full step
  will plausibly land in the low single-digit milliseconds, where those
  floors are tens of percent.
- The real field's sigma spread and spatial distribution differ from the
  fm033 prescription (measure and record the actual sigma histogram, fill,
  and occupancy contrast as the first census output).

## Required work

### 1. Snapshot extraction and references

Deterministic VTP → snapshot extraction (both cases + 3 mid-run steps of
the primary series); sigma/occupancy census of the real field; seeded
200-target sampled-direct U/J references (Float64) per snapshot; standing
1e-3 velocity gate, J diagnostic. Cross-check our direct sums against the
simulation's own `velocity` field (agreement expected only to the
simulation's FMM tolerance — record, don't gate).

### 2. Single-H200 baseline matrix (pre-registered)

Same-job sweep on the primary snapshot (secondary as a check row):
uniform path at `ell ∈ {2, 3, 4}` × the shipped M2L strategy defaults vs
adaptive at `K_max ∈ {16, 32, 64, 128}`; Float32/Float64 at P = 4 (P = 8
contract row); 037f g/h default; graph capture on; verdict-boundary timing
(update + lifecycle + finalize), per-stage breakdown, memory, counters.
Identify the winner configuration and the full-step profile (nearfield /
far-field / refresh / launch-floor shares).

### 3. Optimization cycles (measured, user checkpoint per production change)

Ranked by expected value from the profile, at minimum evaluate:

- **Refresh amortization**: measure real epoch persistence across
  consecutive simulation steps (does the occupied leaf set actually change
  every step at this n?); stable-epoch refresh and any justified
  cheapening of the per-step rebuild at small n.
- **Per-n geometry/error tuning** (`030` lever): depth/K/theta-analog
  retune at the case's own accuracy margin.
- **Strategy mix**: resident M2L strategy choice at this class-occupancy
  regime (the `024` crossovers were measured at larger n).
- **Launch-floor reduction**: graph consolidation opportunities at small
  kernels (the profile decides).
- **041e fused shapes below the envelope**: one bounded re-check only.

Stop when the remaining gap to the floors is quantified or levers are
exhausted; report the best reproducible full step (median of ≥ 5 warmed
steps, two independent job repetitions).

### 4. 8-GPU estimate (modeled, no implementation)

Using the `029` P2 evidence (mirrored-tree falsified at 58–61% by
replicated refresh; partitioned-tree successor staged as `043`–`045`;
measured 0.87 ms/GPU control floor, 0.19 ms comm at 1e6) and this row's
measured single-GPU profile: model the best-case 8-GPU step for THIS case
under (a) the mirrored scheme, (b) an idealized partitioned-tree split, and
state clearly whether 8 GPUs help at n ≈ 7e4 at all (the n-independent
floors say likely not — the model must quantify it, including the smallest
n at which 8 GPUs would break even). This row implements nothing
multi-GPU; `043`–`045` own that arc.

## Deliverables

- `scripts/fm041h_extract_snapshots.jl`, `scripts/fm041h_fullstep_bench.jl`
  (+ submit script), analysis script;
- `data/real_rotor_fullstep/` — committed snapshots, manifest with source
  SHA-256s, census, baseline matrix, optimization-cycle records, best-step
  report, 8-GPU model, checksums;
- Result section here: best achieved full step (ms) vs the pre-optimization
  baseline, per-stage profile, the 8-GPU verdict, and recommendations;
  Done checkbox + independent clear-context approval.
