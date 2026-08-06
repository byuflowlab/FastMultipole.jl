# 029 step 2 — mechanism assessment and ranking (2026-08-06)

Inputs: the step-1 fresh baselines (job 13059710, `cuda029_base_*_13059710.csv`),
the 030 cost-vs-n record (`data/cost_vs_n/report.md`), the 028 record
(`data/feasibility_1m_10ms/report.md` + Staged Continuation Roadmap), and one
new evidence-only H200 profiling job (**13059955**, this step, no `src/`
changes) that attributes the launch/sync floor at the robust baseline config.
All numbers below are measured unless labelled "projection".

## 1. Per-stage budget of the fresh baselines

Robust working baseline `sched6-5-4-4` (verdict **7.436 ms** [7.412, 7.468],
err 1.0300e-3 = 0.87x gate) and knife-edge `sched6-4-4-3` (**7.003 ms**, err
0.9997x gate), n=1e6, ell=5, FP16-WMMA dense M2L, Float32, K=full(904),
counting sort. Stage medians from the step-1 CSVs; splits from job 13059955.

| stage | robust ms | knife ms | device-busy ms (13059955, robust) | notes |
|---|---:|---:|---:|---|
| refresh (counting sort + occupancy) | 0.848 | 0.845 | 0.50 | 108 launches/step inside |
| B2M | 0.589 | 0.587 | 0.58 | single kernel, busy |
| M2M + L2L | **1.956** | 1.966 | **0.47** | 253 launches + 101 small H2D/step; **~1.5 ms is host/launch overhead** |
| M2L total | 2.133 | 2.100 | 2.00 | of which **~0.78 ms is M2L math** (`tensor16` kernel) and **~1.2 ms is route-window regeneration** (flags/scan/compact) + a blocking route-count D2H (1.25 ms sync inside) |
| — M2L leaf | 1.637 | 1.662 | | 6.59M / 6.78M routes |
| L2B + nearfield (standalone) | 2.667 | 2.270 | 2.66 | nearfield kernel 2.25 ms + L2B 0.41 ms; overlap hides ~1.1 ms behind the far-field chain (eval 6.258 vs 7.345 stage sum) |
| finalize + Euler | 0.255 | 0.258 | 0.17 | |
| **verdict** | **7.436** | **7.003** | **6.39 busy / ~1.4–1.7 idle** | 420 kernel launches, 27 syncs, 127 mem-ops per step |

Floor evidence: the identical-geometry n=1e3 control in job 13059955 runs the
complete step at 3.87 ms wall with only **0.82 ms device-busy** — a **~3.0 ms
n-independent host/launch/sync floor**, matching the 030 table (M2M+L2L is
~1.9 ms at ell=5 for *every* n from 1e3 to 1e6; verdict 2.99 ms at n=1e3).
The 030 fit's ~0.44 ms/level launch floor is confirmed and now attributed:
at n=1e6 the floor components are M2M+L2L wall−busy ≈ 1.5 ms, M2L
window-generation sync/idle ≈ 0.5–1.0 ms, refresh/finalize/euler
orchestration ≈ 0.4 ms.

### What ≤1 ms implies

Single H200: today's step is 6.4 ms *device-busy*. Roofline floors at this
geometry: nearfield 0.94e9 interactions ≈ 0.31 ms at FP32 peak (measured
2.25 ms, ~7x off); leaf M2L traffic ≈ 0.15–0.2 ms (measured 0.78, ~4x off);
B2M+L2B ≈ 0.05 (measured 0.99); windows cacheable; M2M/L2L busy 0.05. Ideal
sum ≈ 0.7–0.9 ms plus refresh ~0.3 plus a launch floor. **Single-GPU ≤1 ms
requires simultaneously near-peak versions of every kernel plus a ~10x floor
reduction — possible only in the §5-of-028 "far end" sense.** The multi-H200
track is the credible route: work stages divide by up to 8 intra-node
(NVLink), and only the per-GPU floor and communication must be engineered.

## 2. Ranked mechanism families

Verdict boundary includes refresh + convection + all sync; the 028 closed list
(low-rank M2L, unordered symmetric nearfield, TF32, atomic-only rewrites, B2M
warp-per-cell, stream overlap of saturated kernels) is respected. ~8% run
variance ⇒ gains below ~0.7 ms need repeated measurement.

| # | family | measured remaining cost | plausible complete-step gain | accuracy risk | memory | scope | compat | confidence |
|---|---|---|---|---|---|---|---|---|
| 1 | **Launch/sync floor elimination** (CUDA-graph capture or host-loop fusion of the far-field chain + finalize/euler; kill the 253-launch M2M/L2L pattern and per-step pageable H2Ds) | ~1.5 ms M2M/L2L overhead + ~0.4 orchestration + ~0.3–0.5 idle in M2L windows | **7.44 → ~5.0–5.5 ms** (−2.0–2.5) | none (identical kernels, identical order) | negligible | medium (lifecycle orchestration; graph needs sync-free chain) | keep non-graph fallback | **high** — attribution measured (13059955); mechanism = wall−busy |
| 2 | **Route-window machinery reduction** (windows are ~1.2 ms device-busy of scans/flags/compact + a blocking route-count D2H every step; fold into refresh, regenerate only on occupancy change or rewrite as fused single-pass scan) | ~1.2 ms busy + ~0.5 sync/idle | −0.8–1.5 ms (combined with #1: step ~4.0–4.5) | none if regenerated on every occupancy change; amortization claims must follow the verdict's recurrence rule | small | medium | device-only change | high (kernel-level attribution measured) |
| 3 | **Multi-H200 spatial decomposition** (8x intra-node; own leaderboard track — required for task completion regardless) | whole step | work stages ÷≤8: with #1+#2 landed, projection ~**1.1–1.6 ms** at 8 GPUs incl. comm | partition-exact-once gates required | ~2 GB/GPU | **large** (ownership, halo multipole/body exchange, distributed tests) | new surface, existing single-GPU path untouched | medium (comm volume is small — halo multipoles/bodies ~few MB over NVLink — but orchestration floor per GPU is the risk, hence after #1) |
| 4 | **Nearfield round 2** (ILP/vectorization rewrite: multiple targets/lane, wide loads — a *different* mechanism from closed launch/atomic items; Stage 9a shifted-macrocell only if pair-enumeration cost is exposed) | 2.25 ms busy, ~7x off FP32-peak rate | −0.7–1.2 ms (partially hidden by overlap until #1 shrinks the far field) | none | none | medium | kernel-internal | medium (roofline headroom proven; specific mechanism unproven) |
| 5 | **Leaf M2L round 3** (class-/target-owned batching for operator reuse; WMMA packing reuse) | 0.78 ms busy | −0.3–0.5 ms | low | small | medium | kernel-internal | medium-low (already 2.15x from FP16-WMMA; remaining headroom ~4x but shape is thin) |
| 6 | Refresh/counting-sort reduction beyond #2 | 0.5 busy / 0.85 wall | −0.2–0.4 ms | stale-tree variants blocked by the §4.7 weak-dt caveat | none | small-medium | device-only | medium |
| 7 | Plane-wave/exponential M2L (Stage 9b) | attacks the 0.78 ms M2L math | ≤−0.5 ms | **high** (new translation theory, FP16/F32 stability, P=4 gives little diagonalization benefit) | new tables | **very large** | new theory branch | low at P=4 — deprioritized |
| 8 | Spatial/polyphase FFT M2L (Stage 9b) | same 0.78 ms | ≤−0.5 ms | high; dense-lattice-only, plan/workspace recurring costs | large | very large | fallback complexity | low — deprioritized |
| 9 | Backend-driven depth re-bracketing | — | ~0 | — | — | — | — | closed by measurement (028 §4.1; 030: ell=4 25.0 ms, ell=6 23.6 ms at n=1e6) |

Reopening note (#1): Stage 10 closed "CUDA graphs / launch tuning … unless a
later algorithm makes the step launch-bound." New mechanism-specific
hypothesis, satisfied by measurement: FP16-WMMA + 030 geometry shrank the step
4.3x since that closure, and job 13059955 now measures 420 launches, 27 syncs,
wall−busy ≈ 1.4–1.7 ms plus 1.5 ms of M2M/L2L host overhead — the step *is*
launch-bound at the margin today, which is precisely the reopening condition.
Falsification: if a captured/fused chain saves <0.5 ms, the floor is not
graph-fixable and #1 closes again.

## 3. Bounded prototype designs (top 3)

### P1 — graph-captured far-field chain (+ #2's window fold) — recommended cycle 1
Build (benchmark-side first): a script-local step driver that (a) replaces the
per-level host loops of M2M/L2L with a single stream-ordered launch sequence
free of intermediate syncs and pageable H2Ds (pre-uploaded per-level argument
buffers), (b) captures B2M→M2M→M2L(windows)→L2L→L2B→finalize→euler into a CUDA
graph instantiated once per refresh epoch, replayed per step; refresh stays
outside. The route-count D2H inside window generation must become
device-resident (indirect dispatch via max-size launch + device-side count
guard) — that is also most of #2.
Measures: full verdict A/B (graph on/off) at the robust config, REPS≥15, plus
n=1e3 floor control. Expected: 7.44 → ~4.5–5.5 ms; floor 3.0 → ≤1.0 ms.
Falsifies: <0.5 ms gain ⇒ floor not graph-fixable, close #1.
Cost: ~2–4 days + 1–2 H200 jobs. Accuracy: bit-identical kernels ⇒ gate
unchanged (still re-verified).

### P2 — 2-GPU feasibility slice of the multi-H200 track
Build: octant partition at the level-1 split (each GPU owns 4 of the 8
level-1 subtrees; coarse levels ≤2 replicated), per-GPU resident state, halo
exchange of (i) boundary-cell multipoles before M2L and (ii) halo body buffers
before nearfield, via CUDA P2P/NVLink on one m13h node; verdict = slowest GPU
including all exchange/sync. Measures: parallel efficiency, comm+imbalance
cost, per-GPU floor. Decision numbers: comm+orchestration ≤0.4 ms and
efficiency ≥75% ⇒ commit to the 8-GPU production track (projection ~1.1–1.6 ms
with P1; ≤1 ms requires P1+P4 too). Falsifies: comm/orchestration ≥1 ms at
2 GPUs ⇒ the multi-track cannot reach ≤1 ms intra-node; record and re-scope.
Cost: ~1 week (largest); distributed tests required before any leaderboard
entry.

### P3 — nearfield ILP rewrite (fixed-work A/B)
Build: script-local variants of the direct-pairs kernel on captured direct
lists: 2–4 targets per lane (register-blocked), `float4` source loads, source
tile in shared memory reused across the warp's targets. No change to rsqrt or
the launch shape (both measured saturated in cycle 4).
Measures: isolated kernel ms at fixed work (both baseline configs) + one
end-to-end confirm of the winner. Falsifies: <20% ⇒ close nearfield micro-opt;
macrocell ownership (Stage 9a) then becomes the only remaining nearfield lever
and needs its own pair-coverage evidence first.
Cost: ~1–2 days, can share P1's H200 job.

## 4. What was measured this step (job 13059955)

Evidence-only profiling (CUDA.@profile/CUPTI traces, no production changes):
robust config at n=1e6 and an identical-geometry n=1e3 floor control, REPS=5
per stage, node m13h-1-1, julia 1.11.7 pinned, CUDA 12.8.0. Artifacts:
`cuda029_profile_m13h-1-1_13059955.csv` (per-stage wall/busy/idle/launches/
syncs/mem-ops + untraced control wall), `.kernels.csv` (per-kernel totals),
`fm029p-13059955.out`. Source manifest `993743a2a26c8bf3` = matrix-ops tip
(step-1 manifest `5a5d41312dc113d2` plus the approved 032a stage-A host-side
additions, inert on this scalar device path; step-1's own regression note
covers the parent).
Headline attributions: full step 420 launches / 27 syncs / 127 mem-ops,
device-busy 6.39 ms; M2M+L2L busy 0.47 vs wall 2.49; M2L busy 2.00 of which
window scans ≈ 1.2 and M2L math 0.78; nearfield kernel 2.25; n=1e3 full step
3.87 ms wall vs 0.82 busy (the ~3 ms floor, directly observed).

## 5. Recommended first optimization cycle (for user approval)

**Cycle 1 = P1 (+ the window-generation fold of #2)**: highest measured,
lowest-risk gain (−2.0 to −3.0 ms, 7.44 → ~4.5–5.5 ms), zero accuracy risk,
prerequisite for both leaderboard tracks (the per-GPU floor otherwise caps the
multi-GPU step at ~3 ms regardless of GPU count). P3 can ride the same
evaluation job as an isolated fixed-work A/B without production commitment.
P2 follows as cycle 2 once the floor is fixed. Single-GPU ≤1 ms should be
treated as out of reach on current evidence; the goal path is the multi-H200
leaderboard with cycles 1→2→3 stacked, projecting ~0.9–1.4 ms — a knife-edge
but falsifiable program, with each cycle gated on its own measured verdict.
