# 033 FLOWVPM CPU Baseline Report

Deliverable of task `033-flowvpm-baseline-benchmarks.md`. All numbers measure
FLOWVPM at the `gpu-full` branch-point commit `e2bd487` (2026-05-16, v4.0.4)
with FastMultipole pinned to 2.0.4, at **fixed FMM parameters p=4, ncrit=50,
theta=0.4** (all autotuning off), Float64, on BYU cluster 64-core AMD EPYC
nodes, Julia **1.11.7** (module `julia/1.11.7-6bmogfl`; the 1.12.6 module
default segfaults the host LLVM JIT and is not used).

## 1. Provenance

| Job | Node | Date | Content |
| --- | --- | --- | --- |
| 13051516 | m12-2-1 | 2026-08-04 | cube rows n=1000..1e5 (10 rows) + 7 cube references; cancelled mid-sweep at the ring-to-gaussianerf amendment |
| 13051713 | m12-1-29 | 2026-08-05 | cube rows n=316228, 1e6 (4 rows) + cube profiles at n=1e5; also pre-amendment ring rows (retained history, see section 7) |
| 13058428 | m12-2-18 | 2026-08-05/06 | all 14 wake rows + 7 wake references + wake profiles at n=1e5 ("task 033 complete") |

Case definitions, seeds, and solver settings are recorded in the task file
(cube: uniform unit cube, `MersenneTwister(33025+n)`, `sigma = 2*(1/n)^(1/3)`;
wake: helical wake cylinder D=1, L=5D, `MersenneTwister(33025+7919+n)`,
`sigma = 2*(V_cyl/n)^(1/3)`, `V_cyl = 3.927`; both rVPM, `gaussianerf`,
inviscid, no SFS, default relaxation, transposed, RK3). Harness:
`scripts/benchmark_033_common.jl`, `benchmark_033_cpu.jl`,
`prepare_033_references.jl`, `cpu_033_{submit,run,fetch}.sh`.

Timing policy: `t_uj` = median of repeated `UJ_fmm` evaluations (5 reps for
n<316228, 3 reps above) after a JIT warmup call; `t_step` = median full
`nextstep` RK3 step with relaxation at dt=1e-6 (3 reps, 2 above), warmup
excluded. `t_step ~ 4*t_uj` throughout (3 RK3 stages + relaxation UJ).

Accuracy instrument: sampled-direct Float64 references (`direct!` with
hessian, single-thread — FastMultipole 2.0.4's `direct_multithread!` on the
`(target, source)` path is broken), all targets for n<=1e4, else 512 samples
drawn by `MersenneTwister(33026+n_actual)`. All 14 reference files (7 cube +
7 wake) verify against `references/direct_reference_checksums.sha256`
(`shasum -a 256 -c`: all OK, verified 2026-08-06).

## 2. Cube timings

14/14 rows (jobs 13051516 + 13051713). Times in seconds.

| n | threads | t_uj median | t_step median | u_rel_rms | j_rel_rms |
| --- | --- | --- | --- | --- | --- |
| 1000 | 1 | 0.0568 | 0.228 | 1.4e-15 | 1.4e-15 |
| 1000 | 64 | 0.0566 | 0.229 | 1.4e-15 | 1.4e-15 |
| 3162 | 1 | 0.554 | 2.22 | 2.4e-15 | 2.4e-15 |
| 3162 | 64 | 0.554 | 2.25 | 2.4e-15 | 2.4e-15 |
| 10000 | 1 | 4.58 | 18.2 | 1.07e-2 | 1.84e-2 |
| 10000 | 64 | 0.159 | 0.638 | 1.07e-2 | 1.84e-2 |
| 31623 | 1 | 24.4 | 98.5 | 2.95e-2 | 6.57e-2 |
| 31623 | 64 | 0.647 | 2.61 | 2.95e-2 | 6.57e-2 |
| 100000 | 1 | 113.7 | 452.3 | 4.62e-2 | 7.11e-2 |
| 100000 | 64 | 2.51 | 7.85 | 4.62e-2 | 7.11e-2 |
| 316228 | 1 | 422.3 | 1680.1 | 5.35e-2 | 8.79e-2 |
| 316228 | 64 | 10.4 | 42.9 | 5.35e-2 | 8.79e-2 |
| 1000000 | 1 | 1776.6 | 7065.8 | 6.75e-2 | 8.55e-2 |
| 1000000 | 64 | 56.0 | 189.5 | 6.75e-2 | 8.55e-2 |

Note: the cube cpu64 n=1e6 median (56.0 s) carries large run-to-run variance
(min 34.4 s across 3 reps); the min is likely closer to the node's capability.

## 3. Wake timings

14/14 rows (job 13058428). `n_actual = n_target` exactly at every grid point;
sigma matched `2*(V_cyl/n)^(1/3)` as specified.

| n | threads | t_uj median | t_step median | u_rel_rms | j_rel_rms |
| --- | --- | --- | --- | --- | --- |
| 1000 | 1 | 0.0525 | 0.210 | 1.1e-15 | 1.1e-15 |
| 1000 | 64 | 0.0523 | 0.211 | 1.1e-15 | 1.1e-15 |
| 3162 | 1 | 0.403 | 1.61 | 3.48e-3 | 7.37e-3 |
| 3162 | 64 | 0.404 | 1.62 | 3.48e-3 | 7.37e-3 |
| 10000 | 1 | 2.85 | 11.4 | 8.32e-3 | 2.32e-2 |
| 10000 | 64 | 0.0791 | 0.347 | 8.32e-3 | 2.32e-2 |
| 31623 | 1 | 19.5 | 77.7 | 1.56e-2 | 5.16e-2 |
| 31623 | 64 | 0.409 | 1.68 | 1.56e-2 | 5.16e-2 |
| 100000 | 1 | 102.1 | 407.4 | 3.17e-2 | 1.46e-1 |
| 100000 | 64 | 2.34 | 9.25 | 3.17e-2 | 1.46e-1 |
| 316228 | 1 | 506.1 | 2014.1 | 5.03e-2 | 2.62e-1 |
| 316228 | 64 | 8.53 | 35.8 | 5.03e-2 | 2.62e-1 |
| 1000000 | 1 | 1851.7 | 7523.1 | 6.39e-2 | 4.60e-1 |
| 1000000 | 64 | 31.6 | 125.2 | 6.39e-2 | 4.60e-1 |

## 4. Profiled cost breakdown at n=1e5 and bottleneck identification

`Profile.@profile` capture of one `UJ_fmm` call (tree format, C frames off,
mincount 10; sample counts, ~1 ms/sample). Percentages are of total in-call
samples for cpu1; for cpu64 the profile aggregates all threads and the counts
below are frame maxima across the worker-thread trees.

Single-thread (cpu1):

| Stage | cube samples (% of 112711) | wake samples (% of 102016) |
| --- | --- | --- |
| nearfield (`nearfield_singlethread!`) | 112059 (99.4%) | 101645 (99.6%) |
| — of which `g_dgdr_gauserf` kernel math | 42767 (37.9%) | 38105 (37.4%) |
| — of which `custom_erf64` | 19664 (17.4%) | 17233 (16.9%) |
| — of which `exp` | 14604 (13.0%) | 12788 (12.5%) |
| — of which `set_hessian!` writeback | 15158 (13.4%) | 13807 (13.5%) |
| horizontal pass (M2L) | 339 (0.30%) | 100 (0.10%) |
| tree build (`Tree`) | 33 (0.03%) | 36 (0.04%) |
| interaction lists | 142 (0.13%) | 117 (0.11%) |
| upward pass (B2M+M2M) | 38 (0.03%) | 37 (0.04%) |
| downward pass (L2L+L2B) | 46 (0.04%) | 43 (0.04%) |

64-thread (cpu64), worker-thread frame counts:

| Frame | cube | wake |
| --- | --- | --- |
| nearfield worker (`execute_assignment!`) | 19525 | 22614 |
| — `direct!` inner | 10321 | 11813 |
| — `g_dgdr_gauserf` | 7698 | 8666 |
| — `custom_erf` | 7429 | 8407 |
| — `exp` | 2609 | 2871 |
| — `set_hessian!` | 2801 | 3306 |
| scheduler idle (`poptask`/`wait`) | 2821 | 3059 |
| main-thread `fmm!` orchestration | 41 | 82 |
| M2L (largest far-field frame) | 15 | <10 |

**Bottleneck**: in both cases and at both thread counts the FMM evaluation is
completely dominated by the direct nearfield with the regularized
`gaussianerf` kernel — >99% of single-thread time, and essentially all
worker-thread time at 64 threads (far-field passes barely clear the 10-sample
print threshold). Within the nearfield, roughly two thirds of the time is the
`g_dgdr_gauserf` pair math, of which the erf evaluation (`custom_erf64` ~17%
of total) plus `exp` (~13%) alone are ~30% of the entire UJ solve; Hessian
writeback (`set_hessian!`) is another ~13%. The two cases have nearly
identical profiles — geometry changes the interaction lists but not the
nearfield-bound character at ncrit=50 / theta=0.4 with overlap-2 sigma. The
nearfield parallelizes well (section 6), so the 64-thread runs remain
nearfield-bound rather than shifting the bottleneck to the (serial-cheap)
tree/far-field stages. Consequence for later rows: GPU/nearfield levers
(erf-free g/h evaluation, fused U+J) attack exactly the dominant cost.

## 5. Accuracy vs the fixed 1e-3 gate

Phase gate (START_HERE preamble): sampled relative velocity RMS error
`u_rel_rms <= 1e-3`; `j_rel_rms` is logged as a diagnostic only.

- **Cube**: passes at n=1000 and n=3162 only (~1e-15 — effectively all-direct
  at ncrit=50); **fails from n=1e4 up** (1.07e-2 growing to 6.75e-2 at n=1e6).
- **Wake**: passes at n=1000 only (~1e-15); **fails from n=3162 up**
  (3.48e-3 growing to 6.39e-2 at n=1e6).

At the fixed default parameters (p=4, ncrit=50, theta=0.4), every
genuinely-FMM-active baseline row misses the gate, and errors grow with n
(deeper trees put a larger fraction of interactions through the p=4 far
field). The wake's coherent, aligned strengths make the relative-RMS gate
intrinsically harsher than the random cube at equal convergence (net velocity
is a small residual of large cancelling contributions), and its `j_rel_rms`
degrades faster still (0.46 at n=1e6).

**Explicit consequence** (per the phase policy): all rows above remain in the
record with their measured errors, but **speedup headlines in `035` may only
use gate-passing baseline configurations**. The default-parameter baselines
here pass only at the smallest n; any headline at production n therefore
requires a baseline configuration tuned to meet 1e-3 (parameters are tuned
per case in `035`; no replacement tuned CPU campaign is required in 033).
Everything else stays visible history.

## 6. 64-thread scaling efficiency (t_uj, median-based)

| n | cube speedup | cube eff. | wake speedup | wake eff. |
| --- | --- | --- | --- | --- |
| 1000 | 1.00x | — | 1.00x | — |
| 3162 | 1.00x | — | 1.00x | — |
| 10000 | 28.8x | 45% | 36.1x | 56% |
| 31623 | 37.7x | 59% | 47.6x | 74% |
| 100000 | 45.3x | 71% | 43.6x | 68% |
| 316228 | 40.6x | 63% | 59.3x | 93% |
| 1000000 | 31.7x | 50% | 58.7x | 92% |

At n<=3162 the runs show no parallel speedup at all (identical timings to
single-thread — the work at these sizes does not engage the multithreaded
path effectively). From n=1e4 the nearfield-bound solve scales well; the wake
reaches ~92% efficiency at its largest sizes. The cube's apparent efficiency
drop at n=1e6 (50% on medians) is partly timing variance: min-based speedup
is 51.4x (80%). Full-step (`t_step`) speedups track `t_uj` closely.

## 7. Threats to validity

1. **References share the campaign runs.** Wake references were generated in
   job 13058428 on the same node immediately before the wake timings (cube
   references likewise in 13051516). They are Float64 sampled-direct with
   fixed seeds and sha256-pinned, so the error *measurements* are
   reproducible, but no independent second implementation cross-checked them
   beyond the ~1e-15 agreement of the all-direct small-n rows (which is a
   strong self-consistency check of kernel and metric plumbing).
2. **Reference generation is single-thread by necessity** (FastMultipole
   2.0.4 `direct_multithread!` bug, `UndefVarError: n_source_bodies`,
   `direct.jl:111`). Correctness is unaffected; noted for reproduction.
3. **Ring history.** `cpu_m12-1-29_13051713.csv` and the repository retain
   pre-amendment vortex-ring rows, references, and profiles (ring case
   retired 2026-08-05 in favour of the wake). They are historical record
   only: no ring number feeds any 033 table above or any later gate/speedup.
   Ring rows also have `n_actual != n_target` (e.g. 891 at n=1000), unlike
   cube/wake.
4. **Node heterogeneity.** Cube rows span two nodes (m12-2-1, m12-1-29) and
   the wake a third (m12-2-18), all same-generation 64-core EPYC; small
   cross-node offsets may exist. Wake n=1000/3162 vs cube at the same n
   differ by ~10-30% — within plausible node/config noise at sub-second
   scales.
5. **Timing variance at large n, few reps.** n>=316228 uses 3 UJ reps / 2
   step reps; the cube cpu64 n=1e6 median-vs-min gap (56.0 vs 34.4 s) shows
   the medians can be conservative at the largest sizes.
6. **Profiles capture one call** after warmup (single `UJ_fmm`), so
   sub-percent stages are near the mincount threshold; conclusions are drawn
   only from the dominant frames, which are unambiguous.
