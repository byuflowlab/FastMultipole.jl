# Task 024a benchmark figures

Human-readable figures for the Matrix Operator Refactor's accumulated benchmark
evidence. Every plotted series is generated from the committed CSVs under `../`
and no new benchmarks were run for this task. The prose footnote under each
figure additionally quotes headline numbers and caveats from the dependency task
files; the only in-panel numeric annotations are the panel-4d route-geometry
comparison, whose values come from `tables/fig04d_geometry_bytes.csv`, and the
panel-3c JIT-contamination label, whose value is the plotted bar itself.

Use `../../START_HERE.md` for task order, approval requirements, and phase-gate
rules.

## Regenerating

```bash
bash MATRIX_OPERATOR_REFACTOR/scripts/figures_024a_build.sh
```

Two stages:

1. `scripts/figures_024a_prepare.jl` (Julia stdlib only — no package
   environment is touched) reads the campaign CSVs and writes one tidy, wide,
   pgfplots-ready table per panel into `tables/`. It errors out on a missing
   source, an unexpected header, or an empty selection.
2. `latexmk -pdf` compiles each `fig*.tex` from this directory, so the relative
   `tables/*.csv` paths resolve. PNG copies are made if `magick` or `pdftoppm`
   is available.

Verified with Julia 1.12.5, TeX Live 2025 (`pgfplots`/`pgfplotstable`,
`compat=1.18`), ImageMagick 7. `tables/`, `*.pdf` and `*.png` are all generated
output; `fmfigstyle.tex` and `fig*.tex` are the sources.

**The task 024b campaign is complete** (`2026-07-28`), so `fig09` builds strictly
by default: all 14 CPU cases, 36 feasible GPU cases, the eight-row GPU
capacity-failure ledger, the per-`n` reference-checksum check, and the Float64
accuracy-ratio gate pass unforced, and the figure carries no provisional banner.

`FM024B_ALLOW_PARTIAL=true` is retained only as a resume aid for a re-run of an
incomplete campaign; it relaxes the completeness gates, plots only the particle
counts that have all four comparison modes, and labels the figure `PROVISIONAL`.
Figures 1–8 (task 024a) regenerate identically either way, so that gate is a 024b
completeness check, not a 024a reproducibility problem.

**Known cosmetic item:** `fig05` leaves a large whitespace block between panels
(d) and (e) — the two regime maps have very different row counts and the
2-column `groupplot` cannot equalize them. Every series is legible; this is
presentation polish only, deliberately not fixed.

Shared style lives in `fmfigstyle.tex`: four fixed categorical colour slots
validated colourblind-safe against a white surface in all-pairs mode (worst CVD
ΔE 8.4, worst normal-vision ΔE 16.3, all four ≥ 3:1 contrast), each paired with
a distinct mark shape so no series is identified by colour alone. Reference
rules (break-even, roofline, memory gate) are drawn in the reserved status red
and are never data series.

## Figure index

| Figure | Question it answers | Source CSVs | Machine / regime |
|---|---|---|---|
| `fig01_crossover.pdf` | At which `P` do the isolated per-column and production whole-slab paths overtake their recurrence references under BLAS 1/64? | `smallp_fallback_layout/m12-1-7/crossover_{isolated,stage}_blas{1,64}.csv` | m12-1-7 EPYC 7763, BLAS 1 and 64, Float64 |
| `fig02_speedup_summaries.pdf` | Where does the whole-slab path win vs `P` and route count on CPU, and vs `n` in both the 019b corner and integrated 024 GPU comparison? | `smallp_fallback_layout/m12-1-7/crossover_stage_blas{1,64}.csv`, `smallp_fallback_layout/m13h-1-1/gpu_smallp.csv`, `operator_ab_benchmark/summary/case_rankings.csv` | m12-1-7 EPYC 7763; m12-2-17 EPYC 7763; m13h-1-1 H200 |
| `fig03_gpu_stage_breakdown.pdf` | What did 019 tuning buy per stage, how far is M2L from roofline, what does one-shot setup cost against the step it enables, and what does the actual 023 recurring step contain? | `operator_performance_tuning/cuda_022_baseline.csv`, `operator_performance_tuning/m13h-1-1/cuda_019_phase{ABD,E}_throughput.csv`, `production_integration/benchmark_023_m13h-1-1_20260714-233159.csv` | m13h-1-1 H200, n=1e5, ℓ=4, P=4, chunk 2^17 |
| `fig04_storage_allocation.pdf` | What does the ragged-χ decision save, how large do operators/expansion/scratch/cache footprints get, which strategy peaks highest on-device, and what warmed host allocation remains? | `smallp_fallback_layout/m12-1-7/layout_storage.csv`, `operator_ab_benchmark/summary/memory_feasibility.csv`, `operator_performance_tuning/local_macos_allocations_storage.csv` | analytic (a); m13h-1-1 H200 (b,c,e); local macOS (d,f) |
| `fig05_strategy_selection.pdf` | Which of the four resident M2L strategies should be used, by platform, order, channel, distribution and problem size? | `operator_ab_benchmark/summary/{case_rankings,cpu_crossovers,gpu_crossovers}.csv` | m12-2-17 EPYC 7763 (BLAS 1, 64); m13h-1-1 H200 |
| `fig06_tradeoffs_limitations.pdf` | What do the speedups cost — the χ channel, multi-thread BLAS, construction time — and what accuracy do they run at? | `operator_ab_benchmark/summary/{case_rankings,construction_amortization}.csv`, `operator_ab_benchmark/raw/cuda_*.csv`, `smallp_fallback_layout/m12-1-7/crossover_stage_blas{1,64}.csv` | m12-2-17 EPYC 7763, m13h-1-1 H200, m12-1-7 EPYC 7763 |
| `fig07_baseline_provenance.pdf` | What did 008c project, why was device residency mandatory, and how much of the projection did the CPU operators actually realize? | `impl_performance_baseline/m13h-1-1/{dense_vs_loop_blas1,dense_gpu}.csv`, `impl_performance_baseline/m12-2-5/dense_vs_loop_blas72.csv`, `axis_swap/tmpfac-126-17.et.byu.edu/m2l_variants_blas1.csv` | m13h-1-1 Xeon 8568Y+ / H200; m12-2-5 EPYC 7763; tmpfac-126-17 Apple M2 |
| `fig08_radix_vs_legacy.pdf` | Is the radix path faster than the legacy octree `fmm!` that ships today — the only comparison an end user makes? | `production_integration/benchmark_023_{m13h-1-1,m13h-2-1,mecsrs-MacBook-Pro-188.local}_*.csv` | m13h-1-1, m13h-2-1 (threads=8, H200); local macOS (threads=1); P=4, ℓ=4 |
| `fig09_cpu_gpu_scaling.pdf` | How do legacy CPU 64-thread and resident H200 speedups over legacy CPU single-thread scale from 1e3 to 1e6 particles, and do their sampled-direct errors remain comparable? | `cpu_gpu_scaling/*.csv` | orc 64-CPU node and H200; literature P=4 / code order 3; fixed CPU MAC + manually searched leaf; reviewed compatible GPU stencil; Float64 primary + Float32 extra |

Each figure's own footnote repeats its machine, BLAS regime, source files, the
conclusion it supports, and its caveats, so a PDF is self-describing when read
on its own.

## Panel-to-table map

| Panel | Table(s) |
|---|---|
| 1a, 1b / 1c, 1d / 1e, 1f | `fig01a_abs_{phi,lh}.csv` / `fig01c_wholeslab_{phi,lh}.csv` / `fig01b_ratio_{phi,lh}.csv` |
| 2a, 2b / 2c | `fig02a_concat_speedup_blas{1,64}.csv` / `fig02c_routes_{tinyparent,smallconstp,mediumconstp}.csv` |
| 2d / 2e | `fig02b_gpu_speedup.csv` / `fig02d_integrated_gpu_speedup.csv` |
| 3a / 3b / 3c / 3d | `fig03a_stages.csv` / `fig03b_lifecycle.csv` / `fig03d_oneshot_vs_recurring.csv` / `fig03c_step_split.csv` (panels c and d are deliberately crossed: `fig03c_*` predates the one-shot panel and kept its name so the table history stays stable) |
| 4a / 4b / 4c / 4d / 4e / 4f | `fig04a_layout_overhead.csv` / `fig04b_dense_operator_mib.csv` / `fig04c_cuda_peak_mib.csv` / `fig04e_cache_kib.csv` (+ `fig04d_geometry_bytes.csv`, quoted as text) / `fig04f_concat_buffers_mib.csv` / `fig04g_warmed_allocations_kib.csv` |
| 5a, 5b / 5c / 5d, 5e | `fig05a_relstep_{cpu,gpu}.csv` / `fig05b_wins_{cpu,gpu}.csv` / `fig05{cpu,gpu}_win_*.csv` with `fig05{cpu,gpu}_ticks.tex` |
| 6a / 6b / 6c / 6d | `fig06a_lh_ratio_{cpu,gpu}.csv` / `fig06b_blas_degradation.csv` / `fig06c_break_even.csv` / `fig06d_{potential,gradient}_float{64,32}_p{4,8,12}.csv` |
| 7a / 7b / 7c | `fig07a_dense_speedup_blas{1,72}.csv` / `fig07b_gpu_transfer_floor.csv` + `fig07b_cpu_reference.csv` / `fig07c_axisswap_speedup_blas{1,8}.csv` |
| 8a / 8b | `fig08a_abs_step.csv` / `fig08b_speedup_vs_legacy.csv` |
| 9 speedup / accuracy | `fig09_speedup_vs_n.csv` |

`fig05{cpu,gpu}_ticks.tex` is generated alongside the tables: it carries the
regime tick labels as a `pgfplotsset` style, because pgfplots cannot read
symbolic tick labels from a data table.

`fig02c_concat_speedup_routes.csv` is generated but not currently plotted: it is
the `P=4` route slice in *both* BLAS regimes, while panel 2c plots every `P` at
BLAS=1. The BLAS=64 story is panel 2b and `fig06b`.

`fig06d_gradient_*.csv` are generated but not currently plotted — the gradient
error tracks the potential error closely enough that panel 6d shows only the
potential. They are kept so the panel can be switched without re-deriving data.

## Measurement scope: which figures are end-to-end FMM

This is the first thing to check before quoting any number here, because the
figures mix three scopes and their ratios are **not** composable.

| Scope | Figures | What is inside the timed region |
|---|---|---|
| **Complete `fmm!` evaluation** | fig02c, fig03 (b,c,d), fig05, fig06 (a,c,d), fig08, fig09 | State refresh + B2M → M2M → M2L → L2L → L2B + output writeback. **Nearfield is included**: the direct pairs of the near/self complement are evaluated inside L2B (`_add_host_direct_pairs!` in `src/translate_batched_resident.jl`, `_cuda_direct_pairs_output_kernel!` in `src/translate_batched_cuda.jl`). Task 024 gates every case against an all-bodies `direct!` reference (`direct_scope=all_bodies`), reaching potential errors of 1.2e-8 at `P=4` down to 6.6e-12 at `P=12` — only achievable if the total influence is computed. |
| **Far-field M2L stage only** | fig02a, fig02b, fig06b | One M2L stage pass, per route. No B2M/L2B, no nearfield. |
| **Isolated operator / stage microbenchmark** | fig01, fig07 | A single operator or translation stage on synthetic batches. No `fmm!` call, no tree, no nearfield. |

Two consequences worth carrying into `019a`:

- **fig03's L2B bar is largely nearfield, not a far-field operator.** The largest
  remaining device stage (50 ms at `n=1e5`, untouched by the 019 tuning) is
  dominated by direct-pair evaluation, so no choice of M2L strategy can remove
  it. The roofline gap discussion applies to M2L; L2B is a different problem.
- **Nearfield and far-field cost trade against each other along `P`.** The
  constant-`P` stencil promotes more pairs to the far field as `P` rises, so at
  `n=20000` the direct-pair count falls 227,632 (`P=4`) → 69,664 (`P=8`) →
  39,496 (`P=12`) while L2B's expansion work grows 5.7 → 13.5 → 42.4 ms. Any
  reading of fig05/fig06 vs `P` is reading that trade, not far-field cost alone.

Note that the radix `fmm!` path computes potential + gradient only (no Hessian),
and `target_systems === source_systems` — see the `fmm!(…, ::RadixFMMCache)`
docstring. The legacy octree `fmm!` computes its own nearfield pass as well, so
both paths in `fig08` are complete evaluations — but they are *not* equivalent
work: see the four non-comparabilities in that figure's footnote (threading,
admissibility criterion, absence of an error check at the benchmarked `n`, and
an asymmetric timing boundary).

**`fig08` is the only radix-vs-legacy comparison here.** It is the comparison an
end user actually makes, and its headline is split: the *device-resident* radix
path beats the shipping legacy octree by 1.71× at `n=1e4` and 11.99× at `n=1e5`
(`m13h-1-1`, `P=4`), while the *host* radix path loses to it by 24–152× and, at
`n=1e4`, is ~120× slower than brute-force `direct!`. That is the accepted
consequence of the `019b` decision — legacy stays the CPU default, the radix
cache targets the GPU loop — and there is no roadmap item to optimize the host
radix path.

## What these figures deliberately do not compare

Read together with the caveat lines in each figure:

- **015 microseconds against 019/024 seconds.** Task 015 measured isolated
  scalar per-column kernels on an Apple M2 laptop; 019/024 measure whole-slab
  GEMM lifecycles on HPC nodes. `fig07c` is therefore shown as a ratio only.
- **019's before against 019's after, on equal terms.** The 022 baseline stage
  times are single-shot; the post-tuning columns are min-of-3. The ~25× M2L
  figure mixes timing methods and is annotated as such.
- **`persistent_bytes` / `scratch_bytes` across strategies.** Plan-reported for
  dense and precomputed-y, `Base.summarysize` fallback for concat and factored.
  Only `operator_bytes` (dense) and the CUDA `peak_bytes` device delta are
  compared.
- **CPU `peak_bytes`.** Process-wide `Sys.maxrss()` shared by all four
  strategies, so it is not plotted at all.
- **Occupancy from concat or factored rows.** Those rows report
  `nonempty_classes` and every occupancy column as 0; occupancy must be read
  from precomputed-y or dense rows.

## Not measured anywhere in the project

These gaps are visible as absences in the figures and are carried to the `019a`
final review:

- **Stencil tolerance vs cost.** Every 024 case ran a single fixed
  `ConstantPAnalyticStencil` tolerance, and no other campaign swept it. Panel
  6d is therefore a *precision*-vs-cost view (Float32 vs Float64 at each `P`),
  not an accuracy-vs-tolerance surface. A tolerance sweep would be new work.
- **Julia-thread parallelism in the 024 campaign.** All 024 CPU data is
  `julia_threads=1`; "multi-thread CPU" in figs 01--08 means BLAS threads only.
  Task 024b and fig09 add the separate requested legacy-octree 1-vs-64 Julia
  thread scaling comparison.
- **Dense Float32 at `P=12`.** A hard-coded skip in the 024 harness, not an
  observed failure, so those series end at `P=8`.
- **Three noisy CPU rows.** M2L IQR/median 0.113, 0.195 and 0.713 were recorded
  by 024 as unresolved; they are inside the fig05/fig06 aggregates.
- **Julia-thread parallelism in the radix host path.** `src/fmm.jl` has nine
  `Threads.@threads`/`@spawn` sites; `src/translate_batched.jl` and
  `src/translate_batched_resident.jl` have none. Every `fig08` host-radix bar is
  therefore single-threaded against a multithreaded legacy octree (8 Julia
  threads on the HPC hosts, 1 on the Mac), and BLAS threads were never set or
  recorded in the 023 runs.
- **The 023 GPU construction cost at `n=1e4` is not a construction cost.** It
  reads 23.7 s (`m13h-1-1`) / 24.7 s (`m13h-2-1`) because that case runs first in
  the 023 script and absorbs one-time CUDA kernel compilation; the same hosts
  read 0.126 s / 0.030 s at `n=1e5`. Panel 3c plots both and annotates which to
  trust. No campaign separates JIT from construction directly.
- **Device-side stencil generation.** The 019 "implicit stencil" still
  materializes every route and pair on the host, so `fig03b`'s host list-build
  bar is a real remaining cost, not an artifact.
