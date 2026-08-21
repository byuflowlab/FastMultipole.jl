# 024a Benchmark Visualization

## Objective

Turn the project's accumulated benchmark evidence into a small set of clear,
human-readable figures that communicate (1) where the matrix-operator paths win
and lose (crossover points and speedups), (2) the magnitude and provenance of
the GPU speedup, and (3) the measurements behind the project's recorded
tradeoff decisions — so a human can judge the project's success and limitations
without reading CSVs.

## Dependencies

- `024-impl-operator-ab-benchmark.md` (definitive integrated end-to-end data)
- `019b-exploratory-smallp-fallback-and-channel-layout.md`
- `019-impl-operator-performance-tuning.md`
- `015-impl-axis-swap-benchmarks.md`
- `008c-implementation-performance-baseline.md`

## Required Reading

- `START_HERE.md`
- The Results/data sections of the dependency task files above (for what each
  dataset measured, its environment, and its caveats)

## Data Sources (existing artifacts; do not re-run benchmarks except to fill gaps)

- `data/impl_performance_baseline/<host>/` (008c: recurrence stage baselines,
  dense-vs-loop head-to-head, GPU transfer/fusion floors)
- `data/axis_swap/<host>/` (015: per-column M2L variant sweeps, footprint,
  batch composition)
- `data/operator_performance_tuning/` (019: H200 before/after stage timings,
  chunk sweep, allocations/storage)
- `data/smallp_fallback_layout/{m12-1-7,m13h-1-1,mecsrs-*}/` (019b: small-P
  crossovers isolated + stage-level, GPU tiny-corner sweep, padded-vs-ragged
  layout timing and storage)
- The `024` result artifacts (definitive integrated CPU/GPU A/B)

## Deliverables

- A plotting script (or scripts) under `MATRIX_OPERATOR_REFACTOR/scripts/`
  that reads the CSV artifacts above and regenerates every figure
  deterministically. Plotting dependencies must not leak into the package or
  test environments (use a self-contained script env or document the required
  packages in the script header).
- Figures written under `MATRIX_OPERATOR_REFACTOR/data/figures/` (PNG and/or
  PDF), each with a caption file or embedded title stating machine, BLAS
  regime, and dataset provenance. At minimum:
  1. **Crossover curves:** per-expansion time vs `P` for production recurrence
     vs per-column operators vs the whole-slab dense path (CPU blas1/blasN),
     with the crossover points marked; φ-only and Lamb-Helmholtz panels.
  2. **Speedup summaries:** dense-path speedup over recurrence vs `P` and vs
     batch/routes (CPU), and GPU-vs-host speedup vs problem size `n` (from the
     019b GPU corner sweep and 019/024 data), log scales where appropriate.
  3. **GPU stage breakdown:** before/after 019 tuning per-stage times (B2M,
     M2M, M2L, L2L, L2B) and the remaining roofline gap; lifecycle totals with
     the host list-build and state-build costs alongside, so the end-to-end
     picture (including the 023 recurring-cost target) is honest.
  4. **Storage/allocation:** expansion/cache/scratch footprint comparisons,
     including the ragged-vs-padded layout storage overhead vs `P` and the
     concat-plan geometry reduction (019).
  5. **Tradeoff/limitation views** as the data supports: e.g. LH-vs-φ-only
     cost ratios, multithread-BLAS small-batch degradation, and the platform/
     regime crossovers among whole-slab concat, per-degree factored,
     precomputed-y, and full dense-translation execution from `024`, including
     construction and memory feasibility; also accuracy-vs-cost (stencil
     tolerance / `P`) where `024` provides it.
- A short index note (`data/figures/README.md`) mapping each figure to the
  question it answers and the CSV(s) it was generated from.

## Non-Goals

- No new benchmark campaigns; re-running is limited to filling small gaps that
  a figure exposes (record any such run's commands and environment).
- No production `src/` changes.

## Verification

Record the exact regeneration command(s). Confirm every figure is reproduced
from the committed CSVs by a clean run of the plotting script(s). Confirm no
package/test environment changes. A reader spot-check: each figure's caption
states what is plotted, on which machine/regime, and what conclusion it
supports.

## Implementation Notes (completed `2026-07-25`)

### Toolchain (user-directed)

TikZ/pgfplots in LaTeX, with the plot data in `.csv` files linked from the
`.tex` sources via `\addplot table`. Directed by the user on `2026-07-24` in
preference to Python/matplotlib or Julia plotting packages. This satisfies the
"plotting dependencies must not leak into the package or test environments"
requirement absolutely: nothing is added to any Julia environment, and the
data-prep script uses Julia stdlib only (`CSV`/`DataFrames` are deliberately not
used, and `DelimitedFiles` is no longer a stdlib as of Julia 1.9, so the CSV
parsing is a plain `split` — verified safe, as none of the campaign CSVs contain
quoted fields or embedded commas).

Verified environment: Julia 1.12.5, TeX Live 2025 (`pgfplots` +
`pgfplotstable`, `compat=1.18`), `latexmk`, ImageMagick 7 for the optional PNG
copies.

### Deliverables

- `scripts/figures_024a_prepare.jl` — stdlib-only Julia; reads the campaign CSVs
  and writes 60 tidy, wide, pgfplots-ready tables/styles into
  `data/figures/tables/`. Hard-errors on a missing source, an unexpected header,
  or an empty selection, since a silently empty panel is the failure mode this
  task most needed to avoid.
- `scripts/figures_024a_build.sh` — two-stage regeneration (prepare, then
  `latexmk` per figure from `data/figures/`), plus optional PNG rasterization.
- `data/figures/fmfigstyle.tex` — shared style. Four fixed categorical colour
  slots (`#2A78D6`, `#EB6834`, `#199E70`, `#4A3AA7`), validated colourblind-safe
  against a white surface in all-pairs mode: lightness band PASS, chroma floor
  PASS, worst CVD ΔE 8.4, worst normal-vision ΔE 16.3, all four ≥ 3:1 contrast.
  Each slot is paired with a distinct mark shape so identity never rests on
  colour alone; reference rules (break-even, roofline, memory gate) use the
  reserved status red and are never data series.
- `data/figures/fig01`–`fig08` (`.tex` sources, `.pdf` and `.png` output),
  35 panels total.
- `data/figures/README.md` — figure/panel index mapping each figure to the
  question it answers and its source CSVs, plus explicit
  "deliberately not compared" and "not measured anywhere" sections.

### Figure set

Eight figures cover the five required deliverables:

1. `fig01_crossover` — required item 1. Small-`P` crossover: per-expansion time
   vs `P` for production recurrence / materialized / factored per-column
   operators (φ and Lamb-Helmholtz panels), plus a speedup row where the
   crossover is read off the break-even rule directly. A ratio row replaced
   marked crossover points: for Lamb-Helmholtz no crossover exists in range, so
   a marker-based presentation would have had nothing to mark.
2. `fig02_speedup_summaries` — required item 2. Whole-slab concat host stage vs
   per-route legacy recurrence vs `P` (both BLAS regimes), and H200-vs-host
   speedup vs `n`.
3. `fig03_gpu_stage_breakdown` — required item 3. Per-stage 022 → post-A+B+D →
   post-E device times with the ~10 ms bandwidth roofline, one-shot setup costs
   beside the step they enable, and the 023 resident step split.
4. `fig04_storage_allocation` — required item 4. Padded-vs-ragged χ overhead vs
   `P`, dense operator payload with the 12 GiB feasibility gate, CUDA peak
   device memory per strategy, expansion/scratch footprints, warmed host
   allocations, and the 019 invariant-cache/geometry payloads.
5. `fig05_strategy_selection` — required item 5 (strategy part). Relative step
   cost and win counts per strategy, plus per-regime step-winner maps for CPU
   (28 regimes) and H200 (14 regimes).
6. `fig06_tradeoffs_limitations` — required item 5 (tradeoff part).
   Lamb-Helmholtz cost multiplier, multi-thread-BLAS small-batch degradation,
   dense construction break-even, and accuracy vs cost.
7. `fig07_baseline_provenance` — **added beyond the listed minimum**, because
   dependencies `008c` and `015` are otherwise unplotted and because the
   task's stated goal includes communicating limitations honestly. It shows the
   008c 31–118× dense projection, the host–device transfer floor that made
   device residency architectural rather than optional, and the ≤1.75× that the
   014 per-column operators actually realized on a CPU — i.e. the projection
   that was *not* met by that lineage.
8. `fig08_radix_vs_legacy` — **added beyond the listed minimum** to show the
   end-user comparison available in task 023: the device-resident radix path
   beats the shipping legacy octree on the measured H200 cases, while the host
   radix path loses badly. Its footnote states the known threading,
   admissibility, accuracy, and timing-boundary non-comparabilities.

### Deviations from the plan, and why

- **`operator_bytes` is non-zero only for `dense`.** The planned
  "operator bytes per strategy" panel was impossible: concat, factored and
  precomputed-y all report `operator_bytes = 0` because they build no
  materialized per-class operator. `fig04b` therefore shows the dense payload
  alone against the feasibility gate, and `fig04c` uses the CUDA `peak_bytes`
  device delta — the one memory column task 024 documents as per-strategy and
  comparable — for the cross-strategy view. CPU `peak_bytes` (process-wide
  `Sys.maxrss()`) and `persistent_bytes`/`scratch_bytes` (mixed provenance) are
  excluded, with the reason stated in the figure footnote.
- **A finding the plan did not anticipate:** on the H200 at φ-only `n=20000` the
  *highest* measured device peak is the whole-slab **concat** path
  (405 → 2082 MiB), not dense (136 → 908 MiB); concat's peak is dominated by
  slab scratch rather than by operators. Dense becomes the memory-limited
  strategy only when `P`, Lamb-Helmholtz or clustering inflate its operator
  payload past the gate. `fig04`'s caption states this explicitly rather than
  repeating the intuitive-but-wrong ordering.
- **Accuracy scope (user-directed `2026-07-24`).** Plot what exists and flag the
  gap. Panel 6d is a Float32-vs-Float64 precision-vs-cost view at each `P`; the
  figure footnote and `README.md` both state that no stencil-tolerance sweep
  exists anywhere in the project's data, so tolerance-vs-cost remains unmeasured
  and is carried to `019a`.
- **Regime tick labels are emitted as a generated `pgfplotsset` style**
  (`tables/fig05{cpu,gpu}_ticks.tex`) rather than bare `\def` macros: pgfplots
  expands a style correctly wherever it appears in an axis option list, which it
  does not do for a macro holding a key list.

### Measurement scope (asked and answered `2026-07-25`)

The figures mix three scopes, now stated in each figure's footnote and tabulated
in `data/figures/README.md`:

- **Complete `fmm!` evaluation, far field *and* nearfield** — fig02c,
  fig03(b,c,d), fig05, fig06(a,c,d), fig08 (both the radix and the legacy octree
  bars). The near/self complement's direct pairs are evaluated
  inside the L2B stage on both paths (`_add_host_direct_pairs!` in
  `src/translate_batched_resident.jl:429`; `_cuda_direct_pairs_output_kernel!`
  launched from `src/translate_batched_cuda.jl:2710`). Independent confirmation:
  task 024 gates every case against an all-bodies `direct!` reference
  (`direct_scope=all_bodies`) with potential errors of 1.2e-8 at `P=4` down to
  6.6e-12 at `P=12`, which a far-field-only pipeline could not reach.
- **Far-field M2L stage only** — fig02a, fig02b, fig06b.
- **Isolated operator/stage microbenchmark, no `fmm!` at all** — fig01, fig07.

Two consequences recorded for `019a`:

1. **fig03's L2B bar is largely nearfield.** The largest remaining device stage
   (50 ms at `n=1e5`, untouched by the 019 tuning) is dominated by direct-pair
   evaluation, so no M2L strategy choice can remove it — the roofline-gap
   discussion applies to M2L, not to L2B.
2. **Nearfield and far-field cost trade along `P`.** The constant-`P` stencil
   promotes pairs to the far field as `P` rises: at `n=20000` the direct-pair
   count falls 227,632 (`P=4`) → 69,664 (`P=8`) → 39,496 (`P=12`) while L2B grows
   5.7 → 13.5 → 42.4 ms. Any `P`-axis reading of fig05/fig06 is reading that
   trade, not far-field cost alone.

The legacy octree `fmm!` comparison was initially absent; `fig08` (below) closes
that gap from the same task-023 `production_integration` CSVs.

### Data-integrity rules enforced by the prep script

- The `P=1` rows of `crossover_stage_blas{1,64}.csv` are duplicated in the
  source; the script reduces by key and errors if a duplicate pair disagrees.
- Column-name drift is normalized: 024 uses `p`/`lh` where every other campaign
  uses `P`/`lamb_helmholtz`, and per-unit columns differ
  (`seconds_per_expansion` / `seconds_per_route` / `*_ms_median`). Every emitted
  table carries its unit in the column name plus a provenance comment line.
- Missing points are written as `nan` and the figures set
  `unbounded coords=jump`, so an unrun configuration is a gap rather than a zero.
- Every eligible 024 raw row is asserted to have `correctness_pass=true` while
  the accuracy table is built.

### Verification performed

Clean regeneration from the committed CSVs only:

```bash
cd /Users/ryan/Dropbox/research/projects/tmp3/FastMultipole
rm -rf MATRIX_OPERATOR_REFACTOR/data/figures/tables \
       MATRIX_OPERATOR_REFACTOR/data/figures/*.pdf \
       MATRIX_OPERATOR_REFACTOR/data/figures/*.png
bash MATRIX_OPERATOR_REFACTOR/scripts/figures_024a_build.sh
```

All 57 generated tables/styles and all 8 PDFs (plus PNGs) regenerate; `latexmk` exits 0 for every
figure with no missing-file or empty-plot warnings. Every figure was rendered and
visually inspected: no empty axes, no unlabeled series, log axes wherever the
span exceeds a decade, and every panel carries machine, BLAS regime, source
dataset and caveats in its footnote.

Numeric spot-checks against the dependency task files, all reproduced by the
generated tables:

- `fig03`: M2L 1698 → 66.0 ms, whole lifecycle 1.863 → 0.116 s, host list build
  5.55 → 0.35 s.
- `fig05`: CPU 63 dense / 21 precomputed-y out of 84; H200 20 dense /
  21 precomputed-y / 1 factored out of 42.
- `fig02`: concat host speedup 7.6× at `P=1`, 1.7× at `P=4`, 0.69× for
  Lamb-Helmholtz at `P=8`; H200 ~860× end-to-end at `n=10^4`.
- `fig07c`: 1.75× factored φ-only at `P=20`, 1.23× materialized LH at `P=20`.
- `fig04a`: +33.3% / +38.5% padding overhead at `P_phi=1`.
- `fig06c`: CPU break-even 4–18 930 steps; H200 542–18 208 at `P=4` and
  4120–733 902 at `P=8`.

Environment isolation confirmed: this task added and modified files only under
`MATRIX_OPERATOR_REFACTOR/`. `Project.toml`, `test/Project.toml` and the `src/`
tree were already dirty from earlier rows (`023`–`024`) at the start of this
work and were not touched by it; no `Manifest.toml` was added or changed, and
the figure toolchain requires no Julia packages at all. No new benchmark runs
were performed.

### Review remediation (`2026-07-25`)

The first clear-context review found two significant presentation gaps and, with
user permission, they were corrected from existing CSVs only:

1. Figure 3 no longer calls the 019 host list/state construction bars recurring
   per-step costs. They are labeled as the one-shot setup costs they were; the
   caption now states that task 023 replaced both with the device-side update,
   and panel 3c remains the actual recurring-step split.
2. Figure 1 now places the whole-slab BLAS-1/BLAS-64 crossover beside the
   isolated per-column crossover (separate panels because the harness scopes
   differ). Figure 2 adds an explicit CPU route-count axis and a matched,
   integrated task-024 best-resident CPU/H200 speedup-vs-`n` panel, alongside
   the retained 019b corner sweep.

The regeneration command above was rerun after remediation: 52 generated
tables/styles, seven PDFs and seven PNGs; every TeX source compiled successfully.
The new tables are `fig01c_wholeslab_{phi,lh}.csv`,
`fig02c_concat_speedup_routes.csv`, and
`fig02d_integrated_gpu_speedup.csv`. No benchmark, package, test-environment, or
production-code change was made. Because these are substantive review-driven
edits, task 024a remains unapproved pending a different clear-context reviewer.

### Second review remediation (`2026-07-25`)

A second clear-context review found two further significant presentation gaps
and, with user permission, corrected both from existing CSVs only:

1. Figure 4 now includes the previously omitted footprint/allocation evidence:
   a within-concat H200 comparison of plan-reported expansion and scratch bytes
   versus order/channel, and the task-019 warmed host allocation measurements
   per M2L chunk and M2M/L2L group. The within-strategy and stage-specific-unit
   limits are stated explicitly, avoiding the incomparable cross-strategy
   `scratch_bytes` columns identified by task 024.
2. Figure 6's caption no longer describes the per-order CPU minima (4--10
   steps) as the whole large-problem construction-amortization range. It now
   gives the measured large-`n` ranges: 4--35 steps for uniform Float64, 6--109
   for clustered Float64, and 52--220 for uniform Float32.

The preparation output is now 57 tables/styles and the build emits eight
PDF/PNG figures. No benchmark, package, test-environment, or production-code
change was made. These substantive edits
again leave task 024a unapproved pending a different clear-context reviewer.

### Third increment: radix-vs-legacy and one-shot cost (`2026-07-25`)

Two gaps were closed, again from committed CSVs only and with no new benchmarks:

1. **`fig08_radix_vs_legacy` (new figure, 2 panels).** The end-user question —
   is the radix path faster than what ships today? — had no figure. Built from
   all three task-023 `production_integration` CSVs: (a) absolute per-step wall
   time, log-y grouped bars over host radix / legacy octree / brute-force
   `direct!` / H200 radix, grouped by `(host, n)`; (b) speedup over the legacy
   octree with a break-even rule at 1. The result is split and the caption says
   so plainly: the device-resident radix path is a genuine win at `P=4` —
   1.71× (`n=1e4`) → 11.99× (`n=1e5`) on `m13h-1-1`, 1.83× → 11.76× on
   `m13h-2-1` — while the **host** radix path is 24–152× *slower* than legacy
   and, at `n=1e4`, ~120× slower than brute-force `direct!`, i.e. slower than
   doing no FMM at all. Task 023 records only the `n=1e5` CPU point in prose and
   treats the CPU result as an accepted consequence of the `019b` decision; the
   caption states that there is no roadmap item to optimize the host radix path.
2. **`fig02` panel (c) now covers the whole "vs routes" axis.** It previously
   plotted only the `P=4` slice, which omitted the tiny-batch extreme. It is now
   a scatter over every `(config, P)` point at BLAS=1 — 8 → 1.2e7 routes,
   colour by channel, mark by config — deliberately a scatter because the route
   count is non-monotonic in `P` (`small_constp` runs 8, 59334, 144290, 31254,
   144290, 15492 for `P` = 1,2,3,4,6,8), so a line would draw an ordering the
   experiment does not have. The answer it gives is negative and stated as such:
   route count alone does not predict the win — the largest speedup in the sweep
   (7.6×) is at the *smallest* population (3096 routes), 11.4M routes gives 1.9×,
   and the 8-route point gives 3.2×. Order and channel dominate. The old `P=4`
   two-BLAS table is still generated and is documented in `README.md` as
   generated-but-unplotted.
3. **`fig03` panel (c), one-shot setup vs recurring cost** (fig03 regridded to
   2×2; the pre-existing step-split panel becomes (d)). GPU cache construction
   at `n=1e5` is 0.126 s against an 0.086 s step — device residency amortizes in
   about two steps — while the CPU pays 0.41 s of construction in front of a
   24.6 s step.

**Four non-comparabilities, none of them disclosed in task 023, are stated in
the `fig08` footnote and verified against the source tree:**

1. **Threading.** `src/fmm.jl` has nine `Threads.@threads`/`@spawn` sites;
   `src/translate_batched.jl` and `src/translate_batched_resident.jl` have none
   (verified by grep). Legacy is multithreaded, host radix is not — which is why
   the gap collapses from 152× at `threads=8` to 28× at `threads=1` on the Mac.
   BLAS threads were never set or recorded in the 023 runs.
2. **Different admissibility criteria.** Radix: `ConstantPAnalyticStencil` at
   `stencil_epsilon=1e-4` on a fixed uniform `2^4` grid. Legacy:
   `multipole_acceptance=0.4`, `leaf_size=20`, shrunk adaptive octree. Same
   `P=4`, structurally different work.
3. **No error check at the benchmarked `n`.** The 023 script computes no error
   norms and never uses its `direct!` result as a reference; the only accuracy
   evidence in task 023 is at `n=500`.
4. **Asymmetric timing boundary, against legacy.** Legacy timing includes both
   octree builds and the interaction-list build every call
   (`src/fmm.jl:1037-1058`); the radix `step` excludes its cache construction,
   which is recorded separately and now plotted in fig03(c). The boundary
   therefore favours radix and legacy still wins by 152× — the bias *understates*
   how slow the host path is.

Protocol, also in the caption: times are the **minimum** over reps (radix 5
jittered steps, legacy 3 reps); `direct!` is a **single** un-repeated `@elapsed`
run only at `n=1e4`. Thread counts come from the run sidecar `.md` files, which
the CSVs omit — the prep script asserts the sidecar records the thread count it
labels each case with, so a re-run under different threading fails loudly rather
than mislabeling.

**JIT contamination, plotted rather than hidden** (user decision `2026-07-25`):
the 023 GPU `construct` phase reads 23.7 s at `n=1e4` but 0.126 s at `n=1e5` on
the same host (24.7 s / 0.030 s on `m13h-2-1`). The `n=1e4` case runs first, so
that value is dominated by one-time CUDA kernel compilation. Both bars are drawn
in fig03(c), the contaminated one is annotated in the reserved status colour, and
the caption plus `README.md` state which value is representative.

Regeneration after this increment: **60 generated tables/styles, 8 PDFs and 8
PNGs**, `latexmk` exit 0 for every figure with no missing-file or empty-plot
warnings; every changed figure was re-rendered and visually inspected (the first
render put the JIT annotation under panel 3c's legend, and panel 2c's break-even
rule drew as two stray dots on a marks-only axis; both were fixed). Numeric spot-checks against the source CSVs: fig08 reproduces
8.663 / 0.0568 / 0.0725 / 0.0331 s at `m13h-1-1` `n=1e4` and the 1.71× / 11.99×
GPU-vs-legacy ratios; fig03(c) shows the 23.7 s vs 0.126 s construct pair; fig02(c) places the
`small_constp`/`P=1` point at 8 routes. The
`README.md` index, panel map, scope table and "not measured" section were updated
to match what is actually built, including correcting the now-false statement
that the legacy octree is not benchmarked against the radix path in any figure.
No benchmark, package, test-environment, or production-code change was made;
`git status` shows changes only under `MATRIX_OPERATOR_REFACTOR/`. These
substantive edits again leave task 024a unapproved pending a different
clear-context reviewer.

## Approval Notes

**Approved `2026-07-28`** by a clear-context reviewer (did not perform the work).

Verification actually performed by the reviewer:

- **Reproduction.** Backed up `data/figures/tables/`, reran
  `scripts/figures_024a_prepare.jl`, and diffed: every fig01–fig08 table and
  both `fig05{cpu,gpu}_ticks.tex` regenerate **byte-identically**. All eight
  `fig0[1-8]*.tex` sources then compiled with `latexmk -pdf -halt-on-error`,
  exit 0, no missing-file warnings.
- **Independent numeric recomputation** from the source CSVs, not from the
  generated tables: fig05 win counts recomputed with a standalone awk pass over
  `case_rankings.csv` → 63 dense / 21 precomputed-y (CPU) and 20 / 21 / 1
  (H200), exactly matching `fig05b_wins_*.csv`; fig03a M2L `1.698 s → 6.599735e-2 s`
  and the whole `t_*` stage row confirmed against `cuda_022_baseline.csv` and
  `cuda_019_phaseE_throughput.csv` at the declared `chunk=2^17` key; fig08's
  8.663 / 0.056773 / 0.072500 / 0.033127 s and the 1.71× / 11.99× ratios
  confirmed against `benchmark_023_m13h-1-1_*.csv`; fig04a padding overheads and
  fig02a's 7.62× confirmed against their sources.
- **fig06c filter audit.** The `break_even_steps > 0` filter was checked rather
  than assumed: every dropped row is a `precomputed_y -> dense` pair (dense is
  not the steady winner, so its amortization is undefined, not zero), and every
  retained row is `dense -> precomputed_y`. The panel therefore does plot
  dense's construction break-even as its title and caption claim, and the
  caption's explanation of the absent H200 `P=12` point is correct.
- **Caveat honesty spot-checks.** The post-E B2M bar (14 ms vs 1.4 ms) is a real
  anomaly in the source row and is explicitly explained as sweep jitter in the
  fig03 caveats, with the other two chunk widths cited; the `2^17` chunk is
  disclosed as chosen for least scratch rather than for speed. The fig08
  non-comparabilities and the fig05 noisy-row disclosure match the underlying
  data.
- **Environment isolation.** The prepare script writes only inside
  `data/figures/tables/` (single `mkpath`/`open` target plus the tick/status
  `.tex`), reads only under `data/`, and ran under no Julia project with stdlib
  only. The `Project.toml` / `test/Project.toml` diffs are the CUDA weakdep and
  test-dep changes from `022`/`023`, unrelated to plotting.

Assessment against the review criteria: objectives met (all five required
figure items present, plus two justified additions and the `fig08` end-user
comparison); correctness confirmed by independent recomputation; robustness good
(hard errors on missing source, header drift, empty selection, duplicate-key
disagreement, and a sidecar thread-count assertion; `nan` + `unbounded
coords=jump` so gaps never read as zeros); minimally invasive (no `src/`, no
package or test environment change); readable (self-describing captions stating
machine, regime, provenance, conclusion, caveats and measurement scope).

Two non-blocking observations, neither requiring a change to `024a`'s
deliverables:

1. `scripts/figures_024a_build.sh` currently **fails** at stage 1 with
   `fig09 requires 14 unique CPU cases, found 13`, because in-flight task `024b`
   added `fig09()` to the same prepare script and its campaign is incomplete.
   Setting `FM024B_ALLOW_PARTIAL=true` makes the run succeed and reproduces the
   committed `fig09` tables byte-identically. This is `024b`'s row to close; the
   escape-hatch variable is not yet mentioned in `data/figures/README.md` or the
   build script header.
2. `fig05` panels (d) and (e) leave a large whitespace block between them —
   cosmetic only, every series is legible.

Both observations were documented at user request on `2026-07-28`, after the
approval above: `FM024B_ALLOW_PARTIAL` is now described in
`data/figures/README.md` and in the `figures_024a_build.sh` header, and the
`fig05` whitespace is recorded as a known cosmetic item in
`data/figures/README.md`. These are comment/prose additions only — no plotting
logic, no figure source, no table, no benchmark and no production code changed,
and figures 1–8 still regenerate byte-identically — so they do not invalidate
this approval.
