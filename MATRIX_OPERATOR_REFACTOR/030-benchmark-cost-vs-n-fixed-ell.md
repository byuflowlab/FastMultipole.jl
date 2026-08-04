# 030 Benchmark: Per-Step Cost vs N at 028 Defaults, Fixed-ell Comparison, and Per-N Optimization Recommendations

## Status and Entry Gate

**Added by user request on `2026-08-03`**, after the `019a` final roadmap
Milestone Review was completed and clear-context approved. Not started.

Entry is unblocked: both dependencies (`028` and `019a`) are Done and
clear-context approved. No further approval is required to begin, but the
Mandatory Reading Gate below must be completed and recorded before any cluster
run or analysis conclusion.

Completing this row does **not** reopen `019a`. When this row completes, its
conclusions are recorded as an addendum review note in
`019a-milestone-review-final-roadmap.md`, mirroring the convention adopted for
the deferred `029`. This row neither depends on nor resumes `029`; the `029`
deferral (user direction `2026-08-03`) is unaffected.

## Objective

Measure how the task `028` per-time-step cost scales with body count, at the
`028` shipped-default GPU settings, and turn the resulting per-stage data into
actionable per-`n` optimization recommendations.

Three deliverables:

1. **Cost versus `n`** at the `028` optimal GPU settings, over the `024b`
   body-count grid `n ∈ {1000, 3162, 10000, 31623, 100000, 316228, 1000000}`.
2. **Three fixed-depth series** `ell ∈ {3, 4, 5}` for comparison (user decision,
   `2026-08-03`: each `ell` is held constant across the whole `n` sweep, rather
   than choosing the best `ell` per `n`). `ell = 5` is the `028` optimum at
   `n = 1e6`; the bracket runs **coarser**, not finer, because the optimal depth
   tracks `n` downward — `028` §4.8 measured `ell = 4` as optimal at `n = 2e5`,
   and `024b` selected `ell = 2/3` below `n = 1e5`. Over an `n = 1e3..1e6`
   sweep the coarse side of `ell = 5` is therefore the informative bracket, and
   `ell = 6` was only competitive above the target `n`. (**Revision note:** the
   bracket was `4/5/6` when this row was staged; the user revised it to `3/4/5`
   on `2026-08-03` before any sweep case had run. Job 13035882 was cancelled
   during its preflight and produced no data, so no measurement is affected.)
   Both series run in **both precisions** (user decision,
   `2026-08-03`): the `028` FP16-WMMA/Float32 winner configuration, and a
   Float64 series with `FM028_TENSOR_FORMAT=off`. That is `7 x 3 x 2 = 42`
   cases.
3. **Per-`n` optimization recommendations with estimated savings**, including
   the "fixing the error" lever: the `sched6-5-5-5` radius geometry and `ell = 5`
   were tuned at `n = 1e6`, so at other `n` the delivered accuracy drifts off
   target; retuning the radius schedule and/or `ell` to just meet a stated
   per-`n` accuracy target should recover time. Evidence level (user decision,
   `2026-08-03`): **modeled estimates from the sweep's per-stage data plus a
   small number of targeted H200 spot-check runs** validating the top
   recommendation at 2–3 representative `n`.

The results are plotted as **fig10** in the `024a` figure set.

This is a benchmark and analysis row. It makes **no production `src/` changes**.
Any `src/` change suggested by the analysis requires separate explicit user
approval and would make this row's approval a fresh clear-context pass.

## Dependencies

- `028-performance-feasibility-1m-in-10ms.md`, complete and clear-context
  approved (supplies the winner configuration, the harness, the verdict
  boundary, and the `n = 1e6` anchor).
- `019a-milestone-review-final-roadmap.md`, complete and clear-context approved.
- Transitively: the `024b` scaling study (body-count grid and checksummed
  sampled-direct references), `025`–`027` (hierarchical stencil, the CUDA
  production default), `022`/`023`/`023b`–`023f` (resident lifecycle and
  strategies), and the `024a` figure pipeline.

## Mandatory Reading Gate

Before any cluster submission or analysis conclusion, read all of the following
in full:

1. `START_HERE.md` and `../MATRIX_OPERATOR_REFACTOR.md`.
2. `028-performance-feasibility-1m-in-10ms.md`, including every verification
   entry and the complete failure ledger.
3. `data/feasibility_1m_10ms/report.md` in full.
4. `024b-impl-cpu-gpu-scaling-benchmark.md` — for the body-count grid, the
   sampled-direct reference methodology, the reference checksums, and the
   pre-`028` per-`n` GPU context.
5. `024a-impl-benchmark-visualization.md` and the figure pipeline it defines
   (`scripts/figures_024a_build.sh`, `scripts/figures_024a_prepare.jl`,
   `data/figures/fmfigstyle.tex`).
6. `scripts/benchmark_028_feasibility.jl` and its helpers
   (`scripts/fm028_device_system.jl`, `scripts/benchmark_024b_common.jl`), plus
   the cluster pattern `scripts/cuda_028_submit.sh` / `scripts/cuda_028_run.sh` /
   `scripts/cuda_028_fetch.sh`.

Record the reader, date, files read, and confirmation of completion in the
**Reading Gate Record** below. If any required artifact is absent or
inconsistent with this file, stop and repair the coordination record first.

## Fixed Comparable Workload and Acceptance Boundary

**Workload (frozen for comparability across all 42 sweep cases):**

- Body counts `n ∈ {1000, 3162, 10000, 31623, 100000, 316228, 1000000}` (the
  `024b` grid).
- Deterministic body seed **24025**; sampled-reference seed **24026**; bounds
  `(SVector(-0.01, -0.01, -0.01), 1.02)`.
- Literature **P = 4** (`expansion_order = 3`), `lamb_helmholtz = false`.
- Hierarchical stencil (`HierarchicalRigidStencil`, the CUDA production default
  since `027`); dense strategy (`DenseTranslationM2L()` +
  `MaterializedYRotationM2L()`); `FM028_K = full` (one window per level; the
  shipped default `4096` has the same effect at these sizes); `m2l_threads = 64`;
  `m2l_block_cap = 65536`; counting sort on; symmetric nearfield off; TF32 off.
- Depth/geometry: `ell ∈ {3, 4, 5}` with the radius schedule exactly
  `sched6-5` (`ell = 3`), `sched6-5-5` (`ell = 4`), and `sched6-5-5-5`
  (`ell = 5`, the `028` winner). The schedule must have exactly `ell - 1`
  non-increasing entries; the shipped rule is `(6, 5, …, 5)`.
- Precisions: FP16-WMMA/Float32 (`FM028_TF=Float32`,
  `FM028_TENSOR_FORMAT=fp16`) and Float64 (`FM028_TF=Float64`,
  `FM028_TENSOR_FORMAT=off`).
- `FM028_REPS >= 9`; `FM028_BOUND` includes at least `ab`.

**Acceptance boundary (unchanged from `028`, boundary b):** the reported cost is
the wall-clock median over `REPS` of one complete recurring time step —
tree/route refresh (`update_cuda_radix_state!`) plus evaluation
(`run_cuda_radix_lifecycle!`, covering B2M → M2M → M2L → L2L → L2B+nearfield)
plus output finalization (`finalize_cuda_radix_output!`) plus device Euler
convection (`fm028_euler!`), with zero per-step body H2D/D2H. The harness's
asserted counter contract (`route_uploads`, `operator_uploads`, `body_uploads`,
`influence_downloads`, `metadata_downloads` non-growing after construction;
`expansion_host_copies == 0`) is enforced per case. Boundary (a), eval-only, is
also recorded; boundary (c), the host `fmm!` step, is optional context.

**Accuracy handling.** Error is recorded for every sweep point against the `024b`
checksummed sampled-direct references (512 bodies, seeds 24025/24026,
SHA-256-validated; all seven `n` exist under
`data/cpu_gpu_scaling/references/direct_reference_n<N>.csv`), plus the harness's
on-device Float64 direct cross-check (`ref_cross_check_grad_rel`). There is
**no pass/fail error gate on the sweep points** — the accuracy drift versus `n`
at fixed geometry is itself a deliverable, since the geometry was tuned at
`n = 1e6` (where the FP16 winner delivered gradient rel RMS `1.0593e-3` against
the `n = 1e6`-specific `1.19e-3` gate). Spot-check runs in step 4 must state
their per-`n` accuracy target explicitly and meet it.

**Unconstructible cases.** If a case cannot be constructed or run (device
memory, occupancy, or harness limits), record it in a failures ledger in this
file with the case identity and the specific reason. Do not silently drop it and
do not substitute a modeled value for a measured point.

**Reuse of existing data.** The `n = 1e6`, `sched6-5-5-5`, `ell = 5` points may
be reused from existing `028` artifacts if and only if the configuration matches
this file exactly — verify the manifest and every config column in
`data/feasibility_1m_10ms/cuda_m13h-1-1_20260803-131016.csv` (FP16 winner,
manifest `42a6c254a11ac8a8`, job 13029878, 9.591 ms [9.434, 9.631]) and in the
Float64 `sched6-5-5-5` rows in the same directory. Any mismatch means re-measure.
All other `028` CUDA data is `n = 1e6` or uses a different policy (`hier12`) and
is not directly usable as a series point. The `024b` per-`n` GPU data uses the
pre-`028` flat stencil and different geometry; it is context, not a series.

## Work Plan

### 1. Sweep infrastructure

Add `scripts/cuda_030_run.sh` and `scripts/cuda_030_submit.sh` (and optionally
`scripts/cuda_030_fetch.sh`), cloned from the `028` cluster pattern: rsync
`src test Project.toml MATRIX_OPERATOR_REFACTOR/scripts` plus the reference CSVs
to the staging directory, `Pkg.instantiate` on the login node, then
`sbatch --export=ALL,… --gpus=h200:1 --cpus-per-task=8 --mem=192G`, printing a
source manifest and running the preflight suites as `cuda_028_run.sh` does.

The runner loops the 42 cases over `benchmark_028_feasibility.jl`, which is
already fully parameterized by environment variables (`FM028_N`, `FM028_P`,
`FM028_ELL`, `FM028_K`, `FM028_POLICY`, `FM028_STRAT`, `FM028_TF`,
`FM028_TENSOR_FORMAT`, `FM028_REPS`, `FM028_STEPS`, `FM028_STALE`,
`FM028_BOUND`, `FM028_OUT`, `FM028_REFDIR`, `FM028_SYMMETRIC`). **Reuse that
script unchanged if possible.** If a change is unavoidable, it must keep every
existing `028` mode reproducible, and the change plus its justification must be
recorded in the Verification Notes.

Include completed-case skipping via a CSV prefix grep and a failures ledger,
following the `cuda_024b_run.sh` pattern, so an interrupted sweep resumes
cheaply. Write results to `MATRIX_OPERATOR_REFACTOR/data/cost_vs_n/`.

### 2. fig10

Add `fig10()` to `scripts/figures_024a_prepare.jl` (stdlib-only Julia; use the
script's internal `Table` / `readtable` / `writewide` helpers, no CSV.jl or
DataFrames), writing `data/figures/tables/fig10_cost_vs_n.csv` with the leading
`#` provenance comment, and add `data/figures/fig10_cost_vs_n.tex` following the
shared pgfplots style `data/figures/fmfigstyle.tex`. Update
`scripts/figures_024a_build.sh` if it enumerates figures explicitly.

Suggested panels:

- (i) log-log `verdict_step_ms` versus `n`, six series (`ell` 3/4/5 x
  FP16-Float32/Float64), with the `n = 1e6` FP16 anchor annotated at 9.59 ms;
- (ii) `err_gradient_rel_rms` versus `n` for the same six series, showing the
  fixed-geometry accuracy drift that motivates per-`n` retuning.

### 3. Analysis

From the sweep CSVs (identity/config, structure, construction, stage, boundary,
allocation/counter, and error columns; note that `m2l_per_level`,
`nodes_per_level`, and `routes_per_level` are space-separated sub-fields inside
a single comma column, and nearfield time is fused into `eval_ms`/`l2b_ms`):

- per-`n`, per-`ell`, per-precision stage breakdowns — which stage dominates at
  each point, and where the construction-versus-recurring tradeoff sits;
- per-`n` accuracy headroom or deficit relative to a stated per-`n` accuracy
  target, quantifying the fixed-geometry drift;
- a **per-`n` recommendation table** with columns: `n`, best measured
  configuration and its cost, dominant stage, recommended change, estimated
  saving (ms and %), evidence type (**modeled** or **measured**), and accuracy
  consequence. Candidate levers include retuned `ell`, retuned level-radius
  schedule (the fixed-error lever), precision, and any other lever already
  measured in `028`.

Every modeled number must be labeled modeled. No modeled value may be presented
as a measurement.

### 4. Spot-check validation

Run targeted H200 spot-checks validating the top recommendation at 2–3
representative `n` (suggested: `n = 1e4`, `n = 1e5` or `316228`, and `n = 1e6`).
Report predicted-versus-measured saving for each, and state and verify the
accuracy target used for each spot-check.

### 5. User checkpoint

Present the recommendation table to the user before proposing any change to
production defaults. No production-default change is expected from this row; any
`src/` change requires separate user approval, per the Status section.

## Verification and Compatibility Gates

- The counter contract is asserted for every case (harness default); any
  violation is a failed case, ledgered, not silently accepted.
- Reference checksum validation passes for all seven `n`
  (`reference_source = 024b_csv`), and the on-device Float64 cross-check is
  recorded per case.
- fig10 rebuilds reproducibly from the committed tables via
  `scripts/figures_024a_build.sh`.
- The recommendation table distinguishes modeled from measured for every row.
- Spot-check runs state and meet their per-`n` accuracy targets.
- Production `src/` is unchanged; `git status` shows changes only under
  `MATRIX_OPERATOR_REFACTOR/` and `data/figures/`.
- The existing test suites remain green if any shared script is touched.

## Completion Rule

Complete when:

1. the 42-case sweep is complete, or every missing case is recorded in the
   failures ledger with a specific reason;
2. fig10 is built and reproducible;
3. the per-`n` recommendation table is delivered, with modeled and measured
   entries clearly distinguished and the fixed-error lever quantified;
4. the top recommendation is validated by H200 spot-checks at 2–3 representative
   `n`, with predicted-versus-measured savings reported; and
5. the user checkpoint in step 5 has been held.

Then mark Done and obtain clear-context approval from an agent other than the
one that completed the row. On approval, add the addendum note to `019a`.

## Artifacts or Production Surface

- `MATRIX_OPERATOR_REFACTOR/scripts/`: new `cuda_030_run.sh`,
  `cuda_030_submit.sh`, optional `cuda_030_fetch.sh`; `fig10()` added to
  `figures_024a_prepare.jl`; `figures_024a_build.sh` updated if it enumerates
  figures.
- `MATRIX_OPERATOR_REFACTOR/data/cost_vs_n/` (new): sweep CSVs, `.classes.csv`
  companions, cluster logs, source manifests, failures ledger, and the task
  report.
- `MATRIX_OPERATOR_REFACTOR/data/figures/`: `fig10_cost_vs_n.tex`, its PDF/PNG,
  and `tables/fig10_cost_vs_n.csv`.
- Production `src/` untouched. Data and figure work follows the project's
  TikZ/pgfplots plus stdlib-Julia conventions.

## Reading Gate Record

**Completed `2026-08-03`** by the executing agent (Claude Opus 5, session
implementing this row). Files read in full:

1. `START_HERE.md` (all sections, both phase tables and the protocol rules) and
   `../MATRIX_OPERATOR_REFACTOR.md` (all 326 lines).
2. `028-performance-feasibility-1m-in-10ms.md` (all 866 lines), **including the
   complete Verification Notes for Phase A and Phase B cycles 1–4 and Stages
   5–8, every failure-ledger entry** (jobs 13016917; 13027048, 13027092,
   13027167, 13027174, 13027188, cancelled 13027374; 13028465; failed re-gate
   13031187), and all three Approval Notes sections.
3. `data/feasibility_1m_10ms/report.md` (all 889 lines), including §1–§5 marked
   as the superseded Phase A record, §6b–§6.8, and §7 threats to validity.
4. `024b-impl-cpu-gpu-scaling-benchmark.md` (all 289 lines) — body-count grid,
   sampled-direct reference methodology and checksums, and the per-`n` GPU
   context this row re-measures under the `028` geometry.
5. `024a-impl-benchmark-visualization.md` (all 474 lines) and the figure
   pipeline it defines: `scripts/figures_024a_build.sh`,
   `scripts/figures_024a_prepare.jl`, `data/figures/fmfigstyle.tex`,
   `data/figures/fig09_cpu_gpu_scaling.tex`, `data/figures/README.md`.
6. `scripts/benchmark_028_feasibility.jl` (all 639 lines) and its helpers
   `scripts/fm028_device_system.jl`, `scripts/benchmark_024b_common.jl`; the
   cluster pattern `scripts/cuda_028_{submit,run,fetch}.sh` and the
   skip/failure-ledger idiom in `scripts/cuda_024b_run.sh`.

Confirmed: the full `028` verification history and failure ledger were read.
Facts carried into this row's design, each verified against the source rather
than taken from prose:

- The harness applies **no accuracy gate** (errors are recorded, never
  thresholded), so `030`'s "no pass/fail error gate on sweep points" needs no
  harness change. The `1.19e-3` constant lives in
  `scripts/select_028_stage7_winner.jl:7`.
- `FM028_OUT` is a **file path** opened `"w"` and written only at the end of a
  process (`benchmark_028_feasibility.jl:622-629`); `FM028_TENSOR_FORMAT` is
  process-global (`:87-107`); a `sched*` policy pins one `ell` because the
  schedule must have exactly `ell-1` entries (`:226-227`), and policy/`K`
  validation at `:599` sits outside the `try`. These three facts dictate the
  one-process-per-case sweep structure adopted below.
- A **missing reference silently degrades** a point to
  `reference_source=device_direct` (`:325-340`), so the runner gates on
  reference presence and checksum before any case.
- The counter contract is asserted unconditionally (`:419-434`) regardless of
  `FM028_BOUND`.
- `028` pins `bounds=(BOX_MIN, BOX_SIZE)` at every `n` (`:349-351`), so the grid
  is identical across the `n` sweep and this row's `n`-scaling is internally
  clean — the same property report.md §7 relies on for `028` §4.8.

## Verification Notes

### Pre-sweep local verification (`2026-08-03`)

**Geometry pre-flight.** `HierarchicalRigidStencil` constructs at `P = 4`,
`lh = false` for every series depth in both precisions:
`ell = 3` → `sched6-5` (eps 3.4626), `ell = 4` → `sched6-5-5` (6.9252),
`ell = 5` → `sched6-5-5-5` (13.8503); `ell = 2` (`sched5`) and `ell = 6`
(`sched6-5-5-5-5`) also construct and are available to spot-checks. `FM028_K=full`
resolves to **874 window classes at `ell = 3, 4, 5` alike**, so `window_classes`
is *not* an `ell`-dependent confounder across this sweep (it would be at
`ell = 2`, where the union is 682).

**Driver verification.** `cuda_030_run.sh` was exercised locally through
`FM030_DRYRUN=1`: 42 cases, correct depth→schedule mapping, 21/21 precision
split, all seven `n`; completed-case skip fires only on a `fit=true` file and
correctly retries a `fit=false` one; the `*.classes.csv` companion does not
cause a false skip; `sched6-5-5` and `sched6-5-5-5` do not collide in the
per-case CSV name; unsupported geometries and an unknown mode are ledgered or
rejected. `fit` was confirmed to be field 21 of the 86-column schema against a
real 028 CSV.

**Structure oracle.** `analyze_030_structure.jl` reproduces the measured
structure of **19 independent 028 configurations** (nine uniform radii plus
scheduled policies, `ell = 4/5/6`, up to `n = 1e6`): exact integer agreement on
every saturated leaf grid, including the 028 winner's per-level route breakdown
`1896 / 119784 / 1145544 / 11037576`, `n_nodes = 37449` and
`n_direct = 1729144`. See the harness convection-drift finding below for why
unsaturated grids are gated on a bound instead.

**Harness finding — the structure columns are post-motion.**
`benchmark_028_feasibility.jl` writes its row *after* the timing loops, and
`step!()` advances every body by `FM028_DT` on each call; the boundary-b
samples, the stale probe, the two allocation probes, the counter-contract step
and the `FM028_STEPS` loop together run about `2*REPS + 5 + STEPS` Euler updates
first. Because `fm028_euler!` clamps to `[0,1]`, the motion concentrates bodies
and sparse cells empty out, drifting counts **down** by <0.01%. This is why one
configuration recorded 252483 / 252477 / 252486 occupied cells in three
different 028 jobs. Saturated grids cannot drift and are gated exactly;
unsaturated grids are gated at `DRIFT_TOL = 5e-4` with the observed maximum
(0.0067%) reported. Not stated in the 028 record; recorded here because it
bounds how precisely any structure-keyed model can be held to the CSVs.

**Stage-accounting identities, verified numerically on 028 rows** (four
`sched*` rows in `cuda_m13h-1-1_20260803-131016.csv` and
`cuda_m13h-1-2_20260803-075427.csv`) rather than assumed:

1. **`route_gen_ms` is inside `m2l_ms`, not inside `refresh_ms`.** At the 028
   winner, `refresh_ms = 0.955` against `grid+occupancy+direct_gen+groups =
   1.196`, while adding `route_gen` gives 2.375 — far above the measured
   refresh. Consequence for this row's deliverable: **`route_gen` is 1.179 ms of
   the winner's 2.543 ms M2L, i.e. 46% of "M2L" is window generation, not the
   tensor GEMM.** The dominant-stage column and the recommended lever change
   accordingly — window generation scales as `|V_L|·N_L` and is reduced by fewer
   levels or a smaller radius, not by a faster GEMM.
2. **`eval_ms` is less than the sum of its stages, stably.** `eval − Σ(b2m, m2m,
   m2l, l2l, l2b)` is −1.090, −1.081, −1.057, −1.106 ms across the four rows:
   not noise, but the measured nearfield/L2B overlap gain, since
   `CUDA_OVERLAP_NEARFIELD` is on in the pipeline while the harness times the
   standalone fused L2B. It is carried as an explicit modeled term, never
   absorbed.
3. **Composite identity** `verdict − (refresh + eval + finalize + euler)` is
   +0.11 to +0.15 ms (~1.2%), consistent with median-of-sum vs sum-of-medians;
   carried as `unattributed_ms`.

**Figure pipeline.** Figures 1–9 regenerate byte-identically with `fig10` added.
A synthetic 42-case campaign exercised `fig10()` end to end (13-column table,
compiling PDF); the synthetic inputs and outputs were then removed, and the
prepare script returns to 62 generated files with `fig10` skipping cleanly.

### Campaign

- Job **13035882** (`ell = 4/5/6`) was cancelled 2:44 in, during the lifecycle
  preflight, after the user revised the depth bracket. It wrote no case CSVs;
  the remote campaign directory was verified empty. No measurement is affected.
- Job **13035897** (`ell = 3/4/5`) is the campaign of record. Preflight green
  (`LIFECYCLE_TEST_EXIT=0`, `CONVECTION_TEST_EXIT=0`,
  `COUNTING_SORT_TEST_EXIT=0`), `REFERENCE_GATE_EXIT=0`.
  *(In progress — completion, per-case results and the failure ledger to be
  recorded here.)*

## Approval Notes

To be filled by a different agent after review notes and verification are
complete.
