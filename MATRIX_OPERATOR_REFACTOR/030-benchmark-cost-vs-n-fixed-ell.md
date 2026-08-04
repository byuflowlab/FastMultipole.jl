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
2. **Three fixed-depth series** `ell ∈ {4, 5, 6}` for comparison (user decision,
   `2026-08-03`: each `ell` is held constant across the whole `n` sweep, rather
   than choosing the best `ell` per `n`; `ell = 5` is the `028` optimum and
   `ell = 4/6` bracket it, and `ell = 6` at `n = 1e6` is constructible on the
   hierarchical path per `027`), in **both precisions** (user decision,
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
- Depth/geometry: `ell ∈ {4, 5, 6}` with the radius schedule exactly
  `sched6-5-5` (`ell = 4`), `sched6-5-5-5` (`ell = 5`, the `028` winner), and
  `sched6-5-5-5-5` (`ell = 6`). The schedule must have exactly `ell - 1`
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

- (i) log-log `verdict_step_ms` versus `n`, six series (`ell` 4/5/6 x
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

Not yet completed. Record the reader, date, every required file/artifact, and a
statement that the full `028` verification history and failure ledger were read.

## Verification Notes

To be filled during task execution.

## Approval Notes

To be filled by a different agent after review notes and verification are
complete.
