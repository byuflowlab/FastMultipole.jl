# 036 Milestone Review: Integration Phase

## Status and Entry Gate

**Added by user request on `2026-08-04`; renumbered by roadmap review on
`2026-08-05`.** **Done `2026-08-12`** (review complete, all verdicts recorded
below; clear-context approval pending).

Entry gate: Integration rows `031`, `031a`, `032`, `032a`, `033`, `034`, and
`035` must all be Done and clear-context approved. This is a blocking
Milestone Review: no later row in this or a future phase may start until this
row is complete and clear-context approved.

## Objective

Perform the standard Milestone Review duties from `START_HERE.md` and render
verdicts on the Integration Phase goals.

Review scope:

1. **Interface generality:** verify the shipped device-system interface is
   general-consumer-first and that a third party can connect without reading
   FastMultipole internals.
2. **Correctness and accuracy:** confirm `034` correctness and that every
   `035` winner and speedup numerator passed sampled velocity RMS `≤1e-3`.
   Confirm Jacobian RMS was logged for every reported configuration and was
   clearly labeled diagnostic rather than gating.
3. **Speedup evidence:** verify `035` contains the sole final report, including
   the wake-at-cube-parameters result, stage profiles, eligible speedups, the
   per-U/J-solve `030` ratio, the separate RK3-step cost, and the optimization
   ledger. Ensure no speedup ratio uses a `033` baseline that failed the
   velocity tolerance.
4. **Performance closure:** verify every credible lever with at least 5%
   expected end-to-end U/J-solve gain was implemented or ruled out with
   evidence, and remaining sub-5% items are recorded without extending the
   campaign.
5. **Downstream compatibility:** confirm FLOWVPM CPU users and the
   FLOWUnsteady/VortexLattice-facing public API are unaffected and CPU tests
   pass at the final `gpu-full` head.
6. **Cross-repo hygiene:** confirm FLOWVPM commits live on `gpu-full`,
   FastMultipole commits live here, and cross-repo change pairs are recorded.
7. **Improvement hunt:** record any further significant opportunities as
   follow-on proposals; do not silently expand this completed phase.

## Dependencies and Reading

- `031`, `031a`, `032`, `032a`, `033`, `034`, and `035`, all Done and
  clear-context approved.
- Read all of `../MATRIX_OPERATOR_REFACTOR.md`, this `START_HERE.md`, every
  Integration task and its listed artifacts, `../FLOWVPM.jl/CLAUDE.md`, and
  the relevant final state of FLOWVPM's `gpu-full` branch.

## Accumulated Review Items

Items flagged mid-phase for this review to verify (add here as the phase
progresses; do not resolve them in this file):

- **Consumer migration hazard (from `034`, 2026-08-06):** legacy consumer
  hooks written against the old fixed-row target-buffer layout (switchless
  `set_hessian!`-style setters or hard-coded rows `4`/`5:7`/`8:16` under
  `@inbounds`) silently corrupt memory on `matrix-ops` when any preceding
  standard output is disabled (e.g. `scalar_potential=false`). A "Migrating
  From the Fixed-Row Buffer Layout" section was added to
  `docs/src/advanced_usage.md` (2026-08-06). The review should confirm the
  migration guidance is adequate under scope item 1 (third-party
  connectability) and item 5 (downstream compatibility), and that FLOWVPM's
  shims (`gpu-full` commit `4df2bc0`) match it.
- **FLOWVPM `CLAUDE.md` staleness (from the `034` approval, 2026-08-06):**
  its Phase 4 section claimed H200 validation was pending after jobs
  13061046/13061128 had passed; refreshed 2026-08-06. Confirm it reflects
  the final `gpu-full` state at review time.

## Review Notes

**Reviewing agent:** Claude Fable 5, `2026-08-12`. Reading completed per the
Dependencies list: `../MATRIX_OPERATOR_REFACTOR.md` in full, `START_HERE.md`
(Milestone Review duties + Integration Phase preamble), all seven Integration
task files (`031`, `031a`, `032`, `032a`, `033`, `034`, `035`) in full
including their approval records, `integration-api-spec.md` reviews (via the
`031` record), `docs/src/device_interface.md` and
`docs/src/advanced_usage.md`, `../FLOWVPM.jl/CLAUDE.md`, and the final
FLOWVPM `gpu-full` state (head `5dd0d85`, clean tree). Local verification was
read-only: campaign CSV sha256 re-checks, gate/J column audits over all five
cycle CSVs, FLOWVPM export-surface and commit-scope diffs against the
branch point `e2bd487` and the approved-034 state `80eaf8d`. H200 evidence was
judged from committed logs/CSVs (no cluster access). No production code was
modified.

**Phase-consistency duty (START_HERE items 3–4):** all seven rows are Done
and clear-context approved with recorded user sign-offs at every mandated
checkpoint (031 spec sign-off; 032a default selection; 035 cycles 1, 2, 3A,
3D). Work stayed inside the Integration Phase scope: FastMultipole changes on
`matrix-ops`, FLOWVPM changes on `gpu-full`, no Theory-phase-gate violations,
and the one structural item above the lever bar was handed to `037` rather
than folded in. Task ordering matches the `START_HERE.md` index.

### Scope item 1 — Interface generality: **PASS**

The shipped surface is general-consumer-first. `031`'s spec survived four
clear-context reviews and is written for an arbitrary external consumer with
FLOWVPM as a worked example; `032` promoted it into `src/` as an exported,
documented API (`body_type`, `direct_kernel`, `data_per_body`,
`strength_dims`, `has_vector_potential`, `get_position`, `source_to_buffer!`,
`buffer_to_target!`, `recenter!`, each with ownership/lifecycle/allocation
docstrings). A third party can connect from
`docs/src/device_interface.md` alone (376 lines: consumer surface, delivery
semantics, packed layout, capacity/no-realloc contract, `recenter!`,
`direct_kernel` functor surface incl. custom kernels and the adequacy gate,
LH/vortex guidance, resident-vs-transfer decision framework, v1 restrictions,
worked example) plus the runnable `examples/device_resident_system.jl`; the
page is registered in `docs/make.jl`. The decisive evidence is `034`'s
record: FLOWVPM connected through the published surface and found **zero
gaps in the 032 device interface itself** (the two hazards found were
legacy-consumer migration issues, covered under item 5 and the accumulated
items below).

### Scope item 2 — Correctness and accuracy: **PASS**

- `034` correctness: five deliverables verified on H200 (job 13061046, user
  deliverable-4 sign-off) plus the closing sha256-checksummed 033-reference
  gate (job 13061128, all Float64 u_rel_rms ≤ 1e-3), with the 023 counter
  contract flat; clear-context approved 2026-08-06.
- Every `035` winner and speedup numerator passed the sampled velocity RMS
  ≤ 1e-3 gate. Re-audited locally: `fm035_cycle3d.csv` sha256 matches the
  record (`69e5790b…`), all 16 rows `gate_pass=true`, `counters_flat=true`;
  `fm035_cycle1.csv` 22/22, `fm035_cycle2.csv` 8/8, `fm035_cycle3b.csv`
  12/12 gate-passing. Sweep/3A rows that failed the gate (20/75 and 8/33)
  were never presented as winners — the record explicitly names them as
  rejections (e.g. cube 1e6 ℓ5 q16 at 1.27e-3; P4 at reduced shells).
- Jacobian RMS is logged for **every** configuration (0 missing `j_rel_rms`
  values across all 150 rows of the five cycle CSVs) and is labeled
  "diagnostic" consistently in the pre-registration, every results table, the
  Final Report measurement-policy paragraph, and the Verification Gates.
- Bonus rigor beyond the gate: cycle 3C's cutoff/FMM error decomposition
  (independent host Float64 erf oracle, 1.6e-16–3.0e-15 reproduction of the
  033 references) led to the *stronger* conservative-sum acceptance policy,
  under which the shipped P5/3.668 defaults pass at all four case/scale
  points while the old P4 default demonstrably failed at cube n=1e6.

### Scope item 3 — Speedup evidence: **PASS (with one placement note)**

The `035` Final Report (§ "Final Report (§3, definitive — 2026-08-12)") is
the sole definitive report and contains: final per-case U/J and RK3 tables
with warmup/repeat/median policy stated (§1); the 033-baseline eligibility
policy with the failing historical CPU timings preserved but **no CPU
speedup headlined** — correct, since every FMM-active default-parameter 033
CPU row fails the 1e-3 gate and the gate-passing rows exist only at
n ≤ 3162 with no matched GPU measurement (§2) — so **no speedup ratio uses a
failed 033 baseline** (none uses a 033 baseline at all; headline speedups
are GPU-vs-shipped-034-coupling and same-job anchors); per-stage profiles
and bottleneck movement incl. the counter-free bound-ness analysis (§3);
the matched-n 030 per-U/J-solve ratio with workload differences itemized and
RK3 explicitly excluded from the ratio and reported separately (§4); and the
full cycle ledger with implemented/rejected/remaining-sub-5% levers (§5).
Figures `fig09/10/11_035_*` exist, are tracked (build artifacts gitignored
per commit `d076399`), and the prior clear-context review verified their
tables regenerate with zero diff. **Note:** the wake-at-cube-parameters
result (101.0 ms vs 16.2 ms wake-optimal, 6.2x slower; per-case depth
selection mandatory) lives in the Work Record (initial-sweep Deliverable 2)
rather than being restated inside the Final Report section, and was measured
at the cycle-0 winners, not re-measured at the final P5 defaults. The
conclusion is structural (depth mismatch degenerates the wake toward
all-direct) and unaffected by later cycles; contained-in-035 satisfies the
scope item as written. Non-blocking.

### Scope item 4 — Performance closure (≥5% lever rule): **PASS**

Every credible ≥5% lever was implemented or ruled out with measurement:
implemented — coupling defaults (cycle 1, 4.25x/2.2–2.9x realized), B2M
block-per-cell (cycle 2, stage 9–42x, wake 1.48–1.66x e2e, with an honest
root-cause of the overlap-masked shortfall), P/cutoff/stencil co-design
(cycles 3A–3D, 1.06–2.23x further with errors improved everywhere); ruled
out by measurement — window classes, rho_t 4.789, boosted-coarse schedules,
precomputed-y, FLOWVPM overhead, B2M residual, nearfield
micro-optimization (bound-ness analysis: kernel at 39–60% of the op
ceiling, GPU saturated, DRAM 2–12%; consistent with the falsified 029
nearfield-ILP precedent). The one surviving ≥5% lever — the rectangular
radix grid, 11–23% of the 7.98 ms wake 1e5 solve — was **handed off, not
silently absorbed**: recorded in the ledger and stamped into `037`'s entry
gate (commit `a0bb67d`). Remaining sub-5% items are listed without
extending the campaign. NCU counter profiling is correctly recorded as
blocked-external (ERR_NVGPUCTRPERM) with the driver ready to rerun.

### Scope item 5 — Downstream compatibility: **PASS**

- **Exported-name surface unchanged**: `git diff e2bd487 gpu-full --
  src/FLOWVPM.jl` shows no export-line changes (only the `fmm_radix` include,
  a `relaxation_none` internal refactor, and a save-exclusion symbol).
  `RadixFMMSettings`/`radix_fmm_settings!` are defined only in
  `src/FLOWVPM_fmm_radix.jl` and appear nowhere in `FLOWVPM.jl`'s export
  blocks — the `5dd0d85` "internal/not-exported" claim is **verified**. The
  defaults changed by `5dd0d85` affect only the GPU/radix coupling path;
  `FMM` struct keyword defaults and all exported names are untouched.
- **CPU path**: CPU test files (`runtests_singlevortexring.jl`,
  `runtests_leapfrog.jl`) are bit-identical to the branch point; the entire
  post-034-approval delta (`80eaf8d..5dd0d85`) touches only
  `src/FLOWVPM_fmm_radix.jl` and `test/runtests_gpu_fmm.jl`. `034` verified
  the full CPU suite passing locally against dev'd `matrix-ops`, and the
  radix coupling is `_FMM_HAS_RADIX`-guarded so FLOWVPM still loads against
  registry FastMultipole. **What I did:** judged CPU-test passing
  structurally (zero CPU-path diff since the verified-passing state) rather
  than re-running the multi-hour suite; 035's Part A record (green
  2026-08-12, 41 assertions) covers the coupling file that did change.
- Pre-existing caveat, not caused by this phase: `gpu-full`'s legacy
  `UJ_fmm` already passes `shrink`/`recenter` kwargs absent from registry
  2.0.4, so the branch requires dev FastMultipole for FMM use — documented
  in FLOWVPM `CLAUDE.md` and predating `034`.

### Scope item 6 — Cross-repo hygiene: **PASS (minor note)**

All FLOWVPM Integration commits live on `gpu-full` (clean tree at `5dd0d85`;
the twelve commits `4df2bc0..5dd0d85` map one-to-one to the 034/035 records);
all FastMultipole commits live here on `matrix-ops`. Cross-repo change pairs
are recorded in the owning task files (034 lists its FLOWVPM commits per
session; 035 records both sides per cycle, and its approval note corrected
the one wrong citation — cycle-2 FastMultipole commit is `9412600`, closeout
`d871a19`). The 035 approval's untracked-figure-artifact note was resolved by
`d076399` (pdf/png committed, LaTeX intermediates gitignored). **Minor
note:** untracked `plot_inner_iterations.py` and `plots/` sit in the
FastMultipole repo root, unclaimed by any task record — presumed user
scratch; flagged for disposition, not a phase defect.

### Scope item 7 — Improvement hunt: follow-on proposals (no phase expansion)

Recorded as proposals only; none reopens this phase:

1. **FLOWVPM `CLAUDE.md` defaults paragraph** (small, docs-only): the Phase 4
   section is accurate but silent on the 035 shipped tuning defaults
   (literature P5 / `rho_t=3.668` / dense M2L / partitioned kernel / joint
   auto-geometry, `5dd0d85`). One paragraph would spare the next FLOWVPM
   agent a task-file archaeology trip. See accumulated item 2 below.
2. **13-row functor return-path optimization** (FastMultipole, small-medium):
   032 Benchmark A measured the hessian-capable functor path 4–10% slower
   than hard-coded (13-value tuple return vs in-place accumulation). The
   nearfield now carries 47–88% of solve kernel time, so a few percent of it
   is near the 5% bar for a future phase; measure before implementing.
3. **NCU counter rerun** (external unblock): when Orc grants
   `ERR_NVGPUCTRPERM` access, rerun `profile_035_nearfield_ncu.jl` unchanged
   to confirm or refute the op-count bound-ness model with hardware counters.
4. **Lamb-Helmholtz tensor-path feasibility study** (high risk, unbounded):
   FP16-WMMA (`DENSE_CUDA_TENSOR_FORMAT`) engages only for `!LH && D == 16`,
   structurally excluding FLOWVPM; the 030 FP16 row is 3.7x under the final
   cube F32 solve. A `D == 32` dual-channel tensor format is the only
   remaining route to that class of gain; theory-first if ever pursued.
5. **Per-case wake q-floor guidance** (documentation): the wake-only
   q=12→6-class tuning notes from cycles 2/3 are recorded in 035; if wake
   workloads dominate a future consumer, surfacing per-case tuning guidance
   in `device_interface.md` would be cheap.

### Accumulated item — consumer migration hazard docs (from `034`): **PASS**

`docs/src/advanced_usage.md` § "Migrating From the Fixed-Row Buffer Layout"
is adequate: it states the root cause (compact switch-relative buffers), the
failure mode in plain terms ("silently write the wrong rows or past the end
of the buffer… results are corrupted"), and a four-step migration — replace
switchless setters/hard-coded rows with the switch-aware
setters/getters/range helpers, guard on `PS`/`GS`/`HS` (disabled-output
setters throw, converting the silent bug into a loud one), replace the
removed `get_previous_influence` with metadata rows, and re-run accuracy
checks with at least one output disabled (the configuration that exposes
fixed-row assumptions). An inline warning at the buffer-layout docs (line
166) reinforces it. **FLOWVPM's shims match the guidance exactly**:
`src/FLOWVPM_fmm.jl:72-81` (commit `4df2bc0`) forks at load time on
`isdefined(fmm, :gradient_range)` to the switch-aware accessor forms, and
the `get_previous_influence` overload is guarded at line 118; all hook call
sites route through the `_fmm_get/set_*` shims.

### Accumulated item — FLOWVPM `CLAUDE.md` freshness: **PASS (with note)**

The staleness flagged at the `034` approval ("H200 validation pending") was
fixed by `80eaf8d` (2026-08-06). At the final `gpu-full` head every checked
statement in the Phase 4 section remains true: coupling mechanism, cache
lifecycle, `recenter!` policy, gaussianerf-only + loud-error restrictions,
`nearfield_device` hazard removal, `_FMM_HAS_RADIX` guard, switch-relative
shim description, test wiring, and the dev-FastMultipole pin. It makes no
false claims about defaults — but it also does not mention the 035-shipped
tuned defaults (`5dd0d85`), which is the natural next staleness. Recorded as
follow-on proposal 1 rather than a failure; nothing currently written is
wrong.

### Overall verdict (reviewing agent)

**All seven scope items and both accumulated review items PASS** (items 3,
6, and the CLAUDE.md item with non-blocking notes). The Integration Phase
met its goals: a general device-system interface a third party can adopt
from the docs alone; H200-verified correctness under a fixed, checksummed
1e-3 velocity gate with J diagnostics everywhere; a single definitive,
honest speedup report (4.5x/4.6x cube/wake 1e5 F32 vs the shipped 034
coupling; no ineligible CPU baseline headlined); measured closure of the
lever list with the one surviving lever handed to `037`; and an unchanged
CPU/public surface for FLOWVPM's downstream consumers. Row marked Done;
clear-context approval by a different agent is pending per protocol.

## Clear-Context Approval

**Date:** 2026-08-12. **Reviewer:** clear-context approval subagent (Claude
Fable 5), fresh context, judging per the `START_HERE.md` clear-context
protocol (objectives, correctness, performance, robustness, invasiveness,
readability).

**What was checked:**

1. **Coverage:** the Review Notes address all seven scope items and both
   Accumulated Review Items, each with an explicit verdict and cited
   evidence; the phase-consistency duties (START_HERE items 3–4) are
   recorded.
2. **Accuracy spot-check against the underlying data** (independent,
   read-only): recomputed `sha256(fm035_cycle3d.csv)` = `69e5790b…` matching
   the review; re-audited all campaign CSVs — gate-pass counts 22/22
   (cycle1), 8/8 (cycle2), 25/33 (cycle3a), 12/12 (cycle3b), 16/16
   (cycle3d), 55/75 (sweep); `counters_flat=true` on every row;
   `j_rel_rms` present on every row of every file (0 missing across all 166
   rows). All match the review's claims. One numeric-labeling quibble: the
   phrase "150 rows of the five cycle CSVs" corresponds to
   sweep+cycle1/2/3a/3b (75+22+8+33+12=150), not the five files named
   `cycle*`; the substantive claim (J logged everywhere) holds for every
   file regardless. Not significant.
3. **Downstream compatibility spot-check** (sibling `../FLOWVPM.jl`,
   `gpu-full`, head `5dd0d85`, clean tree): `git diff e2bd487 gpu-full --
   src/FLOWVPM.jl` contains zero export-line changes;
   `RadixFMMSettings`/`radix_fmm_settings!` appear nowhere in
   `src/FLOWVPM.jl`; CPU test files `runtests_singlevortexring.jl` and
   `runtests_leapfrog.jl` are bit-identical to the branch point; the
   post-034-approval delta `80eaf8d..5dd0d85` touches only
   `src/FLOWVPM_fmm_radix.jl` and `test/runtests_gpu_fmm.jl`. The
   switch-aware shim fork (`src/FLOWVPM_fmm.jl`, `isdefined(fmm,
   :gradient_range)` gate and guarded `get_previous_influence`) and the
   `advanced_usage.md` migration section (line 169) were verified to exist
   as described.
4. **Improvement-hunt containment:** five follow-on proposals are recorded
   as proposals only; the 036 commit `128dce1` touches only the task file
   and the START_HERE Done cell — no production code, no phase expansion.

**Verdict:** the review verified what it claims to have verified, its
verdicts are supported by the cited evidence, and its non-blocking notes
(item 3 placement, item 6 untracked scratch, CLAUDE.md defaults silence)
are proportionate. No significant issue. **Approved.**
