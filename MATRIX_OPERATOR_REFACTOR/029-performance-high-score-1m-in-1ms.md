# 029 Performance High Score: 1,000,000 Bodies in 1 ms

## Status and Entry Gate

**Resumed by user direction on `2026-08-05`** (deferred `2026-08-03` to
`2026-08-05`). This task does not block the (already approved) `019a` final
Milestone Review; its results are recorded there as an addendum review note on
completion.

The entry gate is satisfied: task 028 is Done and clear-context approved in
`START_HERE.md`. The Mandatory Reading Gate below must still be completed and
recorded before any profiling, prototype, implementation, or H200 run.

Context accrued during the deferral that the baseline step must absorb: task
`030` (Done `2026-08-05`) retuned the same frozen workload's geometry and
measured **7.092 ms** (`ell=5`, `sched6-4-4-3`, FP16, err 1.18960e-3 — passes
the 1.19e-3 gate by 0.03%) and **7.556 ms** robust (`sched6-5-4-4`, 0.87x
gate) against 028's shipped 9.591 ms — so the 029 single-H200 baseline profile
must include the 030 winner geometry, not only the 028 shipped default.

## Objective

Pursue the lowest reproducible latency beyond task 028's independently
reproduced **9.591 ms** result for the same accurate 1,000,000-body workload.
The high-score goal is a complete resident step in **at most 1 ms**. Maintain
separate leaderboards for one H200 and multiple H200s; a qualifying result on
either track meets the goal, but completion must always report the best valid
result on both tracks.

Initial execution scope is H200 hardware only. Newer GPU models are outside this
task because H200 is the newest hardware currently available. A new hardware
track may be added only if access changes and the user approves the scope change.

## Dependencies

- `028-performance-feasibility-1m-in-10ms.md`, complete and clear-context
  approved
- The approved theory, lifecycle, hierarchical, and benchmark dependencies
  inherited through task 028

## Mandatory Reading Gate

Before any 029 work, read all of the following in full:

1. `START_HERE.md` and `../MATRIX_OPERATOR_REFACTOR.md`.
2. All of `028-performance-feasibility-1m-in-10ms.md`, including every
   verification entry and the complete failure ledger.
3. `data/feasibility_1m_10ms/report.md` in full.
4. The final reproduction log
   `data/feasibility_1m_10ms/fm028-13029878.out` and the reproduced-result CSVs
   `cuda_m13h-1-1_20260803-131016.csv` and
   `cuda_m13h-1-1_20260803-131129.csv`, including their `.classes.csv`
   companions.
5. Any source-manifest information named by those records, and the production
   files and tests identified by task 028 as supporting the retained winner.

Record the reader, date, files read, and confirmation of completion in the
**Reading Gate Record** below before profiling or proposing a change. If any
required artifact is absent or inconsistent, stop and repair the coordination
record before proceeding.

## Fixed Comparable Workload and Acceptance Boundary

- Exactly **1,000,000 bodies** and literature **P=4**
  (`expansion_order=3`).
- Reuse task 028's deterministic body seed **24025**, sampled-reference seed
  **24026**, fixed bounds, and sampled-direct reference methodology without
  alteration.
- Sampled-direct gradient relative RMS error must be **<= 1.19e-3**.
- The verdict is one complete recurring resident step: evaluation, device-side
  convection, tree and route refresh, all required synchronization, and all
  recurring communication. No stage may be omitted, hidden in setup, or
  amortized beyond its real recurrence interval.
- Construction-only work may be reported separately. If a purportedly cached
  operation recurs with motion or time stepping, its amortized recurring cost
  belongs in the verdict.
- Every accepted path must preserve the FP32 fallback on GPUs without supported
  tensor arithmetic. Accelerated execution remains capability-gated and must
  not change the established fallback behavior.

The accuracy threshold, seeds, body count, order, and timing boundary are frozen
for comparability. Any scientifically useful alternative workload must be
reported outside the leaderboards and cannot satisfy this task.

## Leaderboards and Required Record

Maintain two independent tables in this file or a linked report:

1. **Single H200:** exactly one H200 performs the complete step.
2. **Multiple H200s:** two or more H200s; include every communication,
   synchronization, orchestration, and load-imbalance cost in the verdict.

Task completion requires at least one valid measured entry on each leaderboard.
If access to either configuration is lost, record the access failure and leave
the task blocked rather than silently substituting a modeled score.

Every leaderboard entry must record:

- H200 count; node count and node/GPU topology; device identifiers; and the
  interconnect used between GPUs and nodes;
- driver, CUDA toolkit, CUDA.jl, Julia, package/environment, and relevant
  communication-library versions;
- GPU power limit and persistence/performance/clock mode, including whether
  clocks were fixed;
- peak and persistent memory use per GPU and aggregate memory use;
- complete stage timings, communication and synchronization timings, and the
  total verdict median and full measured range over the declared sample count;
- sampled-direct accuracy and reference identity;
- deterministic seeds and complete workload parameters;
- source manifest, working-tree state, launch command, scheduler job/process
  identity, raw logs, and machine-readable result artifacts.

Multi-H200 entries additionally report the partition, per-GPU work and memory,
halo/exchange volume, communication topology, orchestration overhead, and
measured imbalance. A kernel-only or communication-excluding number is not a
leaderboard score.

## Work Plan and Evidence Checkpoints

### 1. Re-establish and profile the retained baseline

Start with a fresh, complete profile of task 028's reproduced FP16 tensor path
in a separately initialized process. Reconfirm accuracy, lifecycle accounting,
stage timings, allocations/transfers, memory, and source manifest before testing
new mechanisms. Use this run as 029's single-H200 baseline; do not substitute
task 028's historical number for a fresh profile.

### 2. Evaluate bounded mechanisms

Profiling, analytical cost models, and bounded prototypes may proceed after the
reading gate. Candidate families include:

- target- or class-owned tensor M2L that improves operator/source reuse;
- WMMA packing, conversion, and reusable packed-layout reductions;
- multi-H200 spatial decomposition with full recurring communication;
- shifted-macrocell nearfield ownership;
- plane-wave/exponential M2L;
- spatial or polyphase FFT M2L;
- backend-driven depth re-bracketing; and
- refresh, synchronization, or launch reductions justified by a fresh profile.

This list is a candidate set, not authorization to implement every branch.
Rank candidates using measured remaining cost, plausible complete-step gain,
accuracy risk, memory, implementation scope, and compatibility impact.

### 3. User approval checkpoints

Bounded prototypes and profiling do not require a production commitment. Before
each **material production rewrite** or **new translation-theory branch**, present
the evidence, expected complete-step gain, risks, affected files, fallback plan,
and verification plan, then obtain explicit user approval. One approval covers
only the stated mechanism and scope. Remeasure after each accepted change before
seeking approval for another material branch.

### 4. Closed 028 results

Treat the following as closed negative or saturated results from task 028:

- low-rank M2L;
- the current unordered symmetric nearfield;
- TF32 cuBLAS;
- atomic-only rewrites;
- B2M warp-per-cell; and
- stream overlap of already saturated kernels.

Reopen one only after recording a new, mechanism-specific hypothesis that
explains why 028's evidence no longer predicts the outcome and identifies the
measurement that can falsify the hypothesis. Renaming or retuning a closed idea
without a new mechanism is not sufficient.

## Verification and Compatibility Gates

Before accepting a single-H200 score, keep the existing lifecycle,
hierarchical, precision, allocation, transfer-counter, and P=4 suites green.
Verify both the accelerated capability gate and the unchanged FP32 fallback.

Before accepting a multi-H200 score, add and pass distributed gates for:

- partition-independent numerical agreement and sampled-direct accuracy;
- exact ownership/coverage with no missing or double-counted interactions;
- deterministic aggregation within the documented floating-point tolerance;
- complete communication, synchronization, transfer, and orchestration
  accounting;
- per-device allocation/residency and transfer-counter invariants; and
- clean teardown and repeated-step behavior in a separately initialized
  distributed process.

Every score must pass the unchanged 1.19e-3 accuracy gate and include all
recurring costs. Any **<=1 ms** claim requires an independent reproduction in a
separately initialized process with the matching source manifest and complete
raw artifacts. The reproducing run must pass the same tests and accounting gates
as the original claim.

## Completion Rule

Stop when either:

1. an independently reproduced leaderboard score reaches **<=1 ms** on the
   single-H200 or multi-H200 track; or
2. measured profiles, bounded prototypes, and documented cost bounds show that
   no credible remaining lever can materially improve the best valid result.

At completion, report the best valid single-H200 result and the best valid
multi-H200 result even if only one track reaches the goal. Explain why work
stopped, list rejected and untested mechanisms, preserve all raw artifacts, and
leave no modeled value presented as a measured score.

## Artifacts or Production Surface

- Task report, leaderboards, profiles, logs, manifests, and machine-readable
  measurements under `MATRIX_OPERATOR_REFACTOR/data/performance_high_score_1m_1ms/`.
- Task-specific benchmark, profiling, submission, fetch, and analysis scripts
  under `MATRIX_OPERATOR_REFACTOR/scripts/`.
- Approved production changes only in the established resident lifecycle,
  hierarchy, and CUDA extension surfaces, with CPU/non-tensor fallbacks intact.
- Distributed tests under `test/` before any multi-H200 result is accepted.

## Reading Gate Record

**Completed `2026-08-05` (late evening MDT)** by the executing agent (Claude
Fable 5, session resuming this row per the user's `2026-08-05` direction).
Files read in full:

1. `START_HERE.md` (all sections, both phase tables, Integration Phase
   preamble, protocol rules) and `../MATRIX_OPERATOR_REFACTOR.md` (all 326
   lines, including the single-level radix finding and the 028/029/030 row
   histories).
2. `028-performance-feasibility-1m-in-10ms.md` (all 866 lines), including
   every Verification Notes entry (Phase A; Phase B levers 1–3, cycles 2–4;
   Stages 5–8), the **complete failure ledger** (jobs 13016917; 13027048,
   13027092, 13027167, 13027174, 13027188, cancelled 13027374; 13028465;
   failed re-gate 13031187), the Staged Continuation Roadmap (Stages 5–10),
   and all three Approval Notes sections.
3. `data/feasibility_1m_10ms/report.md` (all 889 lines): the superseded Phase
   A record §1–§5, §6b–§6.8 cycle-by-cycle progression
   (91.4 → 69.6 → 64.6 → 38.0 → 30.7 → 15.4 → 12.5 → 9.591 ms), the closed
   negative results, the §6.8 default-selection rules, and §7 threats to
   validity (~8% hierarchical run-to-run variance; sub-10-ms lever gains are
   within noise).
4. The final reproduction log `data/feasibility_1m_10ms/fm028-13029878.out`
   (76 lines: full toolchain record, all green gates incl. hierarchy
   33,367/33,367, `STAGE8_REPRO_EXIT=0`) and the reproduced-result CSVs
   `cuda_m13h-1-1_20260803-131016.csv` (FP16: verdict 9.591 [9.434, 9.631] ms,
   err 1.0593e-3, counters 2/2/0/0/0, 2.0 GB persistent, host alloc
   708 KB/step) and `..._131129.csv` (BF16: 9.638 ms, 1.0632e-3), with both
   `.classes.csv` companions (per-level route histograms: L2 1,896 / L3
   119,784 / L4 1,145,544 / L5 11,037,576 routes).
5. Source manifest `42a6c254a11ac8a8` (winner + reproduction) and
   `c0afa01083322fb8` (post-review shipped defaults, job 13031482). Production
   surface supporting the retained winner, confirmed present in the current
   tree this session: `src/translate_batched_cuda.jl` (tiled/grid-stride dense
   M2L, FP16-WMMA path, warp-per-pair nearfield, `_cuda_fast_rsqrt`, counting
   sort, allocation-free scratch validation), `src/containers.jl`
   (`HierarchicalRigidStencil`, shipped `sched6-5-5-5` defaults, tensor-format
   knob), `src/interaction_list_batched.jl` (rigid tables + derived epsilon
   separator), `src/translate_batched_resident.jl` (policy resolution);
   tests `cuda_radix_lifecycle_test.jl`, `cuda_radix_convection_test.jl` (+ six
   028 testsets), `cuda_radix_hierarchical_test.jl`, `cuda_radix_counting_sort_test.jl`,
   `hierarchical_m2l_host_test.jl`.

Confirmed: the full 028 verification history and failure ledger were read.
Key facts carried into this row's design: the verdict boundary and 1.19e-3
gate are frozen; run-to-run variance on the hierarchical path is ~8%, so any
claimed gain below ~10% needs repeated/independent measurement; the closed
negative results list (§Work Plan 4) is binding absent a new mechanism-specific
hypothesis; the 030 campaign (approved after 028) already measured retuned
geometry at **7.092 ms** (`sched6-4-4-3`, err 1.18960e-3, knife-edge) and
**7.556 ms** (`sched6-5-4-4`, 0.87x gate) on this same frozen workload — the
029 step-1 baseline must freshly profile the shipped default *and* these
geometries. One environment note: the cluster's default julia module moved to
1.12.6, which segfaults host LLVM JIT on the device step (clean-env repro, job
13058336); all 029 runs pin `julia/1.11.7-6bmogfl`, the toolchain of every
result of record.

## Single-H200 Leaderboard

Frozen workload: n=1,000,000, seeds 24025/24026, bounds `(-0.01, 1.02)`,
literature P=4 (`expansion_order=3`), LH off, gate `err_gradient_rel_rms
<= 1.19e-3` vs the checksummed 024b reference; verdict = complete resident
step (refresh + eval + finalize + device Euler), medians over REPS=15.

| # | verdict ms [range] | err (gate 1.19e-3) | config | job / node / manifest | date |
|---:|---|---|---|---|---|
| 1 | **7.003** [6.992, 7.034] | 1.18963e-3 (0.9997x — knife-edge) | `sched6-4-4-3`, ell=5, dense FP16-WMMA/F32, K=full(904), counting sort | 13059710 / m13h-1-1 / `5a5d41312dc113d2` | 2026-08-06 |
| 2 | 7.436 [7.412, 7.468] | 1.0300e-3 (0.87x — robust) | `sched6-5-4-4`, ell=5, same stack | 13059710 / m13h-1-1 / `5a5d41312dc113d2` | 2026-08-06 |
| 3 | 9.442 [9.405, 9.453] | 1.0593e-3 | `sched6-5-5-5` (028 shipped default) | 13059710 / m13h-1-1 / `5a5d41312dc113d2` | 2026-08-06 |
| — | 20.461 [20.430, 64.3*] | 1.0497e-3 | `sched6-5-5-5`, Float64/off (context row; *one outlier rep) | 13059710 / m13h-1-1 / `5a5d41312dc113d2` | 2026-08-06 |

Environment record (full block in
`data/performance_high_score_1m_1ms/fm029-13059710.out`): 1x NVIDIA H200
(sm_90a), node m13h-1-1 (8x H200 node, single GPU allocated), driver
580.159.4, CUDA runtime 12.8.0 (local toolkit), CUDA.jl stack per log, Julia
1.11.7 (pinned; 1.12.6 blocked upstream), power limit 700 W (default,
persistence per log), performance state P0, clocks not fixed. Persistent
device footprint and per-stage timings in the four
`cuda029_base_*_13059710.csv` artifacts (+ class companions).

**No entry is a <=1 ms claim; no independent reproduction is therefore yet
required.** The 7.003 ms knife-edge entry reproduces the 030 measurement
(7.092 ms on m13h-1-2) within cross-node variance and its error
(1.18963e-3) is inside the gate by 0.03% on both nodes — the robust
`sched6-5-4-4` row is the recommended working baseline for optimization
work, per the 030 approval's caveat.

## Multi-H200 Leaderboard

No entries yet. Topology probe (2026-08-06): partitions `m13h` (4 nodes x
8x H200) and `eng` (1 node x 8x H200) — the multi-GPU track can run
intra-node up to 8 GPUs; interconnect to be recorded from `nvidia-smi topo
-m` in the first multi-GPU job.

## Verification Notes

### Step 1 — fresh single-H200 baseline (2026-08-06, job 13059710)

`bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_029_submit.sh` -> job 13059710
(m13h-1-1). Preflights green: lifecycle 216/216 + 37/37, convection +
optimized-kernel suites, counting sort; `REFERENCE_GATE_EXIT=0` (checksummed
024b reference); `BASELINE_EXIT=0`, `failed_cases=0`. Harness
`benchmark_028_feasibility.jl` UNCHANGED (one case per process, 030 pattern).
The shipped-default row (9.442 ms) sits inside the record's run-to-run band
(9.591 [9.434, 9.631] job 13029878; 9.448 job 13059638), so the baseline is
re-established fresh rather than inherited, as the Work Plan step 1 requires.
Source manifest `5a5d41312dc113d2` differs from 028's `42a6c254a11ac8a8`
because src/ legitimately advanced through approved rows 032/032a (vortex
interface, radius extension q<=20, PartitionedVortex) — the scalar verdict
path itself is regression-checked by the 032 stage-4 no-regression gate
(9.448 ms / errors identical on this manifest's parent).

### Step 2 — mechanism assessment and ranking (2026-08-06, job 13059955)

Full assessment in
`data/performance_high_score_1m_1ms/step2_mechanism_ranking.md`. One
evidence-only H200 profiling job (13059955, m13h-1-1, julia 1.11.7 pinned, no
production `src/` changes; scripts `profile_029_floor.jl`,
`cuda_029_profile_{submit,run}.sh`) attributed the robust baseline's floor:
full step = 420 kernel launches / 27 syncs / 127 mem-ops with 6.39 ms
device-busy of the 7.44 ms verdict; M2M+L2L is 0.47 ms busy vs 1.96 ms wall
(253 launches + 101 pageable H2Ds); M2L's 2.13 ms is 0.78 ms math + ~1.2 ms
per-step route-window scans + a blocking route-count D2H; the identical-
geometry n=1e3 control runs the complete step at 3.87 ms wall / 0.82 ms busy —
the ~3 ms n-independent launch/sync floor directly observed. Ranking: (1)
launch/sync floor elimination via graph-captured far-field chain (reopens the
Stage-10 CUDA-graphs closure under its own stated condition — the step is now
launch-bound at the margin; expected 7.44 → ~4.5–5.5 ms), (2) route-window
machinery fold, (3) multi-H200 decomposition (2-GPU feasibility slice first),
(4) nearfield ILP round 2, (5) leaf-M2L batching; plane-wave and FFT M2L
deprioritized at P=4. Single-GPU ≤1 ms judged out of reach (roofline + floor);
the goal path is the multi-H200 track with cycles 1→2→3, projecting
~0.9–1.4 ms. Cycle-1 approval request = prototype P1 (+#2 fold), zero
accuracy risk, gated on a ≥0.5 ms falsification threshold.

## Approval Notes

To be filled by a different agent after this task is complete.
