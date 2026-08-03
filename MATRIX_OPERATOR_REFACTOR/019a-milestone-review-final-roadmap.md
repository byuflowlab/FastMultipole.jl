# 019a Final Roadmap Milestone Review

## Objective

Review Implementation tasks `017` through `029` and confirm the completed
Matrix Operator Refactor still matches the background roadmap.

## Dependencies

- `008b-implementation-replan.md`
- `008c-implementation-performance-baseline.md`
- `017-impl-flat-coefficient-buffers.md`
- `018-impl-real-solid-harmonic-basis.md`
- `019-impl-operator-performance-tuning.md`
- `019b-exploratory-smallp-fallback-and-channel-layout.md`
- `022-impl-gpu-device-resident-m2l.md`
- `023-impl-production-integration.md`
- `024-impl-operator-ab-benchmark.md`
- `024a-impl-benchmark-visualization.md`
- `024b-impl-cpu-gpu-scaling-benchmark.md`
- `025-theory-hierarchical-rigid-m2l-stencil.md`
- `026-impl-hierarchical-m2l-host.md`
- `027-impl-hierarchical-m2l-cuda.md`
- `028-performance-feasibility-1m-in-10ms.md`
- `029-performance-high-score-1m-in-1ms.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Completed task files listed above
- Artifacts and production files listed by the completed task files

## Artifacts or Production Surface

Review the production files, tests, benchmarks, generated artifacts, and final
notes listed by tasks `017` through `029`.

## Deliverables

- Final roadmap-alignment notes recorded in this file
- Any required coordination-document fixes identified before the refactor is
  considered complete
- **Final go/no-go (feasibility scoped earlier in `013a`): porting the old
  per-interaction error machinery onto the new expansion operators.** Using the
  `013a` feasibility finding and the `019` performance-tuning evidence, decide
  whether to port the dynamic-`P` / `get_P` / `predict_error` machinery onto the new
  operators or leave the two paths independent (old ops + old error machinery; new
  ops + constant-`P` interaction-list stencil). Record the decision and rationale
  here.
- **Review `008b-implementation-replan.md` (including its re-plan addenda) and
  confirm all recorded decisions and feedback have been incorporated** into the
  completed refactor and coordination documents. Note any gaps and the required
  fixes here.
- **Confirm the small-`P` / tiny-batch fallback and channel-layout decisions from
  `019b`.** These were resolved in `019b` (exploratory benchmark plus user
  discussion). Confirm the chosen fallback policy and padded-vs-ragged `chi` layout
  are implemented and consistent with the coordination documents; note any gaps.
- **Confirm the resident M2L strategy recommendations from `024`.** Record the
  final CPU/GPU and workload-regime recommendations among whole-slab concat,
  per-degree factored, precomputed-y, and full dense-translation execution, and
  note construction/memory constraints, crossover regimes, and integration
  caveats. The reconstructed per-column `Ts(theta)` path is an oracle, not a
  resident candidate.
- **Batched-GEMM speedup verdict (from `016b` watch item 1, deferred `2026-06-24`).**
  `015` observed no batched-GEMM speedup on the macOS host; this was deferred as a
  likely Apple-M2/OpenBLAS artifact, to be re-tested on a non-macOS / different-BLAS
  host in `024`. Record the cross-machine result here as a go/no-go: did the batched
  speedup materialize, and does it change the operator recommendation or the GPU
  (`022`) outlook?
- **Remaining lifecycle cost levers (carried in from the `023` clear-context
  review, `2026-07-15`).** With the `023` per-step update/finalize overhead
  reduced, the resident lifecycle dominates the recurring GPU step (~83% at
  n=1e5/P=4); the identified levers are the `019`-deferred L2B kernel and a
  grouped-GEMM M2L. Using the `024` stage breakdowns, record whether these are
  worth a follow-on task or are formally deferred.
- **Minor observations from the `023` approval (`2026-07-15`, non-blocking).**
  (a) The device per-step update performs a few small blocking scalar downloads
  that no transfer counter tracks (the out-of-box flag read and the route/direct
  prefix-total reads); `metadata_downloads` deliberately counts only the 3
  perm/system/index mirrors. If a future row tightens the per-step sync budget,
  start from these untracked syncs. (b) `_cuda_radix_keys_checked_kernel!`
  writes `oob_flag[1] = Int32(1)` from every out-of-box thread without atomics —
  safe because all writers store the same value; do not extend it to
  multi-value writes without adding atomics. Confirm both remain acceptable or
  note follow-up.
- **End-user scaling evidence (`024b`, added `2026-07-25`).** Use fig09 to
  record how fixed-MAC, manually leaf-searched legacy CPU 64-thread and resident
  H200 speedup over the corresponding legacy CPU single-thread baseline change
  from `n=1e3` through `1e6` at literature `P=4`
  (`expansion_order=3`). The GPU stencil must use the independently reviewed
  ell-scaled equal-cell compatibility rule, and fig09 must show the shared
  sampled-direct relative gradient RMS errors as well as timing. Treat Float64
  as the primary fair comparison and Float32 as an additional throughput
  result; include any dense-to-precomputed-y OOM fallback, Float64 error-order
  failure, or non-monotonic regime in the final recommendation.
- **Hierarchical and high-score evidence (`025`–`029`).** Record the final
  hierarchical-vs-flat and radius/schedule verdict, task 028's independently
  reproduced 9.591 ms FP16 result, and task 029's best reproducible single-H200
  and multi-H200 scores. Confirm that every accepted score uses the unchanged
  1M-body/P=4 accuracy and complete recurring-step boundary, and distinguish
  measured conclusions from modeled opportunities.

## Verification

Confirm completed work matches the background design, hard phase gate, and task
ordering. If `START_HERE.md`, a task file, and `../MATRIX_OPERATOR_REFACTOR.md`
disagree, stop and require a coordination-document fix.

## Approval Notes

To be filled by a different agent after review notes and verification are
complete.
