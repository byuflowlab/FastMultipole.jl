# 044 Implementation: Partitioned Multi-GPU Lifecycle (2 GPUs)

## Status and Entry Gate

Staged `2026-08-11`. Blocked by `043` (Done + Approved, and its kill-switch
acceptance gate passed — a modeled 8-GPU step `<= 2 ms`). Implementation
row; follows the standard placement rules (`*_cuda.jl`, types in
`containers.jl`).

## Objective

Implement the approved `043` decomposition at 2 GPUs on the device-resident
lifecycle and pass the efficiency gate that falsified the mirrored-tree
scheme: **2-GPU efficiency `>= 75%` vs the then-current single-GPU record
at unchanged accuracy**, with the `029` recurring-cost gates (flat `023`
transfer counters, zero per-step allocation) intact.

## Scope

1. Partitioned refresh/upward/downward per `043`: owned-subtree B2M/M2M,
   split-level allreduce for coarse levels, per-level multipole halo
   exchange, partitioned M2L/L2L/L2B, body-halo nearfield.
2. All exchanges inside the captured per-GPU graphs, reusing the `029` P2
   validated mechanism (work-list slicing exactness, perm-aware
   scatter-add, `cuMemPoolSetAccess` pool P2P grant).
3. Ownership migration on occupancy epochs + graph re-record; teleport/
   epoch-invalidation tests.
4. Distributed correctness gates extending `test/cuda_radix_twogpu_test.jl`:
   exact-once across partitions, accuracy vs single-GPU (tolerance-based —
   cross-cache bitwise comparison is invalid per the `029` sort
   non-determinism finding), counter parity, allocation stability, `P=4`
   included per the standing rule.

## Measurement

H200 pair (single node, `gpu:h200:2`), the `029` frozen 1M-body workload
and accuracy gates; per-stage before/after vs the mirrored-scheme rows of
record (job 13066058: 3.993 ms, 58.3%). Report efficiency, comm+orch, and
the per-stage replication savings. Task-owned remote tree per the `034`/
`029` P2 pattern; never submit from a tree with a running job.

## Work Record

(To be filled by the executing agent.)

## Approval Notes

To be filled by a different agent after this task is complete.
