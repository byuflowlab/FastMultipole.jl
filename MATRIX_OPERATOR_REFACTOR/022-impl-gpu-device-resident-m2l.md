# 022 Implementation GPU Device-Resident M2L

## Objective

Implement a device-resident GPU M2L path for the radix interaction list, following
the `008c` device-residency target and the `015` benchmark-gated batching decision,
behind a CUDA extension/flag so the CPU path and public API are unaffected.

## Dependencies

- `008c-implementation-performance-baseline.md` (device-residency target, break-even)
- `008b-implementation-replan.md`
- `015-impl-axis-swap-benchmarks.md` (CPU/GPU batching recommendation)
- `021-impl-constant-p-stencil-and-interaction-list.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- The recorded `015` batching decision and GPU recommendation
- Existing CUDA setup notes and the project CUDA dependency

## Artifacts or Production Surface

- New `src/translate_batched_cuda.jl` (or equivalent `*_cuda.jl`) gated behind a
  package extension or runtime flag per placement rule 4.
- Tests/benchmarks comparing GPU M2L output with the CPU operator path to tolerance,
  skipped gracefully when no CUDA device is present.

## Deliverables

- Upload coefficients to device once, run the full per-offset-class M2L stencil on
  device, download once (no per-operator upload/download).
- Implement the batching shape chosen in `015` (e.g. cuBLAS strided-batched or a
  fused custom kernel) over the translation-invariant offset classes.
- Float64 default; Float32 opt-in only, per `008c`.

## Verification

Run GPU-vs-CPU parity to tolerance and the device-resident throughput benchmark;
confirm the `008c` break-even behavior (device-resident dense beating CPU recurrence
at batch >= 8, best CPU dense at batch >= 64). Record commands, environment, and
result summaries. Confirm the CPU path is unchanged when CUDA is absent.

## Approval Notes

To be filled by a different agent after implementation and verification are
complete.
