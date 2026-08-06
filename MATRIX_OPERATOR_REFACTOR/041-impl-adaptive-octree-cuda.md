# 041 Impl: Adaptive Octree CUDA Device-Resident Lifecycle

## Status and Entry Gate

**Proposed follow-on item; not started.**

Entry gate: `038`, `039`, and `040` complete and approved.

## Objective

Mirror the adaptive octree on the CUDA device-resident lifecycle: device
construction and refresh (sort, prefix split, 2:1 balance, list generation as
flag/scan/compact kernels), device U/V/W/X execution, and H200 measurement
against the uniform-depth path.

## Scope and placement

- Device code in `tree_batched_cuda.jl` / `translate_batched_cuda.jl` behind
  the existing CUDA extension; uniform-depth device path untouched and
  default.
- Replace the dense `Σ 8^L` `node_at` occupancy lookup with the sorted-Morton
  binary-search lookup on device for the adaptive path (and, if cheap,
  expose it to the uniform path to lift the `ell ≤ 8` cap — record the
  decision either way).
- V-list M2L replays the existing device M2L strategies and operator tables;
  construction-only operator/route uploads; `route_uploads`/
  `operator_uploads` constant after construction;
  `expansion_host_copies == 0`.
- W/X kernels: device M2T and S2L with the same accuracy contract as `040`.
- Nearfield load-balance follows from bounded `K_max`: assert/verify that the
  warp-per-pair kernel no longer sees fat-cell serialization; retune the
  symmetric-path threshold if profiling justifies it.
- Refresh: occupancy-epoch caching generalized to the adaptive leaf set
  (rebuild windows only when the leaf key set changes), preserving the CUDA
  graph replay path where applicable.

## Verification

- Host/device parity on all `039`/`040` test distributions, both precisions,
  `P=4` included; transfer-counter and zero-per-step-allocation parity per
  the `023` contract.
- H200 before/after: full resident step and per-stage breakdown on the cube,
  wake, and multi-scale cases vs the uniform-depth path at each case's best
  `ell`; construction and refresh costs; memory footprint vs the
  `min(8^ell, n)` capacity baseline.
- No regression on the shipped scalar benchmark (the `028`/`029` record) —
  the adaptive path is opt-in, so this is a guard that shared code was not
  perturbed.

## Acceptance

Parity and counter gates pass; the multi-scale case shows a measured
end-to-end win consistent with the `038` model; uniform cases regress by no
more than an agreed tolerance (or the adaptive path is documented as
multi-scale-only); a default-selection recommendation (adaptive vs uniform,
per regime) is recorded for the `042` review with evidence.
