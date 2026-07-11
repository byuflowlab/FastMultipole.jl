# 024 Implementation M2L Operator Variant End-to-End Benchmark

## Objective

Decide, by benchmark rather than assumption, which near-term M2L operator variant
the production FMM should prefer per platform:

- **MaterializedYRotationM2L** — `013-impl-axis-swap-operators.md` (reconstructed
  per-angle dense `Ts(theta)` via `build_Ts_from_S!`, production-parity dense
  apply).
- **FactoredRotationM2L** — `013c-impl-factored-rotation-alignment.md` (explicit
  assembled `Z_phi -> S -> Z_theta -> S_inv` alignment, built on the
  `013b-impl-fixed-y-swap-primitives.md` fixed swaps) around the shared `011`/`012`
  M2L core.

The `015` microbenchmark gives an early, isolated read over the `014` stage API.
This task is the **definitive end-to-end** measurement through the fully integrated
FMM (`023`), where memory traffic, batching amortization, threading, and GPU
residency all matter.

## Dependencies

- `013-impl-axis-swap-operators.md` (`MaterializedYRotationM2L` building blocks)
- `013b-impl-fixed-y-swap-primitives.md` (`FactoredRotationM2L` fixed-swap primitives)
- `013c-impl-factored-rotation-alignment.md` (`FactoredRotationM2L` assembled stages)
- `015-impl-axis-swap-benchmarks.md` (two-variant microbenchmark)
- `022-impl-gpu-device-resident-m2l.md` (GPU path for integrated M2L variants)
- `023-impl-production-integration.md` (both variants routed through the real FMM)

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- The `008c` performance baseline and the `015` benchmark harness/results

## Artifacts or Production Surface

- Benchmark scripts under `MATRIX_OPERATOR_REFACTOR/scripts/` and recorded data
  under `MATRIX_OPERATOR_REFACTOR/data/`.
- No production hot-path change (measurement only; both variants already exist
  behind the `014`/`023` dispatch).

## Deliverables

- End-to-end FMM timings for the best benchmarked setup of
  `MaterializedYRotationM2L` and `FactoredRotationM2L` across:
  - single-threaded CPU,
  - multi-threaded CPU (representative thread counts),
  - GPU (device-resident M2L per `022`),
  over a representative range of `N`, expansion order `P`, and interaction-list
  density.
- A per-platform recommendation (which variant wins, and the crossover regimes if
  it is mixed), with the data cited. The comparison must use the best integrated
  implementation of each near-term variant identified by `015`/`019`.
- Input to the `019a` final review's operator recommendation.

## Verification

Report the benchmark methodology, machine/GPU details, and raw plus summarized
numbers. Confirm both variants produce matching FMM results (within the established
tolerance) on the benchmarked cases before comparing their timings, so the
comparison is between two correct paths.

## Approval Notes

To be filled by a different agent after the benchmark is complete.
