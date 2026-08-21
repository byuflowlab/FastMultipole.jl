# Task 019 operator performance tuning — benchmark artifacts

- `cuda_022_baseline.csv` — the pre-019 H200 baseline (022 "H200 Rerun 2026-07-11",
  verbatim from `022-impl-gpu-device-resident-m2l.md`), reformatted to the 019 CSV
  schema. Caveats: 022 stage timings (`t_*`) were **single-shot** with allocator/GC
  jitter (the standalone M2L stage at n=1e4 exceeds the min-of-3 whole-lifecycle
  time); `chunk` was the `ConcatenatedFixedZM2L` default `2^17`; `exec_max` was not
  recorded (NaN).
- `<node>/cuda_019_phaseABD_throughput.csv` — H200 run 2026-07-11 after task-019
  Phases A (dense fixed-stage GEMM concat M2L), B (fused gather/scatter kernels +
  preallocated chunk scratch), and D (implicit constant-P interaction structure +
  per-class plan geometry).
- `<node>/cuda_019_phaseE_throughput.csv` — H200 run 2026-07-11 (later the same
  day) additionally including Phase E (whole-slab M2M/L2L group execution). Same
  environment as `env.md`. Parsed from the `CUDA_019_CSV` block of
  `scripts/cuda_019_tuning.jl` output (min-of-reps stage timings, chunk sweep,
  `exec_max` jitter indicator); one directory per GPU node hostname.
- `local_macos_allocations_storage.csv` — warmed host allocation measurements for
  concat M2L and resident M2M/L2L (Float64/Float32, LH off/on), plus measured
  retained payload for constant-P class geometry and operator-cache components.
  Reproduce with `julia --project=.
  MATRIX_OPERATOR_REFACTOR/scripts/operator_performance_allocations.jl`.

Raw sbatch `.out` logs were not retained in this artifact directory. The H200
pass/fail gates and test counts are therefore recorded external evidence, not
independently auditable local artifacts; the CSV timing blocks are retained here.

Run workflow (user-driven; agent must not ssh):
`bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_019_submit.sh` from the repo root, then
`bash MATRIX_OPERATOR_REFACTOR/scripts/cuda_019_fetch.sh` once `squeue` shows the
job finished. That workflow may create a local `raw/` directory, but none is
present in the reviewed artifact set.
