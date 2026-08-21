# Task 023c host benchmark data

`benchmark_023c_precomputed_y_host.jl` warms construction and application
specializations, then records host/CPU, Julia, BLAS, thread and worktree
provenance; construction time; invariant operator and reusable scratch bytes;
angle/offset occupancy; and warmed M2L/full-step timing and allocation data. It
compares the 023a factored reference, scalar precomputed-y baseline, forced GEMM,
production crossover, and optional focused column-threshold candidates.

The production crossover is 16 columns, selected on an AMD EPYC 7763 from both
one- and 64-thread OpenBLAS sweeps. A global threshold stayed within 5% of the
best focused candidate across the measured `P=4,8,12`, `N=150,2000,20000`, and
LH on/off regimes, so no block-dimension rule was added. See
`hpc_summary_20260718.md` for the decision and representative results.

Artifacts:

- `fm023ccpu-12799879.out` and `host_m12-1-3_blas{1,64}_20260718-090201.csv`:
  candidate-selection job with the provisional production threshold 12.
- `fm023ccpu-12801369.out` and `host_m12-1-3_blas{1,64}_20260718-101021.csv`:
  final confirmation job with production threshold 16 and explicit provenance.
- `smoke_macos_20260718.csv` and `smoke_summary_20260718.md`: local smoke data;
  not production tuning evidence.

Run `scripts/cpu_023c_submit.sh` from the repository root to sync and submit;
fetch with `scripts/cpu_023c_fetch.sh <jobid>`. Benchmark knobs are `FM023C_N`,
`FM023C_P`, `FM023C_REPS`, `FM023C_BLAS_THREADS`, `FM023C_SWEEP_COLS`, and
`FM023C_OUT`.
