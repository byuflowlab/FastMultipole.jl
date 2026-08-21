# Factored resident M2L host benchmark

Reproduce locally (macOS numbers are smoke/laptop reference only):

```sh
julia --project=test MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023a_factored_host.jl
```

Environment knobs: `FM023A_N` (comma list of body counts; large N fills the leaf
grid so offset classes get wide and exercise the GEMM branch), `FM023A_REPS`,
`FM023A_BLAS_THREADS`, `FM023A_SWEEP_COLS` (extra GEMM crossover candidates),
`FM023A_OUT`.

Variants per `(P, LH, N)` row:

- `concat` — default `MaterializedYRotationM2L` / `ConcatenatedFixedZM2L` path.
- `factored_functional` — the allocating functional baseline: the same factored
  plan executed through the generic shared-rotation helpers
  (`_launch_resident_m2l_shared!` over the plan groups). Stage-only; full-step
  columns are NaN/-1 by design.
- `factored_scalar` — grouped factored path with the GEMM branch disabled
  (thresholds forced to `typemax(Int)`).
- `factored_gemm` — GEMM branch forced on for every degree block (thresholds 1/1).
- `factored_optimized` — production defaults (`FACTORED_Y_GEMM_MIN_COLS`,
  `FACTORED_Y_GEMM_MIN_DIM`), recorded in the `gemm_min_cols`/`gemm_min_dim`
  columns.

Required non-macOS single- and multi-thread BLAS measurements come from the BYU
HPC cluster (see `../..//scripts/cpu_023a_run.sh`): sync the tree to
`~/FastMultipole-023`, instantiate `~/fm023env` on the login node, then
`sbatch MATRIX_OPERATOR_REFACTOR/scripts/cpu_023a_run.sh`. The script runs the
sweep twice with `OPENBLAS_NUM_THREADS=1` and `=$SLURM_CPUS_PER_TASK` set at
process start, writing `host_<node>_blas{1,N}_<stamp>.csv`.

File inventory:

- `host_m12-1-21_blas{1,64}_*.csv` — **decision evidence** (job 12760919, AMD
  EPYC 7763): full sweep with the final shipped crossover defaults
  (`gemm_min_cols=16`, `gemm_min_dim=1`) in the `factored_optimized` rows.
- `host_m12-1-10_blas{1,64}_*.csv` — job 12760272, same host class, run before
  the defaults were finalized (`factored_optimized` at the provisional 32/5);
  its `factored_scalar`/`factored_gemm`/`factored_functional` rows are the raw
  crossover evidence the defaults were selected from.
- `host_tmplab-*.et.byu.edu_*.csv` — pre-GEMM macOS (Accelerate) smoke runs
  retained for history; Accelerate ignored the 1-thread request, so both are
  8-thread.
- `smoke_macos_*.csv` — post-GEMM laptop smoke evidence (final defaults).
