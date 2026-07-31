# Dense-translation resident M2L (CUDA, task 023f) data

H200 before/after benchmark CSVs and Slurm job output land here, fetched by
`MATRIX_OPERATOR_REFACTOR/scripts/cuda_023f_fetch.sh <jobid>`.

CSV columns are produced by `benchmark_023f_dense_m2l_cuda.jl`: host/GPU, variant
(concat, factored, precomputed_y, dense, dense_perclass), precision, P, LH, N, ell,
`fit` (false rows record explicit dense memory-gate rejections with the actionable
`note`), route/occupancy statistics, construction time, persistent/peak device
bytes, the dense operator/persistent/estimated-peak byte accounting, chunk width,
whole-pass flag, per-stage (B2M/M2M/M2L/L2L/L2B) and full-step medians, M2L device
allocation, full-step host allocation, and direct-reference potential/gradient
errors.
