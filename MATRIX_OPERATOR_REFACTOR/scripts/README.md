# Exploratory Scripts

This directory holds Theory Phase numerical scripts for checking derivations
and generating comparison data.

Use `../START_HERE.md` for task order, approval requirements, and phase-gate rules.
# Task 024 definitive resident-M2L campaign

Task 024 uses `benchmark_024_common.jl` plus the host/CUDA case drivers to emit
one CSV schema for concat, factored, precomputed-y, and dense translation.
`cpu_024_run.sh` and `cuda_024_run.sh` are the exact Slurm payloads;
`cpu_024_submit.sh` and `cuda_024_submit.sh` synchronize a SHA-256-manifested
snapshot and submit only after explicit user approval.  `fetch_024.sh` fetches
both campaigns and runs `summarize_024.jl` with complete-coverage validation.

# Task 024a benchmark figures

`figures_024a_prepare.jl` (Julia stdlib only) reshapes the committed campaign
CSVs into per-panel tables under `../data/figures/tables/`;
`figures_024a_build.sh` runs it and then compiles the pgfplots figure sources in
`../data/figures/`.  See `../data/figures/README.md` for the figure index and
the recorded limitations.

# Task 024b CPU/GPU scaling

`benchmark_024b_common.jl` reads the checksum-pinned sampled-direct references
and computes the shared five accuracy metrics. `benchmark_024b_cpu.jl` fixes
literature `P=4` / code order 3 and MAC 0.5, manually searches leaf size with a
step-5 refinement, and writes a separate candidate audit. The GPU driver uses
the independently reviewed ell-scaled compatible stencil and sweeps the
occupancy schedule in `cuda_024b_run.sh`. The run scripts refuse to start
unless all seven references pass `direct_reference_checksums.sha256`.

`prepare_024b_direct_references.jl` generates those references and
`verify_024b_mac_stencil_compatibility.jl` re-checks the MAC/stencil gate over
every integer offset for `ell=2:7`. Both write to
`../data/cpu_gpu_scaling/references/`, which is deliberately a subdirectory of
the campaign directory so that `fig09`'s `cpu_*.csv` / `gpu_*.csv` glob never
picks up reference or verification files.

# Task 026 hierarchical host M2L

`benchmark_026_hierarchical_host.jl` runs one isolated host case and records
construction, update subphases, resident stages, per-level M2L/class occupancy,
memory, allocations, counters, and optional sampled-direct accuracy.
`cpu_026_run.sh` executes the 128-case practical window tuning phase, selects the global
eligible width, and then runs the 456-case scaling/clustered and 192-case
accuracy phases. `summarize_026_hierarchical_host.jl` requires a single source
hash, unique cases, per-case level rows, and exactly 776 cases for final
validation. `cpu_026_submit.sh` synchronizes a SHA-256-manifested snapshot into
the isolated ORC environment before submission.

# Task 027 hierarchical CUDA M2L

`benchmark_027_hierarchical_cuda.jl` sweeps `(policy, strategy, precision, P, LH,
n, ell, window_classes)` on the device-resident lifecycle, with
`policy in {flat, hier12, hier3}` and all four resident M2L strategy selections.
Each row records the source manifest, host/GPU/toolchain, per-level occupied node
counts, per-level route counts, per-stage and per-level GPU timings, construction
cost, persistent/peak device bytes, transfer counters, warmed device allocation,
and sampled accuracy vs `direct!`. A companion `<out>.classes.csv` carries one row
per `(case, level)` with the per-class route-occupancy distribution — the input
task 028 needs to evaluate a heterogeneous per-level strategy mix, which task 027
deliberately does not select.

`cuda_027_run.sh` is the Slurm payload (CUDA lifecycle, integration, and
hierarchical test files, then the sweep); `cuda_027_submit.sh` synchronizes the
tree into the isolated ORC environment and submits; `cuda_027_fetch.sh` pulls the
job log and CSVs back into `data/hierarchical_m2l_cuda/`.

`compare_026_regression.jl` now gates `m2l_ms_median` **and** `step_ms_median`
independently (warmed M2L stage and warmed full lifecycle), as the task-027
mandatory old-versus-new gate requires; each metric must satisfy the same
>5% flag / >10% block / geomean <= 1.00 policy, and both must pass.

# Task 041k — brute-force direct UJ(+SFS) ceiling

`fm041k_direct_bruteforce.jl` is a standalone (CUDA + stdlib) driver measuring
how large an N a single H200 can brute-force under 10 s: naive all-pairs O(N²)
evaluation of the FLOWVPM `gaussianerf` workload, N = 1e2 upward by
half-decades, per series {uj, ujsfs} × {naive, tiled} × {F64, F32(+rsqrt)}.
The `ujsfs` series is the stack's first fused direct+SFS evaluation (041b §1.2
factorized identity: UJ pass + O(N) T_q(Γ_q) + fused ζ pass for Ω/Q). Pair
math is transcribed from FLOWVPM with file/line provenance;
`fm041k_erf_vendored.jl` is a verbatim copy of FLOWVPM's GPU-safe erf so CPU
reference and kernels bit-match in both precisions. Writes incremental
`sweep.csv` and gated `accuracy.csv` (F64 max rel err ≤ 1e-11 vs threaded CPU
reference, plus the reordered-vs-pairwise Estr identity cross-check). Without
a functional GPU it runs the CPU/identity self-test only.

`fm041k_submit_gpu.sh` is the H200 Slurm script (copy outside the snapshot,
then sbatch); `fetch_041k.sh <jobid>` pulls the job log and CSVs back into
`data/direct_bruteforce_ceiling/`.
