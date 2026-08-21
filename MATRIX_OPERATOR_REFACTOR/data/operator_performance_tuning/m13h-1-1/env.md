# cuda_019_tuning.jl run environment — 2026-07-11

- Node: `m13h-1-1` (BYU RC, SLURM job `fm019`, sbatch `scripts/cuda_019_run.sh`)
- GPU: NVIDIA H200 (sm_90, 139.8 GiB), driver 580.159.4 (CUDA 13.0 capable)
- CUDA toolchain: runtime 12.8.0 local installation (`CUDA_HOME=/apps/cudatoolkit/12.8.1`),
  compiler 12.9.41 artifact; cuBLAS 12.8.4
- CUDA.jl (CUDACore) 6.2.0, GPUArrays 11.5.8, `CUDA_Runtime_jll.local: true`
- Julia 1.11.7, LLVM 16.0.6; project env `~/fm022env`, repo copy
  `~/projects/FastMultipole-022` (rsync of the local `matrix-ops` working tree with
  task-019 Phases A/B/D applied: dense fixed-stage GEMM concat M2L, fused
  gather/scatter kernels + preallocated chunk scratch, implicit constant-P
  interaction structure with per-class plan geometry)
- Protocol: `exec_time` = min of 3 warmed whole-lifecycle runs; stage timings
  (`t_*`) = min of 3 warmed per-stage launches (022 baseline stage timings were
  single-shot); `exec_max` = max of the 3 lifecycle samples (jitter indicator);
  `ConstantPAnalyticStencil(P, P>=8 ? 1e-8 : 1e-4)`; interaction list via
  `build_radix_interaction_list(LazyMaterializedBatches(32), policy, grid)`.
- Phase-A/B/D gates: `CUDA_019_RESULT=PASS`, `VALIDATION_EXIT=0`, lifecycle tests 204/204 +
  concat parity 19/19, `LIFECYCLE_TEST_EXIT=0`; parity/counters cases all passed
  (host-origin 1 upload/1 download; device-origin 0/0; `expansion_host_copies=0`).
  The later Phase-E completion record reports 205/205 + 19/19 after one
  device-residency assertion was updated; raw logs for either run are not present
  in the local artifact directory.
