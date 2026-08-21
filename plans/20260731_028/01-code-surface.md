# 028 Phase A — Code surface recipe (verified 2026-07-31)

All signatures/paths verified; do not re-explore.

## Construction

```julia
FastMultipole.load_cuda_radix_lifecycle!()   # required before device=true
opts = CUDARadixLifecycleOptions(; precision=TF,           # Float64|Float32
    operator=MaterializedYRotationM2L(),                   # FactoredRotationM2L() for precomputed_y
    m2l_strategy=DenseTranslationM2L())                    # or PrecomputedFactoredYM2L(), ConcatenatedFixedZM2L()
cache = RadixFMMCache(sys; expansion_order=3, ell, max_n_bodies=N,
    bounds=(x_min::SVector{3}, box_size), lamb_helmholtz=false,
    device=true, options=opts,
    near_radius2=12,          # 12 = hier12 (theta=0.5), 3 = hier3 (classic)
    window_classes=K)         # ALWAYS pass explicitly (K-default trap, see 00 file)
# flat oracle instead: stencil_epsilon=... (mutually exclusive with the two above)
```
`src/translate_batched_resident.jl:776-956`; options ctor `src/containers.jl:1561-1582`;
capacity math `translate_batched_resident.jl:852-861`. Strategy combos enumerated at
`MOR/scripts/benchmark_027_hierarchical_cuda.jl:96-105`.

## Per-step (maps 1:1 to the three timing boundaries)

`fmm!(sys, cache; scalar_potential=false, gradient=true)` → `_radix_cache_device_step!`
(`src/translate_batched_cuda.jl:3799-3806`) = exactly:
1. `update_cuda_radix_state!(cache, (sys,))` — **full** in-place Morton rekey + sort +
   tree rebuild + occupancy + direct pairs (hier far-field routes generate lazily
   inside M2L; `n_routes=0` after refresh). No incremental refit exists; the only
   cheaper reuse policy is skipping refresh for k steps (stale tree).
2. `run_cuda_radix_lifecycle!(cache.state)` — B2M→M2M→M2L→L2L→L2B+direct
   (evaluation-only boundary).
3. `finalize_cuda_radix_output!(...)` — device-resident targets stay on device.

## Per-stage timing

Individual launchers (median with `CUDA.@elapsed`; copy `_median_gpu_ms` from
`MOR/scripts/benchmark_027_hierarchical_cuda.jl:85-92`):
`_launch_cuda_b2m!`, `_launch_cuda_resident_m2m!`, `_launch_cuda_resident_m2l!`,
`_launch_cuda_resident_l2l!`, `_launch_cuda_resident_l2b!` (**= L2B + nearfield**).

Host telemetry: `hctx = cache.state.interaction_list  # ::DeviceHierarchicalM2LContext`
(`src/containers.jl:461-495`). Set `hctx.profile_stages=true`, one `fmm!`, read
`hctx.update_stage_ns` (1 grid rebuild, 2 occupancy, 3 direct-gen, 4 route
flag/scan/compact summed over windows, 5 stage-group refresh), `hctx.m2l_level_ns`,
`routes_per_level`, `nodes_per_level`. Usage: `benchmark_027_...jl:234-245`.
Host alloc/step: `@allocated` around the step (027's `full_step_host_alloc_bytes`).
Per-(level,class) route occupancy: `_class_occupancy(state)` harness at
`benchmark_027_...jl:137-159`.

## Outputs, counters, residency

- `state.output::CuMatrix{TF}` 4×maxn, **sorted (Morton) order**: row 1 potential,
  rows 2:4 gradient. Sorted→body maps: `state.body_perm`, `state.body_system_ids`,
  `state.body_indices`.
- Counters `cache.state.counters` (`src/containers.jl:1526-1539`): `body_uploads,
  influence_downloads, expansion_host_copies, route_uploads, operator_uploads,
  metadata_downloads`. Contract: post-construction `route_uploads`/`operator_uploads`
  constant, `expansion_host_copies==0`; verdict boundary additionally requires
  `body_uploads`/`influence_downloads`/`metadata_downloads` flat per step → needs a
  **device-resident source system**: `residency(sys)=DeviceResident()` +
  `source_to_buffer!(device_buffer, sys, sort_index)`; working example
  `test/cuda_radix_lifecycle_test.jl:17-57`. Host-resident variant of the same
  system gives boundary (c) (transfers included) for free.
- Assertion reference: `benchmark_027_...jl:222-226`.

## Convection (must be written — nothing exists in the repo)

CUDA kernel over `1:state.counts.n_bodies`: read gradient from `state.output` rows
2:4 (sorted order), scatter through `body_perm`/`body_indices` into the device
system's position array; proxy `x .+= dt .* v` per the task file. Clamp/choose dt so
bodies never exit the fixed `bounds` box (out-of-box → `ArgumentError`).

## Cluster (BYU ORC; from memory + 027 scripts)

- Templates to copy verbatim and rename: `MOR/scripts/cuda_027_submit.sh` /
  `cuda_027_run.sh` / `cuda_027_fetch.sh` + `benchmark_027_hierarchical_cuda.jl`.
  `REMOTE=orc`, `RDIR=FastMultipole-023`, `ENVDIR=$HOME/fm023env`; needs a live
  `ssh -fN orc` master. Login-node `Pkg.instantiate` is mandatory (compute nodes
  have no internet); CUDA.jl uses the **local toolkit** (artifact downloads fail).
- sbatch: `--gpus=h200:1`; bump to `--mem=128G+`, `--time=12:00:00+` for n=1e6
  (024b used 192G/24h). H200 nodes: m13h-1-1, m13h-2-1/2-2 — record which.
- Accuracy plumbing: `MOR/scripts/benchmark_024b_common.jl` (reference reader
  :36-86, error reducer :88-124) + `MOR/data/cpu_gpu_scaling/references/direct_reference_n1000000.csv`.
- Per-case one-process-per-job/resume/OOM-ledger idioms (if needed at n=1e6):
  `cuda_024b_run.sh` + `benchmark_024b_gpu.jl`.
- Results land in `MOR/data/feasibility_1m_10ms/` both ends; fetch pattern in
  `cuda_027_fetch.sh`.

## Progress

- [ ] (reference only)
