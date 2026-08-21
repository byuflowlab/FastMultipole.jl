# 028 Phase A.1 — Build the harness

No production `src/` changes. All new files are scripts/test-side. If a small
instrumentation hook in `src/` proves unavoidable, gate it behind the existing
`profile_stages`-style pattern and flag it for the approval reviewer.

## Deliverables

1. **Device-resident source system + device Euler convection kernel**, in the
   benchmark script (shared include if a test also needs it). Model the system on
   `test/cuda_radix_lifecycle_test.jl:17-57`; convection recipe in
   `01-code-surface.md`. Add a small correctness test (any new test must run at
   P=4 — automatic here since expansion_order=3).
2. **`MOR/scripts/benchmark_028_feasibility.jl`** — copy conventions from
   `benchmark_027_hierarchical_cuda.jl`: env-var knobs (`FM028_*` comma-lists),
   provenance columns (`manifest/job/host/gpu/julia/cuda/blas_threads/seed`),
   per-case try/catch + empty-row fallback, `GC.gc(); CUDA.reclaim()` between
   cases, manual CSV write, companion `.classes.csv`. Per case, record:
   - construct seconds + persistent device bytes,
   - the three boundaries: (a) evaluation-only (`run_cuda_radix_lifecycle!`
     median), (b) verdict step (refresh + lifecycle + finalize + Euler kernel,
     device-resident, counters asserted flat), (c) transfers-included
     (host-resident variant; H2D up / D2H down timed separately),
   - per-stage CUDA-event medians + `profile_stages` telemetry + per-level M2L +
     class-occupancy histograms + host-alloc bytes/step,
   - all five 024b error columns vs the n=1e6 direct reference, + F32
     admissibility check vs the F64 truncation row.
3. **`cuda_028_submit.sh` / `cuda_028_run.sh` / `cuda_028_fetch.sh`** — copy the
   027 triplet, rename, `--mem>=128G`, `--time>=12:00:00`; run script keeps the
   `FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1` CUDA-test preflight and source-manifest
   print. Output dir `MOR/data/feasibility_1m_10ms/`.

Local check is syntax/include only — the Mac has no CUDA; correctness smoke runs
on the cluster via the run script's test preflight.

## Progress

- [x] Device system + Euler kernel written (`MOR/scripts/fm028_device_system.jl`; bodies replicate `generate_gravitational` seed-24025 stream so 024b references apply; overwrite-semantics `buffer_to_target!`; clamped-broadcast Euler; on-device F64 sampled direct reference kernel)
- [x] Convection test added (`test/cuda_radix_convection_test.jl`, expansion_order=3, F64+F32, counter contract + moved-positions accuracy; skips gracefully without CUDA — verified locally)
- [x] benchmark_028_feasibility.jl written (three boundaries in one row via `merge(ROW_DEFAULTS, ...)` schema; boundary (c) uses a second host-resident cache because host stagings are `nothing` for device-resident construction; `_cache_kwargs` passes `near_radius2`+`window_classes` explicitly — K-default trap avoided; flat rows reproduce the 024b epsilon/box/seed exactly)
- [x] cuda_028_{submit,run,fetch}.sh written (submit takes pilot|sweep mode → sbatch --time + FM028_MODE; run.sh presets respect pre-exported FM028_* overrides; 192G default)
- Notes: parse + bash -n + local graceful-skip all verified 2026-07-31. Boundary-(c) baselines: 024b step_min was measured on a host-resident system, so the flat cross-check compares against `host_step_ms`.
