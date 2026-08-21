# 023b Implementation Factored Resident M2L (CUDA)

## Objective

Implement the CUDA device-resident factored M2L stage mirroring the `023a` host
stage, so the factored candidate in `024` is a real GPU measurement instead of a
forfeit. Today the CUDA path explicitly rejects `FactoredRotationM2L` in two
places: `RadixFMMCache(...; device=true)` throws
(`src/translate_batched_resident.jl:731-733`), and
`_assert_cuda_resident_stage!` throws
(`src/translate_batched_cuda.jl:1133-1140`, "not implemented for the CUDA
resident lifecycle"). This task removes those throws by supplying the
implementation.

This is a two-phase task. First establish functional CUDA parity with the `023a`
host implementation and the existing concat path while satisfying the resident
transfer-counter contract. Then profile, optimize, and retest on an H200.
Completion requires recorded before/after measurements; removing the throws with
an otherwise functional port is not sufficient.

## Dependencies

- `023a-impl-factored-resident-m2l-host.md` (the host stage is the reference
  implementation and parity oracle)
- `022-impl-gpu-device-resident-m2l.md` (device-resident lifecycle, buffer
  residency contract, transfer counters)
- `020a-impl-device-radix-grid-construction.md` (device radix metadata)
- `023-impl-production-integration.md` (`RadixFMMCache` device mode, counter
  contract, in-place per-step refresh)

## Required Reading

- `START_HERE.md`
- The dependency task files above
- `src/translate_batched_cuda.jl` (`_launch_cuda_resident_m2l!`, the concat
  device path, `run_cuda_radix_lifecycle!`, `_assert_cuda_resident_stage!`)
- The `023a` implementation and its notes

## Artifacts or Production Surface

- Production code: `src/translate_batched_cuda.jl` (device stage), plus the two
  guard sites above; any new types in `src/containers.jl` per the placement
  rules. GPU code stays behind the existing runtime-flag loading
  (`load_cuda_radix_lifecycle!()`); the CPU path and public API are unaffected
  when CUDA is absent.
- Tests: `test/cuda_radix_lifecycle_test.jl` and
  `test/cuda_radix_integration_test.jl` (extended).

## Design Sketch (implementer must verify against current code)

- Mirror the `023a` host stage: per offset class, batched per-degree
  `U_n`/`V_n` mode-matrix applications (CUBLAS batched/strided GEMM, or the
  fused-kernel patterns established by the `019` tuning if launch overhead
  dominates at production class sizes), shared-angle phase broadcasts, shared
  fixed-`m` z-translation, return alignment.
- Device buffers for factored scratch are allocated once at construction
  (residency contract from `022`/`023`); the per-step refresh reuses them in
  place. No per-step device allocation beyond the already-accepted CUDA pool
  sort scratch.
- Preserve the `023` transfer-counter contract: `route_uploads` and
  `operator_uploads` constant after construction, `body_uploads` one per step
  per host system, `expansion_host_copies == 0`.
- Remove the two `FactoredRotationM2L` throws only after the stage exists;
  device mode must keep rejecting configurations the implementation does not
  cover (if any remain).

## Optimize, Profile, and Retest Phase

Once the functional device path is correct, record an H200 baseline and tune at
least class ordering, grouped/batched/strided GEMM selection, tiny-block launch
overhead, packing and device scratch layout, launch count, and any narrowly
justified fused kernels. Preserve a separately testable reference path when
fusion changes stage boundaries.

Rerun numerical parity, precision/channel coverage, lifecycle counters, and
allocation checks after each retained change. Record before/after stage and
steady-state timings, launch counts, allocations, persistent/peak device memory,
construction/upload cost, and any crossover policy. H200 measurements are
required for both the functional baseline and optimized result.

## Deliverables

- Device-resident factored M2L selectable through
  `RadixFMMCache(sys; device=true, options=CUDARadixLifecycleOptions(operator=FactoredRotationM2L()))`.
- Unchanged behavior for the default materialized/concat device path.
- H200 functional-baseline and optimized validation runs, with per-stage timing,
  allocation, memory, construction/upload, and launch-count data recorded
  (informational; the definitive comparison is `024`).

## Verification

- Extend the CUDA tests with factored cases:
  - GPU-vs-`direct!`: potential `< 1e-4`, gradient `< 1e-2`;
  - GPU-vs-host-factored (`023a`) parity: `< 1e-9` potential / `< 1e-8`
    gradient;
  - transfer-counter assertions per the contract above;
  - Lamb-Helmholtz on and off, Float64 and Float32.
- Cover empty route sets, partial final batches, repeated time steps, and
  fixed-domain capacity reuse without recurring operator/route uploads or
  unaccounted steady-state allocation.
- Run the complete parity and lifecycle checks before tuning and after the final
  optimized implementation, and retain H200 before/after measurements.
- Full CUDA suite on the H200 via sbatch (pattern:
  `MATRIX_OPERATOR_REFACTOR/scripts/cuda_022_run.sh`). Cluster environment:
  `~/fm023env` with local-toolkit CUDA preferences, `#SBATCH --gpus=h200:1`
  (do **not** pass `--partition=eng`), compute nodes have no internet so run the
  login-node instantiate step first.
- **Show the user the Slurm scripts and get explicit permission before
  submitting any job.**

## Implementation Notes (2026-07-17): Phase 1 functional device stage

**Selection surface.** `RadixFMMCache(sys; device=true,
options=CUDARadixLifecycleOptions(operator=FactoredRotationM2L(),
m2l_strategy=ConcatenatedFixedZM2L()))` now builds and runs the factored device
M2L. The `RadixFMMCache` constructor throw was removed;
`_assert_cuda_materialized_operator!` remains only on the one-shot
`cuda_radix_state` builders (message updated to point at the cache path), and
the recurring lifecycle asserts the relaxed `_assert_cuda_supported_operator!`
(materialized or factored). `_radix_cache_device_build` now forwards
`options.operator` into `_radix_cache_workspace`, whose existing
`ResidentM2LFactoredPlan` branch comes out CuArray-backed automatically because
every array is built `_array_like_*` from the device exemplar.

**Plan extension** (`src/containers.jl`). `ResidentM2LFactoredPlan` gained
device-refresh and fused-kernel operator fields: `class_counts` (device Int32
histogram), `host_class_counts` (pinned host mirror), `class_starts`
(host prefix offsets), `class_theta`/`class_phi` (host per-class scalars),
`ym_flat` (flat Plain-H `U/V` re/im mode vectors for both the multipole and
local dressings, `ymode_offset` indexing), and `z_flat` (`m2l_z_block_length ×
nclasses` fixed-m z tables in `m2l_z_blocks!` layout). A 2-arg outer
constructor keeps the one-shot host construction site unchanged; the
`(::Type{TF}, ...)` builder (now also taking the invariant cache) fills them.

**Per-step refresh** (`_cuda_refresh_factored_m2l_routes!`,
`src/translate_batched_cuda.jl`). Device route emission is class-major and
contiguous (chunked ascending class ranges in `_cuda_route_compact_kernel!`),
so no per-group index repack exists on the device path: one atomic histogram
kernel over `route_class[1:n_routes]`, one pinned counts download (uncounted,
like the existing `host_scalar32` staging), a host prefix-sum into
`class_starts`, and `group.count[]` updates. Class k's columns are read
directly as contiguous views of `state.route_sources`/`route_targets`. Tests
assert `issorted(route_class)` plus histogram/starts consistency each step.

**Device stage** (CUDA method of `_launch_resident_m2l_factored_plan!`,
dispatched on `ResidentM2LFactoredPlan{<:CUDA.AnyCuArray}`). Per nonempty
class, 7 launches per channel through the 023a chain on the capacity-sized
workspace slabs (same slab rotation as the host
`_resident_factored_m2l_group_apply!`): the existing fused
`_gather_rotate_z!`/`_rotate_z_scatter_accumulate!` kernels; two fused
block-diagonal kernels per factored y application
(`_cuda_factored_y_phase_kernel!`: per-degree `V_n` contraction + shared-angle
`e^{iνθ}` phase with θ a class-constant scalar;
`_cuda_factored_y_out_kernel!`: `U_re·G′ − U_im·G″`), O(P³)/column; one fused
fixed-m z kernel (`_cuda_ztranslate_fixed_m_kernel!`) reading the class's flat
block column; and the LH row mix as two `_gather_rows!` gathers plus per-row
broadcasts with the group's r-specific local rows. No `Ts(θ)` and no ζ-dressed
013b primitives; no device allocation on the stage (tests assert
`CUDA.@allocated == 0`). The DEBUG[] physical-subspace guard now also runs in
the CUDA pipeline after M2M via a download-based device method of
`_assert_factored_input_physical` (diagnostic-only path).

**Kernel-math validation without a GPU.** A host emulation of the exact kernel
index arithmetic (flat y-mode indexing, degree decode, flat z-table indexing)
was checked against the 023a host references `_factored_y_degree_major_gen!`
and `_ztranslate_shared_precomputed!` for Float64/Float32 × LH on/off ×
P ∈ {4, 8} × both mode dressings × three offset classes: max deviation < 1e-12
(Float64) / < 1e-4 (Float32) — scratchpad script `check_023b_kernel_math.jl`.

**Tests** (`test/cuda_radix_integration_test.jl`). New 023b sections: factored
device parity matrix P ∈ (4, 8) × LH × Float64/Float32 (GPU-vs-host-factored
< 1e-9 potential / 1e-8 gradient at Float64; GPU-vs-`direct!` < 1e-4 / 1e-2 at
P = 8), class-contiguity and histogram/starts assertions, the transfer-counter
contract over a 3-step moving-body loop (route/operator uploads constant,
body +1/step, metadata +3/step, `expansion_host_copies == 0`, array identity),
an all-nearfield cluster (empty/near-empty route set), the one-shot
`cuda_radix_state` rejection, the M2L device-allocation bound, and a DEBUG[]-on
device run. Awaiting H200 execution.

**Local verification.** Full host `Pkg.test()` green (exit 0); host radix
integration 63/63 unchanged. Cluster artifacts:
`scripts/benchmark_023b_factored_cuda.jl` (construction, persistent/peak device
memory, per-stage GPU timings, per-step wall time, M2L device allocation,
accuracy vs `direct!`), `scripts/cuda_023b_run.sh` / `cuda_023b_submit.sh` /
`cuda_023b_fetch.sh`.

## Implementation Notes (2026-07-17): H200 functional baseline and fixes

**Job 12795229** (H200 m13h-1-1, user-approved): lifecycle suite 205/205 +
37/37 (no regression); integration 64/65. The single error was a
**pre-existing 022/023 capacity bug** exposed by the new P = 4 coverage: the
device direct-pair flag grid is `nreject × n_cells` but its buffer holds only
`max_cells × min(nreject, max_cells)`, and at P = 4 the stencil rejects more
offsets than `max_cells`. Fixed by processing the flag→scan→compact in flat
ascending chunks bounded by the flag buffer (emission order elementwise
identical to the host; the single-chunk case reproduces the old behavior
exactly), with new device-vs-host route *and* direct-pair elementwise parity
assertions at P = 4. The benchmark driver's `CUDA.available_memory` was
corrected to `CUDA.free_memory` (CUDA.jl 6.2 API).

**Job 12795598** (H200 m13h-1-1, user-approved): lifecycle 205/205 + 37/37;
integration **194/195** — the entire factored device section passed (parity
matrix P ∈ {4,8} × LH × F32/F64 incl. GPU-vs-host-factored < 1e-9/1e-8 at
Float64, transfer-counter contract, class contiguity, route/direct parity,
empty-route cluster, zero device allocation). The one remaining error was a
method ambiguity of the DEBUG-guard device method (missing
`B<:AbstractOperatorBasis` bound), hit only in the DEBUG[]-on section; fixed.
Benchmark completed (`data/factored_resident_m2l_cuda/
cuda_m13h-1-1_20260717-213754.csv`).

**Functional baseline (per-class factored vs concat, H200).** As predicted the
per-class stage is launch-bound: production stencils give 1.5k–3.3k nonempty
classes at mean width 2–70 (≈40k launches/step). M2L stage medians (non-LH /
LH), N = 20000: P = 4: 51.6 / 124.4 ms vs concat 0.26 / 0.55 ms; P = 8:
110.4 / 290.8 ms vs 3.2 / 6.7 ms; P = 12: 154.5 / 362.3 ms vs 9.0 / 17.9 ms.
The factored plan's **persistent device memory is 5–30× smaller**: 67–235 MB
vs concat 335–5067 MB across the sweep (P = 12 LH: 235 MB vs 5067 MB).
Construction is also cheaper at high P. Full-step and accuracy columns in the
CSV; `expansion_host_copies == 0` and constant route/operator uploads held
throughout.

## Implementation Notes (2026-07-17): whole-pass optimization

The optimize phase targets the launch count: every factored stage is (or can
be) per-column parameterized, so `_launch_resident_m2l_factored_whole!`
processes the flattened routes in `FACTORED_CUDA_CHUNK[]`-column chunks
(default 2^14) with per-column class indirection — per-column θ in the y phase
kernel (`_cuda_factored_y_phase_cols_kernel!`), per-column class column of the
flat z tables (`_cuda_ztranslate_fixed_m_cols_kernel!`), unit local LH rows
scaled per column by gathered class r (linear in r, the concat trick), and the
existing per-column gather/scatter kernels — ≈7 launches per channel per chunk
instead of per class. The bundle (device class geometry tables, per-chunk
column-parameter gathers, chunk-width slabs) is built once at cache
construction (`_cuda_factored_whole_pass_setup!`) into `plan.whole_pass[]`.
The per-class path is retained as the separately testable reference
(`FACTORED_CUDA_WHOLE_PASS[] = false`); tests assert whole-pass vs per-class
parity (< 1e-10 Float64) plus the zero-device-allocation bound on the
production dispatch, and the benchmark now measures
concat / factored (whole-pass) / factored_perclass.

**Job 12795641** failed on a namespacing slip in the new setup function (bare
`CuArray`; the CUDA module is a require-binding, so the error only surfaces at
runtime on the GPU node). Fixed and the whole CUDA file audited for
unqualified CUDA-only names (none remain). Lifecycle stayed green and the
ambiguity fix held on that run.

**Job 12795678** (H200 m13h-1-1, user-approved): **all tests green** —
lifecycle 205/205 + 37/37, integration 195/195 (`INTEGRATION_TEST_EXIT=0`),
benchmark complete (`cuda_m13h-1-1_20260717-220914.csv`). Optimized M2L stage
medians vs concat vs the per-class functional baseline (Float64):

| P | LH | N | factored (whole-pass) | concat | per-class baseline |
|---|----|----|----|----|----|
| 4 | off | 20000 | **0.20 ms** | 0.26 ms | 51.9 ms |
| 8 | off | 20000 | **2.53 ms** | 3.16 ms | 110.2 ms |
| 12 | off | 20000 | **6.73 ms** | 9.00 ms | 153.9 ms |
| 12 | on | 20000 | **15.26 ms** | 17.87 ms | 343.3 ms |
| 4 | off | 150 | **0.055 ms** | 0.267 ms | 30.5 ms |

The whole-pass factored M2L is **faster than concat at every measured
config** (1.1–4.9×; 20–550× over the per-class baseline) while keeping the
factored memory advantage: persistent device memory 67–470 MB vs concat
134–5067 MB (P = 12 LH N = 20000: **470 vs 5067 MB, 10.8×**), and
construction is cheaper at high P. The M2L device-allocation and counter
contracts held.

**Profiling finding (full-step).** The factored full step initially carried a
50–200 ms host-side overhead over concat despite every GPU stage being
faster: `_assert_cuda_resident_stage!` runs 5×/step and its residency walker
recursed into all ~1.6–3.3k factored groups × ~12 fields × (P+1) z-block
matrices with a string interpolation per visit. Since every group is built by
the same construction loop from the same exemplar, the walker now checks one
representative group plus the operative plan arrays.

**Job 12795697 (final validated run, H200 m13h-1-1, user-approved): all tests
green** (lifecycle + integration + benchmark exits all 0; CSV
`cuda_m13h-1-1_20260717-222256.csv`, job output retained alongside). With the
walker fix, the factored whole-pass path is **faster than concat on both the
M2L stage and the full step at every measured config**, with 2–11× less
persistent device memory. Full-step medians (factored vs concat, Float64):

| P | LH | N | factored step | concat step | factored mem | concat mem |
|---|----|----|----|----|----|----|
| 4 | off | 150 | **4.20 ms** | 4.29 ms | 67 MB | 134 MB |
| 8 | off | 2000 | **7.99 ms** | 8.49 ms | 168 MB | 973 MB |
| 12 | off | 2000 | **23.77 ms** | 25.90 ms | 268 MB | 2013 MB |
| 8 | on | 20000 | **40.09 ms** | 40.59 ms | 268 MB | 2517 MB |
| 12 | on | 20000 | **120.23 ms** | 122.68 ms | 470 MB | 5067 MB |

The per-class reference path is retained and measured (`factored_perclass`
rows) as the honest pre-optimization baseline: 30–360 ms M2L, 20–550× slower
than the shipped whole-pass execution. At N = 20000, P = 12 the remaining
full-step gap between the variants is dominated by L2B/direct (~100 ms
shared), consistent with the 019/023 observation that L2B is the next
lifecycle lever (carried in 024/019a notes).

**Task status: complete.** Deliverables:
`RadixFMMCache(sys; device=true, options=CUDARadixLifecycleOptions(operator=
FactoredRotationM2L(), m2l_strategy=ConcatenatedFixedZM2L()))` runs the
genuine factored device-resident M2L; the default materialized/concat device
path is unchanged (its benchmark rows match the pre-023b measurements); both
guard throws resolved (one-shot builders keep a narrowed rejection); counter
and zero-realloc contracts test-asserted; H200 functional-baseline and
optimized measurements recorded here and in
`data/factored_resident_m2l_cuda/` (three CSVs + job outputs, jobs
12795598 / 12795678 / 12795697). Awaiting separate-agent clear-context
approval.

## Implementation Notes (2026-07-18): compact CUDA factored-plan storage

The follow-up plan `plans/023b-compact-cuda-factored-plan.md` removes the
heavyweight per-offset `ResidentOperatorGroup` array from CUDA factored caches.
`_radix_cache_workspace` now requests a compact factored plan only from the
device-cache builder; host construction remains unchanged and still owns the
full groups required by the approved `023a` implementation. A compact CUDA plan
has `isempty(plan.groups)` and retains only its capacity-sized `route_class`,
device/host class counts, host prefix starts, host class `phi`/`theta`/`r`, flat
Plain-H mode tables, device `z_flat`, and the existing whole-pass bundle.

The optimized whole-pass algorithm and public selection surface are unchanged.
The retained independent per-class CUDA reference now iterates
`host_class_counts`/`class_starts`, fills the shared `ws.phis` prefix with the
class scalar `phi`, reads `theta`, `r`, and the `z_flat` column directly from the
compact plan, and applies the unit-radius local Lamb--Helmholtz rows from the
whole-pass bundle scaled by class `r`. Route refresh updates only the compact
histogram/prefix representation; it no longer depends on group counts. The CUDA
residency walker accepts the intentionally empty group vector while continuing
to validate every operative compact/device field.

CUDA integration coverage now checks zero device-plan groups while confirming
that the host oracle still has one full group per class; reconstructs every
stored class angle/radius from its Cartesian stencil offset; compares every
`z_flat` column against a fresh `m2l_z_blocks!`; compares route classes,
histograms, prefix starts, sources, and targets against the host builder at
construction and during moving-body steps; and compares the per-class CUDA
reference directly against `023a` at `locals.phi`/`locals.chi` before L2L/L2B.
The precision/channel/whole-pass matrix now includes `P = 4, 8, 12`. The
benchmark reads occupancy from `host_class_counts` and records
`resident_groups` (expected zero for factored CUDA), so construction time and
persistent-device-memory measurements reflect the compact representation.

Local verification (no CUDA device on this host):

- `julia --project=test test/cuda_radix_integration_test.jl`: CPU-safe gate 1/1.
- `julia --project=test test/radix_fmm_integration_test.jl`: 63/63.
- `julia --project=. -e 'using Pkg; Pkg.test()'`: complete suite green.
- Forced CUDA extension loading could not run because CUDA.jl is not installed
  in the local test environment; the H200 lifecycle/integration run is therefore
  still required.

**H200 validation (user-approved, 2026-07-18).** Job `12797754` established that
the lifecycle suite and benchmark were green, but eight newly added raw-local
buffer assertions used an inappropriate absolute-only tolerance. Local expansion
coefficients grow strongly with order: for example, a `0.3125` absolute
difference at `P=12` was a tiny relative difference, and all end-to-end checks
were already green. The assertion was corrected to the coefficient tolerances
already used by the CUDA lifecycle tests (`rtol=1e-9, atol=1e-10` for Float64;
`rtol=5e-3, atol=5e-4` for Float32); production code was unchanged.

User-approved rerun job `12797796` on H200 node `m13h-2-2` is fully green:
lifecycle `205/205` plus host-concat parity `37/37`; integration
`127148/127148`; benchmark exit 0. The final artifacts are
`data/factored_resident_m2l_cuda/fm023b-12797796.out` and
`cuda_m13h-2-2_20260718-074240.csv`. Every factored row reports
`resident_groups=0`, zero M2L device allocation, unchanged route/operator upload
counters, and `expansion_host_copies == 0` through the test suite.

Representative warmed `N=20000`, Float64 results compare the previous grouped
CUDA plan (job `12795697`) with the compact plan (job `12797796`):

| P | LH | construction grouped -> compact | persistent grouped -> compact | M2L grouped -> compact | full step grouped -> compact |
|---|---|---:|---:|---:|---:|
| 4 | off | 71.70 -> **6.09 ms** | 100.7 -> **67.1 MB** | 0.201 -> 0.202 ms | 10.58 -> 10.56 ms |
| 8 | off | 196.95 -> **11.42 ms** | 167.8 -> **100.7 MB** | 2.519 -> 2.513 ms | 29.70 -> 29.65 ms |
| 12 | off | 281.46 -> **32.83 ms** | 268.4 -> **201.3 MB** | 6.723 -> 6.739 ms | 98.06 -> 98.03 ms |
| 12 | on | 483.85 -> **48.05 ms** | 469.8 -> **335.5 MB** | 15.259 -> 15.252 ms | 120.23 -> 120.33 ms |

Thus the compact representation removes the construction bottleneck (roughly
9-17x on these warmed rows) and saves 25-40% of the factored persistent footprint
without a measurable steady-state regression. At `P=12`, LH, `N=20000`, peak
device footprint also fell from 539.0 MB to 404.8 MB. Final direct-reference
errors remain unchanged (representative `P=12`, `N=20000`: potential
`2.47e-11`, gradient `3.27e-9` without LH; gradient `1.24e-9` with LH).

## Approval Notes

Clear-context approval, 2026-07-18, by a separate reviewing agent.

**Approved.** Reviewed per `START_HERE.md` item 6: this task file, the
production surfaces (`src/translate_batched_cuda.jl` device stage, refresh,
whole-pass bundle, and residency walker; `ResidentM2LFactoredPlan` extension in
`src/containers.jl`), the test extensions, the compact-plan follow-up
(`plans/023b-compact-cuda-factored-plan.md`), and the recorded cluster evidence
in `data/factored_resident_m2l_cuda/`.

1. **Objectives**: met. Both `FactoredRotationM2L` guard throws are resolved
   (the one-shot `cuda_radix_state` builders retain a narrowed rejection that
   points at the cache path), the factored device M2L is selectable through the
   documented `RadixFMMCache(...; device=true, options=...)` surface, and the
   default materialized/concat device path is unchanged. The required two-phase
   structure (functional H200 baseline, then optimize/retest) is satisfied with
   recorded before/after data, ending fully green on job `12797796`
   (lifecycle 205/205 + 37/37, integration 127148/127148, benchmark exit 0).
2. **Correctness**: GPU-vs-host-factored parity < 1e-9/1e-8 at Float64,
   GPU-vs-`direct!` < 1e-4/1e-2, P ∈ {4, 8, 12} × LH × Float32/Float64,
   whole-pass vs per-class parity, class-contiguity/histogram assertions, and
   the transfer-counter contract (`route_uploads`/`operator_uploads` constant,
   `expansion_host_copies == 0`) are all test-asserted. The P=4 direct-pair
   flag-buffer capacity fix corrected a genuine pre-existing 022/023 bug and is
   covered by device-vs-host elementwise route and direct-pair parity at P=4.
   Local host verification rerun by the reviewer: focused suites and full
   `Pkg.test()` green on this working tree.
3. **Performance**: the whole-pass factored path beats concat on the M2L stage
   and full step at every measured config with 2–11× less persistent device
   memory; the compact-plan amendment removes the construction bottleneck
   (~9–17× on warmed rows) with no steady-state regression. The honest
   per-class baseline is retained and measured.
4. **Robustness**: zero-device-allocation bound on the production dispatch,
   empty-route cluster, moving-body multi-step counter checks, DEBUG[]-on
   device run, and the residency walker still validates every operative
   compact/device field. The raw-local-buffer tolerance fix (absolute → the
   established rtol/atol pairs) was the right call; production code unchanged.
5. **Minimally invasive**: device-only compact storage is opt-in via
   `compact_cuda_factored=true` from the device-cache builder; host 023a
   construction and the public API are untouched, and CUDA code stays behind
   `load_cuda_radix_lifecycle!()`.
6. **Readability**: the staged implementation notes, kernel comments, and the
   flat-index layout documentation are clear.

Non-blocking observations (no change made; carry into `024`/`019a` if useful):
the L2B/direct ~100 ms shared full-step floor at N=20000, P=12 remains the next
lifecycle lever, as already noted; `FACTORED_CUDA_WHOLE_PASS`/`FACTORED_CUDA_CHUNK`
remain global Refs, acceptable as established benchmark practice in this
codebase.
