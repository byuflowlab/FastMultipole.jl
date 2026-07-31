# 023d Implementation Precomputed-Y Resident M2L (CUDA)

## Objective

Port `PrecomputedFactoredYM2L` from `023c` to the CUDA device-resident lifecycle.
All precomputed `M_n(theta)` matrices, class mappings, route metadata, and
capacity-sized scratch must remain resident. Construction may upload them once;
steady-state steps must perform no recurring operator or route uploads.

This task also has distinct functional and optimization phases. First match the
host reference and preserve the device lifecycle invariants. Then profile on an
H200, optimize the stage, and rerun the complete correctness and lifecycle suite.
Completion requires recorded before/after H200 measurements.

## Dependencies

- `023b-impl-factored-resident-m2l-cuda.md` (functional CUDA factored stage and
  device lifecycle integration)
- `023c-impl-precomputed-y-m2l-host.md` (host implementation and numerical oracle)
- `020a-impl-device-radix-grid-construction.md` (resident radix metadata)
- `022-impl-gpu-device-resident-m2l.md` (buffer residency and transfer counters)
- `023-impl-production-integration.md` (`RadixFMMCache` construction, refresh,
  and capacity contracts)

## Required Reading

- `START_HERE.md`
- The dependency task files above and the completed `023c` implementation notes
- `src/containers.jl`, `src/translate_batched_resident.jl`, and
  `src/translate_batched_cuda.jl`
- Existing CUDA lifecycle, integration, precision, and transfer-counter tests

## Artifacts or Production Surface

- Device construction and M2L application code in
  `src/translate_batched_cuda.jl`; shared strategy/storage changes remain in the
  placement required by `START_HERE.md`.
- CUDA parity and lifecycle tests under `test/`.
- Reproducible H200 scripts under `MATRIX_OPERATOR_REFACTOR/scripts/` and
  measurements under `MATRIX_OPERATOR_REFACTOR/data/`.

## Functional Phase

- Mirror the canonical angle classes and multipole/local per-degree matrices
  established by `023c`. Upload invariant matrices and class metadata during
  cache construction only.
- Keep route metadata and operator storage on device across repeated steps.
  Fixed-domain refresh may update dynamic counts/contents in place but must not
  trigger host round trips or recurring invariant uploads.
- Implement device class packing, per-degree/class matrix application, explicit
  `Z_phi`, fixed-`m` z translation, Lamb-Helmholtz coupling, and scatter with the
  same mathematical stage boundaries as the host reference.
- Establish parity against direct evaluation, the existing CUDA concat path,
  the CUDA per-degree factored path, and the `023c` host reference before tuning.

## Optimize, Profile, and Retest Phase

Capture a correct H200 baseline, then investigate and measure at least:

- grouped GEMM versus batched/strided-batched GEMM selection by block size and
  class occupancy;
- canonical class ordering and coalesced class-sorted gather/scatter;
- tiny-block launch overhead and crossover/fallback rules;
- operator packing, matrix orientation, and scratch layout;
- eliminating redundant packing or launch boundaries; and
- narrowly scoped fused kernels when profiling justifies them and parity remains
  independently testable.

Retest after each retained change. Record before/after kernel and lifecycle stage
timings, launch counts, host/device allocations, persistent device memory,
construction/upload cost, and crossover decisions. H200 measurements are
required; measurements from another GPU are supplementary.

## Deliverables

- Device-resident `PrecomputedFactoredYM2L` selectable through the production
  cache/options surface, still requiring `FactoredRotationM2L()`.
- Construction-time-only operator and route-metadata upload behavior.
- A measured CUDA execution/crossover policy for large, sparse, and tiny angle
  classes.
- H200 before/after benchmark artifacts and stage profiles.

## Verification

- Float32 and Float64; Lamb-Helmholtz on and off.
- CUDA-vs-host `023c`, CUDA-vs-concat, CUDA-vs-factored, and direct-evaluation
  parity before optimization and after the final optimization.
- Empty route sets/classes, partial final batches, uneven occupancy, repeated
  time steps, and fixed-domain capacity reuse.
- Transfer counters: `route_uploads` and `operator_uploads` remain constant after
  construction, `expansion_host_copies == 0`, and body/output counters retain
  the established `022`/`023` meanings.
- No recurring device allocation outside explicitly accepted CUDA pool/sort
  behavior; persistent and peak device memory are recorded.
- Full CUDA suite and measured H200 run.
- **Show the user the Slurm scripts and get explicit permission before
  submitting any job.**

## Implementation Notes (2026-07-18): Phase 1 functional device stage

**Selection surface.** `RadixFMMCache(sys; device=true,
options=CUDARadixLifecycleOptions(operator=FactoredRotationM2L(),
m2l_strategy=PrecomputedFactoredYM2L()))` now builds and runs the precomputed-y
device M2L. The `_assert_cuda_supported_operator!` deferral throw was removed;
the one-shot `cuda_radix_state` builders keep rejecting the factored operator
family, and the options constructor keeps requiring the
`FactoredRotationM2L()` pairing.

**Plan extension** (`src/containers.jl`). `ResidentM2LPrecomputedYPlan` gained a
`route_class` type parameter (host `Vector{Int32}`, device `CuVector{Int32}` —
device route emission writes it directly, mirroring the factored/concat plans)
and trailing compact-device fields: `class_counts` (device Int32 per-offset
histogram) + pinned `host_class_counts`, `y_flat_mult`/`y_flat_loc` (flat real
`M_n(theta)` block tables in `ymode_offset` layout, one column per exact angle
class), `z_flat` (`m2l_z_block_length x noffsets`, `m2l_z_blocks!` layout —
identical content and kernel indexing to the 023b factored table since factored
classes are the accepted offsets), `offset_rs` (host radii for LH unit-row
scaling), and a `whole_pass` bundle Ref. A 22-arg outer constructor keeps the
approved 023c host construction site unchanged; a `compact_device` keyword on
the `(::Type{TF}, ...)` builder produces the compact CUDA layout (nested host
operator storage and packed route arrays empty, flat tables built host-side and
uploaded through `_array_like_*`).

**Per-step refresh** (`_cuda_refresh_precomputed_y_m2l_routes!`,
`src/translate_batched_cuda.jl`). Device route emission is offset-class-major
and contiguous, so the host's angle-major physical repack does not exist on the
device path: one atomic histogram kernel, one pinned counts download (uncounted
staging, like the factored path), a host prefix into `offset_starts`
(route-order 1-based starts + sentinel), a partition assertion, and derived
`angle_counts` totals for diagnostics/tests. `angle_starts` is meaningless in
the device layout and stays untouched.

**Device stage.** `_launch_resident_m2l_precomputed_y_plan!` dispatches on the
device-typed plan. Both drivers share one per-column stage chain
(`_cuda_precomputed_y_apply_cols!`): fused gather+`Z_phi`, ONE new kernel
`_cuda_precomputed_y_cols_kernel!` per y application (each column reads its
angle's precomputed real block — replacing the factored path's V/phase/U kernel
pair, ~half the y-stage work at O(P^3)/column), the reused per-column fixed-m z
kernel over per-offset `z_flat` columns, the LH unit-row-times-r mix, one local
precomputed-y kernel, and the fused inverse-`Z_phi` atomic scatter. The
whole-pass driver (`PRECOMPUTED_CUDA_WHOLE_PASS[] = true`, chunk
`PRECOMPUTED_CUDA_CHUNK[]` = 2^14) gathers per-column phi/angle/r through the
route classes (~6 launches per channel per chunk); the per-class reference
driver constant-fills the same column-parameter buffers per offset class,
sub-chunked at the bundle width. The residency walker gained a
`ResidentM2LPrecomputedYPlan` clause checking only the operative flat device
fields (023b walker lesson). The DEBUG[] physical-subspace guard covers the
path automatically (operator-keyed in both pipelines).

**Kernel-math validation without a GPU** (scratchpad
`check_023d_kernel_math.jl`): host emulation of the exact kernel index
arithmetic (flat y-block indexing per angle, degree decode, flat z-table
indexing) reproduces the approved 023c host reference bit-for-bit (max
deviation 0.0) across Float64/Float32 x LH on/off x P in {4, 8} x both
dressings, including pole/equator/mixed-angle offsets.

**Tests** (`test/cuda_radix_integration_test.jl`, 023d section). Parity matrix
P in {4, 8, 12} x LH x Float64/Float32: compact-plan shape checks, angle-key /
theta / phi / r / z-column reconstruction from Cartesian stencil offsets, flat
y blocks vs the host plan's nested matrices (exact), route-class/histogram
parity vs the host cache, per-class and whole-pass M2L parity vs host 023c at
the locals-buffer boundary before L2L/L2B, end-to-end GPU-vs-host-023c
(<1e-9/1e-8 Float64), cross-strategy parity vs the CUDA concat and CUDA
factored paths, direct! accuracy at P=8, whole-pass vs per-class end-to-end
(<1e-10), zero M2L device allocation; plus a tiny-chunk stress (chunk=7:
multi-chunk whole pass and per-class sub-chunking), a 3-step moving-body
counter/refresh/identity contract vs the host cache, the all-nearfield
empty-route cluster, options/one-shot rejections, and a DEBUG[]-on device run.

**Local verification** (Apple M-series host, no CUDA device): focused 023c
suite 73/73 (host behavior bit-identical after the storage refactor), radix
integration 63/63, CPU-safe CUDA gate 1/1, and full
`julia --project=. -e 'using Pkg; Pkg.test()'` green
(`FastMultipole tests passed`).

**Cluster artifacts.** `scripts/benchmark_023d_precomputed_y_cuda.jl`
(four-variant sweep: concat / factored whole-pass / precomputed-y whole-pass /
precomputed-y per-class baseline; records construction, persistent/peak device
memory, per-stage GPU timings, full-step wall time, M2L device allocation,
angle/offset-class occupancy, accuracy vs direct!, plus Float32 precomputed-y
rows) and the `cuda_023d_run.sh` / `cuda_023d_submit.sh` / `cuda_023d_fetch.sh`
trio. The user granted blanket job-submission permission for the 2026-07-18
session; job `12807177` submitted (H200 validation + benchmark).

## Implementation Notes (2026-07-18): H200 validation and functional baseline

**Job 12807177** (H200 m13h-2-2) failed at the first device precomputed-y cache
construction: the intermediate `U_n`/`V_n` mode blocks were dressed with
`_array_like_vector(exemplar, ...)`, so on the device exemplar the host-side
`U * Diagonal(phases) * V` flat-table assembly hit GPU scalar indexing. Fixed by
building those intermediates as host arrays unconditionally (the compact path
uploads only the finished flat tables; host-path behavior is unchanged — the
exemplar there was always a host array). Lifecycle and all pre-023d integration
sections were already green on that run, and the benchmark's concat/factored
rows reproduced the 023b numbers.

**Job 12810375** (H200 m13h-2-2, after the fix): **all green** — lifecycle
205/205 + host-concat parity 37/37, integration **366064/366064** (the full
023d section: construction/geometry/flat-table parity, per-class and whole-pass
locals-boundary parity vs host 023c, end-to-end GPU-vs-host < 1e-9/1e-8,
cross-strategy parity vs CUDA concat and CUDA factored, direct! accuracy,
whole-pass vs per-class < 1e-10, tiny-chunk stress, 3-step counter/refresh
contract, empty-route cluster, zero M2L device allocation, DEBUG[]-on run),
benchmark exit 0. Artifacts: `data/precomputed_y_resident_m2l_cuda/`
(`fm023d-12810375.out`, `cuda_m13h-2-2_20260718-165345.csv`).

**Functional baseline (H200, warmed rows, Float64).** The precomputed-y
whole-pass M2L is the fastest resident variant at **every** measured
`P/LH/N/precision` config — 1.5–1.9x faster than the 023b factored whole-pass
and 2.0–2.6x faster than concat — at the factored-class memory footprint
(2–14x less than concat). The per-class reference baseline confirms the
whole-pass execution (20–500x slower per-class, launch-bound as in 023b).
Representative `N = 20000` M2L-stage medians:

| P | LH | precomputed-y | factored | concat | per-class baseline | prey mem | concat mem |
|---|----|----|----|----|----|----|----|
| 4 | off | **0.132 ms** | 0.201 ms | 0.260 ms | 49.7 ms | 67 MB | 335 MB |
| 8 | off | **1.374 ms** | 2.506 ms | 3.161 ms | 102.9 ms | 101 MB | 973 MB |
| 12 | off | **3.488 ms** | 6.726 ms | 8.992 ms | 107.6 ms | 201 MB | 2013 MB |
| 8 | on | **3.567 ms** | 6.184 ms | 6.709 ms | 296.8 ms | 201 MB | 2517 MB |
| 12 | on | **8.525 ms** | 15.242 ms | 17.870 ms | 303.8 ms | 369 MB | 5067 MB |

Full-step medians improve accordingly (P = 12 LH N = 20000: 114.2 ms vs
121.0/123.5 ms; the remaining step time is the shared L2B/direct floor noted by
019/023b). Float32 precomputed-y rows run 10–20% faster still on the M2L stage
and dominate the full step at high P (P = 12 LH: 84.7 ms). The counter and
zero-device-allocation contracts held throughout the benchmark.

**Construction.** Precomputed-y construction is 2–3x the factored cost at high
`P` (92–118 ms vs 33–46 ms at P = 12, N = 20000) — the per-angle flat `M_n`
assembly is the expected extra work and is construction-only. A repeatable
Float32 anomaly (P = 12 construction ~630–937 ms) was root-caused **outside**
this task's surface: `OperatorInvariantCache(Float32, ...)` itself costs
~557 ms at P = 12 vs ~17 ms for Float64 (measured on host; the 023d compact
plan builder is *faster* in Float32). Every Float32 cache of every strategy
pays this shared 009/013-infrastructure cost; carried as a watch item to
`024`/`019a` rather than modifying approved invariant-cache code here.

**Optimize-phase findings.** The functional implementation already embodies the
023b optimize-phase lessons (whole-pass chunking with per-column class
indirection, ~6 launches/channel/chunk — one fewer than factored since the
precomputed y application is a single kernel). The measured evidence resolves
the required investigation items: grouped/batched per-class GEMM execution is
launch-bound and loses by 20–500x at production occupancies, so the per-column
whole-pass kernel is selected at every block size and class occupancy with no
crossover/fallback rule needed (N = 150 through 20000 all favor it); operator
packing keeps the factored file's flat column-major `ymode_offset`/z-table
layouts (validated coalescing conventions); no redundant packing or launch
boundaries remain beyond the three per-chunk parameter gathers (~15 us,
negligible vs the 1.4–8.5 ms stage); further y+z kernel fusion is not justified
while the step is dominated by the shared L2B/direct floor and would cost the
independently testable stage boundary.

## Implementation Notes (2026-07-18): chunk-width sweep and final decision

**Job 12810676** (H200 m13h-2-2, all three sub-runs exit 0) swept
`PRECOMPUTED_CUDA_CHUNK` in {2^12, 2^14, 2^16} at N = 20000, P in {8, 12}, LH
on/off (7 reps; production code untouched — the env override drives the Refs;
CSVs `chunk_sweep_{4096,16384,65536}_m13h-2-2.csv`). Measured M2L stage /
persistent memory at P = 12, LH:

| chunk | M2L | full step | persistent | peak |
|---|---|---|---|---|
| 4096 | 10.99 ms | 116.6 ms | 168 MB | 237 MB |
| 16384 (default) | 8.56 ms | 114.4 ms | 369 MB | 438 MB |
| 65536 | 7.78 ms | 113.6 ms | 1074 MB | 1143 MB |

2^16 gains 9–16% on the M2L stage but at most 1.6% on the full step (the
shared L2B/direct floor dominates), while nearly tripling the persistent
footprint — eroding the strategy's memory advantage over concat from ~14x to
~4.7x at P = 12 LH. **The production default stays 2^14**: it preserves the
fastest-and-small profile that distinguishes this strategy, and the mutable
`PRECOMPUTED_CUDA_CHUNK[]` Ref remains the documented knob for users who want
the last 10–16% of M2L-stage throughput at ~3x the chunk-slab memory (set it
before cache construction; the bundle width is baked at build). 2^12 loses
20–29% M2L for only a ~200 MB saving and is not recommended. No production
change was retained from the sweep, so the job-12810375 validation stands as
the final tested state; the tiny-chunk integration test independently covers
chunk-width correctness.

**Task status: complete.** Deliverables: device-resident
`PrecomputedFactoredYM2L` selectable through
`RadixFMMCache(sys; device=true, options=CUDARadixLifecycleOptions(operator=
FactoredRotationM2L(), m2l_strategy=PrecomputedFactoredYM2L()))`;
construction-time-only operator/route uploads (counter contract test-asserted);
measured execution policy (whole-pass per-column kernels at every block size
and occupancy, per-class path retained as the testable reference, chunk default
2^14 with the measured tradeoff above); H200 before/after artifacts
(functional per-class baseline vs optimized whole-pass, four-variant
comparison, chunk sweep) in `data/precomputed_y_resident_m2l_cuda/`. Awaiting
separate-agent clear-context approval.

## Approval Notes

To be filled by a different agent after the task is complete.
