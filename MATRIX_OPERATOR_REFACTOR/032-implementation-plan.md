# Task 032 — Generalized Device-Resident System Interface (implementation plan)

Repo: `/Users/ryan/Dropbox/research/projects/tmp3/FastMultipole`, branch `matrix-ops`.
All edits land in this repo; **no FLOWVPM changes in this row**.

## Context

Row `031` (interface spec, `MATRIX_OPERATOR_REFACTOR/integration-api-spec.md`) and
`031a` (nearfield theory, `MATRIX_OPERATOR_REFACTOR/theory/kernel-splitting-nearfield.md`)
are Done + Approved, so `032` is unblocked. The shipped device-resident FMM lifecycle
only does scalar point sources: bodies pack into a hard-coded 5-row matrix, B2M never
writes the Lamb-Helmholtz `chi` channel, output is 4 rows (potential + gradient) with
`hessian=true` throwing, and the nearfield is a hard-coded singular `1/r` kernel.
FLOWVPM (row `034`) needs vector strengths `Γ`, per-body smoothing radius `σ`, the LH
channel end-to-end, a 9-component velocity gradient, and a regularized Biot–Savart
nearfield — with zero per-step allocation and zero per-step body transfers.

Outcome: an external device-resident consumer can drive the resident GPU lifecycle
with vector strengths, LH, and full hessian output, through a documented `src/`-level
API, with the shipped scalar path unregressed.

## User decisions (this session, 2026-08-05)

1. **Staged with checkpoints** — four stages, each ending with tests and a user
   checkpoint before the next begins.
2. **`recenter!`: fallback first.** Ship the documented `recenter!` that rebuilds
   geometry-dependent state in place with zero allocation. Attempt normalized
   unit-cube internal coordinates only if it proves non-invasive; otherwise record it
   as an `035` lever.
3. **Near-set adequacy: reject, don't enlarge.** `032` evaluates the geometric test and
   throws naming the measured ratio and admissible depth. Enlarging the deepest-level
   near set (179 → 389 classes) is `032a` work — `032` makes no change to route
   construction.

## Codebase map (already gathered — do not re-explore)

CUDA is a weakdep with **no `ext/` dir**: `src/translate_batched_cuda.jl` (5318 lines) is
`include`d at runtime by `load_cuda_radix_lifecycle!` (`src/FastMultipole.jl:153-171`),
CUDA fetched by UUID at `translate_batched_cuda.jl:7`. Host mirror of the same
lifecycle: `src/translate_batched_resident.jl` (always included). Containers:
`src/containers.jl:1595-1841`.

| thing | location |
|---|---|
| per-step device refresh | `translate_batched_cuda.jl:4165-4285` `update_cuda_radix_state!` |
| per-step device evaluate | `:3367-3380` `run_cuda_radix_lifecycle!`; pipeline `:3185-3209` |
| device cache build (all allocation) | `:3796-3986` `_radix_cache_device_build` |
| pack kernel (5-row truncation, row 4 zeroed) | `:922-936`; launcher `:938-949` |
| per-step fresh `CuArray` (gap 5) | `:894-900` / `:4005` |
| device B2M | `:1038-1067`, `:1073-1104` (leaf-node variant is the cache path); launcher `:1178-1198` |
| host B2M | `translate_batched_resident.jl:164-173` launcher, `:175-204` kernel |
| L2B evaluator (shared math) | `translate_batched_resident.jl:355-424` `_resident_local_eval_flat` |
| device L2B kernel | `translate_batched_cuda.jl:1618-1647` |
| device nearfield | `:1495-1545` (pairs), `:1551-1612` (symmetric), `:1106-1137` (O(N²) ref); launchers `:3122-3157` |
| `_cuda_fast_rsqrt` | `:1476-1482` |
| output alloc (4 rows) | `:3926` (device; sibling 5-row `source_bodies` at `:3925`), `translate_batched_resident.jl:992` (host; `:991` is the 5-row `source_bodies`) |
| hessian throws (3) | `src/fmm.jl:876-878`; `translate_batched_cuda.jl:1163-1164` (inside wrapper `:1159-1176`); `translate_batched_resident.jl:584-585` (inside `:573-600`) |
| output finalize / `buffer_to_target!` dispatch | `translate_batched_cuda.jl:3431-3476`; scatter kernel `:1139-1157` |
| `>= 5` row validators | `translate_batched_resident.jl:908-911`; `translate_batched_cuda.jl:959-961, 967-969, 976-978` |
| counters + contract | `containers.jl:1606-1619`; asserts `translate_batched_cuda.jl:3370-3371`, `:1241-1265` |
| options struct | `containers.jl:1621-1662` |
| cache / state / step counts | `containers.jl:1749-1781`, `:1682-1725`, `:1668-1674` |
| deprecated hooks (gap 7) | `src/compatibility.jl:53, 69`; call sites `translate_batched_cuda.jl:880-911, 3397-3414` |
| device-system reference pattern | `MATRIX_OPERATOR_REFACTOR/scripts/fm028_device_system.jl` |
| legacy hessian math to port/parity against | `src/evaluate_expansions.jl:46-399` (`evaluate_local`, LH-aware, 3×3 `SMatrix` returned at `:398`; wrapper `evaluate_local!` `:33-44` calls `set_hessian!` at `:42`) |
| FLOWVPM fused U+J pair math (reference only) | `../FLOWVPM.jl/ext/FLOWVPMCUDAExt.jl:172-234` |

**Additional sites the map above originally missed** (all must be threaded through the
cache-level row counts — Stage 1a's "magic-5" sweep covers these too):

- *Host pack paths (5-row, row 4 zeroed):* `translate_batched_resident.jl:91-104`
  `_host_radix_body_matrix(::DeviceRadixGrid,…)` (alloc `:93`, row 4 zeroed `:100`);
  `:107-110` — a **fifth** `>= 5` validator; `:1157-1171` `_pack_radix_source_bodies!`,
  the recurring host-resident pack path (row 4 zeroed `:1168`);
  `translate_batched_cuda.jl:955` one-shot `CuArray{TF}(undef, 5, n)`.
- *Extra per-step fresh device allocations (gap 5):* `translate_batched_cuda.jl:891`
  (HostResident `CUDA.CuArray(host_buffer)`) in addition to `:894-900`/`:4005`.
- *4-row output allocations beyond the two in the table:*
  `translate_batched_cuda.jl:3268`, `:3348` (one-shot `cuda_radix_state` variants);
  `translate_batched_resident.jl:511`; and `translate_batched_cuda.jl:3973` — the
  **pinned host download staging** `_pin_host_array(zeros(TF, 4, maxn))`. If the
  staging is not widened together with the output, the `copyto!` in
  `finalize_cuda_radix_output!` (`:3459`) silently mis-strides (see Risks).
- *Output-width source of truth:* `src/derivativesswitch.jl:78` `target_buffer_rows`,
  used at `translate_batched_cuda.jl:3445`, `translate_batched_resident.jl:1401`,
  `tree.jl:333`, `fmm.jl:917`.
- *Chi sizing / LH gating:* `translate_batched_cuda.jl:41` — when `LH=false`, `chi` is
  a **0×0 array**, so vortex B2M must be gated on `has_vector_potential`; a
  `Point{Vortex}` system with `LH=false` is a construction-time error, never a kernel
  OOB. Also `:2875` (`basis_dof_chi` term in the VRAM budget) and
  `containers.jl:802, 813, 820` (`basis_dof_chi` derivation).
- *Row-5 scalar-strength reads that vector strengths break:*
  `translate_batched_cuda.jl:1055, 1092, 1124, 1528, 1574, 1589`;
  `translate_batched_resident.jl:192, 331`. Both the row-count constant **and** the
  strength-row range `5:4+strength_dims` thread from the cache.

Existing traits already in `src/compatibility.jl`: `residency`, `data_per_body`,
`strength_dims`, `get_position`, `has_vector_potential`, `source_to_buffer!`,
`buffer_to_target!`. **New in 032**: `body_type`, `direct_kernel` — exactly the two
the spec defines. There is **no** `smoothing_row` trait (the spec reaches σ through
the packed extra-state rows); `sigma_row` is a `RegularizedVortex` constructor field,
not interface surface.

Placement rules (`START_HERE.md`): new types → `src/containers.jl`; translation ops →
`*_batched.jl`; CUDA → `*_cuda.jl` behind the runtime flag; nothing may break the
CPU/public API when CUDA is absent.

## Stage 1 — Generalized packing, vortex B2M + LH, 9-component hessian

**1a. Packed layout.** Replace the hard-coded 5 with the cache-level
`data_per_body` (max over systems, stored on `RadixFMMCache` at construction).
- Allocate `source_bodies = CUDA.zeros(TF, dpb, maxn)` (`:3925`) and the host mirror
  (`translate_batched_resident.jl:991`).
- `_cuda_pack_radix_body_kernel!` (`:922-936`): copy rows `1:dpb` from the source
  buffer, **including row 4 (radius), which is currently hard-zeroed**. Nothing reads
  row 4 today, so this is safe; it becomes live in Stage 2.
- Relax the five `>= 5` validators (four in the table + `translate_batched_resident.jl:107-110`)
  to `>= 4 + strength_dims`.
- Ship the canonical all-rows layout only (`031` decision (c)); the 5-row core +
  side-table variant is out of scope unless a bandwidth model shows ≥5%.

**1b. `body_type(sys)` trait + vortex B2M.** Default `Point{Source}()` (current
behavior). Add `Point{Vortex}` device and host B2M kernels writing **both** `phi` and
`chi`, mirroring `src/bodytomultipole.jl` for the same body type and the `008h` rule
(`chi` carried at `P+1`; sizing already honored by `basis_info` / `basis_dof_chi`).
Dispatch at compile time (kernel specialized on the body-type singleton), never a
runtime branch inside the loop. Note the launcher at `:1178-1198` already zeroes
`chi` (`fill!` of both `phi` and `chi` at `:1179-1180`; the kernels write `phi` only)
— keep that, then fill it.

**1c. 9-component hessian.**
- Extend `_resident_local_eval_flat` (`translate_batched_resident.jl:355-424`) with a
  `Val(HS)` variant returning potential + 3 gradient + 9 hessian, porting the
  second-derivative recurrences from `evaluate_expansions.jl:46-399` (which already
  covers the LH channel). Column-major 3×3 order matching `set_hessian!`
  (`compatibility.jl:713-737`).
- **Output rows are chosen at cache construction**, not always 13: add
  `hessian::Bool=false` to `RadixFMMCache`, store `n_output_rows = 4` or `13`, and
  specialize the L2B/direct/scatter kernels on `Val(HS)`. This is what keeps the
  shipped scalar path bandwidth- and atomic-identical (Stage 4 gate).
- Direct kernels (`:1495`, `:1551`, `:1106`) and the host mirrors emit hessian rows
  under `Val(HS)`; scatter kernel (`:1139-1157`) writes `hessian_range(switch)`.
- Remove the three throws; `fmm!` instead validates `hessian` against the cache flag.

**1d. Tests** — new `test/device_system_interface_test.jl` (host resident path, no
CUDA needed) + new CUDA test files **in `test/`** (not `test/cuda/` — CUDA tests live
in `test/` and are included inline at `test/runtests.jl:75-77`; `test/cuda/` is only a
gated re-run harness that re-includes them with `FASTMULTIPOLE_FORCE_CUDA_LOAD=1`).
Register new files in **both** `test/runtests.jl` and `test/cuda/runtests.jl` — note
several existing `cuda_*_test.jl` / `radix_fmm_*_test.jl` files are orphaned (included
by neither runner); don't repeat that. Coverage:
- gradient and hessian parity vs the legacy octree `fmm!` on `test/gravitational.jl`
  bodies, Float64 and Float32, **including `P=4`** (per the standing rule: `P=4` uses
  same-`P` parity checks, accuracy tolerances gate at `P=8`);
- vortex B2M → M2M → M2L → L2L → L2B end-to-end vs legacy LH path;
- packed-layout round trip for `data_per_body > 5`.

**Checkpoint 1**: report parity numbers and any accuracy surprises before Stage 2.

## Stage 2 — `direct_kernel` trait + `RegularizedVortex` + adequacy gate

**2a. Trait + functors** (`containers.jl` types, `compatibility.jl` trait):
`direct_kernel(sys) = SingularSource()` by default; `RegularizedVortex(; sigma_row,
rho_t=4.789)` for `gaussianerf` only. Both isbits, GPU-compilable, consumer-supplied
functors allowed. Pair kernels become generic over the functor and are specialized per
kernel type (one extra instantiation, no runtime branch).
Signature deviation from spec §5 to keep it GPU-safe: pass
`(dx, dy, dz, r2, source_bodies, j)` rather than a column view — document it.

**2b. `RegularizedVortex` math** — from `theory/kernel-splitting-nearfield.md`:
- §3 Horner series for `g(ρ)` and `h(ρ)=ρg'−3g` below `ρ=0.5` (6 terms Float32, 10
  Float64) — required for cancellation safety;
- §6.2 **erf-free** outer branch: `ḡ = e^{−ρ²/2}(Aρ + s(ρ))` with a degree-3
  least-squares `s` in `u=1/ρ²` (`data/kernel_splitting/cheap_gbar_fit.csv`), one
  hardware `exp` + four FMAs; `ρg' = Aρ³e^{−ρ²/2}` reuses the same exponential;
- **Mid-range `0.5 < ρ < 2` — measured switch-point decision.** The theory doc is
  internally inconsistent here (§3 validates the series only to `ρ ≤ 0.5` and says use
  `custom_erf` above; §6.2 and the 032 task file say the series is "already erf-free
  below ρ≈2"), and the §6.2 fit is only valid `ρ ≳ 2` — yet in the
  regularized-everywhere baseline this range is the most-populated. Resolve it
  empirically: validate the shipped series (6-term F32 / 10-term F64) on
  `0.5 < ρ < 2` against the 256-bit reference table from
  `scripts/validate_031a_kernel_split.jl`, both precisions. If it meets the §6.2 error
  budget there, ship series-below-2 + erf-free-above-2 (fully erf-free); otherwise the
  mandated benchmark below decides what covers `(0.5, ρ_t]` (`custom_erf` above 0.5,
  or more series terms). Record the measured switch point and per-branch max error in
  the Work Record. Also note there: `START_HERE.md`'s 032 row summary still says the
  baseline "borrows FLOWVPM's `custom_erf`" — superseded by the task file's later
  benchmark-and-choose amendment (no real conflict since `custom_erf` remains a
  candidate, but flag the stale wording rather than silently picking one);
- nine `J` components in the §1 order (transcription-checked against
  `../FLOWVPM.jl/ext/FLOWVPMCUDAExt.jl:172-234`); regularization uses the **source** σ.
- **Mandated benchmark** (`032` deliverable 4 note): erf-free form vs a self-contained
  `custom_erf` port, ship whichever is faster at equal measured accuracy. If erf-free
  wins, no FDLIBM code is copied into FastMultipole.
- Gate the symmetric nearfield kernel (`:1551`) to `SingularSource` only — it assumes
  the scalar same-source/target kernel.

**2c. Near-set adequacy gate.** Before the first evaluation on a regularized kernel,
evaluate `g_min · h_leaf > ρ_t · σ_max` (equivalently `2^ℓ < g_min·L_box/(ρ_t σ_max)`)
from the cache box, leaf `h`, and a device max-reduction over the packed σ row
(`g_min = √5` for the shipped `|o|²≤12` stencil, `1` for the classic). **Throw** on
failure, naming the measured ratio and the admissible `ℓ`. Do **not** implement the
`n/8^ℓ` form — it mis-ranks a clustered field by one to two levels. Reduction scratch
is construction-sized (zero steady-state allocation).

**2d. Functor-vs-hard-coded benchmark** (`031` decision (b)): measure
`SingularSource()` through the new dispatch against the shipped hard-coded kernel on
the `028` workload; report to the user if the abstraction costs measurably.

**Checkpoint 2**: report both benchmarks and the adequacy behavior.

## Stage 3 — First-class device-system API, persistent buffers, `recenter!`, docs

- **Gap 5 fix**: one persistent per-system device source buffer allocated at
  construction and reused each step (replaces the fresh `CuArray` allocations at
  `:891`, `:894-900`, `:4005`), matching the host-resident path at `:3901-3908`.
  Restores zero steady-state allocation for device-resident consumers.
- **Gap 7**: remove or hard-error the deprecated `source_system_to_device_buffer!` /
  `target_system_from_device_buffer!` hooks (`compatibility.jl:53, 69` and their call
  sites `:880-911`, `:3397-3414`).
- **`recenter!(cache, systems; bounds=nothing, padding=0.05)`** per spec §4: explicit
  call between evaluations, never implicit in `fmm!`; caller-supplied `bounds=(x_min,L)`
  is the deterministic fast path and is not padded; derived bounds use `get_position`
  (host) or a device reduction with construction-sized scratch copying only six
  extrema scalars; `ArgumentError` without mutation on empty/non-finite/nonpositive
  `L`/negative padding/changed system count/over-capacity; invalidates step-count
  prefixes until the next refresh; zero allocation after construction.
  Try normalized unit-cube internal coordinates only if non-invasive (positions,
  radius, σ scaled by `1/L`; outputs by `1/L`, `1/L²`, `1/L³`; strengths stay
  physical) — otherwise record it as an `035` lever and ship the rebuild fallback.
- **Promote the `fm028_device_system.jl` pattern into `src/` + docs**: a documented,
  exported device-system surface (`residency`, `data_per_body`, `strength_dims`,
  `body_type`, `direct_kernel`, `source_to_buffer!`, `buffer_to_target!`, capacity
  pre-sizing with `max_n_bodies`), a worked example, and an **external-code connection
  guide** covering both device-resident coupling and the transfer-based fallback with
  the `031` §8 decision framework (transfer ≈ `n·(data_per_body+output_rows)·sizeof(TF)/25 GB/s`;
  residency required at `n ≳ 1e5–1e6` with single-digit-ms step budgets).
  State **delivery semantics** explicitly: the framework always delivers the total
  influence of this evaluation; overwrite-vs-accumulate is the consumer's choice.
- **Gap 4 (spec §10): keep and document the `targets === sources` restriction.** The
  connection guide and API docs must state it explicitly (spec §9(d) defers
  `targets !== sources` and multi-GPU), and the lifecycle raises a validation error —
  never silent misbehavior — if a consumer passes distinct target systems.

**Checkpoint 3**: docs draft + API surface review.

## Stage 4 — H200 validation and no-regression gate

New scripts modeled on the `cuda_027_*` set:
`MATRIX_OPERATOR_REFACTOR/scripts/cuda_032_validation.jl` plus
`cuda_032_{submit,run,fetch}.sh`.
- Correctness on device: vortex + LH + hessian vs an on-device Float64 sampled direct
  reference (reuse the `fm028_direct_sample_reference` pattern — defined in
  `MATRIX_OPERATOR_REFACTOR/scripts/benchmark_028_feasibility.jl`, not the
  device-system script — extended to the regularized U/J kernel).
- **Counter contract**: `body_uploads = 0`, `expansion_host_copies = 0`,
  route/operator uploads construction-only, zero steady-state allocation, for a
  device-resident vortex consumer.
- **No-regression gate**: re-run the shipped `028`/`030` harness configuration
  (scalar, `P=4`, `ell` per shipped defaults, Float64 + FP16-WMMA/Float32) and compare
  per-stage timings before/after. The construction-time output-row choice (Stage 1c)
  is what should make this a null result.
- Before/after cost check of the new packing/hessian kernels on the `028` workload.

Cluster only — never run these locally (`BLAS/GPU benchmarks on HPC` standing rule).

## Bookkeeping at the end

- Complete the 032 task file's **Mandatory Reading Gate** before starting (START_HERE
  incl. the Integration Phase preamble; the 031 spec in full incl. sign-off record;
  `theory/kernel-splitting-nearfield.md` §§3, 5.1–5.2, 6.2; the placement rules) and
  record its completion in the Work Record.
- Write the Work Record into `MATRIX_OPERATOR_REFACTOR/032-impl-generalized-device-interface.md`
  (reading gate, decisions, benchmark numbers incl. the measured `g` switch point,
  deviations such as the `direct_kernel` signature and any `recenter!` fallback, and
  the stale START_HERE `custom_erf` wording), mark the `032` row `[x]` Done in
  `START_HERE.md`, leave `Approved` unticked for a separate clear-context agent.
- Commit FastMultipole changes in this repo (branch `matrix-ops`).
- Offer a notebook entry at the end of the row (do not write without approval).

## Verification

```bash
cd /Users/ryan/Dropbox/research/projects/tmp3/FastMultipole
# CPU-side parity, host-resident path (no CUDA):
julia --project=. --threads=4 test/device_system_interface_test.jl
julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'   # full suite, <=6 threads locally
```
- `031a` numerics: `julia MATRIX_OPERATOR_REFACTOR/scripts/validate_031a_kernel_split.jl`
  (stdlib-only, exit 0) — reuse its 256-bit reference table to unit-test the shipped
  `g`/`h` implementation in both precisions.
- Device: submit `cuda_032_submit.sh` on the H200 node, fetch, and check
  (a) sampled relative gradient RMS ≤ 1e-3 vs direct, (b) hessian/`J` RMS logged as a
  diagnostic, (c) counters, (d) `028`/`030` timings unchanged for the scalar path.

## Risks

- Hessian in `_resident_local_eval_flat` raises register pressure in the L2B kernel —
  watch occupancy; keep the `Val(HS)` specialization so the scalar kernel is untouched.
- Widening the packed matrix touches ~a dozen magic-`5`/`4`-row sites (see the map,
  including the "originally missed" block); introduce the cache-level row counts once
  and thread them, rather than a second magic constant.
- The regularized nearfield will be slower per pair than singular; `031a` §6.3 warns an
  *unbinned split* kernel is worse than no split — but `032` ships the
  regularized-everywhere baseline, so binning is `032a`'s problem, not this row's.
- The pinned host download staging (`translate_batched_cuda.jl:3973`) must be widened
  together with the output rows, or the `copyto!` in `finalize_cuda_radix_output!`
  (`:3459`) silently mis-strides — no error, wrong numbers.
- When `LH=false` the `chi` expansion array is 0×0 (`translate_batched_cuda.jl:41`):
  a `Point{Vortex}` system without `has_vector_potential` must fail at construction,
  never reach a chi-writing kernel.
