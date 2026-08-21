# 023 Implementation Production Integration

## Objective

Route the production FMM through the validated, tuned matrix operators and radix
driver so the refactor delivers a realized end-user speedup, while keeping the legacy
octree + dynamic-`P` path as the default fallback (minimally invasive, backward
compatible).

## Dependencies

- `008b-implementation-replan.md` (parity-only first pass; this row is the deliberate
  replacement step it deferred)
- `008c-implementation-performance-baseline.md`
- `016a-milestone-review-impl-013-016.md`
- `019-impl-operator-performance-tuning.md`
- `019b-exploratory-smallp-fallback-and-channel-layout.md`
- `021-impl-constant-p-stencil-and-interaction-list.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Existing `fmm.jl` pipeline and the public `fmm!` / `tune_fmm` entry points

## Artifacts or Production Surface

- Production wiring in `fmm.jl` (and entry points) that dispatches the FMM through the
  new operator + radix path based on basis type (`AbstractOperatorBasis`) or a `Cache`
  flag.
- Tests confirming end-to-end accuracy against `direct!` and parity with the legacy
  path; a point-mass to `1/r` convergence check mirroring the Theory acceptance
  example.

## Deliverables

- Basis-type / flag dispatch selecting the new path; legacy path remains the default
  and is unchanged when the new path is not selected.
- The constant-`P` radix path wired end to end (clustering, interaction list, batched
  M2L, downward pass, evaluation). When the GPU path is selected, this row **dispatches
  into the `022` device-resident expansion lifecycle** — which owns body/expansion
  buffer allocation and on-device residency (B2M→M2M→M2L→L2L→L2B, download only per-body
  influence) — rather than allocating or transferring expansion buffers itself.
- No signature changes to the public API for existing users; the new path is opt-in.
- **Time-stepping fast path (user requirement, 2026-07-11, added by `019`).** The
  production target is recurring per-step cost: particle positions and strengths
  change every step, and the particle count varies under a known maximum. The `019`
  H200 data shows why this must be designed in: the device evaluation is 0.116 s at
  n=1e5/P=4, but the host interaction-list build (~0.35 s) and state/plan
  construction (~0.62 s) would dominate any loop that reconstructs the state per
  step. Requirements:
  1. Split the state into **step-invariant** and **step-varying** parts. Invariant
     (build once per `Cache` for fixed domain box, `ell`, `P`, and max `n`): the
     constant-`P` stencil offset sets and classification, all dense operators
     (stacked-y, z matrices), the per-offset-class geometry tables (functions of
     offset × cell width only, not occupancy), Lamb-Helmholtz rows, and all scratch
     buffers sized to the known maximum `n` / route count. Step-varying: the radix
     grid (Morton sort — already device-resident via `020a`), the `cell_at`
     occupancy map, route target/source/class arrays, direct pairs, and per-body
     buffers.
  2. The per-step API (via the existing `Cache` pattern) must accept
     "positions/strengths changed, count ≤ max" as the fast path: refresh only the
     step-varying parts and re-run the evaluation — never rebuild the invariant
     operators or reallocate buffers.
  3. For the GPU path this makes the `019`-deferred **device-side route generation**
     (compaction over (offset class, cell) from the device grid + `cell_at`, replacing
     the host `build_radix_interaction_list` + route upload) a required part of this
     row rather than an optional follow-up, so the whole per-step loop stays on
     device. Body upload/output download are not a concern (~4 MB / ~3 MB at n=1e5,
     ~0.1–0.5 ms over PCIe — negligible vs. the evaluation); host-origin per-step
     position/strength updates are acceptable.
  4. Verification must include a mock time-stepping loop: repeated steps with moved
     particles and varying `n` under the max, asserting per-step wall time ≈
     evaluation time (no rebuild-dominated steps), zero buffer reallocation after
     warmup, and accuracy parity on every step.
- **Factored-path physical-subspace guard (from `016b` watch item 2).** When the new
  path dispatches a `FactoredRotation*` operator, wire the debug-gated
  `_assert_factored_input_physical` check (defined in `src/rotate_batched.jl`) at the
  integration boundary on factored-path inputs. The factored operators reproduce
  production exactly only on the physical subspace (m=0 imaginary part == 0); every
  real expansion is physical, but this guard makes a hypothetical non-physical
  intermediate buffer fail loudly instead of silently diverging. It is `DEBUG[]`-gated
  and zero-cost when off, so it must not sit in the inner loop.

## Verification

Run `test/fmm_test.jl` through the new path and confirm it matches `direct!` to the
configured tolerance and matches the legacy path within parity tolerance; run the
point-mass to `1/r` convergence check. Rerun the full suite with threads
(`julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'`). Benchmark the integrated
path against the legacy path on the `008c` baseline machines and confirm the speedup.
Record commands and result summaries.

### Verification record (implementation session, 2026-07-14)

Implemented as an opt-in `RadixFMMCache` + `fmm!(system, cache::RadixFMMCache)`
dispatch (legacy octree path untouched and still the default). Key pieces:

- **Counts plumbing**: `RadixStepCounts` on `DeviceResidentRadixState`; every
  lifecycle launcher iterates count-bounded prefixes of capacity-sized arrays.
  `ResidentOperatorGroup` gained a `count::Ref{Int}` so per-level M2M/L2L groups
  refresh in place.
- **Invariant/step-varying split**: the fixed Morton domain (`x_min`, `h0`,
  `ell`) makes the per-level dense z-operators and the per-offset-class concat
  M2L geometry tables step-invariant; `update_radix_state!` refreshes grid,
  packed bodies, occupancy, routes (`build_radix_routes!`, offset-class-major,
  elementwise identical to the one-shot builder), tree edges, and group columns
  with zero reallocation (~23 KB/step of view/dispatch noise at n=600).
- **Host finalize**: `finalize_radix_output!` (hoisted scatter, CPU-loadable).
- **Device path** (`device=true`): device grid rebuild in the fixed domain,
  flag→scan→compact device route/direct generation (class-chunked, order-parity
  with the host builder), device group-edge refresh kernels; persistent device
  buffers; `route_uploads`/`operator_uploads` constant after construction,
  `body_uploads` +1/step. Coded locally; **GPU verification pending on HPC**.
- **`sizehint!` over-retention fixed**: the one-shot constant-P builder now
  counts before allocating exact-size batch vectors.

Local (macOS, Julia 1.12.5) results:

- `julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'` — full suite green
  (includes new `test/radix_fmm_integration_test.jl`,
  `test/radix_fmm_timestepping_test.jl`, and the self-gating
  `test/cuda_radix_integration_test.jl`).
- Radix `fmm!` vs `direct!`: monotone P-convergence (P=4/8/12 potential errors
  9.4e-6 / 2.8e-7 / 1.0e-8 at n=500, ell=3); legacy-parity and 2-system tuple
  cases pass; mock time stepping (10 steps, n varying in [N/2, N]) passes with
  array-identity preserved across steps and per-step `@allocated` < 512 KB.
- Cache lifecycle output is **bitwise identical** to a fresh one-shot
  `host_radix_state` build on the same fixed grid across moved-body steps.
- `MATRIX_OPERATOR_REFACTOR/scripts/benchmark_023_integration.jl` (CPU +
  `--gpu`) writes CSV/markdown records to
  `MATRIX_OPERATOR_REFACTOR/data/production_integration/`.

HPC session (BYU rc.byu.edu, H200 node `m13h-1-1`, Julia 1.11.7, CUDA 12.8 local
toolkit per `cuda-hpc-setup`; working tree rsync'd to `~/FastMultipole-023`, env
`~/fm023env`; slurm job 12751636):

- `FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia test/cuda/runtests.jl` — **288/288
  pass** on hardware: lifecycle gate 205, concat parity 37, and the new
  `cuda_radix_integration_test.jl` 46 (device route generation elementwise equal
  to the host builder incl. `route_class`; GPU `fmm!` vs `direct!`; host/device
  cache agreement < 1e-9; 5-step loop with `route_uploads`/`operator_uploads`
  constant and `body_uploads` +1/step).
- `benchmark_023_integration.jl --gpu` at n=1e5 / ell=4 / P=4 (H200):
  **per-step total 0.0859 s** = update 0.0071 s + lifecycle 0.0714 s +
  finalize 0.0059 s. Success criterion was ≤ 1.2 × the 0.116 s task-019
  evaluation (0.139 s) — met with margin; the recurring step is now *faster*
  than the 019 evaluation alone (022 concat M2L), and the one-shot ~0.35 s list
  build + ~0.62 s state build are gone from the loop (7 ms device update).
  n=1e4: per-step 0.0331 s. Counters after construct+warm+5 steps:
  body_uploads=7, route_uploads=2 (construction offsets only),
  operator_uploads=1, influence_downloads=6 (one per finalize).
- CPU reference (same script): radix per-step 24.6 s vs legacy 1.03 s at n=1e5 —
  consistent with the 019b decision that the legacy octree remains the CPU
  production default; the radix cache targets the GPU loop.
- Records: `data/production_integration/benchmark_023_m13h-1-1_20260714-233159.{csv,md}`
  (H200) and `benchmark_023_mecsrs-MacBook-Pro-188.local_20260714-231506.{csv,md}`
  (local Mac).
- Threaded suite on the cluster (`julia --project=. --threads=4 -e 'using Pkg;
  Pkg.test()'`, job 12751776, Julia 1.11.7): **passed**. Threaded suite also
  green locally on macOS (Julia 1.12.5).

### Clear-context review + fix session (2026-07-15)

A clear-context review (different agent, per `START_HERE.md` §6, with
user-directed emphasis on further cost reduction) confirmed all deliverables
against the production surface, tests, and benchmark records, and found one
contract violation plus avoidable per-step device overhead. With user approval
("fix all"), the following was implemented in this session:

- **F1 — fixed-box contract enforced on device (correctness).** The device path
  previously *clamped* out-of-box bodies into edge cells (silently wrong
  results) while the docs promised `ArgumentError`; the host path already threw.
  `_cuda_radix_keys_checked_kernel!` now raises a device flag checked once per
  `update_cuda_radix_state!`, which throws the host path's error before any
  persistent grid state is overwritten (the cache stays usable after a caught
  throw). CUDA test asserts throw + recovery.
- **F2 — per-step device update cost.** The update phase reallocated the entire
  device grid every step, ran (ell+1) redundant `CUDA.sort`s (ancestor keys of
  the sorted leaf keys are already sorted under a right shift), made ~10
  blocking scalar/mirror downloads, and left 2.4 MB/step of host metadata
  mirrors uncounted. Now: capacity-sized persistent grid + scratch refreshed in
  place (`_cuda_update_radix_grid_in_place!`; the state wrapper is built once),
  no per-level sorts, per-level counts batched into one download, pinned host
  mirrors/staging with linear-prefix copies, prefix-only output download in
  `finalize_cuda_radix_output!`, and a new `metadata_downloads` counter
  (asserted +3/step in tests). The `DeviceResidentRadixState` array-identity
  guarantee now holds on the **device** path too (CUDA-tested), matching the
  host path; the only per-step device allocation left is CUDA's pool-served
  sort scratch.
- **F4 — docs.** `RadixFMMCache`/section docstrings now state the
  zero-reallocation guarantee accurately for both paths.
- F3 (lifecycle-dominated cost: `019`-deferred L2B kernel and grouped-GEMM M2L;
  CPU radix slower than legacy per the `019b` decision) recorded as notes in
  `024` and `019a`, not implemented here.

Verification (fix session):

- Local (macOS, Julia 1.12.5): targeted radix/CUDA-gate tests 719/719; full
  threaded suite (`--threads=4`) **passed**.
- HPC (rc.byu.edu H200 node `m13h-2-1`, slurm job 12754018):
  `FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1 julia test/cuda/runtests.jl` — **305/305**
  (lifecycle gate 205, concat parity 37, task-023 integration 63 — up from 46
  with the new box-throw, device array-identity, and `metadata_downloads`
  assertions).
- `benchmark_023_integration.jl --gpu` (same job): n=1e5/ell=4/P=4 per-step
  **0.0848 s** = update 0.0049 s (was 0.0071) + lifecycle 0.0709 s + finalize
  0.0075 s; n=1e4 per-step **0.0311 s** with update 0.0043 s (was 0.0057).
  The ~30% update-phase cut confirms the fixed launch/alloc/sync overhead
  diagnosis; success criterion (≤0.139 s) still met with margin. Records:
  `data/production_integration/benchmark_023_m13h-2-1_20260715-070804.{csv,md}`.
- Note: the H200 job ran with a behaviorally identical predecessor of
  `_cuda_sortperm_into!` (a provably dead fallback branch was removed after
  submission; the executed dispatch path is unchanged). The remote tree was
  re-synced to match the final local tree.

Since this session changed production code, final clear-context approval must
be performed by another agent (per `START_HERE.md`).

## Approval Notes

**Approved 2026-07-15** by a clear-context agent (different from both the
implementing and fix-session agents), per `START_HERE.md` §6. Review scope:
`START_HERE.md`, this task file, the production surface, the three new test
files, and the benchmark records — with focus on the fix-session changes
(F1/F2/F4).

Findings against the §6 criteria:

1. **Objectives — consistent.** Opt-in `fmm!(system, cache::RadixFMMCache)`
   dispatch (`src/fmm.jl:868-899`) leaves the legacy octree methods untouched;
   the invariant/step-varying split matches the 2026-07-11 time-stepping
   requirement (`update_radix_state!`, `src/translate_batched_resident.jl`);
   device-side route generation is implemented (`_cuda_generate_radix_routes!`,
   flag→scan→compact, host-order parity CUDA-tested); the device step dispatches
   into the `022` lifecycle without allocating expansion buffers itself; the
   `016b` factored-path guard is wired `DEBUG[]`-gated once per lifecycle,
   outside the inner loop, and exercised with `DEBUG[]` on in tests.
2. **Correctness — confirmed.** F1 throws before any persistent grid state is
   overwritten (keys land in scratch; CUDA test asserts throw + recovery). The
   F2 no-per-level-sort argument (right shift of sorted Morton keys is
   monotone) is valid; `metadata_downloads` is asserted +3/step. The
   timestepping test proves zero reallocation by array identity, on both paths.
   Hardware: 305/305 CUDA tests (H200, job 12754018). Independent local
   spot-check by the approving agent: `radix_fmm_integration_test.jl` 24/24 and
   `radix_fmm_timestepping_test.jl` 694/694 (macOS, Julia 1.12.5).
3. **Performance — criterion met.** H200 per-step 0.0848 s at n=1e5/ell=4/P=4
   vs the ≤0.139 s target; the fix-session ~30% update-phase cut is reflected
   in `data/production_integration/benchmark_023_m13h-2-1_20260715-070804.*`,
   which matches this file's numbers. CPU radix slower than legacy is expected
   per the `019b` decision and is carried to `024`/`019a` (F3).
4. **Robustness / minimal invasiveness / readability — good.** New code follows
   the placement rules (`_batched`/`_cuda` files + `containers.jl`); exports
   are additive; capacity overruns fail loudly; F4 docstrings state the
   zero-reallocation contract accurately for both paths.

Minor non-blocking observations, recorded for the `019a` milestone review (no
fix cycle warranted):

- **Uncounted per-step scalar downloads (device path).** The out-of-box flag
  read and the route/direct prefix-total reads are small blocking downloads
  each step that no transfer counter tracks; `metadata_downloads` deliberately
  counts only the 3 perm/system/index mirrors. The counters exist to audit the
  `022` residency contract, which they do — but if a future row tightens the
  per-step sync budget, these are the untracked syncs to start from.
- **Benign same-value race.** `_cuda_radix_keys_checked_kernel!` writes
  `oob_flag[1] = Int32(1)` from every out-of-box thread without atomics. All
  writers store the same value, so this is safe; noted so a future edit does
  not turn it into a multi-value write without adding atomics.
