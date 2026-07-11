# Matrix Operator Refactor Index

## START HERE

This file is the only required first-read coordination document for Matrix
Operator Refactor work.

Routine task protocol:

1. Read this `START_HERE.md` first.
2. Select the first row, in table order, whose dependencies are complete and
   clear-context approved.
3. Open only the selected task file.
4. Do not read sibling task files.
5. Do not read `../MATRIX_OPERATOR_REFACTOR.md` unless the selected task
   explicitly requires it or the selected task is a Milestone Review.
6. For clear-context approval, read only this `START_HERE.md`, the completed task
   file, the artifacts or production files listed by that task, and the
   verification notes. Focus on the following in order of importance:
   1) consistent with the stated objectives; 2) correctness; 3) performance; 4) robustness (of code and tests); 5) minimally invasive approach; 6) human-readable. If you can think of any significant way to improve any of these, ask my permission and do it. Don't worry about small improvements. If you must change anything, another agent will need to do the clear-context approval.

No separate active-task pointer file should be added. The first unblocked row
in this index is the active task selection mechanism.

## Source Of Truth

Current user instructions are the first source of truth in all cases.
This index is the second source of truth for task order, status, blockers, Milestone
Reviews, and the hard phase gate. Task files may contain task-local
requirements only; they must not redefine global policy.

If this index, a task file, and `../MATRIX_OPERATOR_REFACTOR.md` disagree
during a Milestone Review, the review must stop and require a
coordination-document fix before further task work continues.

Clear-context approval must be performed by a different agent after the task is
finished. The completing agent must not approve its own work.

## Hard Phase Gate

All Theory Phase rows, including Theory Milestone Reviews and the Theory
Addendum rows `008d`, `008e`, `008f`, `008g`, and `008h`, must be marked both
`Done` and `Approved` before any Implementation Phase row starts.

Theory work may create or edit artifacts under:

- `MATRIX_OPERATOR_REFACTOR/theory/`
- `MATRIX_OPERATOR_REFACTOR/scripts/`
- `MATRIX_OPERATOR_REFACTOR/data/`

Theory work must not modify production FastMultipole code under `src/`.

For each concept, derivation precedes script, script precedes generated data,
and all related Theory artifacts precede approval. Implementation may touch
`src/` only after the full Theory gate is approved.

After the Theory gate is approved, `008b-implementation-replan.md` and
`008c-implementation-performance-baseline.md` must both be completed and
approved before task `009` or any later Implementation task starts.

## Theory Phase Acceptance Target

By the end of the Theory Phase, the approved artifacts must specify matrix
operators for both the compressed complex solid harmonic basis and the real
solid harmonic basis. For each basis, the theory must cover M2M, M2L, and L2L
operations using invariant matrices and z-axis rotations only; all non-z
rotation effects must be expressed through approved invariant axis-swap
matrices and fixed operator compositions.

Before task `008a` can approve the Theory gate, the Theory artifacts must also
include an example for a point mass of unit strength that:

1. obtains the source expansion;
2. applies M2M, M2L, and L2L through the approved matrix-operator chain;
3. evaluates the resulting expansion at a target point; and
4. demonstrates convergence to the analytic potential `1/r` as expansion order
   increases.

## Milestone Reviews

Milestone Reviews are blocking tasks. No later normal task may start until the
preceding Milestone Review is complete and approved.

Each Milestone Review requires the reviewing agent to:

1. Read all of `../MATRIX_OPERATOR_REFACTOR.md`.
2. Read this `START_HERE.md`.
3. Inspect completed task files and their listed artifacts since the previous
   milestone.
4. Confirm work still matches the background design, hard phase gate, and task
   ordering.
5. Try to find ways to improve contributions to the main goal of speeding up the
   FMM with GPU.
   Make notes to implement if you find them.
6. Record review notes in the Milestone Review task file and mark the row
   `Done`.
7. Get clear-context approval before downstream tasks continue.

## Theory Phase

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [x] | [x] | `001-theory-z-rotation-operators.md` | Derive z-rotation blocks, inverse blocks, and overwrite/accumulate semantics. | None |
| [x] | [x] | `002-theory-m2l-z-translation-scaling.md` | Derive M2L fixed-`m` z-translation matrices and stable distance scaling. | `001` |
| [x] | [x] | `003-theory-lamb-helmholtz-operator-form.md` | Derive multipole/local Lamb-Helmholtz operator form and channel coupling. | `001` |
| [x] | [x] | `004-theory-axis-swap-conventions.md` | Derive invariant axis-swap signs and active/passive rotation conventions. | `001` |
| [x] | [x] | `004a-milestone-review-theory-001-004.md` | Milestone Review for Theory tasks `001` through `004`. | `001`, `002`, `003`, `004` |
| [x] | [x] | `005-theory-full-m2l-composition.md` | Derive the complete M2L operator composition from approved component theory. | `004a`, `002`, `003`, `004` |
| [x] | [x] | `006-theory-m2m-l2l-extensions.md` | Extend the component theory to M2M and L2L pipelines. | `005` |
| [x] | [x] | `007-theory-coefficient-buffer-layout.md` | Specify coefficient-buffer layout, indexing, and typed view requirements. | `005`, `006` |
| [x] | [x] | `008-theory-real-solid-harmonic-transforms.md` | Derive complex-to-real and real-to-complex transform conventions and tests. | `001`, `007` |
| [x] | [x] | `008a-milestone-review-theory-005-008.md` | Milestone Review for Theory tasks `005` through `008`, immediately before Implementation can begin. | `005`, `006`, `007`, `008` |

## Theory Addendum

These Theory tasks were added by the `008b` Implementation Re-Plan after the
`008a` Theory milestone (and expanded by the `2026-06-13` re-plan addendum
recorded in `008b`, which added `008f`). They are full Theory Phase rows: each
requires clear-context approval, and—like every other Theory row—each blocks
every Implementation task under the Hard Phase Gate. They become unblocked once
`008b` is complete and approved.

The `008f` row is listed before `008d` because `008d` depends on it; the
f-before-d ordering is intentional and dependency-driven.

The `008g` row was added later by user request on `2026-06-13`. It derives the
radix-path interaction-list construction and depends on `008d` (the stencil) and
`008f` (the cell geometry), so it cannot be selected until `008d` is complete and
approved.

The `008h` row was added by user request on `2026-06-15`. It derives the
Lamb-Helmholtz channel accuracy-order relationship (the hypothesis that the `χ`
channel must be carried at `P + 1` while `φ` stays at `P`). It depends on the
Lamb-Helmholtz operator form (`003`), the constant-`P` stencil (`008d`), and the
real-basis kernel derivatives (`008e`); it is independent of `008g` and is listed
last.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [x] | [x] | `008f-theory-radix-sort-clustering.md` | Derive a radix-sort (Morton/Z-order) clustering for large-`N`/GPU producing uniform-grid cells for translation-invariant M2L stencils. | `008b`, `007` |
| [x] | [x] | `008d-theory-dynamic-p-error-m2l-integration.md` | Specify constant-`P` error handling: legacy octree keeps dynamic-`P`; the radix-sort path moves error control into a conservative translation-invariant interaction-list stencil. | `008b`, `008f`, `002`, `005`, `007` |
| [x] | [x] | `008e-theory-real-basis-kernel-derivatives.md` | Derive real-basis evaluation of potential, gradient, and gradient Jacobian (Hessian) for the `1/r` kernel. | `008b`, `007`, `008` |
| [x] | [x] | `008g-theory-radix-interaction-list.md` | Derive the radix-path M2L interaction-list construction: apply the `008d` constant-`P` stencil over `008f` uniform-grid cells, batch M2L by integer offset class, and route the near/self complement to direct. Verify complete, non-double-counted n-body coverage on a test grid. | `008d`, `008f`, `008b`, `007`, `005` |
| [x] | [x] | `008h-theory-lamb-helmholtz-accuracy-order.md` | Derive the Lamb-Helmholtz channel accuracy-order rule (hypothesis: `χ` carried at `P + 1` while `φ` stays at `P`), with stencil/buffer/operator-sizing consequences and a convergence-slope verification. | `003`, `005`, `008d`, `008e`, `007`, `008b` |

## Implementation Re-Plan Gate

These required planning and benchmark tasks are not Implementation tasks. They
must be completed and approved after the Theory gate is approved and before any
production code work starts.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [x] | [x] | `008b-implementation-replan.md` | User-in-the-loop Implementation re-plan after Theory is approved and before production code work begins. | All Theory rows, `008a` |
| [x] | [x] | `008c-implementation-performance-baseline.md` | Pre-implementation performance, allocation/storage baseline, and design gate before production code work begins. | All Theory rows, `008a`, `008b` |

## Implementation Code Placement

Implementation tasks that touch production `src/` must follow these placement
rules:

1. New struct and type definitions go in `src/containers.jl` alongside the
   existing core data structures. Do not create standalone type files.
2. New translation operators go in a new file named after the closest existing
   source file with `_batched` inserted before `.jl` (for example
   `translate.jl` -> `translate_batched.jl`, `rotate.jl` -> `rotate_batched.jl`).
3. New radix-path driver code (uniform-grid clustering, Morton sort, constant-`P`
   stencil, and the radix interaction list) goes in `tree_batched.jl` and
   `interaction_list_batched.jl`, following the same `_batched` naming rule.
4. GPU code must live behind a package extension or runtime flag in a
   `*_cuda.jl` file (for example `translate_batched_cuda.jl`) so the CPU path and
   the existing public API are unaffected when CUDA is absent.

## Implementation Phase

Every Theory Phase row above, including the Theory Addendum rows `008d`,
`008e`, `008f`, `008g`, and `008h`, is a blocker for every row in this section. Do
not start any Implementation task until all Theory rows are marked both `Done` and
`Approved`. Do not start any Implementation task until
`008b-implementation-replan.md` and
`008c-implementation-performance-baseline.md` are also marked both `Done` and
`Approved`.

Implementation task files must list relevant approved Theory dependencies by
filename. This does not narrow the gate: every Theory task and every preceding
Milestone Review blocks every Implementation task.

The `2026-06-18` roadmap review (recorded in
`MATRIX_OPERATOR_REFACTOR/roadmap-review-2026-06-18.md`) added rows `013a`, `020`,
`021`, `022`, `019b`, and `023`, and sharpened the summaries of several existing
rows. The additions close three confirmed scope gaps: the radix-path driver that the
new operators run on (`020`, `021`), a GPU implementation home for the `008c`
device-residency target (`022`), and the production-integration row that turns the
validated operators into realized end-user speedup (`023`). Row `019b` stages the
small-`P`/tiny-batch fallback and the padded-vs-ragged `chi` layout as an exploratory
benchmark plus a required user-discussion decision before implementation.

The `2026-06-19` M2L operator roadmap update recognizes two near-term M2L
strategies and elects to build and benchmark **both**. `013` (done) supplies the
materialized arbitrary y-rotation building blocks: cached invariant `H(pi/2)` /
`S_pos` / `S_neg` data, `build_Ts_from_S!`, and production-parity wrappers that
apply a reconstructed `Ts(theta)`. `013b` supplies the fixed `90 degree`
y-axis swap primitives (`R_y(+pi/2)` / `R_y(-pi/2)`) only; `013c` (done, `2026-06-23`)
then realizes the explicit factored `Z_phi -> S -> Z_theta -> S_inv` alignment and
return alignment in the `T_n(theta) = S_n Z_n(theta) S_n^{-1}` form, preserving
FastMultipole's sign and extra-`pi` convention. **Roadmap amendment (user-directed
`2026-06-23`, "Plain-H"):** a spike proved the literal chain — composing `013b`'s
ζ-dressed `T_y_pos90/neg90` through the production y kernel with a `Z_theta` — is
structurally incapable of reproducing `R_y(theta)` (`(ζS)Z(ζS⁻¹) ≠ ζ(SZS⁻¹)`; ζ does
not commute through the swap). The shipped `S`/`S_inv` are therefore the **plain** fixed
per-degree swap matrices `V_n`/`U_n` with ζ/η dressing absorbed: every angular Fourier
component of the production y-operator is rank 1, so `Y_n(theta) = U_n diag(e^{i nu
theta}) V_n` with fixed, batch-shared `U_n,V_n` and a cheap `e^{i nu theta}` middle —
`O(P^3)`/column, genuinely distinct from the materialized `013` path, and **not** the
ζ-dressed `013b` primitives (those remain only for the `013`/materialized parity
surface). See `013c` Revised Implementation Notes and `theory/axis-swap-conventions.md`.
The shared `011` z-axis M2L block and `012` Lamb-Helmholtz coupling are
applied by `014`'s composition, not by `013b`/`013c`. The folded no-`Ts` y-rotation
path is not a product or benchmark variant; it may appear only as temporary debug
scaffolding while validating explicit stages. `014` composes swappable
`MaterializedYRotationM2L` and `FactoredRotationM2L` interfaces over shared `011`
z-axis M2L blocks, `012` Lamb-Helmholtz coupling, common cache/scratch conventions,
and common production-parity tests. `015` compares only those two near-term
variants; fully dense per-offset M2L, partially folded hybrids around `K_z`,
alternate z-translation cache/scaling policies, and real-basis execution are
deferred options for later final implementation/performance tasks. `024` remains
the definitive single-/multi-threaded CPU and GPU end-to-end comparison after
integration, fed into the `019a` final review.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [x] | [x] | `009-impl-basis-and-operator-cache-types.md` | Define basis and operator-cache types without changing production translation calls. | All Theory rows, `008b`, `008c` |
| [x] | [x] | `010-impl-z-rotation-operators.md` | Implement explicit z-rotation operators and parity tests. | All Theory rows, `008b`, `008c`, `009` |
| [x] | [x] | `011-impl-m2l-z-translation-blocks.md` | Implement fixed-`m` M2L z-translation blocks with approved scaling; size all blocks only through the `009` order accessors and do not bake in a small-`P` fallback (deferred to `019b`). | All Theory rows, `008b`, `008c`, `009` |
| [x] | [x] | `012-impl-lamb-helmholtz-operators.md` | Implement Lamb-Helmholtz transform operators and parity tests. | All Theory rows, `008b`, `008c`, `009` |
| [x] | [x] | `012a-milestone-review-impl-009-012.md` | Milestone Review for Implementation tasks `009` through `012`. | `008b`, `008c`, `009`, `010`, `011`, `012` |
| [x] | [x] | `013-impl-axis-swap-operators.md` | **Materialized y-rotation building blocks.** Precompute invariant `H(pi/2)` / `S_pos` / `S_neg` data into the operator-invariant cache, rebuild per-angle Wigner `Ts(theta)` with `build_Ts_from_S!`, and apply through production-parity y-rotation wrappers; supplies the materialized-`Ts` M2L variant used by `014`/`015`. | All Theory rows, `008b`, `008c`, `012a`, `010` |
| [x] | [x] | `013a-spike-m2l-batching-and-dynamic-p-feasibility.md` | Design spike: assess the M2L batching candidates (using `008c` prototype data) and the feasibility of porting dynamic-`P` machinery onto the new operators, to inform the `014` composition API and `017` storage. | All Theory rows, `008b`, `008c`, `010`, `011`, `012`, `013` |
| [x] | [x] | `013b-impl-fixed-y-swap-primitives.md` | **Fixed y-swap primitives.** Cache fixed `R_y(+pi/2)` / `R_y(-pi/2)` matrices and expose internal multipole/local fixed swap and inverse-swap stages using the production-parity y-apply kernels; assembly of `Z_phi -> S -> Z_theta -> S_inv` is split to `013c`. | All Theory rows, `008b`, `008c`, `012a`, `010`, `013`, `013a` |
| [x] | [x] | `013c-impl-factored-rotation-alignment.md` | **Assembled factored rotation alignment.** Compose separately callable alignment and return-alignment stages from `010` z rotations plus fixed per-degree y-swap matrices, in the `T_n=S_n Z_n(theta) S_n^{-1}` form, preserving FastMultipole's sign / extra-`pi` convention and shaping the API for global batched GEMM. **Plain-H amendment (`2026-06-23`):** the swap `S`/`S_inv` are the plain fixed mode matrices `V_n`/`U_n` (rank-1 Fourier modes of the production y-operator, ζ/η dressing absorbed), not the ζ-dressed `013b` `T_y_*90` primitives, which a spike proved cannot reproduce `R_y(theta)` when composed with `Z_theta`. | All Theory rows, `008b`, `008c`, `012a`, `010`, `013`, `013a`, `013b` |
| [x] | [x] | `014-impl-full-m2l-operator-pipeline.md` | Compose swappable whole-M2L interfaces: `MaterializedYRotationM2L` using `Ts(theta)` from `013`, and `FactoredRotationM2L` using explicit `Z/S/Z/S` stages from `013c`; both share `011` z-axis M2L blocks, `012` Lamb-Helmholtz coupling, common cache/scratch conventions, and common production-parity tests. | All Theory rows, `008b`, `008c`, `013`, `013a`, `013b`, `013c`, `010`, `011`, `012` |
| [x] | [x] | `015-impl-axis-swap-benchmarks.md` | Compare only the two near-term `014` variants: materialized `Ts(theta)` M2L and explicit factored `Z/S/Z/S` M2L; record deferred options for later tasks (fully dense per-offset M2L, partially folded hybrids around `K_z`, alternate z-translation cache/scaling policies, and real-basis execution). `024` remains the definitive end-to-end comparison after integration. | All Theory rows, `008b`, `008c`, `014`, `013b`, `013c` |
| [x] | [x] | `016-impl-m2m-and-l2l-operator-pipelines.md` | Extend the operator structure to M2M and L2L. | All Theory rows, `008b`, `008c`, `014`, `015` |
| [x] | [x] | `016a-milestone-review-impl-013-016.md` | Milestone Review for Implementation tasks `013` through `016` (including `013b` and `013c`). | `008b`, `008c`, `013`, `013b`, `013c`, `014`, `015`, `016` |
| [x] | [x] | `016b-followup-016a-watch-items.md` | Resolve or formally defer the non-blocking watch items raised by the `016a` review (factored physical-subspace invariant + `023` assertion, protect the `013c` anti-collapse test, background-doc pointer, minor cache/env/Float32 notes). The batched-GEMM speedup item is **deferred** (likely a macOS/BLAS artifact, user-directed `2026-06-24`) and carried into `024`/`019a` as a cross-machine check. Additive only; does not change the approved `013`–`016` deliverables. | `016a` |
| [x] | [x] | `017-impl-flat-coefficient-buffers.md` | Introduce flat coefficient buffers and typed views after the operator API is stable; prune the dead `chi` channel for `Val(false)` and keep the padded-vs-ragged `chi` layout swappable behind the `009` accessors (decided in `019b`). | All Theory rows, `008b`, `008c`, `016a`, `016` |
| [x] | [x] | `018-impl-real-solid-harmonic-basis.md` | Add real-basis transforms with parity tests; keep native real-basis operator execution as future work after flat buffers and operator APIs stabilize. | All Theory rows, `008b`, `008c`, `017` |
| [x] | [x] | `020-impl-radix-grid-clustering.md` | Implement the radix uniform-grid clustering, Morton sort, and cell geometry (`008f`) that the new operators run on. | All Theory rows, `008b`, `008c`, `009` |
| [x] | [x] | `021-impl-constant-p-stencil-and-interaction-list.md` | Implement the constant-`P` translation-invariant stencil (`008d`) and the per-offset-class radix M2L interaction list with direct complement (`008g`). | All Theory rows, `008b`, `008c`, `014`, `020` |
| [x] | [x] | `020a-impl-device-radix-grid-construction.md` | Implement CUDA/device-resident radix construction for the GPU lifecycle: device Morton-key generation, device sort, leaf ranges, occupied ancestor nodes, and parent/child metadata. CPU `020` remains the approved reference path; this row prevents item `022` from depending on host-built radix metadata. | All Theory rows, `008b`, `008c`, `009`, `020` |
| [x] | [x] | `022-impl-gpu-device-resident-m2l.md` | Implement a device-resident GPU expansion lifecycle (upload bodies once; keep all expansion buffers on device across B2M→M2M→M2L→L2L→L2B; download only per-body influence, or nothing when bodies originate on device) behind a CUDA extension/flag. This row owns device buffer allocation/residency; generalizes the `008c` horizontal-pass residency target to the whole evaluation, per the break-even target and the `015` two-variant comparison. | All Theory rows, `008b`, `008c`, `015`, `016`, `020a`, `021` |
| [x] | [ ] | `019-impl-operator-performance-tuning.md` | Tune completed operator paths and storage on the integrated end-to-end radix path after flat buffers, real-basis transform parity, and the radix driver exist. | All Theory rows, `008b`, `008c`, `017`, `018`, `021`, `022` |
| [ ] | [ ] | `019b-exploratory-smallp-fallback-and-channel-layout.md` | Exploratory benchmark plus required user-discussion decision on the small-`P`/tiny-batch fallback policy and the padded-vs-ragged `chi` layout, then implement the chosen policy. | All Theory rows, `008b`, `008c`, `015`, `019` |
| [ ] | [ ] | `023-impl-production-integration.md` | Route the production FMM through the validated, tuned operators behind basis-type dispatch / a `Cache` flag, with the legacy path as default fallback, to realize the end-user speedup. | All Theory rows, `008b`, `008c`, `016a`, `019`, `019b`, `021` |
| [ ] | [ ] | `024-impl-operator-ab-benchmark.md` | Definitive end-to-end benchmark of `MaterializedYRotationM2L` (`013`, reconstructed `Ts(theta)`) vs `FactoredRotationM2L` (`013c`, explicit `Z/S/Z/S` stages using fixed per-degree mode matrices `U_n`/`V_n`, ζ/η-dressing absorbed — not the ζ-dressed `013b` primitives) through the integrated FMM (`023`): single- and multi-threaded CPU and GPU. Record the winning operator per platform and feed the recommendation into `019a`. | All Theory rows, `008b`, `008c`, `013`, `013b`, `013c`, `022`, `023` |
| [ ] | [ ] | `019a-milestone-review-final-roadmap.md` | Final roadmap Milestone Review after Implementation tasks `017` through `024`, including the dynamic-`P` porting go/no-go and the `MaterializedYRotationM2L` vs `FactoredRotationM2L` recommendation from `024`. | `008b`, `008c`, `017`, `018`, `019`, `019b`, `022`, `023`, `024` |

## Future Dispatch Cleanup Notes

For new GPU/device-path work, prefer dispatch on source, destination, and policy
objects over boolean residency flags. Item `022` should use destination-driven
output finalization: resident CUDA evaluation writes `state.output` on device, and
copy/writeback dispatch decides whether the destination is a host array, device
array, or device-native target system.

Cleanup candidates for later rows or maintenance work:

- Replace `CUDARadixLifecycleOptions.allow_host_bodies` with source-buffer
  dispatch or explicit source-residency capability checks.
- If the legacy `nearfield_device::Bool` path is revived, replace it with a
  nearfield execution policy tag.
- Consider target/source tree role tags for the legacy `target::Bool` tree role
  argument in a later tree refactor.
- Consider route-selection policy tags for radix `farfield`, `nearfield`, and
  `self_induced` include flags in future radix-only APIs.
