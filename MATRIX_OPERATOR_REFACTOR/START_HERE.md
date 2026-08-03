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
variants; at that stage fully dense per-offset M2L, partially folded hybrids
around `K_z`, alternate z-translation cache/scaling policies, and real-basis
execution were deferred options for later implementation/performance tasks.
Rows `023c` through `023f` now take up the precomputed-y and fully dense options;
`024` is the definitive four-strategy single-/multi-threaded CPU and GPU
end-to-end comparison after integration, fed into the `019a` final review.

Row `024a` was added by user request on `2026-07-14`: after `024`'s definitive
data exists, plot the accumulated benchmark evidence (crossovers, speedups,
stage breakdowns, storage, accuracy/cost tradeoffs) as human-readable figures
that communicate the project's successes and limitations and support tradeoff
decisions; `019a` cites those figures and therefore blocks on `024a`.

Row `024b` was added by user request on `2026-07-25`: measure end-user speedup
versus particle count at literature `P=4` (`expansion_order=3`), comparing the
legacy octree on 64 CPU threads and the resident H200 lifecycle (Float64
primary, Float32 extra) against legacy single-thread CPU. CPU MAC is fixed at
0.5 and leaf size is manually searched per case to step-5 resolution. The GPU
uses an independently reviewed, ell-scaled analytic stencil that matches the
legacy equal-cell discrete cutoff; every row records sampled-direct error.
Its scaling/accuracy figure extends the `024a` set as fig09, and `019a`
therefore also blocks on `024b`.

Rows `025`, `026`, and `027` were added by user direction on `2026-07-28`, after
the `024b` clear-context review found the radix far field is **single-level**:
`build_radix_routes!` stamps a constant `route_levels = ell` and routes both
endpoints through `leaf_to_node`, so every M2L is leaf-to-leaf, `M2M`/`L2L` run
but feed nothing, and M2L cost is `O(C^2)` in occupied cells. With direct cost
`~|near|*n*(n/C)` the optimum is `C ~ n^(2/3)` and the total is `O(n^(4/3))`;
the measured GPU exponent is `1.46`. That, not allocation sizing, is why `024b`'s
`n=1e6` point regressed to `94x` and why eight `ell=6/7` cases were
unconstructible — the route reservation is tight to within 4% of actual, so no
lazy-allocation change can help. `025` derives a rigid, source-major,
phase-indexed, level-invariant stencil (the `024b` `epsilon ∝ 2^ell` scaling
fixes one lattice cutoff at every depth, which is what makes it level-invariant)
plus an exact-once coverage proof and the level-scaling law that lets one
operator table serve all levels; `026` implements it on the host resident
lifecycle with windowed generation so the full pair list is never compiled; `027`
mirrors it on CUDA. The stencil is parameterized by a single near radius, so the
same construction yields both the `024b` `theta=0.5` stencil (`|o|^2 <= 12`;
179/1253) and the classic FMM stencil (`|o|^2 <= 3`, exactly `|o|_inf <= 1`;
27/189), and `026`/`027` measure the two against each other and against the
shipped flat path. The flat `ConstantPAnalyticStencil` is retained as a
selectable oracle and stays the default until measurement justifies switching —
it is genuinely cheaper below the hierarchical crossover (`C ~ 1430` occupied
cells at `theta=0.5`). `025` is a scoped derivation row confined to `theory/`,
`scripts/`, and `data/` artifacts and is listed in the Implementation Phase
table: it does **not** reopen the Theory Phase hard gate and does not re-block
any completed Implementation row.

Row `028` was added by user request on `2026-07-28`, staged after the planned
`025`–`027` interaction-list improvements. It is a rigorous feasibility study
for solving 1,000,000 particles in at most 0.01 seconds, on a single H200 at
literature `P=4` (`expansion_order=3`) with the `024b` sampled-direct error
methodology; Float32 qualifies for the verdict when its sampled error is
within the `P=4` Float64 truncation error (user decision `2026-07-28`). The
verdict timing boundary is the **per-time-step cost**: one full resident
evaluation plus everything a real step repeats — device-side convection
update and tree/route refresh — with no per-step body H2D/D2H; steady-state
evaluation-only and including-transfers boundaries are also measured and
reported but do not decide the verdict. The study must consolidate
the existing benchmark record and run targeted new benchmarks where evidence
is missing; produce an end-to-end and per-stage time breakdown; determine
whether each important stage is compute-bound, memory-bandwidth-bound,
transfer-bound, launch/latency-bound, or otherwise constrained; measure host
to device and device to host transfer costs separately from resident compute;
and assess whether particle state must remain permanently on the GPU across
convection and other time-stepping operations. Because the `024` strategy
defaults were measured under the flat leaf-only list, the study must evaluate
a per-level heterogeneous M2L strategy mix on the hierarchical path (e.g.
factored/concatenated at sparse coarse levels, fused dense at the leaf level)
using the per-level occupancy data from `026`/`027`. The study must identify
whether one algorithm stage bottlenecks the solve or time is broadly
distributed, quantify the remaining gap and uncertainty at all three
boundaries, and give evidence-backed, prioritized recommendations (with
expected gains and risks) for reaching the target; the optimization list
requires user approval before implementation begins. This
row must then implement the approved optimizations, verify
correctness and accuracy, and repeat the end-to-end and per-stage measurements.
Improving the production path toward the 0.01-second target—not merely
describing the gap—is a primary outcome. Continue the measured
profile/optimize/retest cycle until the target is reached or the remaining
barriers and next steps are rigorously quantified, with a user checkpoint at
the end of every cycle.

Rows `023a` and `023b` were added by user direction on `2026-07-15`, after `024`
planning found its premise stale: the `022`/`023` resident lifecycle shipped only
the materialized-lineage whole-slab dense strategies (`SharedRotationM2L`,
`ConcatenatedFixedZM2L`); `options.operator` is inert in the production
lifecycle (it affects only the retained flat-oracle test launchers and a
`DEBUG[]` assertion), and the CUDA path throws on `FactoredRotationM2L`. The
`024` end-to-end A/B is therefore unrealizable until a genuine factored resident
M2L exists. `023a` builds the host stage (per-degree batch-shared `U_n`/`V_n`
mode-matrix GEMMs applied per offset class — equivalently the grouped-GEMM M2L
lever deferred by `019`/`023` for `019a`), and `023b` mirrors it on the CUDA
device-resident lifecycle; `024` blocks on both.

Rows `023c` through `023f` extend that resident M2L comparison with two more
production strategies, split into host and CUDA stages. `023c`/`023d` precompute
the per-angle, per-degree matrices `M_n(theta) = U_n D_n(theta) V_n` while
leaving azimuth rotation, z translation, Lamb-Helmholtz coupling, and scatter
separate. `023e`/`023f` complete the existing `DenseTranslationM2L` placeholder
as one full coefficient-space matrix `S(Delta r)` per displacement class. Every
row `023a` through `023f` requires functional parity first and then a measured
profile/optimize/retest phase; host tasks require single- and multi-thread BLAS
data from a non-macOS host, and CUDA tasks require H200 before/after data. Task
`024` compares all four resident strategies and may select different defaults by
platform, expansion order, and batch/class-occupancy regime. The traditional
per-column reconstructed `Ts(theta)` path remains a correctness oracle, not a
resident performance candidate.

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
| [x] | [x] | `019-impl-operator-performance-tuning.md` | Tune completed operator paths and storage on the integrated end-to-end radix path after flat buffers, real-basis transform parity, and the radix driver exist. | All Theory rows, `008b`, `008c`, `017`, `018`, `021`, `022` |
| [x] | [x] | `019b-exploratory-smallp-fallback-and-channel-layout.md` | Exploratory benchmark plus required user-discussion decision on the small-`P`/tiny-batch fallback policy and the padded-vs-ragged `chi` layout, then implement the chosen policy. | All Theory rows, `008b`, `008c`, `015`, `019` |
| [x] | [x] | `023-impl-production-integration.md` | Route the production FMM through the validated, tuned operators behind basis-type dispatch / a `Cache` flag, with the legacy path as default fallback, to realize the end-user speedup. | All Theory rows, `008b`, `008c`, `016a`, `019`, `019b`, `021` |
| [x] | [x] | `023a-impl-factored-resident-m2l-host.md` | Implement a factored resident M2L stage on the host lifecycle: honor `options.operator` in `_launch_resident_m2l!` via a factored strategy applying per-degree batch-shared `U_n`/`V_n` mode-matrix GEMMs plus the shared fixed-`m` z-translation per offset class (grouped GEMM, `O(P^3)`/column), with in-place per-step refresh matching the `023` invariant contract. | All Theory rows, `008b`, `008c`, `013c`, `014`, `019`, `023` |
| [x] | [x] | `023b-impl-factored-resident-m2l-cuda.md` | Implement the CUDA device-resident factored M2L mirroring `023a`; remove the `FactoredRotationM2L` throws and preserve the `023` counter contract (`route_uploads`/`operator_uploads` constant after construction, `expansion_host_copies == 0`). | All Theory rows, `008b`, `008c`, `020a`, `022`, `023`, `023a` |
| [x] | [x] | `023c-impl-precomputed-y-m2l-host.md` | Add and optimize host `PrecomputedFactoredYM2L`: precompute per-angle/per-degree `U_n D_n(theta) V_n`, retain separate azimuth/z-translation/LH/scatter stages, and measure construction, memory, allocations, and single-/multi-thread non-macOS performance. | All Theory rows, `008b`, `008c`, `013c`, `014`, `019`, `023`, `023a` |
| [x] | [x] | `023d-impl-precomputed-y-m2l-cuda.md` | Port and optimize `PrecomputedFactoredYM2L` for the CUDA resident lifecycle with construction-only operator/route uploads, transfer-counter parity, Float32/Float64 and LH coverage, and H200 before/after measurements. | All Theory rows, `008b`, `008c`, `020a`, `022`, `023`, `023b`, `023c` |
| [x] | [x] | `023e-impl-dense-translation-m2l-host.md` | Complete and optimize host `DenseTranslationM2L` as one full `N_dof x N_dof` coefficient operator per displacement class, using class-batched large GEMMs with explicit memory limits and measured build/steady-state costs. | All Theory rows, `008b`, `008c`, `005`, `011`, `012`, `014`, `019`, `023`, `023a` |
| [x] | [x] | `023f-impl-dense-translation-m2l-cuda.md` | Port and optimize `DenseTranslationM2L` for CUDA with construction-time-only operator upload, device-memory controls, grouped/batched GEMM and gather/scatter tuning, lifecycle parity, and H200 before/after measurements. | All Theory rows, `008b`, `008c`, `020a`, `022`, `023`, `023b`, `023e` |
| [x] | [x] | `024-impl-operator-ab-benchmark.md` | Definitive construction, memory, steady-state M2L, lifecycle, and end-to-end comparison of four production-resident strategies: whole-slab concat, per-degree factored, precomputed-y, and full dense-translation. Select measured defaults by CPU/GPU, `P`, batch size, and class occupancy; retain reconstructed per-column `Ts(theta)` only as a correctness oracle. | All Theory rows, `008b`, `008c`, `013`, `013b`, `013c`, `022`, `023`, `023a`, `023b`, `023c`, `023d`, `023e`, `023f` |
| [x] | [x] | `024a-impl-benchmark-visualization.md` | Plot the project's benchmark evidence for human consumption: four-strategy resident M2L crossovers and speedups vs `P`/batch/occupancy/`n`, GPU lifecycle stage breakdowns and before/after tuning gains with the roofline gap (`019`/`022`), construction/storage/allocation comparisons, and accuracy-vs-cost surfaces — clearly communicating the project's successes, limitations, and measured tradeoff decisions for `019a`. | `024`, `019b`, `019`, `015`, `008c` |
| [x] | [x] | `024b-impl-cpu-gpu-scaling-benchmark.md` | User-requested (`2026-07-25`) end-user scaling study: speedup vs `n` (`1e3`–`1e6`) at literature `P=4` (`expansion_order=3`) of legacy CPU 64-thread and resident GPU lifecycle vs legacy CPU single-thread. Fix legacy MAC=0.5; manually search leaf size per `(n,threads)` to step-5 resolution; select best GPU `ell` using an independently reviewed ell-scaled compatible stencil; log shared sampled-direct error for every case and hard-gate Float64 error order. Float64 primary plus Float32 extra; speedup + accuracy fig09 in the `024a` set. | `022`, `023`, `023f`, `024` |
| [x] | [x] | `025-theory-hierarchical-rigid-m2l-stencil.md` | Derive the rigid source-major, phase-indexed, level-invariant M2L stencil enabling genuine multi-level node-to-node M2L on the radix path, with an exact-once body-pair coverage proof and the level-scaling law that lets one operator table serve every level. Parameterized by a single near radius so the same construction yields both the `024b` `theta=0.5` stencil (`\|o\|^2<=12`) and the classic FMM stencil (`\|o\|^2<=3`, i.e. `\|o\|_inf<=1`) for like-for-like comparison. Scoped derivation row: `theory/`, `scripts/`, `data/` only; does not reopen the Theory gate. | `008d`, `008f`, `008g`, `021`, `024b` |
| [x] | [x] | `026-impl-hierarchical-m2l-host.md` | Implement the `025` stencil on the host resident lifecycle: per-level node occupancy, source-major windowed route generation (full pair list never compiled), `(level, offset)` classes, and level-scaled operator tables across all four resident strategies. Both near radii selectable. Retains the flat `ConstantPAnalyticStencil` as oracle and default. Entry gate: prove `M2M`/`L2L` correct before they become load-bearing. | `025`, `021`, `023`, `023a`, `023c`, `023e` |
| [x] | [x] | `027-impl-hierarchical-m2l-cuda.md` | Mirror `026` on the CUDA device-resident lifecycle: device per-level occupancy scatter, source-major phase-masked flag/scan/compact per window, window-local class-partition assertions, per-offset dense operator tables via the `025` scaling law, and `023` counter/zero-allocation parity. H200 microbenchmarks including whether the `ell=6/7` grids `024b` could not construct now fit; selects the production default policy from measurement. No `024b` end-to-end re-run in this row. | `026`, `020a`, `022`, `023`, `023b`, `023d`, `023f` |
| [x] | [x] | `028-performance-feasibility-1m-in-10ms.md` | Measure, optimize, and retest toward an accurate 1,000,000-particle solve in `<= 0.01 s` at `P=4` on a single H200 (Float32 admissible within the `P=4` truncation error; verdict boundary = per-time-step resident cost including device convection and tree refresh, no per-step body transfers): consolidate existing evidence and fill material gaps; report end-to-end and per-stage timings at all three boundaries, compute/memory/transfer/latency bounds, GPU transfer and persistent-residency costs (including convection/time stepping), the per-level M2L strategy-mix lever, and bottleneck concentration; then, with user sign-off gating each optimize cycle, implement the highest-value justified production optimizations, verify correctness/accuracy, quantify realized gains, and iterate until the target is reached or the remaining feasibility gap and next steps are rigorously established. | `025`, `026`, `027`, `024b`, `024a`, `024`, `019`, `022`, `023` |
| [ ] | [ ] | `029-performance-high-score-1m-in-1ms.md` | Pursue the lowest reproducible complete resident-step latency for the fixed 1M-body, literature-P=4 workload beyond task 028's 9.591 ms result, with separate single-H200 and multi-H200 leaderboards and a high-score goal of `<= 1 ms`; retain the unchanged accuracy and recurring-cost gates, require independent reproduction, and stop at the goal or when evidence closes every credible material lever. | `028` |
| [ ] | [ ] | `019a-milestone-review-final-roadmap.md` | Final roadmap Milestone Review after Implementation tasks `017` through `029`, including the dynamic-`P` porting go/no-go, the platform/regime-specific resident M2L strategy recommendations from `024`, the hierarchical-vs-flat and `theta=0.5`-vs-classic stencil verdicts from `025`–`027`, the 1M-particle/10-ms feasibility conclusions from `028`, and the single-/multi-H200 high-score conclusions from `029`. | `008b`, `008c`, `017`, `018`, `019`, `019b`, `022`, `023`, `024`, `024a`, `024b`, `025`, `026`, `027`, `028`, `029` |

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
