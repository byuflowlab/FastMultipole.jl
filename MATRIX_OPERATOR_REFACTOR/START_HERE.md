# Matrix Operator Refactor Index

## START HERE

This file is the only required first-read coordination document for Matrix
Operator Refactor work.

Routine task protocol:

1. Read this `START_HERE.md` first.
2. Select the first row, in table order, whose dependencies are complete and
   clear-context approved. If a phase explicitly declares parallel lanes,
   select the first eligible row within any lane; table order does not
   serialize independent lanes.
3. Open only the selected task file.
4. Do not read sibling task files.
5. Do not read `../MATRIX_OPERATOR_REFACTOR.md` unless the selected task
   explicitly requires it or the selected task is a Milestone Review.
6. For clear-context approval, read only this `START_HERE.md`, the completed task
   file, the artifacts or production files listed by that task, and the
   verification notes. Focus on the following in order of importance:
   1) consistent with the stated objectives; 2) correctness; 3) performance; 4) robustness (of code and tests); 5) minimally invasive approach; 6) human-readable. If you can think of any significant way to improve any of these, ask my permission and do it. Don't worry about small improvements. If you must change anything, another agent will need to do the clear-context approval.

No separate active-task pointer file should be added. The first unblocked row
in the index—or the first unblocked row in each explicitly declared parallel
lane—is the active task selection mechanism.

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

Row `030` was added by user request on `2026-08-03`, after the `019a` final
review was approved. It measures the per-time-step verdict-boundary cost
versus `n` (`1e3`–`1e6`, the `024b` grid) at the `028` shipped-default GPU
settings, as three fixed-depth series `ell = 3/4/5` (user decision: fixed
`ell` across the sweep, not per-`n` best; the bracket was revised from
`4/5/6` to `3/4/5` on `2026-08-03` before any sweep case ran, because the
optimal depth tracks `n` downward — `028` measured `ell = 4` best at
`n = 2e5` and `024b` chose `ell = 2/3` below `n = 1e5`) in both the
FP16-WMMA/Float32
winner configuration and Float64, and then recommends a per-`n` optimization
(including retuning the level-radius geometry to a per-`n` accuracy target —
the "fixed error" lever) with estimated savings, modeled from the per-stage
data and validated by spot-check H200 runs at 2–3 representative `n`.
Completing `030` does not reopen `019a`; its results are recorded as an
addendum note in `019a`, mirroring the `029` convention. `030` does not
depend on the deferred `029` and does not resume it.

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
| [x] | [x] | `029-performance-high-score-1m-in-1ms.md` | **Closed by user direction `2026-08-11`** (stop rule: evidence closed every credible in-scope lever; single-H200 record 4.546/4.657 ms; mirrored-tree multi-GPU falsified structurally at 58.3% efficiency with comm+orch 0.190 ms passing; successor partitioned-tree path staged as rows `043`–`045`). **Resumed by user direction `2026-08-05`** (deferred `2026-08-03`–`2026-08-05`; still does not block `019a` — results land there as an addendum note). Pursue the lowest reproducible complete resident-step latency for the fixed 1M-body, literature-P=4 workload beyond task 028's 9.591 ms result, with separate single-H200 and multi-H200 leaderboards and a high-score goal of `<= 1 ms`; retain the unchanged accuracy and recurring-cost gates, require independent reproduction, and stop at the goal or when evidence closes every credible material lever. If completed after `019a`, its results are recorded as an addendum review note in `019a`, not a reopened review. | `028` (no longer blocks `019a`) |
| [x] | [x] | `019a-milestone-review-final-roadmap.md` | Final roadmap Milestone Review after Implementation tasks `017` through `028` (`029` deferred by user direction `2026-08-03`), including the dynamic-`P` porting go/no-go, the platform/regime-specific resident M2L strategy recommendations from `024`, the hierarchical-vs-flat and `theta=0.5`-vs-classic stencil verdicts from `025`–`027`, the 1M-particle/10-ms feasibility conclusions from `028`, and a deferral note for the `029` high-score campaign. | `008b`, `008c`, `017`, `018`, `019`, `019b`, `022`, `023`, `024`, `024a`, `024b`, `025`, `026`, `027`, `028` |
| [x] | [x] | `030-benchmark-cost-vs-n-fixed-ell.md` | Measure per-time-step verdict-boundary cost vs `n` (`1e3`–`1e6`) at the `028` shipped defaults as three fixed-`ell` series (`3/4/5`) in FP16-WMMA/Float32 and Float64, using the `024b` checksummed references; plot as fig10 in the `024a` set; then recommend per-`n` optimizations (including fixed-error geometry retuning) with modeled savings validated by H200 spot-checks at 2–3 representative `n`. Benchmark/analysis row: `scripts/`, `data/`, figures only; no production `src/` changes. | `028`, `019a` |

## Integration Phase

This phase was added by user request on `2026-08-04`, after the `019a` final
roadmap Milestone Review was approved. Now that the matrix operators deliver
decent GPU speedups, the goal is to streamline the API so external codes can
connect easily. FastMultipole provides a **generalizable** device-resident
system interface; `../FLOWVPM.jl` (branch `gpu-full`) is optimized as its first
consumer, and updating FLOWVPM to run its FMM velocity/Jacobian solve on the
GPU as fast as possible is the main objective of the phase.

Design intent: a consumer's particle states are stored and maintained on the
GPU so no per-step host/device body transfer is required. Alternatively, if
transfers prove cheap for a given consumer, this phase publishes measured
guidelines for the transfer-based coupling instead. Memory allocation
procedures are key: there must be no per-timestep reallocation when a maximum
particle count is known a priori — the `RadixFMMCache` capacity contract
(`max_n_bodies` plus derived capacities, zero per-step allocation) is the
existing mechanism, and this phase generalizes and documents it rather than
reinventing it. The storage layout of positions, strengths, extra states (such
as the smoothing radius for finite-core models), potential, gradient, and
hessian is a first-class design topic. The hessian is stored as **9
components** (user decision `2026-08-04`): the 6-component symmetric option is
skipped because the velocity gradient contributed by the Lamb-Helmholtz /
vector-potential channel is not symmetric in general, and FLOWVPM requires
Lamb-Helmholtz.

Repository policy (user decision `2026-08-04`): FLOWVPM edits are committed on
the `gpu-full` branch of `../FLOWVPM.jl`; FastMultipole edits are committed in
this repository. All task files and coordination stay in
`MATRIX_OPERATOR_REFACTOR/` and reference `../FLOWVPM.jl` paths. **Any agent
working on FLOWVPM must read `../FLOWVPM.jl/CLAUDE.md` in full before touching
that repository.** FLOWVPM public API changes (exported names, keyword
defaults) can break the downstream consumers FLOWUnsteady and VortexLattice —
the CPU path and public surface must be preserved.

Benchmark ground rules: the baselines are single-thread and 64-thread CPU runs
of FLOWVPM **at the `gpu-full` branch-point commit `e2bd487`** (2026-05-16,
v4.0.4), profiled to show what costs what and where the bottlenecks lie, so the
phase ends with clear documentation of every speedup achieved relative to those
baselines. Two test cases are used throughout: (a) a random vortex particle
field in a unit cube with average smoothing radius chosen to give an overlap
factor of 2; (b) a **helical wake cylinder** — a solid cylinder of length
5 diameters, particles uniform in its volume at overlap 2, strengths tangent to
a helix of pitch `p = D` with tip-weighted magnitude `|Γ| ∝ r` (user direction
`2026-08-05`, replacing the vortex ring; rationale and full definition in
`033`'s wake amendment). The ring was abandoned because it is locally thin
(near sets half-empty, breaking the `031a` §6 occupancy model) and has no single
overlap factor (`σ/rl = 3` radially, `≈1.58` azimuthally); the wake is locally
dense and isotropic with one `β`, while keeping the coherent aligned strengths
the cube lacks. The wake still fills only 3.14% of its bounding cube, which is
retained deliberately as representative of real wakes and staged as a
production lever in `035`. The accuracy gate is a
**fixed tolerance: sampled relative gradient (velocity) RMS error ≤ 1e-3**
against the direct references (user decision `2026-08-05`, superseding the
`2026-08-04` "match FLOWVPM default FMM parameter accuracy" gate — no
default-settings accuracy check is required anymore; the sampled-direct
references remain as the measurement instrument). The tolerance gates GPU
winners and speedup eligibility; fixed historical CPU rows remain visible
with their measured errors even when they fail. Parameters are tuned
separately per case for optimal performance; the wake case is
additionally benchmarked at the unit-cube-optimal parameters. CPU
baseline runs go on cluster CPU nodes, never the local machine.

Standing directive: **if significant speedup levers are identified during this
phase, invest in them** — row `035` is the measured tuning and optimization
campaign, with a user checkpoint gating each production optimization cycle.
Only levers expected to improve end-to-end U/J-solve time by at least 5% enter
that campaign.

Gating uses explicit parallel lanes; table order does not serialize them:

- interface: `031 -> 032`;
- nearfield: `031 -> 031a`, then `031a + 032 -> 032a`;
- CPU baseline: `033`, independent after `019a`;
- consumer integration: `032 + 033 -> 034`, which may run in parallel with
  `032a`;
- performance join: `030 + 032a + 034 -> 035`;
- milestone review: every Integration row above -> `036`.

This phase does not reopen the Theory or Implementation gates.

The fixed phase accuracy gate applies to sampled relative gradient (velocity)
RMS error only: `U <= 1e-3`. Every reported configuration must also log the
sampled Jacobian `J` RMS error as a diagnostic, but `J` does not select or
disqualify a winner. Historical `033` CPU timings remain in the record even
when they miss the velocity gate; a speedup ratio or headline may use only a
baseline configuration that passes it. No additional tuned CPU baseline is
required.

**Amendment (user direction `2026-08-05`, revised after numerical review):**
the resident vortex nearfield gets two candidate strategies. (i)
*Regularized-everywhere*: evaluate the regularized Biot-Savart kernel for
every direct-nearfield pair, borrowing FLOWVPM's GPU polynomial erf
(`custom_erf`, an FDLIBM rational port) and fused U+J math. (Superseded in
part by the `032` task-file amendment and settled by measurement `2026-08-05`:
the shipped `g`/`h` evaluation is erf-free — theory §3 series + §6.2 one-`exp`
outer form — after the mandated H200 A/B measured it 1.5x faster than the
`custom_erf` port at equal delivered accuracy; the fused U+J math is borrowed
as stated, and no FDLIBM code ships in `src/`.) (ii)
*Partitioned replacement*: keep the singular FMM far field; ensure the direct
geometry contains every pair inside the conservative smoothing cutoff
`r/σ_src ≤ ρ_t`; evaluate those pairs once with cancellation-safe
regularized U/J formulas and use the cheaper singular kernel for remaining
direct pairs. Row `031a` derives the cutoff, stable small-`ρ` series, and
exact-once geometry contract; row `032a` implements it and selects the default
by H200 measurement. (iii) *Two-pass additive correction* (added by user
direction `2026-08-05` during the `031`/`031a` clear-context review): leave the
FMM entirely unmodified and add a second pass carrying only the regularization
deficit over the cutoff shell. It touches no `025` routing invariant, but its
subtraction lands in the accumulator across two kernels, so it requires either
Float64 accumulation of the singular direct term and its correction, or the
`ρ_c = 2` hybrid; `031a` §6.1 derives it, bounds the conditioning, and shows
that two-pass and partitioning have opposite depth trends. An earlier
singular-minus-correction proposal had been removed as ill-conditioned as
`ρ → 0`; the derivation now quantifies that failure as a Float32 tail effect
and supplies the two remedies. **Kernel scope (user decision `2026-08-05`): only the FLOWVPM
default `gaussianerf` kernel is supported in this phase** — it is the sole
kernel compatible with the `CoreSpreading` viscous model
(`_kernel_compatibility`), so nothing viscous-capable is lost; `winckelmans`
support is dropped. (The `033` ring case was first redefined to `gaussianerf`
before its large cases ran — job 13051516 partial ring data discarded — and the
ring was then retired entirely later the same day in favour of the wake
cylinder; the replacement case is `gaussianerf` throughout.) The
SFS (`ζ`/`Estr`) kernel derivation is deferred to a later, not-yet-staged
row.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [x] | [x] | `031-integration-api-design.md` | User-in-the-loop design of the generalizable device-resident system interface: gap analysis (vector strength `Γ` + `σ` packing, Lamb-Helmholtz channel end-to-end, 9-component hessian/`J` output, capacity/no-realloc contract, residency-trait promotion), FLOWVPM 46×N ↔ FastMultipole buffer mapping, resident-vs-transfer coupling guidelines, and the storage-layout spec. The signed-off spec was corrected across three review rounds and approved on `2026-08-05`. | `019a` |
| [x] | [x] | `031a-theory-kernel-splitting-nearfield.md` | Derive the partitioned `gaussianerf` nearfield: stable regularized U/J inside a conservative `ρ_t` cutoff, singular U/J outside, exact-once direct/M2L geometry, per-precision cutoff/error bounds, cost model, and numerical validation. SFS kernel deferred. Derivation row: `theory/`, `scripts/`, `data/` only; does not reopen the Theory gate. | `019a`, `031` |
| [x] | [x] | `032-impl-generalized-device-interface.md` | Implement the approved interface in FastMultipole `src/`: vector-strength/Lamb-Helmholtz body packing and device B2M, 9-component hessian output on the resident path, first-class documented device-system API (promote the `fm028_device_system.jl` pattern into `src/` + docs), external-code connection guidelines, parity tests incl. `P=4`, and a no-regression gate on the shipped scalar path. Nearfield: the regularized-everywhere `RegularizedVortex` (`gaussianerf` only) baseline with FLOWVPM's fused U+J pair math and the erf-free `g`/`h` evaluation (task-file amendment; the mandated H200 A/B measured erf-free 1.5x faster than a `custom_erf` port at equal accuracy, so no FDLIBM code ships); the partitioned alternative and default selection are `032a`. | `031` |
| [x] | [x] | `032a-impl-split-nearfield-comparison.md` | Implement `031a`'s partitioned resident nearfield (stable regularized U/J for direct pairs inside `ρ_t`, singular U/J outside, singular FMM far field, exact-once cutoff coverage) **and its two-pass additive-correction alternative** (`031a` §6.1: unmodified singular FMM plus a deficit-only second pass, in Float64 or the `ρ_c=2` Float32 hybrid), A/B both against `032`'s regularized-everywhere `RegularizedVortex` on H200 at fixed adequate geometry using a profile-triggered benchmark ladder, and ship the measured winner as the resident vortex-nearfield default. | `031a`, `032` |
| [x] | [x] | `033-flowvpm-baseline-benchmarks.md` | Baseline CPU benchmarks of FLOWVPM at the `gpu-full` branch-point commit `e2bd487`: single-thread and 64-thread, both test cases, profiled per-stage cost breakdown and bottleneck identification, plus the checksummed sampled-direct references used to evaluate the phase's fixed 1e-3 relative-gradient-error tolerance. | `019a` |
| [x] | [x] | `034-flowvpm-gpu-integration.md` | Modify FLOWVPM (`gpu-full` branch) to drive the resident GPU lifecycle: `CuArray`-backed `ParticleField` coupled device-to-device (no per-step body transfers), `RadixFMMCache` reuse across time steps, resolve the `nearfield_device` hazard, end-to-end correctness vs `UJ_direct` on both test cases. Read `../FLOWVPM.jl/CLAUDE.md` first. | `032`, `033` |
| [x] | [x] | `035-flowvpm-performance-campaign.md` | Tune each case, benchmark the wake at cube-optimal parameters, rank profiled levers, and run user-approved optimization cycles. Own the sole definitive speedup/profile report and specify whether the equal-physical-cell rectangular radix grid in `037` clears the 5% end-to-end U/J-solve bar; structural radix implementation is owned by `037`. Compare one U/J solve to matched-`n` `030` by ratio (full RK3 step separate). | `034`, `030`, `032a` |
| [x] | [x] | `036-milestone-review-integration-phase.md` | Milestone Review for Integration rows `031`–`035` including `031a`/`032a`: interface generality, FLOWVPM correctness/performance verdicts, speedup-documentation completeness, and downstream compatibility (FLOWVPM CPU users, FLOWUnsteady, VortexLattice). | `031`, `031a`, `032`, `032a`, `033`, `034`, `035` |
| [x] | [x] | `037-impl-rectangular-isotropic-radix-grid.md` | Add an optional rectangular radix-grid path with approximately cubic physical cells for elongated domains: generalized quantization/keying, occupied-cell metadata, routing, CUDA refresh, and lifecycle parity. Start with a fixed-resolution rectangular leaf grid; generalize the hierarchy only if measured results require it. Preserve the cubic path and zero-allocation/device-residency contracts. | `035`, `036` |

## Adaptive Octree Phase

This phase was staged by user direction on `2026-08-06`, following the
sparsity-suitability review of that date. The review found the radix path is
occupancy-compacted (compute scales with occupied cells) but rigidly uniform
— one global leaf width, one cubic box, one depth `ell`, no leaf-population
bound — so it serves uniformly sparse fields well but multi-scale density
(dense clusters plus diffuse regions; wake rollup; `CoreSpreading`-grown σ)
poorly: the global `σ_max` geometry gate forces a globally shallow tree, and
an unbounded fat cell serializes `O(K²)` nearfield work in a single warp and
B2M in a single thread. The remedy is a **2:1-balanced adaptive Morton
octree** in the PVFMM/ExaFMM-T style: with 2:1 balance, same-level (V-list)
M2L remains a finite translation-invariant offset-class set, so the `025`
level-scaling law, the per-`(level, offset)` class batching, and the
existing resident M2L strategies and operator tables carry over unchanged;
the new pieces are the adaptive near-field lists (U direct, W/X via M2T/S2L)
and device tree construction from the sort/scan/compact primitives already
in the CUDA path.

The whole phase is gated behind `035`, `036`, and `037`. By user direction
(`2026-08-13`) the three nearfield-reduction rows `037a`/`037b`/`037c` are
staged BEFORE the adaptive-octree derivation: they appear first in the table
below, so the first-unblocked-row rule activates them ahead of `038`. `037c`
is conditional — it opens only if `037b`'s recorded verdict recommends it
(and the user agrees); otherwise it is closed by pointer and `038` follows
`037b` directly. `037b`'s rotor-wake case also supplies (or refutes) the
multi-scale-density evidence `038`'s entry gate asks for. The `038`
theory row carries an explicit evidence-or-waiver entry gate: multi-scale
density must be measured as a binding cost (or the user must waive that
requirement) before derivation begins. `038` is a scoped derivation row
(`theory/`, `scripts/`, `data/` only) and does not reopen the Theory Phase
hard gate or re-block any completed row. Row `041a` (user-directed
`2026-08-06`) is the phase's reporting row: publishable figures comparing
old (uniform grid) and new (adaptive) machinery on uniform and non-uniform
fields in time and memory, at matched stated accuracy, following the
standing TikZ/CSV figure conventions and extending the `024a` set.

Row `037d` was added by user request on `2026-08-14`, after `037b` falsified
two-pass deficit splitting and recommended `037c` NO-GO. It is a cheap,
paper-only cost/error estimate of evaluating the full `gaussianerf`-regularized
U/J field by particle-mesh (spread/FFT/interpolate — no direct pass, no
kernel split), with the Ewald-split variant as a secondary comparison. It
exists to decide for ~zero hardware cost whether a Fourier-space nearfield
replacement is worth staging as an implementation row; it does not gate `038`
and changes no production code.

Rows `037e` and `037f` were added by user request on `2026-08-14`, after the
`037b` approval, from the nearfield-lever discussion of that date (the
symmetric/Newton-3 pair lever was considered and dropped). Both are
constant-factor levers on the shipped partitioned nearfield, independent of
(and parallel to) the structural `038` lever: `037e` prunes the partitioned
direct list toward the exact `rho_t`-ball geometry (the mechanism `037a`
validated on the two-pass correction list — 56–65% candidate elimination,
bit-identical errors — pointed at the path we actually ship); `037f` cheapens
the regularized pair kernel itself under the lenient 1e-3 gate. Both are
off-by-default until measurement passes the promotion gate, and any default
change requires explicit user approval. They do not gate `038` or each other,
but they are implemented as one coordinated effort because they touch
adjacent nearfield surfaces.

Row `041b` was added by user direction on `2026-08-15`. It closes two
nearfield questions with the cheapest evidence first: whether reducing the
U/J output surface can materially reduce complete solve cost while preserving
all FLOWVPM consumers (including SFS), and whether strategic-target sampling
plus interpolation can economically replace any remaining direct work. It is
a measurement/theory row only. The existing `035`, `037d`--`037f`, and `041`
records supply its A0 and Stage-0 inputs; a production implementation is
permitted only as a separately reviewed successor if the registered rank,
accuracy, and complete-solve gates pass.

Row `041c` was added by user direction on `2026-08-15` from the `041b`
strategic-target discussion. It tests a more analytic alternative: treat each
current terminal U-list pair as a multilevel refinement queue, peel newly
admissible source/target subpairs into M2L/M2T/S2L shells at successively finer
levels, and leave only the irreducible nearest-neighbor complement to direct
P2P. This is not a second adaptive-tree implementation and must state its
increment over `038`--`041`: it refines *inside the residual terminal U-list*,
including virtual subleaves or target filtering where useful. The first row is
proof/census only; production work requires a separately reviewed successor.

Row `041d` was added by user direction on `2026-08-17` from the `041c` NO-GO
discussion. It closes or funds the remaining smooth-representation nearfield
alternatives: (1) a registered census pre-kill of regularized-basis P2M/M2P
substitution of residual direct pairs (the "regularized solid harmonics"
proposal), reusing the `041c` machinery and calibration — the `041b` rank
probe does not bound this regime because it ran at `sigma/h ~ 1e-3` where
regularization is inert; and (2) a paper cost estimate of a sigma-adaptive
multilevel smooth representation (AMR-VIC / multilevel summation) for the
sigma-heterogeneous rotor regime that `037d`'s global-mesh verdict could not
serve. Theory/measurement row only; it does not gate `042`, and any
successor (derivation or implementation) requires separate user staging.

Row `041e` was staged by user direction on `2026-08-17` to attempt the most
promising remaining direct-kernel redesign: replace independent warp-per-U-edge
execution with a target-owned fused traversal of each target leaf's complete
U-neighbor adjacency. The hypothesis is reuse and work-shape improvement, not
the already-falsified atomic-only lever. It preserves the shipped kernel and
exact U list and is off by default through measurement.

Row `041f` was staged by user direction on `2026-08-17` to resolve 041d's one
remaining open nearfield architecture: replace the placeholder
cost band for sigma-adaptive AMR-VIC / multilevel summation with an actual
level/patch/halo census on the deterministic rotor snapshot, a complete
Gaussian level-decomposition and U/J error budget, exact-once ownership, and
a bounded AMR-FFT-versus-MSM optimizer. It is theory/measurement only;
production work requires a separately staged successor. A
`2026-08-17` review amendment (recorded in the task file, binding) adds the
uniform-sigma unified-solver verdict (does the adaptive mesh subsume the
funded 037d global VIC, so a future successor stages one mesh implementation,
not two)
and a mandatory priced FMM-retained banded-hybrid configuration (keep the
shipped singular far field and U/V routing; mesh only the sigma-affected
bands).

Row `041g` was staged by user direction on `2026-08-17`. It closes the one
untested harmonic gap left by `041c`: the shipped sigma demotion is
whole-pair, and `041c` barred M2L on demoted lineages while measuring the
rotor's whole-leaf pure-singular ceiling at 58.7–62.7% — on proxies, never on
the actual rotor sigma field. `041g` derives a predictive singular-M2L
admissibility criterion from the kernel-difference (regularization-tail)
bound already in 031a §4 plus the constant-P truncation bound, generalizes it
via dimensionless collapse variables (leaf width in sigma units
`K^{1/3}/beta`, neighborhood-local sigma spread; validated against a
registered parametric sweep over overlap, spread, and sigma correlation
length), and censuses sigma-class M2L re-admission (class-filtered
multipoles, predeclared logarithmic bins) on the actual rotor snapshot with
the existing calibration. Deliverable even on NO-GO: the admissibility map —
when singular M2L works for the regularized problem and when it cannot. On
gate pass it proposes the per-class demotion interaction list as a
separately staged successor. Theory/measurement only.

Row `041h` was staged by user direction on `2026-08-18`. It moves from
synthetic reconstructions to a REAL simulated rotor wake: the final-step
particle fields of the FLOWPanel `rotor_hover` simulations
(`../../FLOWPanel.jl/data/`, n = 67,745 primary / 37,165 secondary, with
mid-run steps for a true refresh/epoch-persistence measurement), and asks
what the full per-step cost can be driven to on ONE H200 using the whole
038–041e toolbox (adaptive octree, sticky demotion, per-n geometry tuning,
graph capture, 037f g/h, the 041e fused nearfield + selector). At this
n ≈ 4–7e4 scale the problem is latency/refresh-floor-dominated, not
throughput-dominated; the row must also deliver a modeled (not implemented)
8-H200 estimate grounded in the 029 floors, including the break-even n.
Production tuning changes are opt-in and individually user-gated (035
convention).

Rows `041i` and `041j` were staged by user direction on `2026-08-18`, from
the sigma-contamination discussion of that date. The user hypothesized that
large-particle sigma contaminates the multipole representation (expansions
approximate singular `1/r`, not the regularized kernel actually evaluated),
inflating the direct list and under-utilizing expansions. The session's
evidence review found the hypothesis contradicted for the rotor (`041g`
zero sigma demotions; 58.7–62.7% of rotor direct pairs purely geometric;
leaf-local sigma spread ~1.01; adaptive nearfield only ~10% of lifecycle)
but identified two unmeasured channels: the `041g` census ran at `q=12`,
not the shipped `near_radius2=5`/`rho_t=4.789` operating point (a 2.24x
tighter margin where rotor `ell=7` leaves would demote), and the sigma
split-veto/depth-cap channel (`src/tree_batched.jl:802-813`) was never
censused. `041i` closes both cheaply. `041j` benchmarks a
kernel-independent FMM (PVFMM, the strongest maintained open-source KIFMM;
no maintained GPU KIFMM exists) as an external CPU baseline: it documents
the work required to host the regularized `gaussianerf` kernel (all KIFMMs
use the same singular-far/regularized-P2P split we ship), empirically tests
the user's regularized-check-surface accuracy hypothesis against the 031a
basis-independent bound, and delivers a written per-stage GPU-limitations
prediction. `041j` does not gate `042` (external-comparison row, `037d`
convention). Neither row is started; `041h` remains the first unblocked
implementation row.

Row `041k` was staged by user direction on `2026-08-20`: measure how far
sheer hardware power carries a naive O(N²) direct evaluation on a single
H200 — N from `1e2` by half-decades until median wall time exceeds 10 s —
for the realistic FLOWVPM workload (regularized `gaussianerf` U/J, and
U/J plus SFS via the `041b` §1.2 factorized identity), in F64 and F32 with
established levers (`rsqrt`, shared-memory tiling). Direct evaluation
carries no polynomial/multipole approximation (exact up to rounding), so
the row also anchors the brute-force/FMM crossover and builds the first
fused direct+SFS kernel in the stack (today `sfs=true` is a hard error on
the GPU path), pricing SFS enablement with data. Standalone benchmark row
(`scripts/`, `data/` only); does not gate `042`.

**042 scope override (user direction `2026-08-20`).** The adaptive milestone
review covers only `038`–`041a`. Rows `041b`–`041j` neither gate nor belong to
that review; `041k` is independently reviewed and also does not gate it. This
override supersedes the earlier staging notes that placed some of those rows
before `042`.

Row `042a` was added by user direction on `2026-08-20` from the `042`
milestone's highest-value GPU follow-on. It optimizes the adaptive X-list S2L
execution shape—target-major grouping, source/target reuse, reduced atomic
retirement, and a bounded `P=4` specialization—without changing the tree,
interaction list, operator mathematics, or accuracy budget. Same-job H200
stage and complete-step gates decide promotion; it must finish and be approved
before the partitioned multi-GPU phase derives against the final single-GPU
lifecycle.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [x] | [x] | `037a-nearfield-followup-plan.md` | Rectangular-grid and nearfield follow-up to `037`: center automatically snapped rectangular bounds, measure cubic/rectangular lattice parity, select the smallest accuracy-safe regularization cutoff with an exact tail/FMM error decomposition, and screen the existing two-pass correction at smaller primary stencils. H200 evidence gates any default change. **Done `2026-08-13`:** centered rectangular is only 0.3-1.2% faster; accuracy requires P6 for the tested two-pass geometry. Exact AABB pruning removes 56-65% of correction candidates and cuts the `n=1e6` two-pass solve 12.9%, but it remains 4.6% slower there and 39% slower at `n=1e5`; no default change or further uniform-grid cycle. | `037` |
| [x] | [x] | `037b-explore-twopass-deficit-geometry.md` | Measurement-first exploration of two-pass deficit splitting (singular math at expansion-validity geometry + pairwise `(g-1)` correction within `rho_c·sigma ≈ 2σ`) at its OWN co-designed (ell, q, rho_c, P) — superseding the matched-geometry `032a` comparison. Builds the checksummed realistic rotor-wake case (e.g. DJI 9443 via `~/…/FLOWPanel.jl`), extends the 3C error-decomposition oracle to the two-pass field, and produces an exact per-approach speedup report (same-job anchors, critical-path pricing). Renders the go/no-go verdict for `037c` and the multi-scale-density evidence for `038`. Default changes need explicit user approval. | `037a` |
| [x] | [x] | `037c-impl-mesh-deficit-fourier-nearfield.md` | **Closed by pointer (user direction `2026-08-14`): `037b` measured NO-GO — no compact-support regime at the 1e-3 gate (cutoff floor 3.2–3.7σ) and a meshed pass bounded at ≤18 ms vs a ≥28 ms shortfall; no campaign run; `038` follows `037b` directly; the no-split Fourier question moved to `037d`.** Original scope: conditional on the `037b` verdict + user go. PME-style mesh evaluation of the smooth Gaussian deficit: theory-first spectral/interpolation error model, then device-resident spread/convolve/interpolate (rectangular mesh; FFT vs local stencil by measurement) under the capacity/zero-allocation/counter contracts, benchmarked against BOTH the shipped baseline and the `037b` winner so each approach's exact speedup is reported separately. Off-by-default; default changes need explicit user approval. | `037b` |
| [x] | [x] | `037d-theory-fourier-nearfield-cost-model.md` | Paper-only cost/error estimate of full particle-mesh (VIC-style) evaluation of the `gaussianerf`-regularized U/J field — spread/FFT/interpolate on the rectangular bounds, no direct pass — priced against the `037b` same-job anchors (cube/wake/rotor, `n=1e5`/`1e6`) under the 1e-3 velocity gate; must model mesh resolution vs `sigma_min`, free-space padding, transform count for U+J, spread/interpolate cost and order, and σ-heterogeneity handling (σ-binned/multi-mesh), plus the Ewald-split variant as a secondary bound. Verdict: stage an implementation row or close the Fourier direction. `theory/`, `scripts/`, `data/` only; no hardware runs required. | `037b` |
| [x] | [x] | `037e-impl-nearfield-finebin-pruning.md` | Prune the shipped partitioned direct nearfield toward the exact `rho_t sigma` ball: fine-bin/sub-Morton candidate generation and/or exact source-directed AABB-gap predicates on the production direct list (the `037a`-validated mechanism, applied to the partitioned path), preserving exact-once coverage, capacity/zero-allocation, counters, and graph capture. Pre-registered H200 screen on cube/wake/rotor at `n=1e5`/`1e6`; promotion gate: velocity RMS ≤ 1e-3, ≥5% faster end-to-end on a material case, no >3% regression elsewhere; off-by-default until the gate passes and the user approves. | `037b` |
| [x] | [x] | `037f-impl-nearfield-pair-kernel-cheapening.md` | Cheapen the regularized `gaussianerf` U+J pair kernel under the 1e-3 gate: candidate mechanisms are reduced-order `g`/`h` polynomials, shared-memory/texture lookup+interpolation, and reduced-precision inner distance math with FP32 accumulation; select by measured A/B at equal delivered accuracy (the `032` methodology). Same promotion gate, contracts, and off-by-default rule as `037e`; `P=4` and both-precision tests required. **Done `2026-08-14`: `:fp32` passed the full gate (+7.4-7.8% wake, worst regression -2.5%, delivered-error deltas ~1e-8) and the user approved the default flip the same day — `CUDA_NEARFIELD_GH_MODE` now defaults to `:fp32` (no-op on Float32 configs; `:shipped` = control/opt-out); `:reduced`/`:reduced_fp32`/`:lut` ship opt-in with recorded ceilings.** | `037b` |
| [x] | [x] | `038-theory-adaptive-radix-octree.md` | Derive the 2:1-balanced adaptive Morton octree: construction as sort/scan/compact, U/V/W/X interaction lists with an exact-once coverage proof at both near radii, M2T/S2L operators with constant-`P`-consistent error bounds and Lamb-Helmholtz coverage, a per-cell σ geometry gate replacing the global `σ_max` form, cost/capacity model (incl. a synthetic multi-scale case), and the refresh/rebuild policy. V-list M2L must reuse the `025` level-scaled operator tables unchanged. Entry gate: `035`/`037` evidence that multi-scale density binds, or explicit user waiver. | `035`, `036`, `037` |
| [x] | [x] | `039-impl-adaptive-octree-construction-host.md` | Implement host adaptive-tree construction (Morton-prefix split on `K_max`, depth cap, 2:1 balance) and U/V/W/X list generation, with V lists in the existing `(level, offset)` class format, per-cell geometry gate, capacity-sized buffers with zero per-step refresh allocation, exact-once brute-force verification on uniform/wake/clustered fields, and uniform-limit parity with the existing hierarchical routes. | `038` |
| [x] | [x] | `040-impl-adaptive-octree-lifecycle-host.md` | Run the full host resident lifecycle on the adaptive tree: V-list M2L through the unchanged resident strategies and operator tables, M2M/L2L over adaptive ancestor levels, new M2T/S2L kernels (φ+χ, `008h` order rule) for W/X, U-list direct through the existing nearfield kernels. Accuracy gates (velocity RMS ≤ 1e-3, `P=4` and `P=8`, both precisions) on cube, wake, and multi-scale cases; uniform-limit lifecycle parity. | `038`, `039` |
| [x] | [x] | `041-impl-adaptive-octree-cuda.md` | Mirror the adaptive octree on the CUDA device-resident lifecycle: device construction/refresh as flag/scan/compact kernels, sorted-Morton binary-search occupancy lookup replacing the dense `Σ8^L` table (record whether it also lifts the uniform path's `ell ≤ 8` cap), device M2T/S2L, occupancy-epoch caching over the adaptive leaf set, `023` counter and zero-allocation parity, and H200 before/after per-stage measurements vs the uniform-depth path on all three cases. | `038`, `039`, `040` |
| [x] | [x] | `041a-benchmark-adaptive-vs-uniform-figures.md` | Publishable benchmark report and figures: old uniform grid vs new adaptive octree on uniform and non-uniform fields (cube, wake, multi-scale contrast sweep, σ-heterogeneous variant), time and memory, host and H200, every row at stated sampled-direct accuracy under the 1e-3 gate. Figures (TikZ/pgfplots + CSV, extending the `024a` set) must show the old approach's failure mechanism (fat-cell/forced-shallow costs, capacity memory) and the new approach's measured gains. Benchmark/analysis row: `scripts/`, `data/`, figures only. | `040`, `041` |
| [x] | [x] | `041b-nearfield-output-strategic-target-feasibility.md` | Audit the true FLOWVPM U/J/SFS output requirement; analytically pre-kill uneconomic output-width policies; prove the factorized-SFS identity; and run arithmetic plus offline SVD/QDEIM screens to decide whether strategic-target interpolation has a niche beyond VIC and the adaptive U-list. Measurement/theory artifacts only; no production changes. | `035`, `037d`, `037e`, `037f`, `041` |
| [x] | [x] | `041c-theory-multilevel-nearfield-shells.md` | Derive and census a multilevel residual-nearfield traversal: recursively refine current terminal U-list pairs, route newly expansion- and regularization-admissible subpairs to same-level M2L or cross-level M2T/S2L shells, and evaluate only the terminal complement by direct P2P. Prove exact-once coverage and use existing snapshots/timings for a route-only promotion model before any production code. **Done `2026-08-17`: NO-GO — ideal promotion exists on cube/multiscale, but virtual rectangles are below measured expansion-route crossovers; conservative selector falls back to direct everywhere.** | `025`, `037e`, `038`, `041`, `041b` |
| [x] | [x] | `041d-theory-smooth-nearfield-alternatives.md` | Registered census pre-kill of regularized-basis P2M/M2P nearfield substitution (any linear basis, bracketing admissibility policies, `041c`-identical cases and calibration) plus a paper cost estimate of sigma-adaptive multilevel smooth representation for the sigma-heterogeneous regime. **Done `2026-08-17`: part 1 NO-GO — break-even needs 28–1222-source clusters vs an admissibility cap of ~7 inside the sigma floor, zero conservative promotions, selector chooses direct on all rows; part 2 OPEN — `N_mesh ~ 0.75n` independent of sigma spread, no sign flip in the priced band, derivation row recommended before any implementation.** Does not gate `042`. | `037d`, `041b`, `041c` |
| [x] | [x] | `041e-impl-target-owned-fused-nearfield.md` | Reorganize the shipped partitioned U/J direct kernel around exclusive target-leaf ownership: target CSR over the exact U list, fused traversal of all source neighbors, shared source tiling, target reuse, reduced ragged edge tails, and one final target write. Same-job H200 A/B against the 037f-enabled baseline; off by default; exact-once, accuracy, graph, capacity, zero-allocation, and <=3% fallback gates. | `035`, `037e`, `037f`, `041` |
| [x] | [x] | `041f-theory-sigma-adaptive-smooth-nearfield.md` | Replace 041d's placeholder AMR overhead with a deterministic real-rotor level/patch/halo census and full Gaussian multilevel U/J derivation; compare patch-local free-space AMR FFTs against multilevel summation, prove exact-once ownership, price refresh/capacity/fragmentation on the complete critical path, and render GO/REGIME-ONLY/NO-GO before any production implementation. Review amendment `2026-08-17`: also render the uniform-sigma unified-solver verdict vs 037d global VIC, and price an FMM-retained banded hybrid. | `037d`, `038`, `041a`, `041d` |
| [x] | [x] | `041g-theory-singular-m2l-admissibility.md` | Derive a predictive singular-M2L admissibility criterion for the regularized problem (constant-P truncation + 031a kernel-difference tail bound, evaluable from refresh-time statistics), verify the dimensionless collapse (`K^{1/3}/beta` leaf-width-in-sigma, neighborhood-local sigma spread) on a registered overlap/spread/correlation sweep, and census sigma-class M2L re-admission of whole-pair-demoted work (class-filtered multipoles, log bins `{1,2,4,8}`) on the actual rotor snapshot with existing calibration and exact-once class-partition oracle. **Done `2026-08-18`: NO-GO — both registered rotor counts have zero sigma demotions; the reduced collapse map fails; and the existing scalar bound does not certify delivered U/J with live phi/chi budgets.** | `025`, `031a`, `037e`, `038`, `041`, `041a`, `041c` |
| [ ] | [ ] | `041h-impl-real-rotor-simulation-fullstep.md` | Optimize the complete per-step U/J cost on the real FLOWPanel `rotor_hover` wake snapshots (n = 67,745 / 37,165 + mid-run steps) on a single H200: deterministic VTP snapshot extraction with committed provenance, sigma/occupancy census, pre-registered uniform-vs-adaptive baseline matrix, profiled optimization cycles (refresh amortization, per-n geometry, strategy mix, launch floors, bounded 041e re-check) with user-gated production changes, and a modeled 8-GPU estimate with break-even n. | `030`, `037f`, `041`, `041a`, `041e` |
| [ ] | [ ] | `041i-census-sigma-closure.md` | Close the sigma-contamination question at the shipped operating point: re-run the `041g` demotion census at `near_radius2=5`/`rho_t=4.789` (with a `q=12` reproduction control) and census the never-measured sigma split-veto/depth-cap channel (gate-on vs gate-off trees: depth, leaf occupancy, `u_pairs` delta), on rotor/cube/wake; H200 timing spot-check only if the pair delta exceeds 1%. Verdict: signed statement of what sigma costs (pairs, ms) at the shipped defaults. Measurement row: `scripts/`, `data/` only. | `041g`, `041a` |
| [ ] | [ ] | `041j-benchmark-kifmm-external-baseline.md` | Benchmark PVFMM (kernel-independent equivalent-density FMM; no maintained GPU KIFMM exists) as an external CPU baseline on the registered rotor/cube/wake snapshots at the 1e-3 gate: build + Julia bindings on a cluster CPU node, gaussianerf U/J via singular Biot-Savart far field + per-particle-sigma regularized P2P (documenting exactly what custom kernel work is required), the regularized-check-surface P2M experiment vs the 031a basis-independent bound, CPU timings vs `033`/`035`, and a written per-stage GPU-limitations prediction. Does **not** gate `042`. | `041a`, `033` |
| [x] | [x] | `041k-benchmark-direct-bruteforce-ceiling.md` | Brute-force ceiling: naive all-pairs O(N²) direct evaluation of the realistic FLOWVPM workload (`gaussianerf` U/J, and U/J+SFS via the `041b` factorized identity — the stack's first fused direct+SFS kernel) on one H200, N from `1e2` by half-decades until median > 10 s, per {uj, ujsfs} × {naive, tiled} × {F64, F32(+`rsqrt`)}; exactness gated against a Float64 CPU reference (direct has no multipole approximation); reports largest-N-under-10s frontier, marginal SFS-fusion cost, and the brute-force/FMM crossover vs 041-series timings. **Done `2026-08-20` (job 13246033): F64 gate passed at ≤4e-15 (FLOWVPM cross-check ~1e-14); 10-s frontier ~1.4e6 particles (uj/f32/tiled; ~1.24e6 with fused SFS, ~0.7–0.9e6 F64) at a transcendental-bound 2.0e11 pairs/s; fused SFS costs only +26–31% over U/J; against the measured `041a` fig15 best-uniform FMM curve the crossover is n ≈ 4–5e3 (FMM 5× faster at 1e4, 62× at 1e6), so brute force is an exactness reference above that, not a performance alternative. Amendment (job 13246522): `opt` variant (far-field singular switch + fast intrinsics + 2-target blocking) gates-clean at 1.6–1.7× F32 — frontier ~1.8e6, 3.3e11 pairs/s FMA-bound, crossover only moves to ~5.5e3.** Does **not** gate `042`. | — |
| [x] | [ ] | `042-milestone-review-adaptive-octree.md` | Milestone Review for the adaptive octree arc (`038`–`041a`, per the `2026-08-20` user scope override excluding `041b`–`041j`): exact-once and balance evidence, operator-table reuse, M2T/S2L accuracy and Lamb–Helmholtz coverage, performance/reporting audit, default and uniform-depth-cap decisions, lifecycle contracts, deferred dual-grid disposition, and consumer/API documentation requirements. **Done `2026-08-20`: adaptive remains opt-in—recommended for clustered/multiscale density, severe σ heterogeneity, or memory-constrained deep grids; uniform remains default for uniform/small-n/sparse-wake regimes. Do not lift the uniform `ell<=8` cap without a concrete consumer. Close the dual-grid candidate. Adaptive S2L tuning is staged as `042a`; the remaining follow-on is an evidence-based policy/K selector plus documentation.** | `038`, `039`, `040`, `041`, `041a` |
| [ ] | [ ] | `042a-impl-adaptive-s2l-cuda-optimization.md` | Optimize adaptive CUDA X-list S2L without changing interaction coverage or mathematics: target-major grouping, source/target reuse, reduced atomic retirement, and a bounded `P=4` specialization; preserve scalar/LH, F32/F64, `P=4/8`, graph, capacity, counter, and zero-allocation contracts. Same-job H200 promotion requires >=25% lower S2L time and >=3% lower complete adaptive step on the `n=1e6` F64 wake with <=3% fallback regressions elsewhere. | `042` |

This phase does not reopen the Theory, Implementation, or Integration gates.

## Production Integration Phase

Staged by user direction on `2026-08-20`, gated on `042` (approved
`2026-08-20`) and placed BEFORE the Multi-GPU Scaling Phase. Objective: use
this project's matrix-ops machinery to accelerate the FLOWPanel item-018
campaign (`../../FLOWPanel.jl/BRAINSTORM/INDEX.md`). The 023 profiling of an
018 production step (2026-08-20) found 170–230 s/step on 64 cores, per-step
cost ~linear in particle count, and **~75% of a production step spent in the
Dynamic-SFS estimator `Estr_fmm!` near-field walk** — the single largest
lever.

**Budget arithmetic.** The stated stretch target — 30 revolutions in under
1 hour — means 1080 steps in <1 h, i.e. **≤3.3 s/step average, a 52–70×
speedup over today**. The CPU body-pass floor alone (~36 s,
kerneloffset-radius-bound) breaks that budget, so reaching it requires
accelerating BOTH the particle side (GPU UJ+SFS — the 75% lever; `041k`
measured fused SFS at only +26–31% over a U/J pass) AND the panel passes
(a GPU panel kernel, or a drastically cheaper panel-pass configuration).
The user's fallback resource envelope — 1×GPU + 64 CPU threads — bounds
achievable time rather than guaranteeing the 1 h mark. **Escape hatch:** if
`052`'s measured wall time misses <1 h, its verdict may pull specific
Phase-Q levers (`054`/`055`) forward as an addendum rather than waiting for
the `053` milestone — the budget is tight enough that stranding the known
1.6–1.7× nearfield levers behind a phase boundary is not acceptable.

**Ordering rationale.** The branch merges (`046`) come FIRST so that all
subsequent GPU work (hardening, SFS, coupling) lands directly on the
branches FLOWPanel consumes — avoiding building on the tmp3 clone and then
pushing every diff through a 174-commit merge — and so the riskiest task is
retired first. `047` and `048` then run in parallel (both block only on
`046`).

**Conventions and checkpoints.** Production changes require explicit user
checkpoints (the `035` convention); FLOWVPM's CLAUDE.md constraints apply to
work on that repo. The consolidated user checkpoints for this phase: repo
layout after the merges (retire tmp3 vs re-point; in `046`), the residency
default (`049`), and any default-behavior change (`053` review). `041h`
stays in the Adaptive phase as its first unblocked row; this phase does not
duplicate it and cites its results as soft inputs, not gates.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [x] | [x] | `046-impl-branch-unification.md` | **Done `2026-08-20`: both merges landed (`flowpanel-20260817` ← matrix-ops, only 2 conflicts; `flowpanel` ← gpu-full, 11 hunks + repo-wide dedicated-vorticity completion `bc9b9a6`); FastMultipole + FLOWVPM suites green; FLOWPanel 16/18 with both failures in the user's uncommitted WIP (radius_inflation), not merge-caused; 018 CPU smoke passed (116 clean steps); tmp3 retired per user direction (all branch tips + stash preserved as `tmp3/*` refs and a tag; see decisions log D2).** Original scope: the merges (user direction), front-loaded: fetch tmp3 FastMultipole `matrix-ops` into `projects/FastMultipole` and merge into **`flowpanel-20260817`** (176 commits vs 14; conflict hotspots: fmm!/tree internals vs FmmPlan/NearfieldInfluenceCache/autotune-perturb — the doc carries the complete 14-commit list); fetch tmp3 FLOWVPM `gpu-full` into `projects/FLOWVPM.jl` and merge into **`flowpanel`** (23 vs 10; flowpanel's `9fd25e6` Estr_fmm! source/target selection is load-bearing for 018). Safety protocol: tag/backup both sides of each merge, perform the merge on a scratch branch, and fast-forward the real branch only after the test gate passes. Gates: FastMultipole, FLOWVPM, FLOWPanel test suites green post-merge; the 018 driver still runs CPU-only unchanged (short smoke, not a campaign); user checkpoint on repo layout (retire tmp3 vs re-point). | `042` |
| [x] | [x] | `047-impl-production-settings-hardening.md` | **Approved `2026-08-21` after clear-context remediation review: all 31 tunables classified and production reads consolidated through `radix_setting`; construction-lock drift remains loud at device-step entry; atomic `set_radix_settings!` makes FLOWVPM grouped overrides all-or-nothing; tiled-thread validation is CUDA-shape exact; typed source/target tree, host/device nearfield, and radix-route policies replace production Boolean selection while compatibility shims remain. The earlier 046 entry-gate violation and D4's improper dispatch deferral are explicitly recorded and D4 is superseded. Focused host/static gates and FLOWVPM host suite green; no new HPC job.** Original scope: consolidate + robustify the GPU production surface (on the unified branch): the ~25 process-global `Ref` tunables get validated, documented behind one consolidated settings surface, and either made per-cache (`CUDARadixLifecycleOptions`/`AdaptiveTreePolicy`) or construction-time-locked with loud errors on late flips (today they silently keep the old mechanism, `translate_batched_cuda.jl:2247-2252`) — plus a regression test that a post-construction flag flip errors loudly; Future Dispatch Cleanup Notes items (`allow_host_bodies`, `nearfield_device::Bool`, `target::Bool` tree-role arg, route-selection flags → dispatch-on-object); generalization/robustness sweeps (F32/F64 × adaptive/uniform × P=4/P=8 parity — the P=4 test rule; capacity/out-of-box error paths; recenter contract); full FastMultipole suite green. No new performance work. | `046` |
| [x] | [x] | `048-impl-gpu-sfs-enablement.md` | **Approved `2026-08-22`: H200 job 13303399 passed all stages 1–5 (artifacts + sha256s in `data/gpu_sfs_enablement/`); production SFS settings selected per D14 (P=6, rho_t=4.789, derived q) and implemented as the coupling defaults in `FLOWVPM_fmm_radix.jl` (expansion_order 4→6, `_PARTITIONED_RHO_T_DEFAULT` 3.668→4.789; default-assertion tests updated); regression coverage of the new defaults rode in 049's job 13305555.** Prior state: host candidate matrix green; device/timing pending. `sfs=false` executes no TG/ζ kernels and row 9 restores CPU static semantics. Job 13294119 completed P=4/P=8 × F32/F64 × rho_t=4.211/4.789: all CPU `Estr_direct!`/`Estr_fmm!` gates pass (F64 9.18e-5–4.09e-4 ≤5e-4; F32 9.20e-5–4.09e-4 ≤1e-3; mechanical parity ≤4.1e-7). It stopped only because obsolete `@test_broken` declarations treated passes as errors; now fixed. Corrected device contracts and real-p018 A/B remain pending; prior 0.3 ms is invalid delivery overhead. No default changed. See D12–D13 and `data/gpu_sfs_enablement/`. | `046` |
| [x] | [x] | `049-impl-rotor-field-gpu-verification.md` | **Accepted `2026-08-22`: H200 job 13305555 ran all five stages; artifacts (sha256-verified) in `data/rotor_field_gpu_verification/results-13305555/`. All calibration anchors met or exceeded (u 1.1e-4 at P4; residency parity 150/150 at 1e-11; counters 144/144; resident RK3 291 ms/step at production P6 — 11x under the 3.3 s target); the run's 49 harness-gate FAILs were root-caused to harness gate misapplication (see 049 doc Results 2026-08-22 + D15) — lifecycle-layer alloc and error-bounded replay measurement deferred to `053`. User selected **upload-per-step** residency (D15: monitor compatibility; +12 ms/step, +4.1%).** Historical job 13247848 remains evidence only: UJ 3.4e-4 passed, but SFS timing/reference and arithmetic residency estimate were invalidated. Corrected harness hard-gates direct U/J integrity, runs P4/P8 × F32/F64 × both per-pair cutoffs, checks contracts/budgets, and performs a true interleaved same-job residency A/B without auto-selecting a default. | `047`, `048` |
| [x] | [x] | `050-theory-panel-multisystem-scoping.md` | **Approved `2026-08-22` (B' pricing reconciled against the delivered 049 budget, `results-13305555/fm049_budget.csv`: particle side D15 0.306 s median/0.373 worst = 9.3–11.3% of the 3.3 s budget; pessimistic stack leaves 0.89 s solve headroom; verdict and 051 shape unchanged; user notes multi-system generalization stays an eventual goal — B' tentatively adopted as a step toward it). Done `2026-08-21` (verdict: `theory/panel-multisystem-scoping.md`): option B' selected — keep FLOWPanel's 3-pass structure; cross passes become rectangular GPU brute-force kernels (wake→panels ~0.02-0.04 s at 041k rates; panels→particles ~0.4-2 s, retiring the 36-s floor); panel self-solve via device-resident dense NearfieldInfluenceCache matvec (10.8 GB fits H200; iterations = 051 measurement); radix framework untouched (A rejected: homogeneity excludes panel sources regardless of the targets===sources lift; unnecessary at these sizes); C priced as fallback ceiling ~60-80 s/step. The panel work is justified independently by the measured ~36 s CPU body-pass floor; the corrected 049 particle budget remains pending.** Original scope: decision row (user: "check if system-on-system looks significantly easier before doing it"). Options, priced with recorded facts: (A) lift the radix v1 `targets===sources` restriction (`translate_batched_resident.jl:1735-1743`; `target_bodies` aliased at `translate_batched_cuda.jl:5447`) to support distinct target sets incl. panel centers/probes; (B) individual system-on-system GPU evaluations mapped onto FLOWPanel's existing 3-pass structure (`FLOWPanel_simulate.jl:673-712`) — **a-priori favorite: least invasive, the CPU path already runs separate passes**; choose A only if B's measured pass overhead is material; (C) hybrid: particles on GPU, panel passes on 64-thread CPU (bounded by the ~36 s body-pass floor → cannot reach 30 rev/h; states what it CAN reach). Also scopes the panel GPU kernel itself (FLOWPanel `direct!` overload `FLOWPanel_abstractbody.jl:1260`; element types constant source/doublet tris + vortex rings/sheets/filaments) and where FmmPlan/NearfieldInfluenceCache fit. Verdict names the `051` implementation shape. | `046`, `049` |
| [x] | [x] | `051-impl-panel-particle-gpu-coupling.md` | **Approved `2026-08-24` (user pre-approval + clear-context audit APPROVE, zero discrepancies). Done `2026-08-24`: B′ stack fits the 3.3 s budget at 3.124 s (particle worst 0.373 + pass 1 0.124 + pass 2 U-only 2.020 + solve 0.607, 5.3% margin; jobs 13391706/13395348). Panel↔particle coupling via rect GPU seam (parity gates green); body solve repriced from 7.3 s FMM-path to 0.607 s via opt-in source-potential matrix S on `Backslash` (lossless: self pair is all-direct, bitwise == DirectBackend; the on-cluster exact-0 checks were vacuous — claim rests on the local nonzero-σ probe). Device gemv / Float32 S deferred to `052+`. Commits across all three repos in handoff eleventh-session-part-2.** Original scope: Implement the `050` verdict: panel↔particle GPU coupling (panel GPU direct kernel, or system-on-system passes, or hybrid), wired through FLOWPanel's `influence!`/`FastMultipoleBackend`. Fallback resource envelope per user: 1×GPU + 64 CPU threads. Gates: pass-by-pass parity vs CPU at the 018 operating point, no regression on FLOWPanel CPU tests, harness respects the step-head gotcha (maneuver!/update_TE! before influence eval). | `050` |
| [x] | [x] | `052-impl-flowpanel-018-driver-gpu.md` | **CLOSED `2026-08-26` (Ryan): all phases complete except the 1,080-step acceptance, which MOVES TO `052c` (after the performance overhaul; see `052c-plan-2026-08-26.md`). Final evidence: recenter!-sfs fix (FastMultipole `8479d0f2`) validated by h200 chain `13484013` — stages a–c, mature memory gate, and mature-continuation tolerance gates (count_tol 16 vs pinned CPU reference) ALL PASSED; stage d ran clean far past the prior crash point and timed out at 4 h at step 473/1079 on pace (~66 s/step at N≈230k), a performance issue not a correctness one (root-caused in `052c-plan`).** Prior state: **IN PROGRESS `2026-08-25` (PM): combined campaign in queue.** Protected H200 GPU `13468358` FAILED at 6m33s (stale cluster `FLOWPanel_solver.jl` called nonexistent `CUDA.MemoryInfo`; gate `13468360` cancelled); local guarded fix (`CUDACore.MemoryInfo` fallback) verified in-env and `rsync --checksum`-synced to all five cluster trees, along with the corrected UUID-key probe. Per user directives (qos=eng for H200 — served by partition `eng`, since `m13h` rejects it; combine stages into one allocation; tight time limits): live jobs are `13477875` (H200 chain a b c + inline gate + d, eng/eng, 4h), `13477876` (H100 probe→smoke→mature chain, cs2/gstandby, 3h), `13477878` (B200 chain, optional, 3h), `13477880` (L40S probe, 1h), `13477881` (H200 supplemental probe, eng/eng, 1h). All arms run to allocation (user rescinded the 1-day cancel budget). CPU mature reference (13468359, 38m13s exit 0) intact and reused. See the 2026-08-25 afternoon addendum in the impl doc. Architecture/job-ID-qualified probe → smoke → mature harness and cross-report tests are locally green. Isolated H100, B200, L40S, and supplemental H200 source triplets plus x86 environments are prepared/instantiated. Ryan-only launch wrappers cover H100 and optional B200 probe/smoke/mature plus probe-only L40S and supplemental H200; all use `--no-requeue`, and H100/B200 directly request `standby` (the scheduler may report effective `gstandby`). L40S should record official low-memory rejection; B200 is optional. **GH200 is not part of 052 and must not be attempted here; all ARM/GH200 work is reserved for future item 052a.** No alternative stage submits a 1,080-step run. | `049`, `051` |
| [x] | [x] | `052a-impl-gh200-arm-unified-memory.md` | **CLOSED `2026-08-26` (Ryan): gh200 arm COMPLETE (probe, smoke, membench, Phase C closed-negative, mature `13483367` PASSED); h100 PASSED all gates (`13477876`, incl. CPU-reference provenance gate); l40s CLOSED-ineligible by design (`13477880`: 48.3 GiB < 49.46 GiB bound); b200 support DROPPED by decision (`13477878` cancelled before start, 2026-08-26). Cross-checkout consolidation + notebook entry move to `052c` prep (frozen-tree constraint lifted with the b200 drop). Full evidence trail in `052a-plan-2026-08-25.md`.** Original scope: **STAGED `2026-08-25`.** GH200 (`mgh`/`gh200`, `--constraint=arm`) execution split out of 052's multi-arch extension: (A) offline aarch64 environment — populate the per-slug depot from the x86 login node via Pkg platform-override (compute nodes have no internet; shared `/home`; user's aarch64 Julia 1.11.7 at `/home/rander39/julia/julia-1.11.7/bin/julia`); (B) ARM-native probe → smoke → mature on the unchanged discrete-memory path; (C) fused-memory (Grace ~480 GB LPDDR5X over NVLink-C2C) effectiveness assessment with a decision gate — targeted unified-memory use only on a demonstrated capacity/bandwidth win, construction-time buffers only, never the graph-captured step (error-900 constraint, `translate_batched_cuda.jl:6535-6551`); (D) full 1,080-step acceptance under the unchanged 052 policy. | `052` harness (arch scripts) |
| [ ] | [ ] | `052b-impl-multirotor-ige-gpu.md` | **Phase A.1 COMPLETE (verified) `2026-08-26`: warm-start NaN root-caused + fixed (`FLOWPanel_warmstart.jl` §5.0); suites warmstart 153/153, simulate 199/199, solver 412/412; restart-guard removal accepted (test-covered); setup-only parses green 4r-IGE + 1r-OGE after fixing a load-killing driver ParseError at :2040; diff-overlap audit + `git diff --check` clean. ALL UNCOMMITTED; no deploys/jobs. A.2+ still gated on the 052 acceptance (now owned by 052c).** Prior: STAGED `2026-08-25`; Phase A.1 IN PROGRESS `2026-08-26` concurrently with 052/052a (shared-pfield diff adopted + warmstart fixes) — subtask checklist in `052b-plan-2026-08-26.md`. Multi-rotor (1/2/4) + in-ground-effect GPU extension serving FLOWPanel `BRAINSTORM/022` Phase 6: 10 revolutions per case for all six shapes (1/2/4 rotors × OGE/IGE) in ≤ 2 h walltime each (≤ 20 s/step; 4-rotor IGE is binding), major cost steps (wake FMM, panel passes, solve) on GPU. Phases: (A) driver conformance (022 ruling 7: single shared particle field for all rotors — per-rotor pfield at driver `:815` must be fixed FLOWPanel-side first, minimal change) + measured gap audit (wake↔bodies rectangular passes in BOTH directions — B′ is per system pair, so 5 body systems at 4r IGE ⇒ ~5 passes each way; wake→bodies direct kernel scales panels×particles ≈ 16× ⇒ over budget alone, levers = 041k F32 hatches or FMM-rectangular; bodies→wake needs once-per-step shared particle tree reuse; multi-body block-GS solve vs 052's single-body S gemv); (B) implementation with pass-by-pass parity gates; (C) 1-rev probes, all six shapes, one combined H200 sbatch chain; (D) the six 10-rev acceptance runs (= 022 Phase 6 cases, blocked on 022 Phase-3 particle-policy verdict). Discrete-memory path per 052a verdict. | `052` acceptance, `052a` verdict |
| [ ] | [ ] | `052e-impl-hybrid-wake-potential-experimental.md` | **STAGED `2026-08-27`: `HybridWakePotential` moved off the 052b critical path.** Preserve the API and host regressions; treat the existing ~10% trace and ~60% velocity errors as characterization only. Production-shaped accuracy, CUDA routing, dense Green/Hodge cost, nonlinear 414-step projection, and a separate promotion ruling are required before it can become an acceptance formulation. | `052b` host orchestration |
| [ ] | [ ] | `052c-plan-2026-08-26.md` | **IN PROGRESS `2026-08-26`: P1 wiring audit COMPLETE with a headline premise correction — stage d ran with a device-resident CuArray pfield and (evidence-consistent) the SFS-armed radix GPU FMM already serving particle self-UJ, so the superlinear `influence_pass1` term likely = radix near-field densification under the pinned triple, making P3 attack both dominant categories (see the plan's "P1 RESULTS" section; [O] items need one probe with the 052b fine timer seams). P1.5 COMPLETE (Ryan-ruled): Float32 uncompressed visual series + full-precision `_particles_fp64` checkpoint sidecar preferred by the warm-start loader (`FLOWPANEL_PARTICLE_FP64_KEEP` bounds disk); warmstart 153/153 + simulate 199/199 green. P0 sync DONE: all four kept cluster FM trees verified carrying 8479d0f2 (an earlier "missing" claim was a bad probe); consolidation decision + notebook entry pending Ryan. I3 APPROVED with a revisit-if-still-expensive caveat; I2/I4/I5/I6 unruled.** Stage-d performance overhaul + the 1,080-step acceptance inherited from `052`. Root cause (Phase-1 telemetry, 473 steps of `13484013` + source grounding): `wake_influence` = 86% of step time — `influence_pass1` is a DENSE O(N_t×N_s) `direct_rectangular!` pass with particles as targets (~N^1.6, 14.8 s/step at 230k) and `wake_sfs` is CPU-hosted `Estr_fmm!` near-field on a tree pinned at p=4/ncrit=50/theta=0.4 never tuned for plateau N (~N^1.5, 29.5 s/step). Plan: P0 consolidation inherited from `052`/`052a` closure (sync `8479d0f2` to arch trees, retire redundant checkouts, notebook entry); P1 wiring audit; P2 particle self-UJ + SFS through the device-resident SFS-armed radix GPU FMM (`targets===sources` holds — no 052b Part-1 lift), dense rectangular only for fixed-small target sets, panels-on-self AIC cached a priori; P3 retune a fixed deterministic FMM triple at plateau N; P4 36-step mature gates per change → regenerate pinned CPU reference → 1,080-step acceptance at 12 h limit. Target ≤10 s/step at N≈230k (was ~66). | `052`, `052a`, `052b` A.1 |
| [ ] | [ ] | `053-milestone-review-production-integration.md` | Milestone review of `046`–`052` (contracts, defaults changed only with user approval, test suites in all three repos, the 018 speedup verdict, punch list for the Peak Efficiency Phase). | `046`, `047`, `048`, `049`, `050`, `051`, `052`, `052a` |

This phase does not reopen any earlier gate.

## Peak Efficiency Phase

Staged by user direction on `2026-08-20`, after the Production Integration
Phase and still BEFORE the Multi-GPU Scaling Phase: implement as many small
optimizations as possible to bring the single-GPU code as near peak hardware
efficiency as possible before scaling out.

**Efficiency-gap analysis (2026-08-20 discussion).** The FMM's effective
efficiency sits **~10× below the brute-force kernel's measured 38%-of-peak**
(`041k`: 3.3e11 pairs/s FP32 FMA-bound with the opt levers). The losses live
in M2L scatter/gather, small memory-bound GEMMs, the ~50 µs/window launch
floors (`027`), the ~0.87 ms per-GPU control floor (`028`/`029`), and
refresh — **NOT in P2P**, which is already partitioned
singular/regularized + `037f` fp32 + `041e` target-owned CSR. The governing
cost model is T(K) ≈ αKN + βN/K + floors with the GPU optimum at K=256
(`027`): **every kernel cheapening must be followed by leaf-size retuning to
harvest** — cheapening α lets the autotuner raise K*. `028`'s finding that
leaf M2L is load-bound, not atomic-bound, directs the scatter/gather effort
to the load side.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [ ] | [ ] | `054-impl-nearfield-opt-lever-port.md` | Port the `041k` opt levers into the production partitioned U-list kernels (both UJ and the `048` ζ/SFS pass, which was written lever-ready): far-field singular switch + fast/libdevice transcendentals for the regularized remainder, 2–4-target register blocking, then autotune/leaf-size re-sweep to move K*. Same-job A/B on the 018 operating point + standard cases; `037f`-style promotion gate (≥5% end-to-end on a material case, ≤3% regression elsewhere, 1e-3 accuracy, P=4 both precisions). `041k` measured ceilings recorded in the doc (3.3e11 pairs/s ≈ 38% FP32 FMA peak; 1.6–1.7× from these levers on all-pairs F32; F64 opt inert below n≈3e4). | `053` |
| [ ] | [ ] | `055-impl-scatter-gather-efficiency.md` | The scatter/gather program: exclusive-ownership (gather) accumulation for M2L-accumulate and L2B; read-side densification of ragged M2L gathers beyond the `023d` precomputed-y tables; deeper whole-pass fusion + graph consolidation to shave the ~50 µs windows and ~0.87 ms control floor; `028` finding (leaf M2L load-bound, not atomic-bound) directs effort to the load side; bounded persistent-mega-kernel spike (explicitly time-boxed, kill if it fights graph capture). Each lever same-job A/B'd, individually promoted/rejected. | `053`, `054` |
| [ ] | [ ] | `056-benchmark-roofline-accounting.md` | Per-stage roofline accounting on H200 (achieved vs attainable flops/bandwidth per stage at n = 1e5/2.1e5/1e6 + the 018 operating point), the "fraction of peak" scoreboard before/after `054`–`055`, re-run of the `052` driver to record the end-to-end delta, and a written statement of what remains (with est. ceilings) feeding `043`'s multi-GPU targets. | `054`, `055` |
| [ ] | [ ] | `057-milestone-review-peak-efficiency.md` | Milestone review of `054`–`056`; hands the single-GPU baseline to the Multi-GPU phase. | `054`, `055`, `056` |

This phase does not reopen any earlier gate.

## Multi-GPU Scaling Phase

Staged by user direction on `2026-08-11` at the closure of `029`. The `029`
P2 prototype falsified mirrored-tree multi-GPU scaling structurally
(replicated refresh/upward/finalize bound the 2-GPU wall at ~3.8 ms ≈ 61%
efficiency even with zero communication) while validating the communication
mechanism itself (bitwise-lockstep work-list slicing + allreduce, 0.190 ms
comm+orch at n=1e6; `cuMemPoolSetAccess` pool P2P grant for 234 GB/s
NVLink). This phase pursues the successor: a **partitioned-tree
decomposition** — costed Morton-range ownership, local subtrees below a
split level with replicated+allreduced coarse levels, per-level multipole
halo exchange inside captured graphs, body halos for the boundary
nearfield, and ownership migration on occupancy epochs.

**Targets (user direction `2026-08-11`): `<= 1 ms` per resident step for the
fixed 1M-body literature-`P=4` workload on up to 8 H200s is the goal;
`<= 2 ms` still counts as a win.** The recorded feasibility bound: perfect
8-way compute splitting (~0.58 ms) plus the measured ~0.87 ms n-independent
per-GPU control floor plus ~0.2 ms comm lands at ~1.0–1.2 ms, so the 1 ms
goal additionally requires shaving the per-GPU launch floor. `029`'s
accuracy, recurring-cost, and independent-reproduction rules carry over
unchanged.

The phase is gated behind the Adaptive Octree Phase and its S2L optimization
follow-on (`042a`) by user direction ("after the adaptive tree and other
improvements"): the adaptive
octree changes tree construction, occupancy lookup, and list generation —
the very surfaces a partitioned tree must split — so partitioning is derived
once against the final tree machinery rather than twice.

| Done | Approved | Task | Summary | Blocking |
| --- | --- | --- | --- | --- |
| [ ] | [ ] | `043-theory-partitioned-multigpu-decomposition.md` | Derive the partitioned-tree decomposition: costed ownership, split-level scheme, per-level halo sets with exact-once coverage proof, migration policy, graph-capture/comm plan reusing the validated `029` P2 exchange, and a measured-floor-calibrated cost model. Kill-switch acceptance gate: modeled 8-GPU step `<= 2 ms` with an identified path to `<= 1 ms`, else recommend not proceeding. Derivation row: `theory/`, `scripts/`, `data/` only. | `042a`, `053`, `057` |
| [ ] | [ ] | `044-impl-partitioned-multigpu-lifecycle.md` | Implement the `043` design at 2 GPUs on the device-resident lifecycle: partitioned refresh/upward/downward with graph-captured per-level halo exchange, body halos, epoch-based ownership migration, distributed correctness gates extending `test/cuda_radix_twogpu_test.jl`. Gate: 2-GPU efficiency `>= 75%` vs the then-current single-GPU record at unchanged accuracy. | `043` |
| [ ] | [ ] | `045-benchmark-multigpu-highscore.md` | Scale to 4/8 GPUs: scaling ladder, per-GPU launch-floor reduction as a measured lever if the `<= 1 ms` goal demands it, independent reproduction, final leaderboard and verdict (`<= 1 ms` goal / `<= 2 ms` win), results recorded as `019a`/`029` addendum notes. | `044` |

This phase does not reopen any earlier gate.

## Future Dispatch Cleanup Notes

For new GPU/device-path work, prefer dispatch on source, destination, and policy
objects over boolean residency flags. Item `022` should use destination-driven
output finalization: resident CUDA evaluation writes `state.output` on device, and
copy/writeback dispatch decides whether the destination is a host array, device
array, or device-native target system.

Cleanup disposition (047 remediation; compatibility shims remain for downstream callers):

- `allow_host_bodies` is absent; source residency dispatches through
  `HostResident` / `DeviceResident` and source-buffer traits.
- The legacy nearfield path accepts `HostNearfield` / `DeviceNearfield`.
- Production tree construction uses `SourceTree` / `TargetTree` roles.
- Production radix generation uses `RadixRouteSelection` policy objects.
