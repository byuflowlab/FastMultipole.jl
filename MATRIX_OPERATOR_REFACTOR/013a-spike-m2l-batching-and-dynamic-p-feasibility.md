# 013a Spike M2L Batching And Dynamic-P Feasibility

## Objective

Design spike (investigation, no production hot-path change) that resolves two
questions early enough to shape downstream tasks:

1. Which M2L batching candidate the `014` composition should be built toward, so the
   full pipeline is composed as separable stages that the later `015` decision can
   fold without a rewrite.
2. Whether the legacy per-interaction dynamic-`P` machinery (`get_P`,
   `predict_error`, truncation) can be ported onto the new operators, so `017`
   storage and the `023` integration are designed with the answer known rather than
   discovering it at `019a`.

> **Superseding roadmap note (2026-06-19).** This spike remains useful background
> evidence, but the later M2L Operator Variant Roadmap Update narrows the near-term
> product/benchmark scope. Downstream tasks should use `013` for the materialized
> `Ts(theta)` building-block path, `013b` for explicit fixed `Z/S/Z/S` primitive
> stages, `014` for swappable `MaterializedYRotationM2L` and `FactoredRotationM2L`
> whole-M2L interfaces, and `015` to compare only those two near-term variants.
> References below to folded dense/no-`Ts` y-rotation paths, broad batching
> candidate sets, z-translation cache-policy benchmarking, or real-basis execution
> are historical recommendations and are now deferred to later
> implementation/performance work unless a later task explicitly reactivates them.

## Dependencies

- `008c-implementation-performance-baseline.md` (prototype batching data)
- `008d-theory-dynamic-p-error-m2l-integration.md` (constant-`P` vs dynamic-`P` paths)
- `008b-implementation-replan.md`
- `010-impl-z-rotation-operators.md`
- `011-impl-m2l-z-translation-blocks.md`
- `012-impl-lamb-helmholtz-operators.md`
- `013-impl-axis-swap-operators.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- `008c` prototype harness and its recorded data

## Artifacts or Production Surface

- Spike notes and any throwaway benchmark scripts under
  `MATRIX_OPERATOR_REFACTOR/scripts/` and data under `MATRIX_OPERATOR_REFACTOR/data/`.
- No production hot-path change.

## Deliverables

- A recommended target batching structure for `014` (with rationale from `008c`
  prototype evidence), framed so the final benchmark-gated `015` decision remains a
  composition choice over separable stages.
- A feasibility finding on porting dynamic-`P` onto the new operators: feasible /
  feasible-with-constraints / not feasible, with the constraints it imposes on `017`
  buffers and `023` integration. The final go/no-go remains at `019a`.

## Verification

Spike conclusions must cite the `008c` prototype data and the approved operator
theory. Record the analysis and the recommendations. No parity gate (no production
change).

## Spike Notes (completion pass)

Investigation only. No `src/` change; no production hot-path touched. Conclusions
cite the approved `008c` baseline data
(`data/impl_performance_baseline/{m13h-1-1,m12-2-5}/`) and the approved operator
theory (`theory/full-m2l-composition.md`, `theory/axis-swap-conventions.md`,
`theory/radix-interaction-list.md`, `theory/constant-p-error-stencil.md`). No new
benchmark script was required: the `008c` prototype already measured the four
forms (`recurrence`, `dense`, `dense_packed`, `compiled_block_loop`) per stage
across single-thread CPU (Xeon BLAS=1), multithread CPU (EPYC BLAS=72), and GPU
(H200 custom kernel), which is exactly the evidence both deliverables need.

### The two structural facts the spike turns on

1. **The radix M2L list is already a per-offset-class batch.**
   `theory/radix-interaction-list.md` builds `Batch(d, targets, sources)`: every
   pair in a batch shares the integer offset `d`, hence the same separation `c`,
   the **same forward/back rotation angles** `(phi, theta)`, and the **same
   fixed-`m` z-translation block** at the constant `P`. This batch is the natural
   batched-GEMM unit; the operator is loaded once per offset class and applied
   across the whole batch.

2. **The M2L chain (`theory/full-m2l-composition.md` §Operator Chain) splits into
   exactly the stage classes `008c` priced separately:**

   ```text
   M1 = Z_phi M                  # diagonal z-rotation        (angle = const per d)
   M2 = S_M Z_theta S_M^-1 M1    # axis swap: fixed S + diagonal Z_theta
   L1 = K_z(r) M2                # fixed-m z-translation block (block = const per d)
   L2 = H_L(r) L1               # optional Val(true) LH coupling (structured)
   L3 = S_L Z_theta S_L^-1 L2    # axis swap back
   L  += Z_phi^-1 L3             # inverse diagonal z-rotation, accumulate
   ```

   In every stage, the only angle/`r` variance is carried by the **diagonal
   z-rotations** (`Z_phi`, `Z_theta`) and the **per-`d` z-translation block**
   `K_z(r)`. The dense axis-swap `S_M`/`S_L` is the **fixed pi/2 swap** (theory
   `004`: angle-independent `S_n` blocks) — identical for every interaction and
   every offset class.

## Deliverable 1 — recommended target batching structure for `014`

**Historical recommendation, superseded in scope by the 2026-06-19 roadmap
update:** build `014` toward the variant-B stage decomposition, batched over the
per-offset-class radix `Batch(d, …)`, with each stage a separable,
individually-callable operator. `014` now owns two swappable whole-M2L interfaces
instead: `MaterializedYRotationM2L` and `FactoredRotationM2L`. Rationale, stage by
stage, from `008c`:

| stage | form (from `008c`) | what is shared across the batch |
|---|---|---|
| `Z_phi`, `Z_theta` (z-rotation) | fused **diagonal** scaling, no GEMM (cheapest stage, `~2.4e-7`/exp P=20) | the `(phi,theta)` angle is **constant within an offset class** → one diagonal reused across the whole batch |
| `S_M`, `S_L` (axis swap) | **dense per-block batched GEMM** (the dominant win: 31× unbatched / **118× batched** single-thread Xeon; 65× multithread; dominant GPU stage) | the fixed pi/2 swap `S` is **identical across *all* offset classes and the whole batch** — precompute once, reuse forever |
| `K_z(r)` (fixed-`m` z-translation) | **dense per-block batched GEMM** (10–22× batched; best-scaling GPU stage `7.2e-9`/exp) + `compiled_block_loop` no-BLAS/unbatched fallback | the block depends only on `d` → **one block per offset class**, shared across that batch |
| `H_L(r)` (Lamb-Helmholtz) | **structured two-channel banded** update, *not* densified | channel coupling only (same-degree + one neighbor degree, theory `003`/`008h`) |

**Why factored stages remain important, while still benchmarking the materialized
path.** The materialized path
(`013`, reconstructed `Ts(theta)`) collapses `S · Z_theta · S^-1` back into a
single angle-dependent dense Wigner `Ts`. In the radix path this is **not** inherently
per-pair: every pair in `Batch(d, ...)` shares the same translation vector, so one
assembled `Ts` can be reused across that offset-class batch. The limitation is
that the dense angle-dependent operator is not globally shared across different
offset classes, and it hides the separable `S` / diagonal `Z_theta` structure.
`FactoredRotationM2L` keeps `S` fixed and globally batch-shareable and
pushes angle variance into diagonal `Z` stages. Those z-rotation phases/operators
may themselves be computed on demand or cached later; the 2026-06-19 update defers
alternate z-rotation and z-translation cache/scaling policies out of `015`.

**Batch unit and layout (forced by `008c`).**
- Unit: the per-offset-class `Batch(d, …)`. Gather the source-cell expansion
  columns (via `007`'s `expansion_index` per cell) **contiguous and GEMM-ready**;
  scatter target columns the same way. `008c` measured a **1.7–3.6× gather/scatter
  penalty** (`dense_packed` vs `dense`) when operands are copied around block
  GEMMs — so the buffer must let each block read/write in place or by strides, not
  via `copyto!`. This *supports* the `007` contiguous layout; it is a layout
  constraint, not a FLOP one.
- One batched GEMM per block: per degree `n` for the swap, per order `m` for the
  z-translation, over a `block × (channel·batch_width)` matrix.
- **Batch width is a tunable, not a constant** (`008c`): single-thread CPU peaks
  near batch ≈ 64 (cache eviction after), multithread keeps improving to ≈ 32768,
  GPU device-resident keeps improving to 262144. `014` must size by max `P` and
  let `015`/`019` pick the width per regime.
- **Parallelism**: across offset-class batches with **single-thread BLAS per
  thread** (pin `OPENBLAS_NUM_THREADS=1` at process start — runtime
  `set_num_threads` was unreliable in `008c`). Never call multithreaded BLAS on
  one operator block: `008c` showed EPYC batch-1 `dense` *loses* to the recurrence
  for m2m/l2l.

**Historical `015` framing, now narrowed.** The stages are separable and
individually callable, so the following candidates were originally identified as
possible composition choices. The 2026-06-19 update narrows `015` to the two
near-term variants only and defers the rest:
- per-block batched GEMM **per offset class** (the default above);
- **global** stage batching of the shared `S` swap across all offset classes, plus
  grouped diagonal `Z` batches keyed by offset class or shared direction;
- materialized-path per-offset-class assembled `Ts` or folded dense operators
  (shared within one translation-vector class, not globally across offset classes);
- factored-path z-rotation reuse at several granularities: on-the-fly recurrence,
  cached per offset class, cached per shared direction, and cached per level where
  the angle convention is identical;
- z-translation reuse separated into core block structure and distance-dependent
  pre/post scaling; shared directions do not by themselves imply shared distance
  scaling, so reuse must be keyed by the physical `r` / offset norm / level as
  appropriate;
- a **merged padded single GEMM** over the block-diagonal operator (explicit
  off-block zeros) — `008c` explicitly did **not** measure this; it is a `015`
  call;
- **strided-batched GEMM** where the per-`d` `K_z` blocks differ but share shape;
- the φ/χ **channel-merge** special case of the merged-GEMM trade-off.

Historically, this implied that `014` should expose each of `Z`, `S`, `K_z`, `H_L`
as a stage that accepts a batch and a width. Under the superseding roadmap update,
`014` instead exposes the two whole-M2L interfaces first, and `024` makes the
end-to-end call between `MaterializedYRotationM2L` and `FactoredRotationM2L`.

**GPU note (hard `008c` constraint, inherited by `022`).** The per-offset-class
batch structure is exactly what device residency needs: upload expansions once,
run **every** offset-class batch on device (single fused launch per block beats
many per-block launches below batch ~10⁵), download once. Per-operator
host/device transfer floors GPU throughput at CPU-dense levels (`~6e-7`/exp
Float64) and must be avoided. The factored path is the natural device-resident
form (fixed `S` shared, diagonals cheap). Float64 stays default (Float32 only
~1.2× on H200).

## Deliverable 2 — feasibility of porting dynamic-`P` onto the new operators

**Finding: feasible-with-constraints, and the constraints are path-dependent.**
Per-pair dynamic-`P` is feasible on the legacy octree path only (at the
*unbatched* per-pair speedup tier). On the first-pass radix batched path,
per-pair dynamic-`P` is **not** in scope: `008d` adopts one constant `P` and moves
error control into the stencil. A coarser per-offset-class / per-level `P`
selection based on cheap classical relative-error metrics is a plausible future
optimization, but it is not the approved first-pass radix behavior. **The final
go/no-go remains at `019a`.**

**Why per-pair dynamic-`P` and the radix batch are structurally incompatible.**
`get_P` (`src/dynamic_expansion_order.jl:7`) is **per-pair and data-dependent**:
it reads each source branch's actual monopole/dipole coefficients
(`q_monopole`, `dipole_from_multipole`) and the actual pairwise geometry
(`r_min,r_max,ρ_min,ρ_max,ΔC2` via `get_r_ρ`) and returns a per-interaction
`P ≤ PMAX`. Two pairs sharing one offset class `d` will in general get **different
`P`** (different source strengths/positions). But the entire radix batching win
depends on **one shared operator at one constant `P` per offset class**
(`theory/radix-interaction-list.md` §Batched M2L List Layout). Per-pair `P` inside
a batch would require one operator size per pair → the batch dissolves back into
per-pair GEMMs, forfeiting the `008c` 10–118× batch results. This is not a tuning
gap; it is the reason `theory/constant-p-error-stencil.md` **deliberately moves
error control out of `get_P` and into the translation-invariant interaction-list
stencil** (constant `P`, conservative analytic bound `B(P,d,A)` per offset class):
"The radix path never calls `get_P` or `predict_error` during M2L batching."

**Path-by-path feasibility:**

- **Legacy octree path — FEASIBLE-WITH-CONSTRAINTS (unbatched tier).** `008d`
  keeps `get_P`/`predict_error` here unchanged. The new operators can serve this
  path as **per-interaction** operators sized at that pair's `P`: a pair-specific
  `Ts` / operator apply is the natural fit, and `008c` shows it
  still wins **even unbatched** (axis-swap 31× single-thread at batch 1;
  z-translations 2–4.7× with tuned BLAS, with the `compiled_block_loop` fallback
  for the multithread-BLAS batch-1 case that loses). So dynamic-`P` + new
  operators is feasible on the legacy path but realizes only the **per-pair**
  speedup tier, not the batch tier. Constraint: each call sizes its operator
  blocks to the pair's `P`, so the operator stages must accept a per-call `P ≤
  PMAX` (the `009` order accessors already provide this).

- **Radix path — first pass is constant `P`; dynamic/coarse-`P` is exploratory.**
  Keep error control in the `008d` stencil for the approved path. A limited
  dynamic granularity may be feasible without breaking batching if `P` varies
  **by offset class or by grid level** (farther/coarser classes may accept a
  smaller `P`; the stencil's accepted sets are monotone in `P`), so one operator
  size is still shared within a batch. That is a candidate for later benchmark and
  accuracy work, not a requirement for `014`/`021`. Full per-pair `P` on the radix
  path remains a no for the batched/device-resident design.

**Constraints imposed on `017` buffers:**
- Buffers size to **max `P` (PMAX)** but must expose a **truncation length** as a
  typed view, contiguous/GEMM-ready at that length:
  - legacy path: a **per-interaction** truncated view sized by the active
    basis-specific `basis_dof(P)` (compressed complex: `2*Ncomplex(P)` real lanes;
    real basis: `(P+1)^2`) of the `basis_dof(PMAX)` storage, since `P` varies per
    pair;
  - radix path: a **per-batch** (per-offset-class, or per-level) uniform
    truncation length carried as batch metadata, so one length sizes the whole
    batch's block GEMM.
- Keep the `008c` storage budgets: drop the dead `χ` channel for `Val(false)`
  (halves expansion storage), and ragged `P_chi = P+1` per `008h`. A per-pair
  truncated view must respect the ragged φ/χ split.
- Aliasing per `007`: overwrite stages need distinct src/dst/scratch; only the
  final inverse z-rotation accumulates — unchanged by truncation.

**Constraints imposed on `023` integration:**
- Dispatch (basis-type / `Cache` flag) must route: **legacy octree → per-pair
  dynamic-`P` operators (unbatched tier, keeps `get_P`/`predict_error`)**; **radix
  → constant-`P` (optionally per-level `P`) batched operators that never call
  `get_P`/`predict_error`** (per `008d`). The legacy path stays the default
  fallback.
- `023` must **not** attempt to make a radix batch honor per-pair `P`; that is the
  explicit non-goal. The two error-control regimes coexist behind dispatch, not
  merged.

**Recommendation to carry into `019a` (final go/no-go owner):**
- Adopt per-pair dynamic-`P` **only** as the legacy path's operator mode
  (unbatched tier), explicitly flagged as forgoing the batch win.
- On the radix path, do **not** port per-pair dynamic-`P`; if dynamic granularity
  is wanted there, add **per-level (or per-offset-class) constant-`P`** selection,
  which preserves batching and reuses the `008d` stencil monotonicity.

### Verification

No parity gate (no production change). Conclusions are traceable to:
- `008c` measured forms/numbers (axis-swap 31×/118×; z-translation 10–22×; GPU
  device-resident `7.2e-9`/exp and the `with_transfer` floor; gather/scatter
  1.7–3.6×; batch-width regime peaks);
- `theory/full-m2l-composition.md` (stage chain), `theory/axis-swap-conventions.md`
  (angle-independent `S`), `theory/radix-interaction-list.md` (per-offset-class
  batch), `theory/constant-p-error-stencil.md` (error control out of `get_P`);
- `src/dynamic_expansion_order.jl:7` (`get_P` per-pair, data-dependent inputs).

## Approval Notes

Clear-context review approved.

Reviewed under the `START_HERE.md` clear-context protocol: this file, the
listed artifact surface, and the verification notes. The spike satisfies both
deliverables: it recommends a separable per-offset-class batching structure for
`014` while leaving `015` as the benchmark-gated composition decision, and it
records a path-dependent `feasible-with-constraints` finding for dynamic-`P`
porting with explicit `017` buffer and `023` integration consequences. No
production parity gate is required because the task is investigation-only.
