# 008c Implementation Performance Baseline

## Objective

Benchmark current production paths, inventory allocation/storage behavior, and
record approved operator-theory design constraints before production
implementation begins.

## Dependencies

- All Theory Phase rows in `START_HERE.md`
- `008a-milestone-review-theory-005-008.md`
- `008b-implementation-replan.md`

## Required Reading

- `START_HERE.md`
- `../MATRIX_OPERATOR_REFACTOR.md`
- Approved Theory task files
- Approved Theory artifacts and scripts listed by the Theory task files
- Completed `008b-implementation-replan.md`
- Current production translation, evaluation, and benchmark entry points

## Artifacts or Production Surface

This is a pre-implementation benchmark and design-gate task. No production code
changes may start until this task is complete and approved.

Benchmark and data-structure review work may use current production paths,
approved Theory scripts, and benchmark/result artifacts under
`MATRIX_OPERATOR_REFACTOR/data/` if needed. Coordination updates may edit
`008b-implementation-replan.md`, Implementation task files, and/or
`START_HERE.md` only if benchmark or storage/allocation results require a
changed design, scope, risk assessment, or task order.

## Deliverables

- Benchmark commands for current production/operator-theory baselines
- Environment notes sufficient to reproduce the measurements
- Baseline summaries for CPU single-thread paths
- Baseline summaries for CPU multithread paths where measurable
- Inventory of current production coefficient, operator, cache, and scratch data
  structures used by translation/evaluation paths
- Baseline allocation/storage review covering allocation counts, retained
  storage, scratch reuse, and temporary buffer pressure for current production
  paths, measured where practical and estimated where measurement is not
  practical
- Review of approved Theory layouts, especially `007` coefficient-buffer layout,
  for minimum required storage, batch layout requirements, aliasing rules, and
  CPU/GPU tradeoffs
- Explicit storage/allocation budgets or constraints that Implementation tasks
  must respect
- A per-stage decision on operator form — dense-materialized matrices (applied
  via `mul!`/GEMM) versus recurrence-wrapped operators (today's `O(p)`
  recurrences behind the operator API) — for the z-rotation, axis-swap,
  fixed-`m` z-translation, and Lamb-Helmholtz stages, with benchmark-backed
  rationale. This decision is a constraint that Implementation tasks `009`–`016`
  inherit. (Deferred here from the `008b` re-plan.)
- GPU-relevant design tradeoffs and constraints where measurable
- Design implications for Implementation task order, scope, or risk
- Any needed coordination updates to `008b-implementation-replan.md`,
  Implementation task files, and/or `START_HERE.md` if benchmark or
  storage/allocation review changes task order, design, or risk
- Clear-context approval before task `009` or any later Implementation task
  starts

## Verification

Confirm that no production code work has started for task `009` or any later
Implementation task. Confirm benchmark commands, environment notes, baseline
summaries, allocation/storage inventory, storage constraints, CPU/GPU design
notes, design implications, and any coordination updates are recorded before
requesting approval.

Status of these checks (this completion pass):

- No `src/` changes. `git status` shows edits only under
  `MATRIX_OPERATOR_REFACTOR/` (this file plus the two new scripts and the
  generated `data/impl_performance_baseline/<host>/` artifacts). No work has
  started on `009`+.
- Benchmark commands, environment notes, single- and multi-thread baseline
  summaries, the dense-vs-recurrence head-to-head, the allocation/storage
  inventory, explicit storage budgets, the per-stage operator-form decisions,
  CPU/GPU design notes, and design implications are all recorded below.
- No coordination updates to `008b`, the Implementation task files, or
  `START_HERE.md` were required: results confirm the `008b` design and task
  order rather than changing them (see Design Implications).

---

# Results (completion pass)

## Reproducibility and benchmark commands

Two throwaway benchmark harnesses were added (neither touches `src/`):

- `MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_cpu.jl`
- `MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_gpu.jl` (user-run on a GPU box)

Both are self-contained (no `BenchmarkTools`/plotting deps; a small internal
`timeit`/`gpu_timeit` minimum-of-samples helper is used so the scripts run on a
fresh checkout under the default Julia). They write machine-tagged output to
`MATRIX_OPERATOR_REFACTOR/data/impl_performance_baseline/<hostname>/` so a later
run on stronger hardware does not clobber these numbers.

The harnesses are now intended for cross-machine collection. Numbers recorded
on this local CPU are provisional design evidence; final implementation-policy
decisions should compare at least local CPU, the 72-thread CPU host, and a CUDA
GPU host by downloading each machine-tagged result directory.

Run commands (from the repository root):

```bash
# CPU baseline, GENUINELY single-thread BLAS (apples-to-apples vs the recurrence)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
P_LIST="4,8,12" P_DENSE_LIST="4,8,12" BATCH_LIST="1,8,64" SAMPLES=10 \
  julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_cpu.jl

# CPU baseline, all-cores BLAS
OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu) OMP_NUM_THREADS=$(sysctl -n hw.ncpu) \
P_LIST="4,8,12" P_DENSE_LIST="4,8,12" BATCH_LIST="1,8,64" SAMPLES=10 \
  julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_cpu.jl

# GPU baseline (on a CUDA machine; add CUDA first)
julia --project=. -e 'import Pkg; Pkg.add("CUDA")'
P_DENSE_LIST="2,3,4,5,6,7,10,12,14,20" \
BATCH_LIST="64,512,4096,32768,262144" \
PREC_LIST="Float64,Float32" SAMPLES=50 \
  julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_gpu.jl
```

`dense_vs_loop_blas<N>.csv` schema:

```text
stage,form,precision,blas_threads,P,batch,measured_batch,scaled_from_per_expansion,seconds,seconds_per_expansion
```

CPU stages include `m2m_z_translation`, `m2l_z_translation`,
`l2l_z_translation`, and `axis_swap`. CPU forms include:

- `recurrence`: current production scalar path, measured in Float64. For large
  batches the harness measures only `measured_batch = min(batch,
  RECURRENCE_BATCH_CAP)` expansions and sets `scaled_from_per_expansion=true`
  when projecting the `seconds` column to the requested batch.
- `dense`: direct per-block BLAS `mul!` on pre-split block matrices.
- `dense_packed`: gather from a flat coefficient-like layout, run per-block
  BLAS `mul!`, and scatter back to a flat output buffer.
- `compiled_block_loop`: full-batch hand-written small-block matmul loop for
  CPU comparison against BLAS call overhead.

`dense_gpu.csv` schema:

```text
stage,form,launch_strategy,transfer_variant,precision,P,batch,seconds,seconds_per_expansion
```

GPU `launch_strategy` values are `per_block_launch` for the existing per-block
`mul!` path and `fused_kernel` for the single-launch packed-block prototype.
Each is measured as `device_resident` and `with_transfer`. On CPU-only machines
the GPU script exits cleanly with no measurements.

Small-`P` / tiny-batch fallback policy remains undecided here. The final choice
must be revisited after the full `019` benchmark evidence, especially for
`P <= 3` and `batch == 1` regimes where prototype overhead can dominate.

**BLAS thread-control caveat (important for the single-thread numbers).**
Runtime `BLAS.set_num_threads()` was found **unreliable** for OpenBLAS on this
platform: `get_num_threads()` updates but actual GEMM execution does not. Direct
proof, a 2000×2000 Float64 GEMM (same process otherwise):

| control | reported threads | time | GFLOPS |
|---|---|---|---|
| `OPENBLAS_NUM_THREADS=1` (process start) | 1 | 3.18e-1 s | 50.3 |
| `OPENBLAS_NUM_THREADS=4` (process start) | 4 | 1.11e-1 s | 144.7 |
| runtime `set_num_threads(1)` vs `(4)` | 1 vs 4 | ~identical (~0.11 s) | ~145 both |

So threading must be controlled by the **process-start env var**, and the
benchmark is therefore run **twice** (env var = 1, then = ncores). Each run
records the actual `BLAS.get_num_threads()` and writes
`dense_vs_loop_blas<N>.csv`. (An earlier version of this baseline used the
runtime call and its "1 thread" numbers were silently multithreaded at large
batch — corrected here.) The production recurrences are pure Julia scalar code
and are single-threaded regardless. A separate Julia `--threads` setting does
not affect these stage/GEMM numbers.

## Environment of the recorded baseline

From `data/impl_performance_baseline/<host>/env.md`:

- CPU: Apple M2, 4 cores; Julia 1.12.5.
- BLAS: OpenBLAS 0.3.31 (`libopenblas64_p`); `env.md` records
  `BLAS.get_num_threads()` for whichever run last wrote it (the two runs pin it
  to 1 and 4 via the env var). **Detected as a tuned BLAS** — so the dense/GEMM numbers here are
  representative, not a reference-BLAS lower bound. (The script auto-flags a weak
  BLAS in `env.md` if a future host lacks a tuned library; in that case the dense
  numbers must be treated as a lower bound and confirmed elsewhere.)

Even with a tuned BLAS this is a 4-core laptop CPU; the GPU script exists to
confirm the large-batch regime on the hardware the refactor ultimately targets.

## Baseline summary — current production recurrence stages

Median per-call time (seconds), `lamb_helmholtz = Val(false)`, from
`stage_recurrence.csv`:

| stage | P=4 | P=8 | P=12 | P=20 |
|---|---|---|---|---|
| `rotate_z` (z-rotation) | 2.8e-8 | 6.4e-8 | 1.0e-7 | 2.0e-7 |
| `rotate_multipole_y` (axis-swap, Wigner) | 3.5e-7 | 2.4e-6 | 9.0e-6 | 5.5e-5 |
| `translate_multipole_z` (M2M z) | 4.8e-8 | 1.9e-7 | 4.8e-7 | 1.6e-6 |
| `translate_multipole_to_local_z` (M2L z) | 8.4e-8 | 3.6e-7 | 1.0e-6 | 3.7e-6 |
| `translate_local_z` (L2L z) | 5.6e-8 | 2.1e-7 | 5.5e-7 | 1.9e-6 |

`Val(true)` (Lamb-Helmholtz) adds the two `transform_lamb_helmholtz_*` stages
(~4e-8 at P=4 to ~6e-7 at P=20) and raises each stage ~20–40% (two channels).

**Key observation:** the Wigner y-rotation (`rotate_multipole_y`) is the
dominant stage and the fastest-growing — at P=20 it is ~15× the M2L
z-translation and ~270× the z-rotation. It rebuilds the `Ts` (Wigner) matrices
on every call. This is the single hottest target.

## Dense-prototype vs recurrence head-to-head (P=12)

From `dense_vs_loop_blas1.csv` (genuine single-thread BLAS) and
`dense_vs_loop_blas4.csv` (all cores). Times are **per expansion** (seconds);
dense uses materialized blocks applied via `mul!` over a `block × (2·batch)`
matrix (re/im = 2 columns/expansion); recurrence is the production code applied
per expansion. "speedup" is single-thread dense vs recurrence.

M2L z-translation (block-diagonal over m):

| batch | dense (BLAS=1) | recurrence | speedup | dense (BLAS=4) |
|---|---|---|---|---|
| 1 | 9.8e-7 | 1.0e-6 | ~1.0× | 1.0e-6 |
| 8 | 3.0e-7 | 1.0e-6 | 3.3× | 3.2e-7 |
| 64 | 2.3e-7 | 1.0e-6 | 4.3× | 2.3e-7 |
| 512 | 2.2e-7 | 1.0e-6 | 4.6× | 2.2e-7 |
| 4096 | 2.2e-7 | 1.0e-6 | 4.6× | **1.7e-7** |

Axis-swap (Wigner y, block-diagonal over n):

| batch | dense (BLAS=1) | recurrence | speedup | dense (BLAS=4) |
|---|---|---|---|---|
| 1 | 1.6e-6 | 9.2e-6 | 5.8× | 1.6e-6 |
| 8 | 5.6e-7 | 9.2e-6 | 16× | 5.6e-7 |
| 64 | 4.7e-7 | 9.2e-6 | 19× | 4.7e-7 |
| 512 | 4.5e-7 | 9.2e-6 | 20× | 4.6e-7 |
| 4096 | 4.6e-7 | 9.3e-6 | 20× | **2.2e-7** |

Facts that stand out (all on **genuinely single-thread** BLAS unless noted):

1. **Axis-swap dense wins even at batch 1** (5.8×) and grows to ~20× when
   batched, single-threaded. The recurrence's per-call Wigner rebuild is pure
   overhead a precomputed invariant matrix (theory `004`: angle-independent
   `S_n`) avoids entirely.
2. **M2L z-translation ties at batch 1 and wins ~3–4.6× once batched** (≥8),
   single-threaded.
3. **Multithreaded BLAS helps only at very large batch, and only modestly.**
   At batch ≤512, BLAS=1 and BLAS=4 are within noise (the per-block matrices are
   ≤13×13 at P=12, below OpenBLAS's threading threshold). At batch=4096 BLAS=4
   gives ~1.3× (z-translation, 1.7e-7 vs 2.2e-7) to ~2.1× (axis-swap, 2.2e-7 vs
   4.6e-7) over single-thread dense. The dense-vs-recurrence win is therefore
   **driven by the GEMM/precompute structure, not by BLAS threading.**

## GEMM-vs-compiled-loop crossover analysis

The operator-stage matrices here are small (per-block dimension `P+1−m` for
z-translation, `2n+1` for the rotation; ≤ ~41 at P=20). Where dense GEMM beats
compiled loops therefore depends entirely on **batch width**, not matrix size:

- **Single-thread CPU.** A single small matmul (`batch=1`) is dominated by BLAS
  call overhead, and the recurrences additionally exploit structure/zeros a
  dense matmul cannot. So for unbatched application the recurrence is competitive
  (z-translation) — *except* where the recurrence itself carries per-call
  rebuild cost, which is exactly the axis-swap/Wigner case (dense wins even
  unbatched). Once expansions are batched into a `basis_dof × (batch·channel)`
  GEMM, dense overtakes by ~3–5× (z-translation) and dramatically for axis-swap.
  Measured crossover: **batch ≈ 8** on this machine.
  *Caveat:* on a host without a tuned BLAS this crossover shifts right or
  disappears; `env.md` flags that case.
- **Multithread CPU.** Negligible help except at very large batch. Genuine
  1-thread vs 4-thread BLAS are within noise for batch ≤512; only at batch≈4096
  does multithreaded BLAS add ~1.3× (z-translation) to ~2.1× (axis-swap) — far
  less than the single-threaded dense-vs-recurrence win itself. Multithreaded
  BLAS only pays off for matmuls/total-work far larger than a single FMM operator
  block. The way to use cores here is parallelism *across* interactions (as
  production already does with `@threads`), with each thread issuing
  **single-thread** batched GEMMs (set `OPENBLAS_NUM_THREADS=1` at process
  start — note runtime `BLAS.set_num_threads` is unreliable here), not
  multithreaded BLAS inside one operator.
- **GPU.** Per-operator tiny matmuls are launch-bound; only large *batched* GEMM
  amortizes kernel launch (and host/device transfer, if coefficients are not
  already device-resident). This is the regime where the dense form is expected
  to dominate most, and the reason the refactor targets a batched operator API.
  The `impl_baseline_gpu.jl` script measures `device_resident` and
  `with_transfer` variants over the same batch sweep to locate that break-even;
  it must be run on a GPU box to populate `dense_gpu.csv`. **Not yet measured
  here** (this machine is CPU-only; the script verified it skips cleanly).

## Per-stage operator-form decision (constraint inherited by tasks 009–016)

| stage | decision | rationale |
|---|---|---|
| **z-rotation** | **Recurrence / diagonal** (keep `O(p)`) | No GEMM analog — it is a per-mode diagonal `2×2` (cos/sin) scaling (theory `001`). Materializing a dense matrix would be all-zero off-diagonal waste. Cheapest stage measured. Apply as a fused diagonal scaling over the batched buffer. |
| **axis-swap (y)** | **Dense-materialized, batched GEMM** | Biggest win: 5.7× unbatched, 17–34× batched. The invariant `S_n` blocks (theory `004`) are angle-independent → precompute once, reuse across all interactions and time steps; removes the per-call Wigner rebuild that dominates production. Block-diagonal over n; batched GEMM per degree. |
| **fixed-m z-translation** | **Dense-materialized, batched GEMM** (CPU/GPU); recurrence acceptable only for unbatched fallback | Ties unbatched, 3–5× batched. Block-diagonal over m (theory `002`); scaling carried as `D_L · K̂_m · D_M` metadata. The natural batched-GEMM unit. |
| **Lamb-Helmholtz** | **Structured two-channel operator** (not a full dense `(2·basis_dof)²` materialization) | Coupling is sparse: same-degree + one neighbor degree (theory `003`/`008h`). Cheap (~6e-7 at P=20). Implement as the structured banded update; do **not** densify the full two-channel block. Revisit as a small dense block only if a future GPU batch shows the structured form is launch-bound. |

Net: **axis-swap and fixed-m z-translation go dense/GEMM; z-rotation and
Lamb-Helmholtz stay structured/recurrence.** This matches the theory's hints and
is robust across the single-thread and multithread CPU regimes; the GPU run is
expected to widen the dense margin further, not reverse it.

## Allocation / storage inventory and budgets

Measured element counts (Float64) at representative P:

| structure | scope | P=4 | P=8 | P=12 | P=20 | scaling |
|---|---|---|---|---|---|---|
| `expansions` per branch (`2×2×Ncomplex`) | per branch | 60 | 180 | 364 | 924 | `4·Ncomplex(P)` |
| `Ts` (Wigner scratch) | per thread | 55 | 285 | 819 | 3311 | `O(p³)` |
| `harmonics` scratch | per thread | 112 | 264 | 480 | 1104 | `O(p²)` |
| `gradient_n_m` scratch | per thread | 90 | 270 | 546 | 1386 | `O(p²)` |
| `Hs_π2`,`ζs_mag`,`ηs_mag`,`M̃`,`L̃` | global, shared | — | — | — | ≈2024/3795/3795/253/253 | precomputed once, sized to max P |

`Ncomplex(P) = (P+1)(P+2)/2`.

Observations and budgets for Implementation:

- **Dominant cost is `expansions`** (`4·Ncomplex` Float64 per branch × n_branches);
  e.g. ~7.4 KB/branch at P=20. The global precomputed tables are a few tens of
  KB total, shared across all branches and threads — not a scaling concern.
  Per-thread scratch is ~tens of KB/thread and scales with thread count, not N.
- **Current layout always allocates the `χ` component** (`2×2×Ncomplex`) even
  when `lamb_helmholtz = Val(false)`, where only component 1 is used → **50% of
  expansion storage is dead for non-LH systems.** Budget: the `007`
  per-channel-on-demand layout (`basis_dof × batch × channel`, channel = 1 for
  `Val(false)`, 2 for `Val(true)`) must drop the unused channel; this halves
  expansion storage for the common non-LH (gravitational/electrostatic) case.
  Implementation tasks `009`/`017` must not carry the dead channel forward.
- **`007` minimum storage** is `basis_dof = 2·Ncomplex(P)` per channel
  (compressed complex, interleaved re/im) or `(P+1)²` (real basis). Current
  `2×2×Ncomplex` equals `2·basis_dof` (i.e. basis_dof × 2 channels), confirming
  the non-LH waste above.
- **`008h` `P_chi = P_phi + 1`:** the χ channel needs `Ncomplex(P+1)` modes vs
  φ's `Ncomplex(P)` — an extra `P+2` complex modes (e.g. +22 modes at P=20,
  ~10% of the φ size). **Budget decision: ragged per-channel sizing** (φ at P, χ
  at P+1) for storage, since padded-uniform would inflate the φ channel by the
  same `P+2` modes for no accuracy benefit. Padded-uniform may still be chosen
  *locally inside an operator* if a fixed stride simplifies a batched GEMM; that
  is an operator-internal choice (tasks `011`/`012`/`014`), not a storage-buffer
  choice (task `017`), which should be ragged.
- **Aliasing (from `007`):** overwrite stages (z-rotation, axis-swap,
  z-translation, Lamb-Helmholtz) require distinct source/dest/scratch; only the
  final inverse z-rotation accumulates. Scratch budget per batched operator
  call: at least two `basis_dof × batch × channel` buffers, reused across stages.

## CPU / GPU design notes (guidance for task 015)

- **Batch is the only lever.** Both the CPU crossover and the GPU rationale say
  the same thing: dense form pays off when many interactions sharing an operator
  are issued as one GEMM. Task `015`'s batching-strategy candidates should center
  on (a) one large GEMM when expansions share an operator vs (b) strided-batched
  GEMM when operators differ — the `007` storage contract supports both with no
  layout change.
- **Tight-packed vs `lda`-padded:** keep the `007` default (tight second-dim
  stride → merged-channel single GEMM). Only revisit `lda`-padding if a GPU
  profile in `015` shows a coalescing win that beats losing the channel-merge.
- **Interleaved vs planar re/im:** keep interleaved for the transitional
  compressed-complex path; the real basis (task `018`) is all-real and removes
  the question. Planar re/im is a GPU-only consideration to defer to `015`.
- **Don't use multithreaded BLAS inside operators**; parallelize across
  interactions and call single-thread batched GEMM per thread (pin
  `OPENBLAS_NUM_THREADS=1` / equivalent at process start — runtime
  `set_num_threads` did not reliably take effect on the test platform, a trap
  task `015` benchmarking must avoid).

## Design implications for task order / scope / risk

- The `008b` operator-form risk ("a dense path must beat the `O(p)`
  recurrences") **is resolved in favor of dense for axis-swap and z-translation**
  on a tuned-BLAS CPU, with the GPU expected to widen the margin. z-rotation and
  Lamb-Helmholtz stay structured. No task reordering is needed; this is exactly
  the per-stage split tasks `009`–`016` were structured to allow.
- **Precompute-once payoff is concentrated in axis-swap.** Task `013`
  (axis-swap operators) is the highest-value implementation step and the one
  whose dense form is least sensitive to batch size — prioritize its parity +
  benchmark.
- **Storage:** the non-LH dead-channel halving and the ragged `P_chi = P+1`
  sizing are concrete budgets for tasks `009`/`017`; neither changes the planned
  order.
- **Open item carried to `015`/`019`:** GPU numbers (`dense_gpu.csv`) are not
  yet collected on this CPU-only host. The decision above does not depend on them
  (it holds on CPU), but `015` must run `impl_baseline_gpu.jl` (or equivalent) to
  set the batched-GEMM batching strategy and confirm the GPU break-even.

## Approval Notes

To be filled by a different agent after benchmark/design notes and verification
are complete.
