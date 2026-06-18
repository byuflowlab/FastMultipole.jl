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

Cross-machine collection is **complete**. The decision basis below uses three
genuinely measured regimes from two target hosts:
`data/impl_performance_baseline/m13h-1-1/` (Intel Xeon 8568Y+ single-thread BLAS
+ NVIDIA H200 GPU) and `data/impl_performance_baseline/m12-2-5/` (AMD EPYC 7763,
72 BLAS threads). The earlier Apple M2 laptop run was a caveated single point and
is superseded by these.

Run commands (from the repository root):

These are the exact commands that produced the recorded artifacts (the full
sweep is the script default; the lists below are shown explicitly to match
`env.md`/`env_gpu.md`). Run the CPU script **once per BLAS regime**, controlling
threads via the process-start env var (runtime `set_num_threads` is unreliable —
see the caveat below).

```bash
# CPU baseline, GENUINELY single-thread BLAS (m13h-1-1 / Intel Xeon)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
P_LIST="4,8,12,20" P_DENSE_LIST="2,3,4,5,6,7,10,14,20" \
BATCH_LIST="1,8,64,512,4096,32768,262144" SAMPLES=50 \
  julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_cpu.jl

# CPU baseline, multithread BLAS (m12-2-5 / AMD EPYC ran at 72 threads)
OPENBLAS_NUM_THREADS=72 OMP_NUM_THREADS=72 \
P_LIST="4,8,12,20" P_DENSE_LIST="2,3,4,5,6,7,10,14,20" \
BATCH_LIST="1,8,64,512,4096,32768,262144" SAMPLES=50 \
  julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_cpu.jl

# GPU baseline (on a CUDA machine; add CUDA first)
julia --project=. -e 'import Pkg; Pkg.add("CUDA")'
P_DENSE_LIST="2,3,4,5,6,7,10,14,20" \
BATCH_LIST="1,8,64,512,4096,32768,262144" \
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

The decision basis is now **three genuinely measured regimes on two target
machines** (the original Apple M2 laptop run was a caveated single data point and
is superseded). From `data/impl_performance_baseline/<host>/env.md`:

| host | role / regime | CPU / GPU | Julia | BLAS | threads |
|---|---|---|---|---|---|
| `m13h-1-1` | **single-thread CPU** + **GPU** | Intel Xeon Platinum 8568Y+ (96 cores) / **NVIDIA H200** (140 GiB, CUDA 12.8) | 1.11.7 | OpenBLAS (`libopenblas64_`), **optimized** | `OPENBLAS_NUM_THREADS=1` |
| `m12-2-5` | **multithread CPU** | AMD EPYC 7763 (128 cores) | 1.11.7 | OpenBLAS (`libopenblas64_`), **optimized** | `BLAS.get_num_threads()=72` |

Both CPUs report a tuned/optimized BLAS, so the dense/GEMM numbers are
representative, not a reference-BLAS lower bound. (The script auto-flags a weak
BLAS in `env.md`; on such a host the dense numbers would be a lower bound to be
confirmed elsewhere.) The single-thread regime comes from the Xeon with
`OPENBLAS_NUM_THREADS=1` pinned at process start; the multithread regime from the
EPYC at 72 BLAS threads. The GPU regime comes from the H200 on the same `m13h-1-1`
box, so its CPU recurrence baseline is directly comparable to its GPU numbers.

Tables below report **P=20 Float64** as the representative high-order case
(P-sweep `[2,3,4,5,6,7,10,14,20]` is in the CSVs; the qualitative ordering is
stable across P). All times are **per expansion** (seconds).

## Baseline summary — current production recurrence stages

Per-call time (seconds), `lamb_helmholtz = Val(false)`, from
`m13h-1-1/stage_recurrence.csv` (Xeon single-thread; the EPYC numbers in
`m12-2-5/stage_recurrence.csv` are within ~10–20%):

| stage | P=4 | P=8 | P=12 | P=20 |
|---|---|---|---|---|
| `rotate_z` (z-rotation) | 3.1e-8 | 6.8e-8 | 1.1e-7 | 2.4e-7 |
| `rotate_multipole_y` (axis-swap, Wigner) | 6.3e-7 | 4.1e-6 | 1.5e-5 | 8.6e-5 |
| `translate_multipole_z` (M2M z) | 6.8e-8 | 2.6e-7 | 6.8e-7 | 2.6e-6 |
| `translate_multipole_to_local_z` (M2L z) | 1.3e-7 | 5.4e-7 | 1.4e-6 | 5.6e-6 |
| `translate_local_z` (L2L z) | 8.5e-8 | 3.2e-7 | 8.2e-7 | 3.1e-6 |

`Val(true)` (Lamb-Helmholtz) adds the two `transform_lamb_helmholtz_*` stages
(~5e-8 at P=4 to ~6e-7 at P=20) and raises each stage ~20–40% (two channels).

**Key observation:** the Wigner y-rotation (`rotate_multipole_y`) is the
dominant stage and the fastest-growing — at P=20 it is ~15× the M2L
z-translation and ~270× the z-rotation. It rebuilds the `Ts` (Wigner) matrices
on every call. This is the single hottest target.

## Dense-prototype vs recurrence head-to-head (P=20)

From `m13h-1-1/dense_vs_loop_blas1.csv` (Xeon, genuine single-thread BLAS) and
`m12-2-5/dense_vs_loop_blas72.csv` (EPYC, 72 BLAS threads). Four forms are
measured per stage: `recurrence` (production code, per expansion), `dense`
(materialized blocks applied via `mul!`, one `mul!` per block over a
`block × (channel·batch)` matrix), `dense_packed` (**the same per-block `mul!`
but with a `copyto!` gather from a flat buffer before and scatter after** — i.e.
it measures layout/copy traffic around block GEMMs, *not* a merged single GEMM),
and `compiled_block_loop` (hand-written loop over the same block structure,
batch-independent). Times are **per expansion** (seconds); "speedup" is best-dense
(`dense` or `dense_packed`) vs `recurrence`.

**What was *not* measured (deferred to task `015`):** (a) a true **merged padded
single GEMM** over the full block-diagonal operator (one big GEMM with explicit
zeros off-block) — none of the forms here build that, so this baseline says
nothing for or against the `007` merged-channel single-GEMM idea; (b) **cuBLAS
batched / strided-batched GEMM** on the GPU (the GPU prototype is a custom scalar
kernel, see below). Both are batching-strategy questions owned by task `015`.

**Single-thread CPU (Xeon, BLAS=1):**

| stage | recur | dense b=1 | dense b=64 | dense b=4096 | best speedup (batch) |
|---|---|---|---|---|---|
| axis_swap | 8.6e-5 | 2.8e-6 | **7.3e-7** | 1.5e-6 | **118× (b=64)** |
| m2l_z_translation | 5.6e-6 | 1.2e-6 | **2.6e-7** | 5.9e-7 | **22× (b=64)** |
| l2l_z_translation | 3.1e-6 | 1.2e-6 | **2.6e-7** | 5.8e-7 | **12× (b=64)** |
| m2m_z_translation | 2.6e-6 | 1.3e-6 | **2.6e-7** | 5.9e-7 | **10× (b=64)** |

**Multithread CPU (EPYC, BLAS=72):**

| stage | recur | dense b=1 | dense b=64 | dense b=262144 | best speedup (batch) |
|---|---|---|---|---|---|
| axis_swap | 8.2e-5 | 1.1e-5 | 2.3e-6 | **1.3e-6** | **65× (b=262144)** |
| m2l_z_translation | 7.9e-6 | 5.8e-6 | 9.5e-7 | **4.9e-7** | **18× (b=32768)** |
| l2l_z_translation | 4.3e-6 | 5.8e-6 | 9.5e-7 | **4.8e-7** | **10× (b=32768)** |
| m2m_z_translation | 2.7e-6 | 5.8e-6 | 9.5e-7 | **4.9e-7** | **6× (b=32768)** |

Facts that stand out:

1. **Axis-swap dense is the dominant win — even unbatched.** On single-thread
   Xeon it is **31× at batch 1** and **118× at batch 64**; the recurrence's
   per-call Wigner rebuild is pure overhead a precomputed invariant matrix
   (theory `004`: angle-independent `S_n`) avoids entirely. This is the strongest
   result and the highest-value implementation target.
2. **Single-thread optimized BLAS now wins for *every* stage at batch 1**
   (z-translations 2–4.7×, vs the earlier M2 laptop where batch 1 was a tie).
   With a good BLAS the GEMM call overhead is small enough that even one tiny
   matmul beats the recurrence.
3. **Multithread BLAS hurts the tiny unbatched case.** On the EPYC at 72 threads,
   `dense` at batch 1 **loses** to the recurrence for m2m (0.47×) and l2l (0.75×)
   and only ties for m2l (1.4×) — BLAS thread spin-up dominates a single
   `≤41×41` block. It needs **batch ≥ 8** to win, then scales well to very large
   batch. → never call multithreaded BLAS on one operator block; batch first, or
   keep BLAS single-threaded and parallelize across interactions.
4. **`compiled_block_loop` is a robust, batch-independent fallback** (~2.8e-6 at
   P=20 across both machines). It beats the recurrence for the z-translations at
   batch 1 on the multithread box (where `dense` loses), and never needs a BLAS
   call. Useful for the unbatched path and as a no-BLAS portability floor.
5. **Gather/scatter copy traffic around block GEMMs costs 1.7–3.6×.**
   `dense_packed` runs the *same* per-block `mul!` as `dense` but adds a
   `copyto!` gather from a flat buffer before and a scatter after; it is
   consistently slower than `dense` (~3.6× single-thread, 9.4e-7 vs 2.6e-7 at
   b=64; ~1.7× multithread). The lesson is about **layout, not FLOPs**: a buffer
   that keeps each block's operands contiguous and GEMM-ready (so the operator
   can read/write in place or via strides without copying) avoids this cost —
   which *supports* the `007` contiguous-layout goal. This says **nothing** about
   a merged padded single GEMM (not measured here; deferred to `015`).
6. **Optimal CPU batch differs by regime.** Single-thread peaks at **batch ≈ 64**
   then degrades at very large batch (cache eviction once the working set exceeds
   L2); multithread keeps improving out to **batch ≈ 32768** before flattening.
   Implementation should treat batch width as a tunable, not a fixed constant.

## GEMM-vs-compiled-loop crossover analysis

The operator-stage matrices here are small (per-block dimension `P+1−m` for
z-translation, `2n+1` for the rotation; ≤ ~41 at P=20). Where dense GEMM beats
compiled loops therefore depends entirely on **batch width**, not matrix size:

- **Single-thread CPU (measured: Xeon BLAS=1).** With a tuned single-thread
  BLAS, dense wins at **every batch including batch 1** (z-translations 2–4.7×,
  axis-swap 31×), and peaks at **batch ≈ 64** (z-translations ~10–22×, axis-swap
  118×) before cache eviction erodes the very-large-batch numbers. The recurrence
  is no longer competitive anywhere at P=20. *Caveat:* on a host without a tuned
  BLAS this crossover shifts right or disappears, and the `compiled_block_loop`
  form (batch-independent, no BLAS) is the portability floor; `env.md` flags the
  weak-BLAS case.
- **Multithread CPU (measured: EPYC BLAS=72).** Multithreaded BLAS **hurts** the
  unbatched case — at batch 1 `dense` loses to the recurrence for m2m/l2l
  z-translation (0.47×/0.75×) because thread spin-up dominates a `≤41×41` block.
  The crossover is **batch ≈ 8**, after which dense scales to **65× (axis-swap)**
  and **6–18× (z-translation)** at batch ≈ 32768–262144 — a wider top-end than
  single-thread but a worse small-batch floor. Conclusion: the right way to use
  cores is parallelism *across* interactions (as production already does with
  `@threads`), each thread issuing **single-thread** batched GEMMs (pin
  `OPENBLAS_NUM_THREADS=1` at process start — runtime `BLAS.set_num_threads` was
  unreliable on the test platforms), not multithreaded BLAS inside one operator.
- **GPU (measured: NVIDIA H200 — custom kernel, *not* cuBLAS).** The GPU
  prototype is a hand-written single-launch scalar CUDA kernel over the
  block-diagonal structure (`fused_block_kernel!`), deliberately **not cuBLAS**,
  so these numbers measure **launch fusion and device residency**, not a
  vendor-GEMM path. With that caveat it confirms the refactor's premise: the
  fused-kernel device-resident form scales from launch-bound `~1.1e-5`/exp at
  batch 1 down to **`7.2e-9`/exp at batch 262144** for the z-translations and
  `1.9e-8` for axis-swap — **~36–40× faster than the best CPU dense** and ~700×
  faster than the CPU recurrence. Crossover details:
    - **Device-resident** beats the CPU recurrence at **batch ≥ 8**, and beats
      the best CPU dense at **batch ≥ 64**.
    - **Host/device transfer is the killer.** The `with_transfer` variant floors
      at **~6.1e-7/exp (Float64 z-translations)** and **~1.13e-6/exp (Float64
      axis-swap)** at batch 262144 (min-over-batch ~4.6e-7 at batch 4096; the
      Float32 floor is ~3.2e-7 / 5.9e-7) — i.e. *worse than CPU dense*, regardless
      of batch. The GPU only wins if coefficients stay **device-resident across
      the M2L stencil** (upload once, run the whole horizontal pass on device,
      download once). This is a hard architectural requirement for the GPU path,
      not a tuning detail.
    - **Single fused launch ≫ many per-block launches.** `per_block_launch`
      (one kernel launch per block) is catastrophic at small batch (`4.6e-4`/exp
      at batch 1 from many tiny launches) and only edges out the single fused
      launch at the extreme batch 262144 (`3.3e-9` vs `7.2e-9`). The single fused
      launch is the robust default; per-block launching is a large-batch-only
      micro-optimization. **Whether a cuBLAS batched/strided-batched GEMM beats
      this custom kernel is unmeasured and is a task `015` decision** — these
      numbers do not establish a cuBLAS strategy.
    - **Float32 is not the lever.** On the H200, dropping to Float32 gives only
      **1.2–1.27×** at large batch and is *slower* (<1×) at small batch. The
      payoff is batching + device residency, not precision. (Confirmed on CPU too:
      Float32 ~1.2× there.) Keep Float64 as default; Float32 is an opt-in for
      memory-bound very-large-batch runs only.

## Per-stage operator-form decision (constraint inherited by tasks 009–016)

| stage | decision | rationale |
|---|---|---|
| **z-rotation** | **Recurrence / diagonal** (keep `O(p)`) | No GEMM analog — it is a per-mode diagonal `2×2` (cos/sin) scaling (theory `001`). Materializing a dense matrix would be all-zero off-diagonal waste. Cheapest stage measured. Apply as a fused diagonal scaling over the batched buffer. |
| **axis-swap (y)** | **Dense-materialized, per-block batched GEMM** | Biggest win: **31× unbatched, 118× batched** single-thread; 65× batched multithread; the dominant GPU stage. The invariant `S_n` blocks (theory `004`) are angle-independent → precompute once, reuse across all interactions and time steps; removes the per-call Wigner rebuild that dominates production. Block-diagonal over n; one GEMM per degree (per-block; whether to instead merge into a single padded GEMM is unmeasured and deferred to `015`). |
| **fixed-m z-translation** | **Dense-materialized, per-block batched GEMM** (CPU/GPU); `compiled_block_loop` for the unbatched / no-BLAS fallback | 2–4.7× unbatched (single-thread, tuned BLAS), 10–22× batched; on GPU device-resident it is the best-scaling stage (`7.2e-9`/exp). Block-diagonal over m (theory `002`); scaling carried as `D_L · K̂_m · D_M` metadata. The natural batched-GEMM unit. Note multithread BLAS *loses* unbatched, so the fallback path matters. |
| **Lamb-Helmholtz** | **Structured two-channel operator** (not a full dense `(2·basis_dof)²` materialization) | Coupling is sparse: same-degree + one neighbor degree (theory `003`/`008h`). Cheap (~6e-7 at P=20). Implement as the structured banded update; do **not** densify the full two-channel block. Revisit as a small dense block only if a future GPU batch shows the structured form is launch-bound. |

Net: **axis-swap and fixed-m z-translation go dense/per-block-batched GEMM;
z-rotation and Lamb-Helmholtz stay structured/recurrence.** This matches the
theory's hints and is now confirmed **robust across all three measured regimes**
(single-thread CPU, multithread CPU, GPU) — the GPU widens the dense margin to
~36–40× over the best CPU rather than reversing it. Two refinements the data
forces onto Implementation: (a) keep each block's operands **contiguous and
GEMM-ready** so the operator avoids gather/scatter copy traffic (the measured
`dense_packed` gather/scatter penalty was 1.7–3.6×); (b) carry a
`compiled_block_loop` fallback for the unbatched / weak-BLAS / multithread-batch-1
path, where a BLAS `mul!` on one tiny block can lose. The choice between per-block
and a single merged padded GEMM (and, on GPU, custom kernel vs cuBLAS
batched/strided-batched) is **unmeasured here and deferred to task `015`**.

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

- **Batch is the only lever** (confirmed in all three regimes). Dense pays off
  when many interactions sharing an operator are issued as one batched GEMM.
  Task `015`'s batching-strategy candidates should center on (a) batched GEMM
  *over the block structure* (one GEMM per n- or m-block, batched across
  interactions) vs (b) strided-batched GEMM when operators differ — the `007`
  storage contract supports both with no layout change.
- **Keep block operands GEMM-ready to avoid gather/scatter.** The `dense_packed`
  form (same per-block `mul!` as `dense`, but with a `copyto!` gather before and
  scatter after) was 1.7–3.6× slower than `dense` purely from the copy traffic.
  So the buffer layout should let each block's operands be read/written **in
  place or via strides without copying** — this *supports* the `007`
  contiguous-layout goal and is a perf note for `015`/`017`, not a change to the
  `007` storage layout.
- **Merged single-GEMM is an open `015` question, not decided here.** Whether a
  single padded GEMM over the whole block-diagonal operator (explicit zeros
  off-block) beats N per-block GEMMs depends on small-matrix BLAS overhead vs the
  wasted FLOPs, and was **not measured** in this baseline. Likewise the φ/χ
  **channel-merge** is a special case of the same trade-off. Task `015` must
  benchmark per-block vs merged padded GEMM (CPU) and the cuBLAS
  batched/strided-batched vs custom-kernel paths (GPU) before fixing a default.
- **Interleaved vs planar re/im:** keep interleaved for the transitional
  compressed-complex path; the real basis (task `018`) is all-real and removes
  the question. Planar re/im is a GPU-only consideration to defer to `015`.
- **GPU coefficients must stay device-resident across the horizontal pass.** The
  `with_transfer` measurement shows per-call upload/download floors GPU
  throughput at CPU-dense levels (~6e-7/exp Float64; ~3e-7 Float32). The GPU
  operator API (`015`) must upload expansions once, run the full M2L stencil on
  device, and download once — per-operator transfer defeats the purpose. Prefer a
  **single fused launch** over many per-block launches (except at extreme batch);
  whether a cuBLAS batched/strided-batched GEMM beats the custom kernel is a `015`
  measurement (the prototype here is a custom kernel, not cuBLAS).
- **Float64 default; Float32 opt-in.** Float32 buys only ~1.2× even on the H200
  and is slower at small batch — keep Float64 the default precision; expose
  Float32 only for memory-bound very-large-batch runs.
- **Don't use multithreaded BLAS inside operators**; parallelize across
  interactions and call single-thread batched GEMM per thread (pin
  `OPENBLAS_NUM_THREADS=1` / equivalent at process start — runtime
  `set_num_threads` did not reliably take effect on the test platforms, a trap
  task `015` benchmarking must avoid). Multithread BLAS measurably *loses* on a
  single unbatched block (EPYC batch-1 m2m/l2l).

## Design implications for task order / scope / risk

- The `008b` operator-form risk ("a dense path must beat the `O(p)`
  recurrences") **is resolved in favor of dense for axis-swap and z-translation
  across all three measured regimes** (single-thread CPU, multithread CPU, GPU).
  z-rotation and Lamb-Helmholtz stay structured. No task reordering is needed;
  this is exactly the per-stage split tasks `009`–`016` were structured to allow.
- **Precompute-once payoff is concentrated in axis-swap.** Task `013`
  (axis-swap operators) is the highest-value implementation step (31–118× CPU,
  dominant on GPU) and the one whose dense form is least sensitive to batch size —
  prioritize its parity + benchmark.
- **GPU break-even is now known and is an architectural constraint, not a tuning
  item.** Device residency across the horizontal pass is mandatory (per-operator
  transfer floors throughput at CPU-dense levels), a single fused launch beats
  many per-block launches below batch ~10⁵, and Float64 stays default (Float32
  ~1.2×). These hold for the measured **custom kernel**; the cuBLAS
  batched/strided-batched comparison is left to `015`. Task `015` inherits the
  device-residency requirement and the recorded break-even (device-resident dense
  beats CPU recurrence at batch ≥ 8, beats best CPU dense at batch ≥ 64).
- **Implementation should provide both `dense` (per-block batched GEMM) and a
  `compiled_block_loop` fallback** for the operator stages, since the BLAS path
  loses on a single unbatched block under multithreaded BLAS and on weak-BLAS
  hosts. Keep block operands contiguous/GEMM-ready to avoid the measured
  gather/scatter penalty; the per-block-vs-merged-GEMM choice is a `015` decision.
- **Storage:** the non-LH dead-channel halving and the ragged `P_chi = P+1`
  sizing are concrete budgets for tasks `009`/`017`; neither changes the planned
  order.

## Approval Notes

Approved on 2026-06-17 by clear-context review.

Reviewed `START_HERE.md`, this completed task file, the changed benchmark
harnesses, generated baseline artifacts under
`data/impl_performance_baseline/`, and the relevant production
translation/evaluation storage surfaces. No blocking issues found.

Checks performed:

- Confirmed the phase gate is intact: no `src/` files are modified, and no task
  `009`+ production implementation work has started.
- Confirmed recorded CPU/GPU environment notes and CSV schemas match the task
  text.
- Spot-checked the P=20 headline CPU and GPU numbers against the recorded CSVs.
- Confirmed the production expansion/scratch inventory matches current storage:
  `initialize_expansion`/`initialize_expansions` allocate `2×2×Ncomplex`, `Ts`
  length follows the recorded cubic formula, and evaluation uses the recorded
  `gradient_n_m` scratch shape.
- Confirmed the remaining open items are explicitly delegated to later benchmark
  tasks (`015`/`019`) rather than silently deciding unmeasured strategy choices.
