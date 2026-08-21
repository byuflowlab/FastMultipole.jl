# 015 Axis-Swap (M2L variant) Benchmark Results

Isolated-operator comparison of the two near-term full-M2L variants shipped by
task `014`, against each other and against the current production recurrence M2L.
This is the `015` measurement deliverable. The definitive end-to-end comparison
is deferred to `024`; actual GPU benchmarks are deferred to `022`/`024` (no GPU
operator path exists yet, so the GPU recommendation here is analytical).

## Harness and reproduction

Script: `MATRIX_OPERATOR_REFACTOR/scripts/impl_015_m2l_variants.jl` (CPU only, no
`src/` changes, self-contained `timeit` min-of-samples; same conventions as the
`008c` baseline harness).

Run twice for the two BLAS regimes (thread count set at process start, output
tagged by `BLAS.get_num_threads()`):

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_015_m2l_variants.jl
OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu) OMP_NUM_THREADS=$(sysctl -n hw.ncpu) \
  julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_015_m2l_variants.jl
```

Sweeps (ENV-overridable): `P ∈ {4,8,12,20}`, `batch ∈ {1,8,64,512,4096}`,
`lamb_helmholtz ∈ {false,true}`, `Float64`, `SAMPLES=50`.

### Environment (this run)

- host `tmpfac-126-17.et.byu.edu`, Apple M2, Julia 1.12.5
- BLAS `libopenblas64_p-r0.3.31` (tuned), regimes recorded: 1 thread and 8 threads
- git HEAD `2ded526` (dirty: refactor branch in progress)

### Artifacts (machine-tagged, under `data/axis_swap/<host>/`)

- `env.md` — environment + BLAS metadata + GPU-deferral note. **Note (016b):**
  `env.md` reflects only the **last** BLAS-regime run (8-thread); the per-regime
  timing CSVs (`*_blas1.csv` / `*_blas8.csv`) preserve both regimes.
- `m2l_variants_blas1.csv`, `m2l_variants_blas8.csv` — variant vs production timing sweep
- `footprint.csv` — per-call allocation + cache/scratch byte footprint
- `batch_composition_blas1.csv`, `batch_composition_blas8.csv` — offset-class /
  shared-direction / shared-norm probe (columns `n_dir, n_norm, n_offset_class` are
  the true realized distinct counts, swept over P, batch, and both LH policies)

### Correctness gate

The harness aborts before timing unless `MaterializedYRotationM2L` reproduces the
production `multipole_to_local!` to `< 1e-6`. Observed: max abs diff `0.00e+00` at
both P=4 and P=20 — we are timing the parity-validated operators (the `014`
`test/m2l_operator_test.jl` suite covers the full `Val(true)`/cross-variant parity).

## Timing results

Per-expansion time is flat across batch width (the operators are per-column
kernels), so the table reports the asymptotic `seconds_per_expansion` at
`batch=4096`, single-thread BLAS, in microseconds. Speedup = production / variant
(`>1` means faster than current production M2L).

| P | LH | materialized µs | factored µs | production µs | mat speedup | fac speedup |
|---|----|----------------:|------------:|--------------:|------------:|------------:|
| 4 | false | 0.738 | 0.953 | 0.674 | 0.91× | 0.71× |
| 4 | true  | 1.523 | 2.843 | 0.917 | 0.60× | 0.32× |
| 8 | false | 3.432 | 3.962 | 3.601 | 1.05× | 0.91× |
| 8 | true  | 6.027 | 10.384 | 4.678 | 0.78× | 0.45× |
| 12 | false | 10.168 | 10.292 | 12.175 | 1.20× | 1.18× |
| 12 | true  | 16.266 | 25.317 | 15.149 | 0.93× | 0.60× |
| 20 | false | 46.788 | 40.419 | 70.791 | 1.51× | **1.75×** |
| 20 | true  | 66.951 | 90.648 | 82.222 | **1.23×** | 0.91× |

### Findings

1. **Order crossover.** Both variants only beat production at higher order:
   `Val(false)` becomes a win at `P ≳ 12`; at `P ≤ 8` the production recurrence is
   still faster. This directly motivates the deferred small-`P`/tiny-batch fallback
   policy (`019b`) — at low `P` the new operators should fall back to the legacy path.

2. **Channel crossover (the central result).** The best variant depends on the
   Lamb-Helmholtz policy:
   - `Val(false)` (φ-only): **factored wins** at high P (1.75× vs production at P=20;
     materialized 1.51×).
   - `Val(true)` (Lamb-Helmholtz): **materialized wins** decisively — it beats
     factored at every P and is the only variant that beats production at high P
     (1.23× at P=20). Factored is the worst option under LH (0.91× at P=20, i.e.
     slower than production). The factored path's extra per-channel `Z/S` stages
     plus the `χ`-at-`P+1` carry penalize it heavily once the second channel is live.

3. **No batched-GEMM speedup yet (structural).** `blas1 ≈ blas8` (P=20 φ-only:
   41.20 µs vs 40.56 µs) and per-expansion time is batch-independent — the current
   `014` CPU operators are scalar per-column kernels, **not** the dense per-block
   GEMM form prototyped in `008c`. So the large `008c` dense projections (31–118×
   for axis-swap) are **not** realized by today's CPU operators; the CPU advantage
   here is the modest (≤1.7×) gain from a leaner composition. Realizing the GEMM
   speedup is the job of operator performance tuning (`019`) and the GPU path
   (`022`).

4. **Allocation / footprint.** `@allocated` for a steady-state
   `m2l_operator_batch!` call is **0 bytes** for both variants across every
   `(P, lh, batch)` — the preallocated-scratch design holds. Static footprint
   (P=20, batch=4096): invariant cache 1.30 MB (φ-only) / 1.52 MB (LH); scratch
   57.8 MB (φ-only) / 63.3 MB (LH), dominated by the two `[2,2,nh,batch]` work
   buffers. Both variants share the same cache and scratch types, so storage does
   not distinguish them.

5. **Batch composition (offset-class / shared-direction / shared-norm probe).**
   The probe sweeps order (P ∈ {8,20}), batch (∈ {64,4096}), the number of *distinct*
   directions `n_dir` and *distinct* norms `n_norm` actually present in the batch
   (levels {1, 8, 64, batch}, realized exactly via `pool[mod1(j,n)]`), the resulting
   distinct offset-class count `n_offset_class`, and both LH policies. Findings:
   - **Materialized is flat** across every diversity axis: e.g. P=20, `Val(false)`,
     batch=4096, per-expansion time is 46.66–46.72 µs as `n_dir` goes 1 → 4096
     (< 0.2% spread), and is likewise insensitive to `n_norm` and `n_offset_class`.
   - **Factored has a mild (~5–6%) increase with distinct-direction count**: same
     case, 39.20 µs at `n_dir=1` → 41.0 µs at `n_dir=8` → ~41.4 µs at `n_dir≥64`
     (saturating), and a slighter rise with `n_norm`. This is the cost of recomputing
     its per-angle z-rotation diagonals/scalings when fewer angles repeat.
   - **Crucially, neither variant exploits offset-class sharing.** Even at the
     fully-shared corner (`n_dir = n_norm = 1`, `n_offset_class = 1`) both pay
     essentially the full per-column cost (factored 39.2 µs, materialized 46.7 µs at
     P=20) — the same order as the fully-distinct corner. The current API recomputes
     the rotation/translation per column and never reuses one materialized operator
     across a batch that shares a single `(phi, theta, r)`.
   This is unrealized headroom: a future per-offset-class form that materializes one
   operator and reuses it across the offset-class batch (the `013a` batching unit)
   would collapse the small-`n_dir` cost dramatically and benefit the materialized
   variant most. Recorded as future work, not a `015` deliverable.

## Recommendations

### CPU

- **Retain both variants behind dispatch** (as the roadmap already mandates), and
  select by Lamb-Helmholtz policy: **factored for φ-only (`Val(false)`)** at
  moderate/high P, **materialized for Lamb-Helmholtz (`Val(true)`)** at all P.
- **If a single CPU default must be chosen, choose `MaterializedYRotationM2L`.** It
  is the robust choice: it wins every LH case, never loses badly (worst case 0.60×
  at P=4 LH, where the small-P fallback applies anyway), and trails factored by only
  ~16% on the φ-only high-P corner where factored is best. Factored's advantage is
  confined to the φ-only high-P regime; its LH penalty disqualifies it as a single
  default.
- **Gate on order:** below P ≈ 8–12 the production recurrence still wins, so the
  new operators should be enabled only above that crossover (or behind the `019b`
  small-P fallback).

### GPU (analytical — staged for `022`/`024`)

No GPU operator path exists yet (`022` deferred), so this is a structural argument
from the `008c` GPU baseline (`data/impl_performance_baseline/m13h-1-1/dense_gpu.csv`),
to be confirmed on real hardware by `022`/`024`:

- **Recommended GPU candidate: `MaterializedYRotationM2L`.** Per `008c`, the dense
  per-block axis-swap GEMM reaches device-resident throughput only at very large
  batch (262144). The materialized variant maps onto **fewer, larger** dense
  operators — one materialized `Ts(theta)` per offset class, reused across the whole
  batch (the `013a` batching unit) — which means far fewer kernel launches and
  better occupancy. The factored variant is many **small** diagonal `Z` scalings
  plus small mode-matrix products, which is launch-bound on a GPU.
- **Prerequisite:** today's CPU operators are scalar, not GEMM. The GPU win is only
  realized once `022` implements the device-resident dense-batched form; `024` then
  decides the winner per platform with real measurements.

## Deferred options (recorded, not benchmarked as `015` deliverables)

Per the task spec, these are noted for later final implementation/performance tasks
(`019`/`022`/`024`) and were intentionally not benchmarked here:

- fully dense per-offset M2L matrix
- partially folded hybrids around the `K_z` z-translation block
- alternate z-translation cache/scaling policies
- native real-basis operator execution
- per-`m` block-batched z-translation
- non-rotation operator families
- the per-offset-class shared-angle reuse identified in finding 5 (one materialized
  operator applied across an offset-class batch)
