# 041k report — brute-force direct UJ(+SFS) ceiling on one H200

**Job 13246033**, node `m13h-1-2`, NVIDIA H200, 2026-08-20. Driver
`scripts/fm041k_direct_bruteforce.jl` (+ `_gpu.jl`), snapshot
`~/FastMultipole-041e`, env `~/fm041eenv`, `julia -t 8`. Bodies per the
pre-registration: `MersenneTwister(123)`, unit cube, `|Γ| ~ 1/N`, uniform
`σ = 2 N^{-1/3}`. Raw data: `sweep.csv`, `accuracy.csv` (this directory).

## Accuracy (gate: F64 max rel err ≤ 1e-11 — PASSED)

Direct evaluation carries no multipole/polynomial approximation; the only
error is rounding. GPU (tiled, fused UJ+SFS) vs the threaded Float64 CPU
reference of the identical formulas, max relative error per block:

| n | prec | U | J | Ω | Q | E |
| --- | --- | --- | --- | --- | --- | --- |
| 1e3 | f64 | 3.4e-16 | 1.5e-15 | 3.7e-16 | 7.4e-16 | 1.6e-15 |
| 1e4 | f64 | 3.8e-16 | 2.4e-15 | 3.9e-16 | 1.2e-15 | 4.1e-15 |
| 1e3 | f32 | 1.2e-06 | 1.9e-05 | 1.7e-06 | 1.2e-06 | 1.6e-05 |
| 1e4 | f32 | 4.2e-06 | 9.2e-05 | 2.2e-06 | 3.1e-06 | 4.1e-05 |

The `041b` §1.2 reordered-E identity vs the original pairwise `Estr_direct`
form agrees to 4.6e-15 (n=1e3) / 1.2e-14 (n=1e4) on the CPU reference, and
the transcription itself was validated against production FLOWVPM
(`UJ_direct(sfs=true)`, gaussianerf, transposed) at ~1e-14 before submission
(`scripts/fm041k_crosscheck_flowvpm.jl`). F32 delivers ~1e-5 worst-case on
J/E — comfortably inside the phase's 1e-3 velocity gate (U at ~4e-6).

## Frontier: how far does 10 s of H200 go?

Median wall time (s) per series; sweep stops per series after the first
median > 10 s. All rows in `sweep.csv` carry the job ID.

| kernel | variant | prec | t(1e5) | t(3.16e5) | t(1e6) | t(3.16e6) | largest N ≤ 10 s (measured) | implied 10-s N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uj | naive | f64 | 0.266 | 1.86 | 15.2 | — | 3.16e5 | ~8.1e5 |
| uj | tiled | f64 | 0.227 | 1.65 | 13.3 | — | 3.16e5 | ~8.7e5 |
| ujsfs | naive | f64 | 0.335 | 2.58 | 21.9 | — | 3.16e5 | ~6.8e5 |
| ujsfs | tiled | f64 | 0.270 | 2.05 | 17.4 | — | 3.16e5 | ~7.6e5 |
| uj | naive | f32 | 0.112 | 0.879 | 7.24 | 65.1 | 1e6 | ~1.24e6 |
| uj | tiled | f32 | 0.095 | 0.723 | 5.76 | 49.9 | 1e6 | ~1.42e6 |
| ujsfs | naive | f32 | 0.163 | 1.30 | 11.1 | — | 3.16e5 | ~9.5e5 |
| ujsfs | tiled | f32 | 0.112 | 0.878 | 7.28 | 65.0 | 1e6 | ~1.24e6 |

("Implied 10-s N" = n·sqrt(10/t) from the terminal row; throughput is fully
saturated there, so the N² extrapolation over a half decade is safe.)

**Headline: one H200 brute-forces the complete FLOWVPM per-step physics —
velocity, full Jacobian, and SFS, exactly — for ~1.2 million particles in
10 s (F32 tiled), ~0.76 million in F64.** The 3.16e7 grid cap was never
reached; every series died on the 10 s budget, not memory (device footprint
at 3.16e6 F32 is ~0.35 GB — memory is irrelevant to this workload).

Small-N floor: a bare 1–3-kernel launch sequence costs 70–200 µs
(n = 100 rows), far below the ~0.87 ms full-lifecycle control floor
measured in 028/029 — brute force has essentially no fixed overhead.

Scaling is clean N² from n ≈ 3e4 upward (throughput saturates around
n = 3e5: e.g. uj/tiled/f32 delivers 1.05e11 pairs/s at 1e5, 1.38e11 at
3.16e5, 1.74e11 at 1e6, 2.00e11 at 3.16e6).

## Throughput and levers

- Peak measured: **2.0e11 pair-interactions/s** (uj/tiled/f32, n=3.16e6),
  ~15.6 TFLOP/s nominal non-transcendental flops — the kernel is
  transcendental-bound (one `exp` + one polynomial `erf` per pair through
  the SFU/software path), so ~23% of the 67 TFLOP/s FP32 FMA peak is the
  expected ceiling shape, not an inefficiency.
- F32 (+`rsqrt`) is worth **2.3×** over F64 at the terminal sizes (F64
  `exp`/`erf` are software-emulated); tiling is worth another **1.15–1.30×**
  over naive. Both levers behave as expected; nothing exotic remains on the
  table short of intrinsic-`erff`/tensor-core reformulations.

## Marginal cost of fused SFS

The stack's first direct+SFS measurement (today `sfs=true` hard-errors on
the GPU path). Structure: UJ pass + O(N) `T_q(Γ_q)` + one fused ζ pass for
Ω and Q (041b identity) — the ζ pass cannot merge into the UJ pass because
`Estr` needs completed Jacobians.

At the saturated sizes, **ujsfs / uj = 1.26–1.31×** (tiled: 7.28/5.76 f32 at
1e6; 17.4/13.3 f64). The entire exact SFS model costs ~30% of a UJ pass —
consistent with the ζ pass sharing the pair list and needing only one `exp`
(no `erf`, no `rsqrt`) and 12 FMAs of payload per pair. This is direct
evidence for the 041b SFS-enablement prior: a fused Ω/Q ζ-convolution is
cheap relative to U/J wherever the pair list already exists (the production
U-list case adds atomics, which this benchmark does not price).

## Amendment: `opt` variant A/B (job 13246522, same node class, 2026-08-20)

User-directed levers implemented as variant `opt` and A/B'd against `tiled`
in one job (results in `sweep_tiled_opt.csv`; `tiled` reproduced within
noise of job 13246033): (1) far-field singular switch at the GaussianErf
saturation cutoff (ρ² > 42.25 F32 / 81 F64 — exact at working precision;
ζ pairs beyond it skipped entirely), (2) fast/libdevice transcendentals for
the remaining near pairs (`__nv_fast_expf` + `__nv_erff` F32; accurate
`exp` + `__nv_erf` F64), (3) two targets per thread (each shared-memory
source load reused twice).

**Accuracy**: `f64_opt` passes the 1e-11 gate with max rel err 2.7e-14
(J, n=1e4); `f32_opt` is equal or slightly better than `f32` tiled on every
block (the switch replaces noisy saturated `g−1` terms with exact zeros).

Median seconds, tiled → opt (speedup):

| series | 1e5 | 1e6 | 3.16e6 | implied 10-s N |
| --- | --- | --- | --- | --- |
| uj f32 | 0.0949 → 0.0579 (1.64×) | 5.75 → 3.40 (1.69×) | 49.9 → 30.6 (1.63×) | 1.42e6 → **1.81e6** |
| ujsfs f32 | 0.112 → 0.0771 (1.45×) | 7.28 → 4.23 (1.72×) | 65.0 → 37.3 (1.74×) | 1.24e6 → **1.64e6** |
| uj f64 | 0.227 → 0.160 (1.42×) | 13.3 → 10.04 (1.32×) | — | 0.87e6 → **~1.0e6** |
| ujsfs f64 | 0.270 → 0.205 (1.32×) | 17.4 → 12.2 (1.42×) | — | 0.76e6 → **0.90e6** |

Peak throughput rises to **3.27e11 pairs/s** (uj/opt/f32 at 3.16e6, ~25.5
nominal TFLOP/s ≈ 38% of FP32 FMA peak — the kernel is now FMA/load-bound,
not transcendental-bound). The gain is 1.6–1.7×, not the ~4× transcendental
ceiling, because the singular path still carries the full 78-flop U/J tail
per pair. Caveats: at n ≤ 1e4 the F64 switch is inert (r_cut = 9σ ≈ 0.84
spans the unit cube, so nearly all pairs stay "near") and 2-target blocking
costs registers, making `ujsfs/opt/f64` up to 16% *slower* there — `opt`
is a large-n lever. F32 `opt` wins at every n ≥ 100.

## Brute-force vs FMM crossover

Anchor: the measured best-uniform full-step FMM curve from `041a` fig15
(`data/figures/fig15_adaptive_time_vs_n/unitcube_gpu_uniform.csv`, same
H200 class, 1e-3 gate): 1.31 ms at 1e4, 2.71 ms at 3.16e4, 7.40 ms at 1e5,
23.6 ms at 3.16e5, 92.3 ms at 1e6. Against best-case direct (uj/tiled/f32:
0.75 ms at 1e3, 2.20 ms at 3.16e3, 6.53 ms at 1e4, 94.9 ms at 1e5, 5.76 s
at 1e6):

- the crossover is **extrapolated, not measured** — the 041a FMM sweep
  stops at n = 1e4, where direct-opt (4.41 ms) is still 3.4× above FMM
  (1.31 ms). Extending FMM leftward brackets the intersection with
  direct's exact N² line: FMM flat at its 1e4 value gives n ≈ 5.5e3;
  continuing its measured ~n^0.63 slope down to the 028/029 ~0.87 ms
  per-GPU control floor gives n ≈ 4.1e3. **Extrapolated crossover
  n ≈ 4–5.5e3** (tiled: ~3.5–4.5e3); a measured closure would need an FMM
  run at n = 1e3–1e4;
- at 1e4 the FMM is already **5× faster** (3.4× vs opt), at 1e5 **13×**
  (7.8×), at 1e6 **62×** (37×).

An earlier draft of this report placed the crossover at n ≈ 2–5e4 by
assuming a 10–30 ms latency-dominated FMM step; the measured fig15 curve
(1.3–2.7 ms at those sizes) supersedes that assumption. At the real rotor
sizes (n ≈ 3.7–6.8e4, `041h`), brute force costs ~30–90 ms vs a measured
FMM step of ~3–5 ms on comparable uniform cases — brute force is a
correctness reference there, not a performance alternative.

## Verdict

- A single H200 brute-forces the exact FLOWVPM workload to ~1.2e6 particles
  per 10 s (F32), ~0.76e6 in F64 — a useful exactness/reference
  instrument, and the natural evaluation route only below n ≈ 4–5e3 where
  it undercuts the FMM's control/refresh floor with zero approximation
  error and zero fixed overhead (70–200 µs at n = 100).
- Fused SFS is nearly free in pair-list terms (+26–31% over U/J alone),
  strengthening the 041b SFS-enablement case for the production nearfield.
- The `opt` levers (far-field switch + fast intrinsics + register blocking)
  buy 1.6–1.7× (F32), lifting the 10-s frontier to ~1.8e6 (U/J) / ~1.6e6
  (U/J+SFS) at 3.3e11 pairs/s, now FMA/load-bound. Remaining headroom is
  incremental (4-target blocking, tensor-core r² GEMM); the N² shape means
  even large throughput gains move the frontier only by their square root.
  The FMM remains mandatory for n ≳ 1e4 and is ~37× ahead by 1e6 even
  against `opt`.
