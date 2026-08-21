# 041k Benchmark: Brute-Force Direct UJ(+SFS) Ceiling on One H200

## Status and entry gate

**Staged `2026-08-20` (user direction). Done `2026-08-20` (same day); see
Completion notes. Clear-context approved `2026-08-20`.**

Entry gate: none — standalone benchmark row, independent of `041h`/`041i`/
`041j`, and does **not** gate `042`. Benchmark/analysis row: artifacts under
`scripts/` and `data/` only; no production `src/` or FLOWVPM changes. GPU
runs on the H200 via the standing cluster loop; local work at most four
threads.

## Motivation

User direction (`2026-08-20`): how far can sheer hardware power carry a
**naive O(N²) direct evaluation** of the N-body problem on a single H200,
before FMM machinery earns its complexity? Direct evaluation carries **no
polynomial/multipole approximation at all** — it is exact up to
floating-point rounding — so it has none of the far-field truncation error
the expansion path manages; the only accuracy knob is F32-vs-F64 rounding.

The swept workload is the realistic FLOWVPM per-step physics, not a toy
gravitational kernel (user decision at staging):

1. **UJ kernel** — regularized `gaussianerf` velocity + full 9-component
   Jacobian, transcribed from the production pair math
   (`../FLOWVPM.jl/src/FLOWVPM_fmm.jl:132-198`, `FLOWVPM_kernel.jl:51-57`).
2. **UJ+SFS** — the same plus the subfilter-scale vortex-stretching model.
   **No fused direct+SFS kernel exists anywhere in the stack today** (the
   GPU/radix path hard-rejects `sfs=true`, `FLOWVPM_fmm_radix.jl:499`; CPU
   runs `Estr_direct!` as a separate pass). This row builds the first one,
   using the `041b` §1.2 factorized identity

   $$
   \Omega_p=\sum_q\zeta_{pq}\Gamma_q,\qquad
   Q_p=\sum_q\zeta_{pq}\,T_q(\Gamma_q),\qquad
   E_p=T_p(\Omega_p)-Q_p,
   $$

   with `T` the transposed scheme (FLOWVPM default). Because `Estr` needs
   the **completed** Jacobians of target and source, the ζ convolution
   cannot merge into the UJ pass; the minimal exact structure is two O(N²)
   passes — pass 1 (U + J), an O(N) per-source `T_q(Γ_q)` precompute, and
   pass 2 (fused ζ pass emitting Ω and Q together, per the identity). The
   measured UJ-vs-UJ+SFS delta prices SFS enablement with data, feeding the
   `041b` SFS-enablement track.

Secondary payoff: the (N, time) frontier against the 041-series FMM timings
answers "below what N is FMM not even worth it" for the true workload,
including the known ~0.87 ms n-independent launch floor (`028`/`029`) that
dominates small N.

## Objective

Measure the largest N a single H200 can brute-force under a 10 s wall-time
budget, per kernel (UJ, UJ+SFS), variant (naive, shared-memory tiled), and
precision (F64, F32), starting at N = 1e2 and increasing by half-decades;
validate exactness against a Float64 CPU reference; and render a verdict on
the brute-force/FMM crossover and the marginal cost of fused SFS.

## Method (pre-registered)

### Bodies

Seeded `MersenneTwister(123)`; positions uniform in the unit cube; strength
components `Γ_i ~ U(-1,1)·(1/N)`; **uniform** smoothing radius
`σ = 2·N^{-1/3}` (σ enters only transcendental arguments; timing is
σ-independent). Same body set shared by every series at a given N.

### Kernels

All pair math is transcribed with file/line provenance from FLOWVPM
(`gaussianerf`: `g_dgdr` from `FLOWVPM_kernel.jl:54-57`, ζ from `:51`; U/J
accumulation from `FLOWVPM_fmm.jl:132-198`; `Estr` transposed scheme from
`FLOWVPM_subfilterscale_models.jl:16-41`). `erf` is the vendored
GPU-compatible polynomial (`scripts/fm041k_erf_vendored.jl`, a verbatim copy
of `FLOWVPM_gpu_erf.jl`) so CPU reference and device kernels share bitwise
formulas in both precisions.

- **UJ series**: one all-pairs pass, per-target registers accumulate U(3) +
  J(9); self-pair (r=0) skipped as in production.
- **UJ+SFS series**: UJ pass + O(N) `T_q(Γ_q)` kernel + fused ζ pass
  accumulating Ω(3) and Q(3) (self-pair kept: its contribution to
  `E_p = T_p(Ω_p) − Q_p` cancels identically). E formed on host, O(N),
  excluded from timing (negligible).
- **Variants**: (a) naive thread-per-target; (b) shared-memory source-tiled
  (tile staged cooperatively per block).
- **Precisions**: Float64; Float32 with `CUDA.rsqrt` (F64 path never uses
  `rsqrt` — known ~1e-7 accuracy).

8 series total = {uj, ujsfs} × {naive, tiled} × {f64, f32}.

### Sweep

N = round(10^e), e = 2.0, 2.5, …, 7.5 (1e2 … 3.16e7 cap). Per series
independently: stop after the first N whose median exceeds **10 s**. Timing:
median of 5 synchronized reps after 2 warm-ups; if a single warm rep exceeds
15 s, fall back to 1 warm-up + 2 reps (documented in the CSV `reps` column).
H2D/D2H measured separately. Report pairs/s (primary) and nominal GFLOP/s
using a stated flops/pair count (transcendentals counted separately).

### Accuracy

At N = 1e3 and 1e4: GPU (tiled, both precisions) vs an in-script threaded
Float64 CPU reference of the identical formulas, per output block (U, J, Ω,
Q, E). Gate: **F64 max relative error ≤ 1e-11** (expect ~1e-13; residual is
accumulation order only). F32 errors reported, not gated (documented lever).
Additionally the CPU reference cross-checks the `041b` identity itself:
`E` via the reordered form vs the original `Estr_direct` pairwise form.
Before submission, `scripts/fm041k_crosscheck_flowvpm.jl` validates the
transcription against production FLOWVPM (`UJ_direct(sfs=true)`, gaussianerf,
transposed) at n = 1e3 locally (same ≤ 1e-11 bar).

### Amendment: `opt` variant (user direction 2026-08-20, post-completion)

After the base run (job 13246033) the user directed implementing brute-force
levers 1–3 as a third variant, `opt`, benchmarked as a same-job A/B against
`tiled` (job runs variants `tiled,opt` only; results land in
`sweep_tiled_opt.csv`, never overwriting the base `sweep.csv`):

1. **Far-field singular switch** — beyond the GaussianErf saturation cutoff
   (`ρ² > 42.25` F32 / `81` F64, where `|1−g|` and `dgdr` are below working
   precision) the pair takes the singular path (g=1, dgdr=0) and ζ pairs are
   skipped entirely. Exact at working precision, so the F64 1e-11 gate
   applies unchanged to `opt`. With σ = 2N^(−1/3) only ~1% of pairs at
   N = 1e6 keep transcendentals.
2. **Hardware transcendentals for near pairs** — F32: SFU `__nv_fast_expf`
   + libdevice `__nv_erff`; F64: accurate `exp` + libdevice `__nv_erf`.
3. **Register blocking** — two targets per thread; each shared-memory
   source load reused twice.

Accuracy for `opt` (both precisions) is gated/reported in `accuracy.csv`
with tags `f64_opt`/`f32_opt` at the same N = 1e3/1e4 points.

## Gates and verdict

- Accuracy gate above must pass before any timing is reported as valid.
- Deliverables: (N, median time) frontier per series; **largest N under
  10 s per series**; measured marginal cost of SFS fusion (ujsfs/uj ratio
  vs the naive two-separate-pass expectation); pairs/s vs H200 peak sanity
  check; verdict paragraph locating the brute-force/FMM crossover against
  the `041a`/`041e`/`041h`-era FMM timings and stating the implication for
  the `041b` SFS-enablement track.

## Artifacts

- `scripts/fm041k_direct_bruteforce.jl` (driver; CPU core + conditionally
  included `fm041k_direct_bruteforce_gpu.jl` CUDA half),
  `scripts/fm041k_erf_vendored.jl` (verbatim FLOWVPM erf, provenance header),
  `scripts/fm041k_crosscheck_flowvpm.jl` (local production cross-check),
  `scripts/fm041k_submit_gpu.sh`, `scripts/fetch_041k.sh`.
- `data/direct_bruteforce_ceiling/{sweep.csv, accuracy.csv, report.md}`.
- All cluster timings carry slurm job IDs.

## Verification

- `accuracy.csv` passes the F64 gate on all blocks at both N.
- `sweep.csv` shows the small-N launch-floor plateau, then clean ~N²
  scaling; every row carries the job ID.
- Report sanity-checks achieved throughput against H200 FP32/FP64 peaks.

## Completion notes (2026-08-20, lead agent)

Job **13246033** (node `m13h-1-2`, H200); full results and verdict in
`data/direct_bruteforce_ceiling/report.md`. Summary:

- **Accuracy gate PASSED**: F64 GPU vs CPU reference ≤ 4.1e-15 on all
  blocks (gate 1e-11); 041b identity holds at ≤ 1.2e-14; pre-submission
  cross-check vs production FLOWVPM `UJ_direct(sfs=true)` agreed at ~1e-14.
  F32 worst-case ~1e-5 (J/E), ~4e-6 (U) — inside the phase 1e-3 gate.
- **Frontier**: largest N under 10 s = 1e6 measured / ~1.4e6 implied
  (uj/tiled/f32); fused UJ+SFS ~1.24e6 implied (7.28 s at 1e6); F64
  ~0.68–0.87e6. Every series terminated on the 10 s budget by n ≤ 3.16e6;
  peak throughput 2.0e11 pairs/s (transcendental-bound). F32+rsqrt worth
  2.3× over F64; tiling 1.15–1.3×.
- **Fused SFS is cheap**: ujsfs/uj = 1.26–1.31× at saturated sizes — the
  exact SFS model rides on the pair list for ~30% of a UJ pass (no atomics
  priced), evidence for the 041b SFS-enablement track.
- **`opt` amendment (job 13246522)**: levers 1–3 pass the gates
  (`f64_opt` ≤ 2.7e-14; `f32_opt` ≤ tiled everywhere) and deliver
  1.6–1.7× (F32) / 1.3–1.4× (F64) at saturated sizes — 10-s frontier
  ~1.81e6 (U/J) / ~1.64e6 (U/J+SFS) F32 at 3.27e11 pairs/s, now
  FMA/load-bound. F64 `opt` is a large-n lever only (switch inert and
  blocking net-negative at n ≤ 3e4). Crossover vs FMM moves just
  4.5e3 → ~5.5e3 (√-scaling). Data: `sweep_tiled_opt.csv`.
- **Crossover** (corrected 2026-08-20 against the measured 041a fig15
  best-uniform curve, which shows 1.3–2.7 ms FMM steps at n = 1e4–3.16e4,
  not the 10–30 ms first assumed): **n ≈ 4–5e3**. FMM is 5× faster at 1e4,
  13× at 1e5, 62× at 1e6; brute force is a correctness reference above the
  crossover, not a performance alternative.

## Clear-context review (2026-08-20)

**APPROVED.** Reviewed only this task and its listed artifacts, per
`START_HERE.md` and the user's explicit scope. The implementation matches the
registered objective and production formulas: the U/J pass, transposed SFS
factorization, precision-specific reciprocal-square-root policy, per-series
10 s stopping rule, and post-completion `opt` amendment are represented in the
driver and CUDA kernels. Both shell scripts pass `bash -n`; a four-thread local
CPU smoke rerun reproduced the `identity_E` errors at `4.646e-15` (`n=1000`)
and `1.153e-14` (`n=10000`). Independent CSV checks found 75 valid base rows
and 76 valid A/B rows, consistent job IDs, and a worst reported F64 GPU error
of `2.683e-14`, safely below the `1e-11` gate. The report's terminal-size,
throughput, SFS-cost, and extrapolated-crossover claims agree with the raw
CSVs and clearly label the crossover as extrapolated. No significant defect
or improvement requiring a correction was found.
