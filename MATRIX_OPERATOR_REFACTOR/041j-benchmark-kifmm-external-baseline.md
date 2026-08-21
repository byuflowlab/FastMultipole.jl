# 041j Benchmark: Kernel-Independent FMM External Baseline (PVFMM)

## Status and entry gate

**Staged `2026-08-18` (user direction). Not started.**

Entry gate: `041a` complete and approved (checksummed snapshots and anchors),
`033` complete and approved (CPU baseline conventions). Independent of `041i`
and does **not** gate `042` (external-comparison row, mirroring the `037d`
convention). Benchmark/analysis row: artifacts under `scripts/`, `data/`, and
an external build prefix on the cluster; no production `src/` or FLOWVPM
changes. CPU baseline runs go on cluster CPU nodes, never the local machine;
local work at most four threads.

## Motivation

User direction (`2026-08-18`): evaluate a kernel-independent FMM (KIFMM) as
an alternative architecture for the regularized `gaussianerf` problem. The
staging research found **no maintained GPU KIFMM exists** (PVFMM's CUDA path
is 2015-era and M2L-only; `bempp/kifmm` GPU P2P is an open issue, #143;
ScalFMM-3 StarPU is "under development"; exafmm-t is stale, CPU, scalar), so
the plan is: benchmark the strongest open-source **CPU** KIFMM — **PVFMM**
(active 2026, LGPL, in-tree Julia bindings, built-in singular Biot-Savart
3->3, per-stage kernel hooks via `BuildKernel`) — and *predict* the GPU
limitations in writing.

Two questions this row answers with evidence, beyond raw timings:

1. **Required work for our kernel.** Every existing KIFMM handles a
   per-particle-`sigma` regularized kernel the same way FastMultipole's
   shipped partitioned nearfield does: singular kernel in the far field,
   `sigma` only in P2P (STKFMM "kernel aggregation"; its README requires
   `epsilon` much smaller than the smallest leaf — the same constraint as our
   split veto). The row documents exactly which PVFMM stages need custom code
   for gaussianerf U/J.
2. **The user's accuracy hypothesis.** Hope: kernel-independent expansions
   could represent the *regularized* field, allowing accuracy with a smaller
   direct list. The prior record bounds this — outside the blob support the
   regularized field is harmonic (any exterior representation is equally
   expressive); inside the Gaussian tail (`r <~ rho_t sigma`) the field is
   non-harmonic and the 031a §4 kernel-difference bound is basis-independent;
   `041d` part 1 measured NO-GO for "any linear basis" (break-even 28–1222
   sources vs admissibility cap ~7). One genuine nuance remains untested:
   KIFMM's P2M can evaluate the **true regularized kernel on the upward check
   surface** (source-side `sigma` handled exactly), which no solid-harmonic
   B2M can. Stage 2 tests this empirically instead of closing it by argument.

Reference context recorded at staging (for the report's comparison table):
equivalent-density KIFMM (Ying–Biros–Zorin; PVFMM, exafmm-t, kifmm-rs) uses
**no polynomial basis** — point-source equivalent densities on check/
equivalent surfaces; P2M/M2M/L2L are kernel evaluations plus precomputed
pseudo-inverses; M2L is a dense per-offset kernel matrix accelerated by
FFT-Hadamard products or SVD-compressed GEMMs; P2P is unchanged. The
interpolation/black-box branch (bbFMM/PBBFMM3D, ScalFMM Chebyshev, TBFMM
uniform) **does** use polynomials (Lagrange interpolation on Chebyshev tensor
grids). Operator reuse across levels requires kernel scale-invariance, which
a `sigma`-regularized kernel lacks (`scale_invar = false` forces per-level
operator tables in PVFMM).

## Objective

Benchmark PVFMM as a like-for-like external CPU baseline on our registered
cases at the phase's 1e-3 velocity gate, document the kernel-porting work,
empirically test the regularized-check-surface hypothesis, and deliver a
written per-stage GPU-limitations prediction — ending in a verdict on whether
any KIFMM lever merits further investment.

## Method

### Stage 0 — build

Build PVFMM (develop branch, CMake) plus its in-tree Julia bindings
(`julia/src/PVFMM.jl`) on a cluster CPU node; record the exact recipe
(modules, flags, commit hash) in the report. Fallback if the Julia bindings
prove immature: a small C++ driver reading our snapshot CSVs.

### Stage 1 — kernel work

- Far field U: built-in singular Biot-Savart (3->3).
- Far field J (`grad u`): choose by measured effort — (a) preferred: write
  the 3->9 Biot-Savart-gradient micro-kernel via `BuildKernel` (PVFMM's
  Stokes `vel_grad` 3->9 is the template), or (b) three Laplace-gradient
  passes on the vector potential.
- Nearfield: per-particle-`sigma` regularized gaussianerf U/J in the S2T/P2P
  hook (STKFMM RPY pattern: `sigma` as a 4th source dimension), using the
  shipped `g`/`h` forms from 031a/032.
- Deliverable: a "required work" section enumerating precisely which stages
  needed custom code, how much, and where PVFMM's abstractions helped or
  fought.

### Stage 2 — regularized-check-surface experiment (user hypothesis)

On isolated regularized clusters (drawn from the rotor `sigma` statistics and
a `beta = 2` synthetic), build the upward equivalent density two ways —
singular-kernel P2M vs regularized-kernel-on-check-surface P2M — and measure
delivered U/J accuracy versus target separation. Plot both against the 031a
kernel-difference tail bound. Verdict: does the `sigma`-aware P2M enlarge the
admissible region beyond the basis-independent bound, or confirm it?

### Stage 3 — benchmark

Accuracy/cost sweep (equivalent-surface order `m`, tree depth / max leaf
size) on the rotor, cube, and wake snapshots at the 1e-3 sampled-direct
velocity gate (same checksummed reference instrument as `033`/`041a`).
Report single-thread and full-node CPU timings against the `033`/`035` CPU
baselines, and — explicitly labeled apples-to-oranges — the ratio to the
H200 resident numbers.

### Stage 4 — GPU-limitations prediction

Written per-stage analysis of a hypothetical KIFMM GPU port, grounded in our
measured H200 stage costs: FFT-Hadamard vs SVD-GEMM M2L against our
per-offset-class GEMM machinery; pseudo-inverse P2M/M2M/L2L solves vs our
B2M/M2M kernels; identical P2P; per-level operator tables under lost
scale-invariance; refresh/latency floors (which dominate the rotor regime per
`041e`/`041h` staging evidence). Cite the no-maintained-GPU-KIFMM finding.

## Gates and verdict

- Accuracy gate: any timed PVFMM configuration must pass the 1e-3 sampled
  relative velocity RMS gate to be speedup-eligible (J logged as diagnostic,
  phase convention).
- Verdict: a recommendation — invest further in a KIFMM direction (with the
  specific lever named and a staged successor proposed), or close the
  direction with the measured evidence. Include the per-stage
  KIFMM-vs-FastMultipole comparison table (representation, operators,
  precomputation, kernel-dependence, polynomial basis).

## Artifacts

- `scripts/fm041j_*` drivers (snapshot export, PVFMM run harness, Stage 2
  experiment, plots per the standing TikZ/CSV conventions).
- `data/kifmm_external_baseline/` — build recipe, timing/accuracy CSVs,
  Stage 2 curves, `report.md` with the comparison table and verdict.
- External build prefix on the cluster (path recorded in the report; no
  third-party source vendored into this repository).

## Verification

- PVFMM outputs validated against the checksummed sampled-direct references
  used by `033`/`041a` (same instrument, same gate).
- Stage 2 curves reproduce the analytic 031a bound in the singular-P2M
  control before the `sigma`-aware variant is interpreted.
- All cluster timings carry job IDs; single-thread and full-node runs are
  same-node, same-build.
