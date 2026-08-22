# 050: Panel / multi-system GPU scoping — pricing and verdict

2026-08-21. Inputs: recorded facts (050 doc), 049 measurements (H200 job
13247848), 041k pair-rate ceilings, 023 profiling of the 018 step.

## What the 049 budget table changed

The historical production arm (wake U/J + SFS) observed 0.199 s/step
device-resident at n=210k. The old 0.3 ms
UJ/UJ+SFS difference is invalid as marginal cost: both arms executed ζ, so it
measured delivery overhead. Corrected same-state p018 A/B is pending for
rho_t=4.211 and 4.789. Until that rerun, neither the corrected particle budget
nor the binding pass is established. Independently, the measured ~36 s CPU
body-pass floor and ~16–21 s solve share establish that panel-involving passes
must move off the CPU:
(kerneloffset-radius-bound) plus the ~16–21 s solve share. Any option that
leaves a panel pass on the CPU at production shape misses the target.

## The structural fact that decides A vs B

Option (A) — lifting the radix v1 `targets === sources` restriction
(`translate_batched_resident.jl` `_assert_radix_targets_are_sources`;
`target_bodies` aliased) — is **insufficient on its own** for the panel
passes: the radix path also requires source homogeneity (one shared
`body_type`, `strength_dims`, `direct_kernel` across concatenated systems),
and panel elements (constant source/doublet tris, vortex
rings/sheets/filaments) are not `Point{Vortex}`. Making panels first-class
radix sources means a panel B2M + panel device direct kernel + heterogeneous
strength plumbing INSIDE the radix framework — strictly more work than (A)'s
already-large aliasing surgery, all before the first panel evaluates.

## The size regime makes cross-pass FMM unnecessary

The cross passes are rectangular and moderate:

- **wake → panels** (+trailing-wake targets): 2.1e5 point sources ×
  3.7e4 panel-center targets ≈ 7.7e9 gaussianerf pairs. At the 041k measured
  device rates (2.0–3.3e11 pairs/s) that is **0.02–0.04 s brute force** —
  already far under budget with zero tree machinery. An FMM would save at
  most ~30 ms/step here.
- **panels → particles** (the 36 s CPU floor): 3.7e4 panel sources × 2.1e5
  particle targets ≈ 7.7e9 panel-kernel pairs. Panel influence is ~5–20×
  the point-pair cost (tri geometry + doublet/source terms + ring
  filaments), giving an estimate of **0.4–2 s brute force** on one H200.
  Within budget; and this pass is where option (A) helps least (heterogeneous
  sources — see above).
- **panel self-solve** (9.3% ≈ 16–21 s CPU): flowpanel-20260817's
  `NearfieldInfluenceCache` already expresses the near-field as packed dense
  blocks; the production matrix (36,752² F64 ≈ 10.8 GB) fits H200 memory
  outright, so a device-resident dense (or blocked) matvec under the
  existing FastGaussSeidel/Krylov iteration is the natural lever
  (~3.6 ms/matvec at HBM bandwidth). Iteration counts at the production
  operating point are a 051 measurement, not assumed here.

## Options priced

| Option | Covers 36-s floor? | Est. dev cost | Est. step contribution | Verdict |
| --- | --- | --- | --- | --- |
| (A) radix targets≠sources surgery | NO (homogeneity still excludes panel sources) | large (aliasing surgery + panel-in-radix) | n/a alone | REJECT as the vehicle; re-scope only if (B') measures out |
| (B') system-on-system passes, cross passes as rectangular GPU brute-force kernels, solve via device-resident dense nearfield-cache matvec | YES | moderate (2 rectangular kernels + solve matvec port; no FMM framework changes) | ~0.05 + 0.4–2 + solve | **SELECTED** |
| (C) particles GPU, panels CPU 64-thread | NO | small | ≈ 60–80 s/step (body 25.3% + solve 9.3% of 170–230 s) ⇒ ceiling ≈ 2.6–3.4× | fallback only; states its ceiling honestly |

(B') is also the least invasive against FLOWPanel: it keeps the existing
3-pass `influence!` structure (`FLOWPanel_simulate.jl:673-712`) and the
per-pass kerneloffsets/derivative switches; the GPU replaces the *evaluation*
inside each pass, not the orchestration. The a-priori favorite from staging
survives contact with the measurements, with one amendment: the cross passes
use brute-force rectangular kernels rather than FMM evaluations (the
targets≠sources capability the GPU path lacks turns out to be unnecessary at
these sizes and budgets).

## Named 051 implementation shape

1. **Rectangular device kernel 1 — points → arbitrary targets**: gaussianerf
   U/J(+optional potential) from a particle set onto an arbitrary
   target-position set (panel centers, trailing-wake probes). Clone of the
   041k tiled kernel with separate source/target arrays; F64 with F32(+rsqrt)
   opt-in. Wired as the wake→bodies evaluation inside pass 1.
2. **Rectangular device kernel 2 — panels → arbitrary targets**: port
   FLOWPanel's `direct!` element influences (constant source/doublet tris +
   vortex ring filaments, the 018 driver's element set;
   `FLOWPanel_abstractbody.jl:1260` as the semantic reference) to a CUDA
   kernel over (panel, target) pairs. This retires the 36-s floor. Gate:
   pass-by-pass parity vs the CPU `direct!` at the 018 operating point.
3. **Solve**: measure first (051 stage): FGS/Krylov iteration counts and the
   flowpanel-20260817 nearfield-cache matvec cost at production shape; then
   device-resident dense/blocked matvec if it prices in, else CPU 64-thread
   solve as the bounded remainder (~16–21 s/step would still miss <1 h — so
   the measurement decides how hard to push; a partial lever is
   solve-every-N-steps/warm-start policies, priced in 051, decided with the
   user at 053 if it changes physics).
4. **FmmPlan / NearfieldInfluenceCache disposition**: reuse as-is on the CPU
   side (panel solve bookkeeping); the GPU rectangular kernels sit beside
   them, not inside the radix framework. RadixFMMCache remains
   particles-only (targets===sources), untouched.
5. Estr stays on the 048 radix SFS path; its corrected marginal cost is pending
   and must not be described as free.

## Watch items carried to 051/052

- SFS delivered accuracy on the real wake (0.666 vs exact-J reference,
  J-error-bound; production gate is 052's CT/Γ(r/R) vs the CPU arm, which is
  itself J-approximate — like-for-like comparison happens there).
- The 049 CPU-vs-GPU direct cross-check discrepancy (2.6e-4) — resolve
  before it can contaminate 051 parity gates.
- Residency: no recommendation yet. Present the corrected same-job A/B and
  ask the user which mode should ship; 051 must support either choice until
  that checkpoint is answered.
