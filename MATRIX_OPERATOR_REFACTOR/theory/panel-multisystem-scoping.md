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

## Reconciliation against the delivered 049 budget — 2026-08-22

The corrected particle budget now exists (H200 job 13305555,
`data/rotor_field_gpu_verification/results-13305555/fm049_budget.csv`,
production settings D14: P=6, rho_t=4.789; residency mode D15
upload-per-step selected by the user). This section re-derives the B'
arithmetic against it; nothing above is rewritten.

**Delivered particle-side numbers (n ≈ 209.6k–210.1k, steps 710–719):**

| Quantity | median | min | max |
| --- | --- | --- | --- |
| full_resident_rk3 (s/step) | 0.294 | 0.286 | 0.361 |
| full_upload_rk3 = D15 selected (s/step) | 0.306 | 0.299 | 0.373 |
| h2d_46xn / d2h_46xn (s) | ~0.0044 each | | |

Internal consistency: 3 × ujsfs_complete (0.0939) + rk3_integrator_residual
(0.0092) = 0.2910 = full_resident_rk3 at step 710 — the stage table sums.
Stage detail per UJ+SFS eval (step 710): nearfield 0.0557, SFS 0.0281,
m2l 0.0037, b2m 0.0043, l2b 0.0045, m2m/l2l/tree_refresh < 1 ms.

**What the delivered budget confirms or changes in the B' pricing:**

1. **Particle count assumption confirmed.** The 2.1e5 figure used for both
   cross-pass pair counts matches the measured np (209.6k–210.1k), so
   7.7e9 pairs stands for both rectangular passes.
2. **Particle side is 9.3% of budget (worst step 11.3%).** D15
   upload-per-step costs 0.306 s/step median (0.373 worst) against the
   3.3 s target, leaving ≈ 2.9–3.0 s for the panel-involving passes. The
   old 0.199 s figure (invalid marginal-cost arm) is superseded; the
   corrected number is *larger* but still small — no B' conclusion moves.
3. **B' total prices in with margin.** Pessimistic stack: 0.373 (particles,
   worst step) + 0.04 (wake→panels) + 2.0 (panels→particles, 20× panel
   multiplier) = 2.41 s ⇒ **0.89 s solve headroom ≈ 246 dense matvecs at
   the ~3.6 ms HBM estimate**. Optimistic stack: 0.30 + 0.02 + 0.4 =
   0.72 s ⇒ 2.58 s headroom. Even the pessimistic case leaves solve room
   far above plausible FGS/Krylov iteration counts; the 051 solve
   measurement remains the deciding gate but is no longer at risk of being
   squeezed out by the particle side.
4. **Estr/SFS marginal cost is now measured, not pending:** SFS is
   0.0281 s per UJ+SFS eval (~0.084 s per RK3 step, 2.6% of budget) —
   small but not free, as required by item 5 of the 051 shape.
5. **Residency watch item resolved:** the user selected D15
   (upload-per-step, +12 ms/step, +4.1%, parity 150/150 at 1e-11) for
   compatibility with monitors that trim/modify particles between steps.
   051 designs against D15 as the production mode; resident remains
   available.

**Effect on the verdict and the 051 shape: none.** B' stands as written;
the four-stage 051 shape is unchanged. The delivered budget strengthens
(C)'s rejection unchanged and removes the last pending input flagged in
"What the 049 budget table changed" above.

**Strategic note (user direction, 2026-08-22):** the multi-system radix
generalization (option-A-like unified `fmm!` with heterogeneous
source/target systems on the GPU) is a goal we intend to reach eventually,
in this phase or phase Q. B' is tentatively adopted as the path — it is a
step *toward* that generalization (rectangular targets≠sources kernels and
a device panel `direct!` kernel are prerequisites A would need anyway),
not a substitute that forecloses it.
