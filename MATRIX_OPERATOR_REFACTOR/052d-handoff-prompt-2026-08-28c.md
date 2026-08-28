# Handoff prompt — compact filament/panel regularization derivation (2026-08-28, post P3.1–P3.3)

Task (Ryan, 2026-08-28): review the doublet/vortex-ring panel regularization
and the filament regularization in FLOWPanel and DERIVE a more compact one —
more compact in physical error footprint, therefore more performant in the
FMM (smaller guard radius → smaller near field). Example target: a TRUE
integration of the particle erf/Gaussian regularization over a line segment.
Ryan's framing: "what exists is only there for convenience's sake" — i.e.
the current families are ansatz denominators, not integrals of a physical
core, and a kernel change is explicitly on the table.

## Why this matters (the P3.2 finding that motivates it)

The 052d cross-pass prototypes (P3.1–P3.3, complete 2026-08-28; findings in
`052d-plan-2026-08-26.md` § "P3.1–P3.3 execution", memo in
`052d-p33-coverage-and-device-interface-2026-08-28.md`) proved the FMM
machinery sound (exact-once certified; truncation 1e-7 at P=8) but found the
production kernel is the accuracy blocker:

- `GaussianRegularization` (shipped default, Ryan ruling 2026-08-20)
  modulates the singular Biot–Savart segment kernel by `g(h) = 1 -
  exp(-h²/2rc²)` where **h = perpendicular distance to the segment's
  INFINITE LINE** (`h² = |r1×r2|²/|r0|²`;
  `../FLOWPanel.jl/src/FLOWPanel_elements_fmm.jl:997-1005`,
  `_bound_vortex_velocity`). The relative deviation from the singular kernel
  is `exp(-h²/2rc²)` — **O(1) inside an rc-cylinder around every edge line,
  extending to ARBITRARY distance along line extensions**. Multipoles
  represent the singular kernel, so this deviation is an irreducible FMM
  far-field error no expansion order can remove.
- Measured aggregate far-mismatch vs exclusion radius R at production
  step-472 shape (36,752 tri panels → 5,000-particle samples; rc =
  core_size_targets = 1e-3):
  - solved strengths: 2.3e-2 @5mm, 1.1e-3 @15mm, 1.36e-4 @3cm, 1.9e-5 @6cm,
    2e-6 @8cm (`prototypes/052d_cross_stencil/p32f_solved.log`);
  - random strengths (worst case, no adjacent-edge cancellation): 8.4e-3
    @5mm, 5.2e-3 @2cm, 1.2e-4 @7.5cm, ~4.7e-5 floor at 8–10cm
    (`p32e_guard.log`).
  - per-pair tail (`p32d_tail.log`): median collapses by ~6rc as a Gaussian
    should, but a max-population (line-aligned probe directions) persists
    O(1) to ~5–10cm.
- Consequence: the chosen cross-pass operating point needs a physical guard
  radius R_guard = 0.06 m demoting close M2L routes to direct (~3.3% of
  dense pairs ≈ 0.11 s device). With a truly compact kernel (error decaying
  with distance to the SEGMENT), R_guard could shrink to ~1cm-scale: the
  near-surface particle population is 0.26% of 242k within 1cm / 0.71%
  @2cm / 1.4% @3cm / 4.5% @5cm / 9.8% @8cm (p32d part B), i.e. guard cost
  0.008–0.05 s instead of 0.11 s, and ×4 better at 4-rotor.
- The shipped `radius_inflation` logic (Δr = rc·√(2z*) ≈ 5.90rc at tol 1e-6)
  bounds TRANSVERSE approach only and never covered the along-line channel —
  so the production θ-MAC host route also carries a residual of this
  mismatch (its measured 2–5e-5 relU comes from directly evaluating ~8% of
  dense).

## Required reading

1. `../FLOWPanel.jl/src/FLOWPanel_elements_fmm.jl:900-1160` — the
   `FilamentRegularization` enum + docstring (all families share numerator
   `c*q`, differ only in scalar denominator `D` (velocity) and `∇D = κ∇A`
   (gradient)); `_bound_vortex_velocity` (`:965-1010`) and the matching
   gradient kernel (`:1040-1060` region); `radius_inflation` for VortexRing
   (`:1154` region). Performance contract at `:958-965`: hot loops read the
   family ONCE per direct! call and cross a `Val{F}` function barrier —
   never per edge (a per-edge Ref read measured +34-49%).
2. `../FLOWPanel.jl/BRAINSTORM/025_kernel_regularization_update/` —
   `phase_00` (matched-core-size comparison: Gaussian peak velocity 0.45 vs
   compact 1.21 vs Vatistas 0.71, units Γ/2π/rc; peak gradient 0.50 / 2.55 /
   1.00) and `phase_01_theory.md` (derivations of the current D families).
   This is the doc trail the new derivation extends.
3. The three shipped families (all in `_bound_vortex_velocity`):
   - Vatistas n=2: `1/h² → 1/√(h⁴+rc⁴)` — algebraic tail, Δr = rc(2/tol)^¼
     (37.6rc pathology — why Gaussian replaced it).
   - Compact: `D = A + (h-rc)²B` for h < rc, EXACTLY A beyond — exactly
     singular for h ≥ rc, Δr = rc, tolerance-independent. NOTE: still
     h-based (infinite-line), so its mismatch lives in a thin infinite
     cylinder — nonzero measure but exactly zero outside h < rc. Quantify
     whether Compact ALREADY solves most of the FMM problem before deriving
     anything new (cheap A/B: `set_filament_regularization!(:compact)` and
     rerun the mismatch curve — see harness below).
   - Gaussian (default): Lamb–Oseen transverse profile, the along-line
     problem described above.
4. Source-panel side: constant-source panels use an erf-regularized kernel
   with `core_size_targets` (the "gradient-aware Gaussian radius rule",
   warning at `../FLOWPanel.jl/src/FLOWPanel_abstractbody.jl:1202`); the
   doublet-velocity kernel uses the compact-support `regularize` family
   (per the enum docstring). Constant-doublet panel ≡ vortex ring on its
   edges, so the panel doublet regularization is the filament one — one
   derivation covers both.
5. FMM-side context (skim): `052d-plan-2026-08-26.md` § "P3.1–P3.3
   execution — 2026-08-28" and the decision surface at its end;
   `052d-p33-coverage-and-device-interface-2026-08-28.md` §3b (how a guard
   radius enters the device route classifier — the smaller R_guard the new
   kernel allows, the cheaper the pass).

## The work

1. **Derive the line-convolved Gaussian kernel.** Convolve the singular
   filament Biot–Savart kernel with the particle Gaussian/erf core (the
   FLOWVPM ζ_σ blob), i.e. the velocity of a straight segment of a
   Gaussian-cored vortex — the physically-consistent object the current
   g(h) approximates. Expect an expression in the prolate/segment
   coordinates with erf/exp corrections that decay with distance to the
   SEGMENT (both transverse AND along-line) — that decay is the entire
   point. Provide: velocity D-form, gradient ∇D (both needed — the enum
   contract is (D, ∇D=κ∇A)), the h→0 and rc→0 limits (must reduce to the
   shipped singular kernel exactly), peak velocity/gradient at matched
   core size (extend the phase_00 table), and the radius_inflation rule
   Δr(tol) — which should now bound the error by SEGMENT distance, closing
   the along-line channel by construction.
2. **Prototype standalone, not in FLOWPanel.** `../FLOWPanel.jl` is a live
   worktree — READ-ONLY unless Ryan authorizes edits. Implement the
   candidate kernel(s) as standalone functions in
   `MATRIX_OPERATOR_REFACTOR/prototypes/` (new sibling dir, e.g.
   `052d_compact_kernel/`), validated against (a) brute-force numerical
   quadrature of the blob-line convolution and (b) the shipped families in
   their limits.
3. **Measure the FMM payoff with the existing harness.** The mismatch-curve
   machinery is built and fast (~10 s/curve):
   `prototypes/052d_cross_stencil/p32e_guard.jl` part 1 (binned pair pass,
   2000 targets; swap the per-pair kernel call for the candidate), and
   `p32d_tail.jl` for per-pair tails. Snapshot data + solved strengths:
   binaries under `/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472/`
   (solved sigma/gamma are cell data in `fm052d_gpu_1080_body1.472.vtu`,
   read via ReadVTK; column order ("sigma","gamma") per
   `FLOWPanel_abstractbody.jl:420`; body construction route incl.
   `calc_normals!`/`calc_controlpoints!`/`_set_core_sizes!` is in
   `p32e_guard.jl:make_body` — the SNAPSHOT_INDEX snippet omits those calls
   and builds a degenerate body). Deliverable: mismatch-vs-R curves for
   {shipped Gaussian, shipped Compact, candidate(s)} × {solved, random}
   strengths, and the implied (R_guard, guard cost) per kernel against the
   1e-4 ceiling with margin (ruling R4: report margins; 0.6 s/step gate).
4. **Report to Ryan**: the derivation (Math Syntax rules: $$ blocks,
   VS-Code renderable), the comparison table, the recommended family +
   R_guard, and the fingerprint consequences — changing the kernel changes
   the physics, so the locked 36-step gate fingerprint (CT ~7e-5 rel,
   Γ rms 5.4e-5) will shift; that acceptance is RYAN'S CALL, as is any
   edit to FLOWPanel. Do not modify FLOWPanel or start integration without
   his go.

## Key facts you'd otherwise re-derive

- Geometry: 36,752 tri panels × 241,986 particles, step-472 snapshot;
  dense 8.89e9 pairs; measured dense A100 rate 2.695e9 pairs/s (3.3 s);
  gate 0.6 s/step; leg is U-only in production; tolerance ceiling 1e-4 relU
  (may tighten, never grow — R4).
- rc = core_size_targets = 1e-3 (production pass-3 activates it via
  `_set_core_sizes!((body,), :core_size_targets)`); core_size_panel is
  R*1e-10 (inactive). Panel circumradius: median 1.79mm, max 2.2mm.
- Kernel body type: `RigidWakeBody{Union{ConstantSource, VortexRing}}`;
  random-strength protocol seeds: Random.seed!(472) for strengths,
  seed 99 / sort(shuffle(1:nt)[1:5000]) for the target sample.
- The 052d cross-pass operating point WITHOUT a kernel change (solved
  strengths): q=12 / ell_x=5 / P=6 / R_guard=0.06 m → 9.7e-6 relRMS,
  10.3× margin, ~0.11 s device near field. The new kernel's win = shrinking
  that R_guard (and removing the same residual from the production θ-MAC
  route). Unguarded fallback: same config, 6.1e-5, 1.65× margin, 0.031 s.
- FMM truncation at these shapes is NOT a limiter (1.1e-4 at P=3 →
  1.2e-7 at P=8, measured); the kernel tail is the whole game.
- Julia 1.12.5 JIT segfault flake in dense panel evaluation — rerun or
  `--check-bounds=yes`.

## House rules

≤4 local threads; delegate runs to julia-test-runner
(`JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:/Users/ryan/.julia
JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl script.jl > log 2>&1`
from the FastMultipole repo root); ../FLOWPanel.jl read-only; no commits,
no cluster submissions without authorization; never read raw CSV/bin —
script summaries only; long output → scratchpad logs; append findings to
`052d-plan-2026-08-26.md`; math per the user CLAUDE.md Math Syntax rules;
lab-notebook entry for the 052d arc still pending Ryan's approval — offer,
don't write. New code in `MATRIX_OPERATOR_REFACTOR/prototypes/`, not src/.
