# Handoff prompt — LineGauss ship-prep + radius_inflation dispatch + 052d redesign plan (2026-08-28)

Context: the 2026-08-28 kernel-derivation session (handoff `052d-handoff-
prompt-2026-08-28c.md`) is COMPLETE. Its findings are appended to
`052d-plan-2026-08-26.md` § "Kernel derivation — line-convolved Gaussian
('LineGauss'), 2026-08-28" — read that section FIRST; it is the compressed
record of everything summarized below. Full artifacts in
`MATRIX_OPERATOR_REFACTOR/prototypes/052d_compact_kernel/`:
`DERIVATION.md` (math), `linegauss.jl` (validated implementation),
`k01_validate.jl` + `k01.log` (validation), `k02_mismatch.jl` +
`k02_random.log` / `k02_solved.log` (mismatch curves).

## What was established (do not re-derive)

- **LineGauss** = exact closed form of the singular segment Biot–Savart
  kernel convolved with the FLOWVPM Gaussian blob, using
  g(t) = erf(t/√2) − √(2/π)·t·e^(−t²/2) (odd extension):
  u = Γ/(4πh)·b̂·[z1·g(R1/σ)/R1 − z2·g(R2/σ)/R2
  − e^(−h²/2σ²)(g(z1/σ) − g(z2/σ))], i.e. shipped singular kernel × scalar
  W(h, z1, z2); D-form D = A/W. 4 erf + 2 exp per edge (local erf in
  linegauss.jl — SpecialFunctions is NOT resolvable in the FLOWPanel env).
- The shipped `GaussianRegularization` is EXACTLY the L→∞ limit of
  LineGauss; σ→0 limit is exactly singular. Deviation decays as
  poly·e^(−d²/2σ²) with d = distance to the SEGMENT (along-line channel
  closed by construction).
- Validation (k01, ALL PASS): quadrature 2.8e-11; singular assembly vs
  FLOWPanel 7e-15 (vel) / 3e-15 (grad, same ∂u_i/∂x_j convention as pnl,
  pinned by T3c); analytic ∇u vs FD 4e-6 (FD-conditioning-limited,
  quadrature-FD 1.6e-9); axis-guard seam 1.1e-7.
- Measured per-pair matching radius (max over directions incl. axial,
  vel AND grad, L̂ = 0.5/1.79/3 all alike): **Δr = 5.25σ / 5.75σ /
  6.0–6.25σ at tol 1e-4/1e-5/1e-6** — vs the shipped Gaussian fixed-point
  rule 4.99/5.47/5.90σ. Same shape, small polynomial-prefactor excess.
- Peaks at matched rc (phase_00 units Γ/2π): infinite line 0.4512 / 0.5000
  (identical to shipped Gaussian); L=1.79σ segment 0.1789 / 0.2205 (lower).
- Aggregate far-mismatch (k02, p32e part-1 protocol, step-472 snapshot,
  rc = 1e-3; ANCHOR: shipped-Gaussian column reproduces
  `prototypes/052d_cross_stencil/p32e_guard.log` bit-for-bit, and solved
  R_guard(3e-5) = 0.06 m = the P3.2 operating point; NOTE the 08-28c
  handoff's "8.4e-3 @5mm random" was a transcription slip — true value
  8.350e-2): **R_guard(≤3e-5): linegauss 6 mm on BOTH solved and random
  strengths** (6 mm also at ≤1e-5; machine-zero floor beyond 8 mm), vs
  gauss 60 mm solved / >100 mm random, compact 16 mm / 45 mm. At 6 mm the
  linegauss curve reads 2.35e-6 solved / 1.91e-6 random (~40–50× margin
  under 1e-4). Compact does NOT close the channel (infinite rc-cylinder).
- Implied payoff: guard population <0.26% of 242k (p32d part B: 0.26%
  @1cm) → guard cost ≲0.008 s vs 0.11 s at 0.06 m; ×4 better at 4-rotor;
  same residual leaves the production θ-MAC host route.
- Enum-contract consequence: ∇D ≠ κ∇A for LineGauss — the gradient needs
  two scalars (∂W/∂h, ∂W/∂z) or the cylindrical assembly
  (`lg_gradient` in linegauss.jl; formulas DERIVATION.md §5).
- Lab-notebook entry for the whole 052d arc: still PENDING Ryan approval —
  offer, don't write.

## The work (Ryan, this session)

1. **radius_inflation dispatch update.** `radius_inflation` at
   `../FLOWPanel.jl/src/FLOWPanel_elements_fmm.jl:1150-1165` already
   branches on `FILAMENT_REGULARIZATION[]`; the task is to make the rule
   family-correct once LineGauss exists as a fourth
   `FilamentRegularization` member (enum + docstring at `:900-943`,
   `set_filament_regularization!` + env hook `FLOWPANEL_FILAMENT_REG`,
   `Val{F}` hot-loop barrier contract at `:957-965` — NEVER read the Ref
   per edge, +34-49% regression). LineGauss's rule: same gradient-aware
   fixed point e^(−z)(1+2z) = tol, Δr = rc·√(2z), but CALIBRATE against
   the measured 5.25/5.75/6.0–6.25σ (the fixed point gives 4.99/5.47/5.90
   — slightly non-conservative; decide a prefactor or a +0.35σ pad and
   verify against k01's T7 scan). Crucially document the semantic upgrade:
   for LineGauss Δr bounds by SEGMENT distance, which is what the FMM MAC
   geometry actually measures — the h-based caveat dies. This is a
   FLOWPanel edit: Ryan's instruction authorizes it this session, but keep
   the diff minimal and show it to him before/at commit time.
2. **LineGauss ship-readiness re-check.** Audit before integration:
   (a) gradient path — port `lg_gradient` into the `_bound_vortex_gradient`
   assembly shape (the (D, κ∇A) contract must be extended; check every
   consumer of κ); (b) hot-loop cost — 4 erf + 2 exp per edge on host AND
   device (CUDA `erf` intrinsic exists; the prototype's series erf is
   host-only — decide what ships); (c) guards — endpoint zero-limit,
   axis series seam (`axis_guard`, threshold 1e-8·(1+min ẑ²)), Float32
   behavior on GPU (prototype is Float64-only; the guard thresholds and
   cancellation margins need a Float32 re-derivation or FP64 edge math);
   (d) semi-infinite wake filaments — `induced_semiinfinite`
   (`:1421-1425`): LineGauss has a clean z2→−∞ limit (g→−1); derive and
   test it if wake legs need the same family; (e) VortexRing
   scalar-potential branch stays compact-regularized (phase_01 known
   limitation) — confirm acceptable or scope the unification; (f) the
   36-step gate fingerprint (CT ~7e-5 rel, Γ rms 5.4e-5) WILL shift —
   physics changes within ~6rc of edges; re-acceptance is RYAN'S ruling,
   plan the A/B run that quantifies the shift; (g) `_set_core_sizes!` /
   `core_size_targets` plumbing unchanged (σ ≡ core_size, matched-rc
   convention). Re-run k01 after any port; keep k02 as the regression
   anchor (its gauss column must stay bit-identical).
3. **052d redesign plan.** Write the plan doc (append to
   `052d-plan-2026-08-26.md` or a new 052f doc — Ryan's naming call) for
   re-designing the cross-pass around the compact kernel. The decision
   surface changes: R_guard 0.06 → 0.006 m makes the guard ball vanish
   (<0.26% population, ≲0.008 s), so re-ask: does the q=12/ell_x=5/P=6
   operating point still win, or does a shallower/cheaper stencil now
   dominate (the guard dominated the old tradeoff; with it gone,
   truncation ~1e-7 at P=8 / 1.1e-4 at P=3 is the only knob)? Also fold
   in: the production θ-MAC host route's 2–5e-5 residual disappears with
   the kernel change — does the device cross-pass still pay for itself,
   and at what P? Answer "is the best architecture clear now?" explicitly:
   recommend ONE architecture (kernel + guard + stencil + P + route
   split), with margins per ruling R4 (report margins; 0.6 s/step gate;
   1e-4 ceiling may tighten, never grow). Use the existing harnesses
   (p32e part 2 accepts an Rg override; k02 for curves) rather than new
   machinery.

## Key facts you'd otherwise re-derive

- Geometry: 36,752 tri panels × 241,986 particles, step-472 snapshot;
  dense 8.89e9 pairs, A100 rate 2.695e9 pairs/s (3.3 s); leg U-only.
- Old operating point (no kernel change): q=12/ell_x=5/P=6/R_guard=0.06 m
  → 9.7e-6 relRMS, 10.3× margin, ~0.11 s near field; unguarded fallback
  6.1e-5, 1.65× margin, 0.031 s.
- rc = core_size_targets = 1e-3 (`_set_core_sizes!((body,),
  :core_size_targets)`); panel circumradius median 1.79 mm, max 2.2 mm.
- Snapshot binaries + solved sigma/gamma:
  `/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472/`
  (VERIFY it still exists — /private/tmp is volatile; if gone, the
  re-export recipe is in `052d-handoff-prompt-2026-08-28c.md`).
  Strength protocol: Random.seed!(472) strengths; seed 99 /
  sort(shuffle(1:nt)[1:5000]) targets; solved via ReadVTK cell data
  ("sigma","gamma") from `fm052d_gpu_1080_body1.472.vtu`; body build incl.
  `calc_normals!`/`calc_controlpoints!`/`_set_core_sizes!` — see
  `make_body` in k02_mismatch.jl.
- Run pattern: from the FastMultipole repo root,
  `JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:/Users/ryan/.julia
  JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl <script> > log 2>&1`.
- Julia 1.12.5 JIT segfault flake in dense panel evaluation — rerun or
  `--check-bounds=yes`.

## House rules

≤4 local threads; delegate runs to julia-test-runner; FLOWPanel edits ONLY
within the task-1 scope above (minimal diff, no commits without Ryan);
no cluster submissions without authorization; never read raw CSV/bin —
script summaries only; long output → scratchpad logs; append findings to
`052d-plan-2026-08-26.md`; math per user-CLAUDE.md Math Syntax ($$ blocks);
notebook entries need Ryan's approval; new prototype code stays in
`MATRIX_OPERATOR_REFACTOR/prototypes/`.
