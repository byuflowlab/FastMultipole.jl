# 052d — Response to plan review (2026-08-26; findings A–G, revised staging)

**Companion documents:** `052d-plan-2026-08-26.md` (original plan), `052d-plan-review-2026-08-26.md` (shared-grid review this responds to).

## Context

Ryan asked for a review of `MATRIX_OPERATOR_REFACTOR/052d-plan-review-2026-08-26.md`
("Revised 052d": shared world grid, role-specific trees, device-resident
cross-system FMM as primary). Goal: highly efficient cross-system
panels↔particles FMM on GPU.

**Correction from v1 of this review (retraction):** v1 claimed panel geometry
is static ("pure code motion"). That was a misread: the "pure code motion"
comments in `FLOWPanel_simulate.jl` (:630, :744, :909) are refactoring
annotations, not physics. In fact `propagate_kinematics!` is called every step
(`FLOWPanel_simulate.jl:1428`) and `rotate_translate!`
(`FLOWPanel_frames.jl:189-208`) physically rotates/translates `body.nodes`.
Panel geometry MOVES every step, and the design target is **4 rotors moving
relative to each other** (Ryan, 2026-08-26); the 052 single-rotor case
(`systems = (rotor,)`, example:1142) is the current acceptance vehicle, not
the end state. Findings below are rebuilt on these facts.

## Verified facts

- Timers at mature window (np≈209k, single rotor, total_step 5.79 s):
  `rotor_panels_to_particles` 1.98, `panel_cross_targets` 0.82,
  `wake_to_rotor_panels` 0.127, `wake_to_probes` 0.100 s/step
  (052c-plan:299–314). N_panels ≈ 3.7e4 (one rotor). At 4 rotors, panel counts
  and the panels→particles leg scale ~4×; cross-body pairs go from 0 to 12
  ordered pairs.
- Panel meshes rotate every step (`FLOWPanel_simulate.jl:1428`,
  `FLOWPanel_frames.jl:189-208`); rotor tip cells are crossed essentially
  every step at rotor tip speeds → panel world-grid occupancy changes per step.
- Solver direction (Ryan): one direct Backslash/ldiv solve per rotor, outer
  block iteration to convergence, cross-panel influence updated each outer
  iteration. Geometry is frozen *within* a step's solve; only strengths change
  across outer iterations.
- `BODY_HESSIAN_TO_PARTICLES` defaults `"false"` (example:441) → body→particle
  leg is U-only in production. Review's correction is right.
- Radix path: fixed geometric offset stencil classified once from grid
  geometry (`interaction_list_batched.jl:38-78`); materialized routes cached,
  regenerated on occupancy-epoch change (`containers.jl:824-836`,
  `translate_batched_cuda.jl:6910-6914`). `targets === sources` + homogeneous
  body_type/strength_dims/direct_kernel enforced (`containers.jl:2323-2325`,
  `FLOWVPM_fmm_radix.jl:46,378-393`).
- World grid derived from the live **particle** bounding box at cache build
  (`_radix_derive_bounds`, `FLOWVPM_fmm_radix.jl:417-444`); `recenter!` on
  out-of-box; growth-gated depth/sigma rebuilds (`:568-607,630-661`).
- `panel_cross_targets` label = pass-3 non-ParticleField targets
  (`FLOWPanel_gpu_influence.jl:648-656`); with one body this may be probes,
  not another body's panels — verify what it measures.

## Findings — improvements to the review doc

### A. State the real cache cadence: per-step rebuild, per-outer-iteration reuse
The review's validity model ("moving bodies within their current cells do not
invalidate a route cache") is the wrong emphasis: rotor panels cross grid
cells every step, so the combined-panel-tree keys/topology and all
panel-source occupancy epochs advance **every step**. The correct and valuable
amortization statement, which the review should make explicit, is:

- **Per step:** rebuild panel-side keys/sort/topology and cross routes once
  (cheap at 37k–150k panels; must still be priced and instrumented — route
  build is on the critical path every step, unlike the particle self-influence
  case where occupancy is comparatively stable).
- **Per outer Backslash iteration (within a step):** geometry frozen →
  strength-only epoch → reuse routes and topology exactly as the review's
  strength-epoch rules say; only B2M/M2M + cached M2L apply per iteration.
  With 4 rotors and k outer iterations, this is where the caching design pays
  hardest — say so, and gate the design on measured per-step route-rebuild
  cost rather than on rigid-motion route preservation.
- Drop or demote to future work the machinery for preserving routes across
  steps under rigid motion (transforming Morton keys, geometry gates for
  "moved within cell") — at rotor tip speeds it never triggers.

### B. Reinstate host-FMM Phase 1 as a decision gate (and possibly the shipped v1)
The review demotes host `fmm!` to "correctness reference" and bans the host
mirror without pricing it. Priced: D2H positions ≈ 6 MB (measured 0.054 s,
possibly already synced), H2D U-only accumulation ≈ 12 MB ≈ few ms; expected
leg cost 0.2–0.5 s on 64 host threads vs the 0.6 s gate; per-step host panel
tree rebuild is cheap at these panel counts. A 5–10 step probe with the host
path costs days, not weeks, and either (i) meets the gate → ship behind the
env flag, defer the device program, or (ii) bounds the (p, error, time)
envelope the device design must beat. Add an explicit kill-switch: device
architecture starts only if the host probe misses the gate at the 4-rotor
target scale (~150k panels → the host ceiling should be probed at that scale,
not just 37k).

### C. Scope the interaction matrix by measured ROI at the 4-rotor target
- particles→panels (`wake_to_rotor_panels`) is 0.127 s/step dense at 1 rotor,
  ~0.5 s at 4. The combined panel-control-point target tree (implementation
  step 4) is defensible at 4 rotors but marginal at 1; sequence it after the
  panels→particles leg proves out, and gate it on a 4-rotor dense baseline
  measurement, not assumption.
- The combined all-panel source tree is well-motivated for the **final
  panels→particles pass** at 4 rotors (one aggregated traversal instead of 4).
  For the **solve's cross-body influence** under Backslash-per-rotor block
  iteration it does not fit directly: each rotor needs "all others at latest
  strengths", i.e. per-ordered-pair routes (12 pairs), or a Jacobi-style sweep
  that can use the combined tree at lagged strengths. The review should pick
  one explicitly; recommend per-pair routes (matches its own block-GS section)
  with the combined tree reserved for the final pass.
- Verify what `panel_cross_targets` (0.82 s at 1 rotor) actually measures
  before designing around it — the label logic catches any pass-3
  non-particle target, possibly probes.

### D. Remove the coupled-Krylov prototype
Ryan's stated formulation is Backslash-per-rotor + outer iteration. Delete the
"Coupled-solve prototype" section and implementation step 6, and correct the
block-GS section's premise that "each body's FGS repeatedly evaluates its own
panel farfield" — with a direct per-body solve there is no repeated
self-farfield; the repeated work is cross-body influence once per outer
iteration (see A for why that's the caching sweet spot).

### E. Specify world-grid ownership and the rebuild cascade
The shared world grid is derived from the live particle bounding box and
mutates under `recenter!` / depth / sigma rebuilds. The review is silent on
the cascade: any particle-driven grid change must invalidate and rebuild all
panel-side keys/topology/routes (rare — 0 recenters in the 36-step mature
probe — but must be fail-loud). Additionally, panel-role admissibility on a
particle-sized grid needs a concrete check: leaf cell size comes from the
particle occupancy heuristic (~np^{1/3} per side); panel geometric extent +
regularization must be checked against near-stencil admissibility at the
panel role's chosen level, allowing the panel role to stop at a coarser level
of the shared level geometry. With 4 rotors the grid must also cover all four
hubs — bounding-box-derived cell sizes may differ substantially from the
single-rotor case; re-derive, don't assume.

### F. Gates should include route-build every step and the 4-rotor shape
The review's performance gates (leg < 0.6 s, ≥2× leg, ≥1.0 s net step) are
stated at the single-rotor shape. Add: (i) route/topology rebuild time counted
inside the leg every step (per A it is not amortized away); (ii) a 4-rotor (or
synthetic multi-body) scaling probe before committing to the architecture,
since that is the shape the design exists for; (iii) outer-iteration count ×
per-iteration apply cost as an explicit reported quantity.

### G. Smaller corrections
- The review silently resolves the tolerance-policy question (hold-locked vs
  re-pin) that the original plan flagged for Ryan's ruling; restore it as an
  explicit open question.
- Keep the review's confirmed corrections: U-only production body→particle
  leg, p=8 starting point, homogeneous-source-per-pass, no full host mirror in
  the device path (host mirror lives only in the Phase-1 host probe).
- The review's per-step execution step 1 (reuse particle upward pass for
  particles→panels) requires the particle upward pass to precede the solve
  (that leg is the pass-1 RHS); confirm step ordering permits it.

## Recommended revised staging

1. Cheap fact checks: what `panel_cross_targets` hits; per-step panel
   occupancy-change confirmation on the probe (expected: every step).
2. Host-FMM probe (original plan steps 1–3, U-only, per-step tree rebuild):
   parity testset + 5–10 step near-peak probe at 1 rotor, plus a host scaling
   estimate at the 4-rotor panel count. Gate: leg < 0.6 s and error compatible
   with the tolerance ruling. Pass → 36-step gate → ship behind env flag;
   device program deferred until 4-rotor scale demands it.
3. Device v1 (only if 2 misses or 4-rotor scale requires it): panel source
   tree(s) on the shared grid + two-occupancy-map implicit-stencil routes to
   the particle target tree; per-step rebuild cadence per A; U-only output;
   grid cascade per E; route-cost instrumentation per F.
4. Solve acceleration: per-ordered-pair cross-body routes with
   strength-only-epoch reuse across Backslash outer iterations (the 4-rotor
   payoff); combined all-panel source tree for the final panels→particles
   pass. Coupled Krylov removed.
5. particles→panels combined target tree last, gated on a 4-rotor dense
   baseline.

## Deliverable / execution plan after approval

Edit `MATRIX_OPERATOR_REFACTOR/052d-plan-review-2026-08-26.md` (or write a
companion `052d-plan-review-response-2026-08-26.md` if the review doc should
stay immutable — Ryan's call) incorporating findings A–G and the revised
staging. No code changes. Verification beyond staging step 1 is probe-job
work, not local.
