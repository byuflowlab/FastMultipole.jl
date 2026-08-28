# Handoff — LineGauss shipped to FLOWPanel host; device cross-pass roadmap (2026-08-28e)

Context: the 2026-08-28 session-2 (handoff `052d-handoff-prompt-2026-08-28d.md`)
is COMPLETE. Its full record is `052d-plan-2026-08-26.md` § "LineGauss
integration (host port) + cross-pass redesign, 2026-08-28 (session 2)" —
read that section FIRST (including its same-day CORRECTION paragraphs).
Do not re-derive anything summarized there.

## How to work this handoff (process contract)

Attack the steps below ONE AT A TIME, in order. For each step: develop your
own lower-level plan for that step only, execute, validate, append findings
to `052d-plan-2026-08-26.md`, then EITHER hand off via a fresh
`052d-handoff-prompt-<date><letter>.md` for a context reset OR continue if
context is still slim. Keep context slim by delegating: test/script runs →
`julia-test-runner`; doc questions → `refactor-docs-librarian`; bulk code
exploration → Explore subagents (haiku for symbol hunts, sonnet for
conceptual). Never pull raw logs/CSVs into context — script summaries only.

## State (do not re-establish)

- **LineGauss is live on the FLOWPanel host path** (uncommitted):
  `../FLOWPanel.jl/src/FLOWPanel_elements_fmm.jl` (+227/−7, 8 hunks —
  enum member `LineGaussRegularization` (Int code 3 → gpu code 4), setter
  `:linegauss`, env `FLOWPANEL_FILAMENT_REG=linegauss`, `_lg_*` helper
  block, velocity branch, `_linegauss_gradient` cylindrical assembly,
  `radius_inflation` = Gaussian fixed point + 0.35rc pad ⇒
  5.34/5.82/6.25rc @1e-4/1e-5/1e-6, bounds SEGMENT distance).
  FastMultipole `src/direct_rectangular.jl` (+7): device route THROWS for
  code ≥ 4 (was: silent Vatistas fallback). NO COMMITS yet — Ryan must see
  the diffs (his standing rule).
- **Validation green**: `prototypes/052d_compact_kernel/k03_flowpanel_port.jl`
  ALL PASS (24 checks; parity residual 4.4e-8 fully attributed to
  erf-backend ulps × q̃-conditioning, κ_cond printed in k03.log); k01
  re-run 32/32; k02 anchor gauss+compact columns bit-identical
  (k02_random_rerun.log). Axis-robustness fix (deterministic C·M·[t̂]×
  fallback when |hvec| ≤ 1e-10·max(R1,R2)) applied to BOTH prototype
  linegauss.jl and the port — k03b/k03c logs document the bug.
- **Architecture (settled, session-2 plan section)**: device cross-pass at
  1-rotor AND 4-rotor (host θ-MAC route FAILED its measured gate: 15.8
  s/step vs 0.6 s, 99.8% pairs demoted; LineGauss does not change its pair
  geometry). Kernel LineGauss; guard demote Rg = 6 mm (suffix-safe ≤3e-5,
  BOTH solved and random strengths, 40–50× local margin); stencil
  q=12/ell_x=5/P=6 (predicted total ~1e-5, ~10× under the 1e-4 ceiling);
  near-field budget ~0.039 s. Expansion-array reuse REJECTED by design
  (panel leg U-only vs particle U+J; independent P) — cross-pass keeps its
  own small locals at ell_x; what IS reused: shared radix tree, occupancy
  lists, batched-M2L machinery.
- Lab-notebook entry for the 052d arc: STILL PENDING Ryan approval — offer,
  don't write.

## Steps

**Step 1 — certify the operating point (host-only, unblocked).**
Re-run p32e part 2 (`prototypes/052d_cross_stencil/p32e_guard.jl`) with
`FLOWPANEL_FILAMENT_REG=linegauss`, Rg_list = {0.006, 0.012} (hardcoded in
the script — small edit), `P32_STRENGTHS` solved AND random. Keep the gauss
run as anchor. Expected: q=12/ell_x=5/P=6 ≈ 1e-5 relRMS, ≥10× margin, both
protocols; P=4 ≈ 8e-5 (fails margin hygiene — confirming P=6). Report
margins per ruling R4 (1e-4 ceiling, may tighten to 5e-5, never grow).
Append the table to the plan doc.

**Step 2 — land the host diffs.**
Add LineGauss coverage to FLOWPanel tests: `runtests_unit_regularization.jl`
(REG_FAMILIES tuple + closed-form transverse profile + setter ArgumentError
update — note :bogus error string changed), `runtests_unit_fmm.jl`
radius_inflation testset (+0.35 pad values). Run the touched testsets. Then
show Ryan the diffs (FLOWPanel elements_fmm + tests; FastMultipole
direct_rectangular; prototype linegauss.jl axis fix + k03*) and commit on
his approval — separate commits per repo.

**Step 3 — LineGauss device arms (small, self-contained).**
Add `Val{4}` arms to `_rect_bound_vortex_velocity/_gradient`
(FastMultipole `src/direct_rectangular.jl:519-627`, shared by host fallback
and the CUDA kernel at `translate_batched_cuda.jl:~9033`); erf via FLOWVPM
`custom_erf64` (fdlibm polynomial, `FLOWVPM.jl/src/FLOWVPM_gpu_erf.jl`) or
CUDA intrinsic — decide and document; Float64-only (existing
RectangularPanelInfluence contract enforces it). Update `_rect_reg_val` /
`RectangularPanelInfluence(:linegauss)` from THROW to the new arm. Gradient
needs the cylindrical assembly (∇D ≠ κ∇A — two scalars ∂W/∂h, ∂W/∂z; port
from FLOWPanel `_linegauss_gradient`). Validate: k03-style parity FLOWPanel
host vs FastMultipole host-fallback rectangular path (CPU, no GPU needed);
GPU parity goes in the Step-5 sbatch.

**Step 4 — device cross-pass port (THE heavy item; write a sub-plan doc
first and get Ryan's look before coding).**
Route B per `052d-plan:571-652` and
`052d-p33-coverage-and-device-interface-2026-08-28.md`: cross-M2L kernel
with per-offset-class cached rotation operators (reuse
`interaction_list_batched.jl` / `translate_batched_resident.jl` /
`translate_batched_cuda.jl` machinery); separate downward pass (L2L +
U-only L2B at ell_x, existing kernels); block-sparse near-field reusing
`direct_rectangular!` + demote guard at Rg=6 mm; panel B2M host-side with
per-step coefficient upload; behind the `PANEL_INFLUENCE_FMM` seam.
Must-carry robustness items: explicit MAC-tie convention (r_S+r_T = θ·d
holds EXACTLY on the shared quantized grid; both ≤/< proven partition-safe
— pick one, document); ell_x > ell particle non-contiguity (host-built
per-cell index lists, ~1 MB Int32 upload, device gather/scatter);
per-step route rebuild (prototype: 0.015–0.05 s/step, exact-once coverage
at 8.89e9 pairs).

**Step 5 — GPU acceptance (cluster; needs Ryan's authorization to submit).**
One sbatch, staged (repo rule: combine GPU jobs): (i) device parity
harness vs host; (ii) 36-step gate A/B gaussian vs linegauss (CT rel,
Γ rms vs the locked fingerprint CT ~7e-5 / Γ rms 5.4e-5) → Ryan's
re-acceptance ruling; (iii) cross-pass timing at step-472 snapshot
(target ~0.039 s near-field, 0.6 s/step gate).

**Step 6 — multi-rotor tail (052b checklist).**
`052b-impl-multirotor-ige-gpu.md`: section D matrix-sharing parity audit;
sections E–F CUDA smoke runs + 414-step six-case acceptance. Re-tune
ell_x/Rg headroom at 4-rotor scale (~×4 guard leverage expected).

## Key facts you'd otherwise re-derive

- Run pattern (from FastMultipole repo root):
  `JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:/Users/ryan/.julia
  JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl <script> > log 2>&1`.
  Julia 1.12.5 JIT segfault flake in dense panel eval — rerun once.
- Snapshot472 binaries + solved sigma/gamma:
  `/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472/`
  (VERIFY it exists — /private/tmp is volatile; re-export recipe in
  `052d-handoff-prompt-2026-08-28c.md`). rc = core_size = 1e-3 m; 36,752
  tri panels × 241,986 particles; dense 8.89e9 pairs @ 2.695e9 pairs/s
  (A100) = 3.3 s.
- Sweep data for step 1 sanity: p32f_solved.log (gauss, Rg=0.06, q12/ellx5):
  P=3 3.79e-4 / P=4 8.51e-5 / P=6 9.74e-6; truncation anchors P=3 1.1e-4,
  P=8 1.2e-7. LineGauss far-mismatch @6 mm: 2.35e-6 solved / 1.91e-6
  random; machine floor beyond 8 mm (k02 logs).
- Hot-loop contract: NEVER read `FILAMENT_REGULARIZATION[]` per edge —
  `Val(family)` barrier (elements_fmm ~:965 comment; +34-49% regression).
- Known pre-existing wrinkle (flagged, unfixed): semi-infinite wake
  `_U_boundvortex_gradient` tracks the active family while its velocity
  sibling hardcodes Vatistas (elements_fmm ~:1552 vs ~:1608). LineGauss
  z2→−∞ limit is derived in the session-2 plan section if unification is
  ever scoped.

## House rules

≤4 local threads; delegate runs to julia-test-runner; no cluster
submissions without Ryan's authorization; no commits without Ryan seeing
the diff; FLOWPanel edits minimal and scoped to the step at hand; never
read raw CSV/bin — script summaries; long output → scratchpad logs; append
findings to `052d-plan-2026-08-26.md`; math per user-CLAUDE.md Math Syntax
($$ blocks); notebook entries need Ryan's approval; new prototype code
stays in `MATRIX_OPERATOR_REFACTOR/prototypes/`.
