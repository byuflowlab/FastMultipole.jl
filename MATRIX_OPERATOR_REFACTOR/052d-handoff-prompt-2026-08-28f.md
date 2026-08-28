# Handoff — LineGauss certified + landed; device arms next (2026-08-28f)

Context: the 2026-08-28 session-3 (handoff `052d-handoff-prompt-2026-08-28e.md`)
completed Steps 1 and 2 of that handoff. Full records are in
`052d-plan-2026-08-26.md` §§ "Step-1 certification — LineGauss guarded
operating point (p32g)" and "Step-2 — LineGauss test coverage + axis-guard
bug fix" (both 2026-08-28, session 3) — read those two sections FIRST.
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

- **Steps 1–2 of handoff-e are COMPLETE and COMMITTED.**
  - FLOWPanel commit `8b07f96` (branch `fastmultipole`): LineGauss port in
    `src/FLOWPanel_elements_fmm.jl` + test coverage in
    `runtests_unit_regularization.jl` / `runtests_unit_fmm.jl`.
  - FastMultipole (branch `flowpanel-20260817`): `src/direct_rectangular.jl`
    device-route THROW for reg code ≥ 4, `prototypes/` (052d_compact_kernel,
    052d_cross_stencil, 052d_shared_radix), and 052d doc updates.
- **Operating point CERTIFIED (Step 1, p32g)**: q=12 / ell_x=5 / P=6 @
  Rg=6 mm → relRMS 7.20e-6 solved / 2.08e-6 random (13.9× / 48× under the
  1e-4 ceiling; ~7×/24× under a 5e-5 ceiling). P=4 = 1.75e-4 solved (over
  ceiling → P=6 confirmed). At ell_x=5, Rg=6 mm demotes ZERO routes and
  Rg=12 mm is bit-identical — guard has free doubling headroom. Kernel
  mismatch floor eliminated (gauss plateaued at 6.1e-5). Logs:
  `prototypes/052d_cross_stencil/p32g_linegauss_{solved,random}.log`.
- **Axis-guard bug FIXED (Step 2)**: `_lg_axis_guard` / prototype
  `axis_guard` now `ĥ² < 1e-7` FIXED (old min-ẑ²-scaled form dropped the
  O(ĥ²) correction for long segments; zero impact on 052d panel-scale
  data). Port + prototype in sync; k01 seam probes moved. Validation green:
  k01 34/34, k03 25/25 (parity 4.44e-8 vel / 1.60e-8 grad — erf-backend
  ulps), regularization tests 345/345, fmm tests 71/71.
- **Architecture (settled)**: device cross-pass at 1-rotor AND 4-rotor
  (host θ-MAC route failed its 0.6 s/step gate at 15.8 s). Kernel
  LineGauss; guard demote Rg = 6 mm; stencil q=12/ell_x=5/P=6; near-field
  budget ~0.039 s. Expansion-array reuse REJECTED (panel leg U-only vs
  particle U+J; independent P) — cross-pass keeps its own small locals at
  ell_x; reused: shared radix tree, occupancy lists, batched-M2L machinery.
- Lab-notebook entry for the 052d arc: STILL PENDING Ryan approval — offer,
  don't write.

## Steps

**Step 3 — LineGauss device arms (small, self-contained).**
Add `Val{4}` arms to `_rect_bound_vortex_velocity/_gradient`
(FastMultipole `src/direct_rectangular.jl:519-627`, shared by host fallback
and the CUDA kernel at `translate_batched_cuda.jl:~9033`); erf via FLOWVPM
`custom_erf64` (fdlibm polynomial, `FLOWVPM.jl/src/FLOWVPM_gpu_erf.jl`) or
CUDA intrinsic — decide and document; Float64-only (existing
RectangularPanelInfluence contract enforces it). Update `_rect_reg_val` /
`RectangularPanelInfluence(:linegauss)` from THROW to the new arm (that
THROW is at `direct_rectangular.jl:127-133` and `:139-146`). Gradient
needs the cylindrical assembly (∇D ≠ κ∇A — two scalars ∂W/∂h, ∂W/∂z; port
from FLOWPanel `_linegauss_gradient`, elements_fmm ~:1098). Carry the
FIXED axis guard (ĥ² < 1e-7), not the old ẑ²-scaled form. Validate:
k03-style parity FLOWPanel host vs FastMultipole host-fallback rectangular
path (CPU, no GPU needed); GPU parity goes in the Step-5 sbatch.

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
- LineGauss kernel structure (both prototype `linegauss.jl` and port):
  velocity u = c·M/(4πσ²L) via `_lg_M` (branches: ĥ²=0 axis / wholly-small
  R̂ series / endpoint split / axis guard ĥ²<1e-7 / general 4-erf+1-exp
  form); gradient = cylindrical assembly from duθdh, duθdz, uθ/h with the
  deterministic C·M·[t̂]× fallback when |hvec| ≤ 1e-10·σ·max(R̂1,R̂2).
  Gaussian family = exact infinite-line limit (test closed form).
- FLOWPanel commit scope note: `src/FLOWPanel.jl` working-tree hunks are
  UNRELATED (HybridWakePotential exports, particle_body_overlap) — leave.
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
