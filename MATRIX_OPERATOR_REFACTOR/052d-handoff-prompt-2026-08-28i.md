# Handoff — Step-4 Stages A–F ALL CODE-COMPLETE; blocked on the Step-5 sbatch (2026-08-28i)

Context: session 6 (handoff `052d-handoff-prompt-2026-08-28h.md`) finished
the Stage-B TE-wake arm, then implemented Stages C (cross-M2L), D (L2L+L2B),
E (block-sparse near field), and F (seam dispatch), each CPU-validated and
recorded. Records: `052d-plan-2026-08-26.md` session-6 sections ("Stage B —
TE-wake dipole-triangle arm", "Stages C + D", "Stage E", "Stage F");
`052d-step4-subplan-2026-08-28.md` STATUS blocks. Read those first; do not
re-derive.

## How to work this handoff (process contract)

Attack steps ONE AT A TIME, in order. For each: develop your own lower-level
plan, execute, validate, append findings to `052d-plan-2026-08-26.md`, then
EITHER hand off via a fresh `052d-handoff-prompt-<date><letter>.md` OR
continue if context is slim. Delegate: test/script runs → `julia-test-runner`;
doc questions → `refactor-docs-librarian`; bulk exploration → Explore
subagents. Never pull raw logs/CSVs into context — script summaries only.

## State (do not re-establish)

- **ALL device stages A–F CODE-COMPLETE, UNCOMMITTED** across two repos:
  - FastMultipole: `src/cross_stencil_host.jl` (CrossStencilTables +
    cross_m2m_operators + NEW cross_m2l_operators/class_slots/level_scales/
    row_degrees + cross_l2l_operators), `src/cross_stencil_cuda.jl`
    (Stage A producers; Stage B B2M kernel WITH the TE-wake dipole arm via an
    8×n_panels wake matrix + M2M; Stage C refresh_cross_locals! —
    reference-level M2L tables + exact 2-power level rescaling, route-matvec
    kernel; Stage D finish_cross_locals! — octant L2L + L2B reusing
    _resident_local_eval_flat (classic-basis equivalence PROVEN by CPU test);
    Stage E apply_cross_near! — block-per-direct-block, production
    _rect_panel_pair math, wake arm, optional shift),
    `test/cross_stencil_test.jl` (487,683 assertions, ALL PASS).
  - FLOWPanel: `src/FLOWPanel_gpu_influence.jl` "052d STAGE F" section +
    dispatch hook (device route first, host-fmm! fallback, per-body cached
    state w/ frozen root box, XVERIFY relU print).
- **Per-step device call order**: refresh_cross_producers! →
  refresh_cross_multipoles!(xs, ctx, d_pbuf, d_wake) →
  refresh_cross_locals!(ls, ctx, xs) → finish_cross_locals!(ls, ctx, d_pos)
  → apply_cross_near!(ls, ctx, d_pbuf, d_pos; d_wake_buffer, wake_tag, reg).
  Output: ls.d_out (4×np: potential, ∇φ), original column order.
- **Key parity rulings established by reading production** (plan-doc
  session-6 records): NO wake_strength_shift anywhere on the host-FMM leg
  (dense pack_panels! path differs — pre-existing asymmetry); wake B2M
  coefficients land in the BODY PANEL's node (never the appended wake
  columns); tag-3 body panels remain a recorded far-field limitation.
- **GPU oracles staged** (prototypes/052d_cross_stencil/): p34 (Stage A),
  p35 (Stage B; case 3 = TE-wake vs the ACTUAL RigidWakeBody overload —
  requires FLOWPanel env), p36 (Stage C), p37 (END-TO-END expansions), p38
  (Stage E near + timing). CPU scratchpad harnesses (this session's
  28bb2295 scratchpad): wake_arm_cpu_check.jl (1.1e-16),
  wake_near_cpu_check.jl (6.3e-16).
- **Step-5 sbatch STAGED, NOT SUBMITTED**:
  `MATRIX_OPERATOR_REFACTOR/scripts/fp052d_step5_oracles_run.sh` — one H200
  job: p34→p38 + Stage-F XVERIFY production-shape smoke. Needs (a) Ryan's
  authorization, (b) both repos rsynced to the cluster
  ($HOME/FastMultipole-052-h200, $HOME/FLOWPanel-052-h200, fm052env-h200
  pattern, julia/1.11.7), (c) snapshot472 rsynced + SNAPDIR set. XVERIFY
  relU EXPECTATION ~1e-5 (two different approximations), NOT 1e-10.
- **Local test caveat**: FLOWPanel runtests_unit_panel_fmm.jl segfaults on
  local Julia 1.12.5 in the UNTOUCHED dense-reference stage (documented JIT
  flake, 4/4 incl. --check-bounds=yes) — validate on cluster 1.11.7 only.
- Operating point (certified, p32g): q=12 / ell_x=5 / P=6 / Rg=6 mm; seam
  envs PANEL_INFLUENCE_FMM_XQ/_XELL/_XP/_XRG (+_DEVICE, _XVERIFY).
- Lab-notebook entry for the 052d arc: STILL PENDING Ryan approval — offer,
  don't write.
- House rules unchanged: run pattern JULIA_DEPOT_PATH=/private/tmp/
  flowpanel-052b-depot:...; snapshot472 in the 1a9c539a session scratchpad
  (VERIFIED present 2026-08-28; re-export recipe in handoff-c); ≤4 local
  threads; NO cluster submission and NO commits without Ryan; math
  $$-blocks; prototypes under MATRIX_OPERATOR_REFACTOR/.

## Steps

**Step 1 — get Ryan's go-ahead, then run Step 5.** Ask Ryan to authorize the
combined sbatch (and whether to commit the session-6 work first — two repos'
worth of uncommitted changes is fragile). Then: rsync repos + snapshot472,
submit fp052d_step5_oracles_run.sh, digest results. Expected failure modes
worth pre-planning: (a) oracle tolerance misses from atomics ordering →
check whether worst-node errors sit just above 1e-10 (loosen with evidence,
don't hide); (b) p38 timing above the 0.039 s target → the recorded
optimization paths are per-class GEMM batching (Stage C) and warp-per-target
tiling (Stage E); (c) XVERIFY relU >> 1e-4 → real port bug, bisect stage by
stage with p36/p37/p38 configs at the production shape.

**Step 2 — Stage G leftovers** (after the sbatch is green): the θ-MAC
second-partition oracle (subplan G(iv), CPU-runnable) and the end-to-end
sampled dense reference at step-472 (relRMS ≤ 7.2e-6 + margin) if the
XVERIFY smoke isn't judged sufficient. Wire the p34 list bit-compare into
`PANEL_INFLUENCE_FMM_XVERIFY` if Ryan wants the per-step guarantee rather
than the job-time one.

**Step 3 — performance pass** (only if Step-5 timing misses): Stage-C GEMM
batching over route classes; Stage-E warp-per-target; particle-occupancy
capacity headroom to avoid per-step ctx rebuilds (recorded in the Stage-F
plan record); segmented LSD sort (Stage-A v1 deviation).

**Step 4 — multi-rotor tail (052b checklist)** — unchanged: sections D–F of
`052b-impl-multirotor-ige-gpu.md`; re-tune ell_x/Rg headroom at 4-rotor
scale.

## Key facts you'd otherwise re-derive

- M2L table memory trap: K=1740 push offsets → per-level dense is ~175 MB;
  the shipped design probes ONE level and rescales exactly (plan-doc Stage
  C+D record has the math). Rebuild triggers: h0 change only.
- The resident "flat" local basis IS the classic row basis
  (flat_basis_index, containers.jl:1224) — proven equivalent for LOCAL
  evaluation; the Stage-B warning still applies to operator/multipole
  RESIDENT tables elsewhere.
- Wake triangle composition (both far and near): tri1 (v[idx1], v[idx2],
  v[idx1]+Da), tri2 (v[idx1]+Da, v[idx2], v[idx2]+Db); strength = panel
  dipole strength (s1 for tag 2, s2 for tags 4/5); far = always Dipole;
  near = wake-kernel tag (3 ring / 2 doublet). Quad-diagonal choice is
  exactly immaterial.
- Route semantics: target = source + offset (flat node ids, level in
  route_levels, class = push-offset id); demoted+near blocks share one list
  (first n_demoted are demoted); node_ranges give contiguous subtree body
  ranges at every level.
- Hot-loop contract (FLOWPanel seam): NEVER read FILAMENT_REGULARIZATION[]
  per edge — Val(family) barrier (+34-49% regression otherwise).
- Julia 1.12.5 JIT segfault flake in dense panel eval — cluster 1.11.7 only.
