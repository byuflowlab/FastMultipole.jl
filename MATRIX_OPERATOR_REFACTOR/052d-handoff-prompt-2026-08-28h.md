# Handoff — Step-4 Stages A+B code-complete (device-native); B needs the TE-wake arm; C next (2026-08-28h)

Context: session 5 (handoff `052d-handoff-prompt-2026-08-28g.md`) settled
D1–D3 with Ryan, committed Step 3, and implemented Step-4 Stages A and B.
Records: `052d-plan-2026-08-26.md` §§ "Step-4 Step-0 — D1–D3 ruled",
"Stage A — device-native cross producers", "Stage B — device panel B2M +
upward M2M", "wake-partition ruling"; and `052d-step4-subplan-2026-08-28.md`
(stages carry STATUS blocks). Read those first; do not re-derive.

## How to work this handoff (process contract)

Attack steps ONE AT A TIME, in order. For each: develop your own lower-level
plan, execute, validate, append findings to `052d-plan-2026-08-26.md`, then
EITHER hand off via a fresh `052d-handoff-prompt-<date><letter>.md` OR
continue if context is slim. Delegate: test/script runs → `julia-test-runner`;
doc questions → `refactor-docs-librarian`; bulk exploration → Explore
subagents. Never pull raw logs/CSVs into context — script summaries only.

## State (do not re-establish)

- **Rulings (session 5)**: D1 = DEVICE-NATIVE producers v1 (Ryan overruled
  host recommendation). D2 = ≤ everywhere (near iff |o|² ≤ q; guard demote
  strict-<; REVISIT after the port — see FUTURE_IMPROVEMENTS.local.md, could
  strict-< improve near/far ratio?). D3 = `PANEL_INFLUENCE_FMM_XVERIFY=1`.
  Wake partition (Ryan): TE-wake dipole triangles (as composed by the
  RigidWakeBody B2M overload, FLOWPanel_liftingbody.jl:712-782) ARE part of
  the cross-pass B2M; wake rows behind them are NOT (they go to the
  wake-on-all step with the single TE filament row).
- **Step 3 COMMITTED**: `1f34e3c3` (LineGauss reg-code-4 arms, 4 files).
- **Stage A CODE-COMPLETE, UNCOMMITTED**: `src/cross_stencil_host.jl`
  (CrossStencilTables: far/demoted/near per-level phase masks, box-gap guard
  strict-<) + `src/cross_stencil_cuda.jl` (DeviceCrossOccupancy,
  DeviceCrossProducerContext, refresh_cross_producers! — keys → sortperm! →
  per-level occupancy → particle node_at → windowed flags/scan/compact for
  far routes + demoted blocks + near blocks; union-root-box fallback).
  KEY REUSE FACT: `_cuda_hier_route_flags_kernel!` takes node_at (target)
  and node_coords (source) separately → reused verbatim for two-occupancy;
  only `_cross_node_first/count_kernel!` and `_cross_route_compact_kernel!`
  (compact with running base) are new. Wired: eager include of the host file
  in src/FastMultipole.jl; lazy include of the cuda file in
  load_cuda_radix_lifecycle!. v1 deviation: full-key device sortperm! (not
  segmented LSD; identical result). Local: test/cross_stencil_test.jl PASSES
  (447,132 assertions incl. Stage-B operators); GPU oracle
  `prototypes/052d_cross_stencil/p34_stageA_oracle.jl` staged for the
  Step-5 sbatch (3 configs; bit-compare vs untouched CrossStencil.jl).
- **Stage B CODE-COMPLETE except the TE-wake arm, UNCOMMITTED**:
  `_cross_panel_b2m_kernel!` (thread-per-panel over the 17-row buffer, tags
  1/2/4/5 → Source/Dipole/SourceDipole phi-only, tag 3 & nv<3 counted in
  `skipped`, quads split (1,2,3)+(1,3,4)); `_crossb2m_*` device
  transcriptions of bodytomultipole.jl:39-278 recurrences (MArray scratch
  2×2×45, P ≤ 8); `cross_m2m_operators(P, h0, ell_x)` (host) — dense
  [D×D×8×ell_x] octant-class M2M probed from host multipole_to_multipole!
  with unit vectors (occupancy-independent → device-native upward; rebuilt
  iff union-box changed h0); `_cross_m2m_kernel!` bottom-up;
  `DeviceCrossExpansionState` + `refresh_cross_multipoles!` (REQUIRES
  refresh_cross_producers! same step). Coefficient convention: CLASSIC phi,
  row = 2·(harmonic_index−1)+re/im — Stage C tables must consume this (or
  add an explicit conversion; do NOT silently assume the resident basis).
  GPU oracle `p35_stageB_oracle.jl` staged (snapshot tag-4 case + synthetic
  mixed-tag/quad/skip case, tol 1e-10, independent host reference).
- Operating point (certified, p32g): q=12 / ell_x=5 / P=6 / Rg=6 mm.
- Lab-notebook entry for the 052d arc: STILL PENDING Ryan approval — offer,
  don't write.
- Run pattern + snapshot472 location + house rules: unchanged from handoff-g
  (JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:... ; snapshot in the
  1a9c539a session scratchpad, VERIFY it still exists, re-export recipe in
  handoff-c; ≤4 local threads; no cluster submission or commits without
  Ryan; math $$-blocks; new prototypes under MATRIX_OPERATOR_REFACTOR/).

## Steps

**Step 1 — finish Stage B: the TE-wake dipole-triangle arm (per the wake
ruling).** Read the RigidWakeBody B2M overload (FLOWPanel_liftingbody.jl:
712-782) to extract exactly which wake triangles the FIRST wake row
contributes and what per-panel data they need (the native buffer's trailing
`end-7..end` block). Then: (a) define the upload path — extend the seam
packing or a supplementary device buffer (FLOWPanel edits minimal); (b) add
the kernel arm (Panel{Dipole} triangle contributions per TE panel, same
`_crossb2m_*` recurrences); (c) add a p35 case with TE-wake triangles whose
host reference is the RigidWakeBody overload itself; (d) update the plan doc.

**Step 2 — Stage C: cross-M2L device kernel (THE long pole).** Own
per-(level, offset-class) operator tables at cross P=6 consuming the Stage-B
CLASSIC-phi coefficient convention; reuse the per-class GEMM drivers
(`translate_batched_cuda.jl:3560-4667`) only after shakedown-reading them —
`_build_cuda_hierarchical_context` (cuda:8130) is tied to the self-pass
grid/plan, so instantiate a cross context; the same identity-probing trick
used for `cross_m2m_operators` (probe host `multipole_to_local!` per
(level, offset) class) is the convention-safe way to build M2L tables and is
already validated by the Stage-B pattern. Targets = cross locals at ell_x
levels 2..ell_x over the Stage-A route lists (route_class = push-offset id k;
assign class bases when building tables). Then Stage D (L2L via the same
octant trick + U-only L2B through perm_x, no-hessian output kernel cuda:3372).

**Step 3 — Stage E (block-sparse near field)** wrapping `_rect_panel_pair`
(LineGauss Val{4}) over the Stage-A direct blocks (demoted + near), body
ranges via node_ranges + perm gather. ~0.039 s target at step-472 shape.

**Step 4 — Stages F–G**: seam dispatch (PANEL_INFLUENCE_FMM +
PANEL_INFLUENCE_FMM_XVERIFY), oracle harnesses, end-to-end sampled dense at
step-472 (relRMS ≤ 7.2e-6 + margin). All GPU validation (p34, p35, wake
case, C/D/E parity, timing) accumulates into ONE Step-5 sbatch — needs
Ryan's authorization to submit.

**Step 5 — multi-rotor tail (052b checklist)** — unchanged: sections D–F of
`052b-impl-multirotor-ige-gpu.md`; re-tune ell_x/Rg headroom at 4-rotor
scale.

## Key facts you'd otherwise re-derive

- Device kernel reuse map + panel B2M/host M2M call signatures: in the plan
  doc Stage-A/B records (session 5). Oracle call pattern:
  `multipole_to_multipole!(pe, pb, ce, cb, w1, w2, Ts, eimϕs, ζs_mag, Hs_π2,
  P, Val(false))` with `_cross_dummy_branch(center)` branches; update_Hs_π2!
  + update_ζs_mag! first (and ηs_mag/M̃/L̃ for M2L — see p32_harness.jl
  warmup block for the full M2L/L2L/L2B call set).
- 17-row buffer: row1 tag, row2 nv, rows3:14 verts (last repeated), 15:16
  s1/s2, 17 core_size (FLOWPanel_gpu_influence.jl:252-266). Tags:
  1=Source, 2=Doublet, 3=VortexRing, 4=Source+VortexRing, 5=Source+Doublet.
- Panel B2M is phi-only in production (vortex ring → dipole equivalence);
  chi never touched → cross pass carries no Lamb-Helmholtz.
- Stage-A lists: route/block class = plain push-offset id k (no class_base
  yet); blocks = demoted (first n_demoted) ++ near, at their emission level;
  subtree body ranges contiguous via the per-set sorted order (node_ranges).
- Hot-loop contract (FLOWPanel seam): NEVER read FILAMENT_REGULARIZATION[]
  per edge — Val(family) barrier (+34-49% regression otherwise).
- Julia 1.12.5 JIT segfault flake in dense panel eval — rerun once.
