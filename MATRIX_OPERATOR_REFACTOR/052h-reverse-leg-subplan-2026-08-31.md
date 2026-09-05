# 052h reverse-leg sub-plan — particles→panels cross-pass FMM — 2026-08-31

> **STATUS (session 2026-08-31s, same-day):** R0 DONE (LH builders in
> cross_stencil_host.jl + 554-test testset in cross_stencil_test.jl, all
> green; §2 covariance RESOLVED — diagonal blocks scale as n+n'+1, φ←χ as
> n+n' (one power weaker), χ←φ ≡ 0, giving a still-separable law with χ-row
> degree n+1 / χ-col degree n−1, `cross_m2l_level_scales_lh`). R1 DONE
> code-wise (build_reverse producer context + reversed sweeps + reverse
> download; p34 oracle extended with reverse bit-compare — GPU pending).
> R2/R3/R4 DONE code-wise (`DeviceCrossReverseState`, reverse Stage B
> vortex B2M via gathered sorted buffer + radix kernel reuse, LH M2L kernel
> with row/col scales, LH L2L + dual-channel L2B, Stage E
> `_cross_near_vortex_kernel!` + `apply_cross_reverse_near!`), all
> parse-clean, NOT yet device-run. HOST ORACLE p52 (prototypes, 6/6 PASS):
> end-to-end far+near vs production `fmm.direct!` with clean geometric
> P-convergence (max relU 3.0e-1 → 1.1e-3 over P=4→12 at q=3, ell=4; mean
> 2.1e-5 at P=12). Device replay p53 written for the GPU bundle (run p34 +
> p52-lib-driven p53 + stage-7 cuda_radix_interface_test together).
> R5 FLOWPanel wiring DONE code-wise same session (gates PANEL_WAKE_FMM/
> _DEVICE/_XVERIFY, _ReverseWakeEntry cache, pass1 hook in
> _gpu_rect_influence!; body targets only, probes stay dense; loads clean).
> **R6 GPU validation DONE (job 13518724 on h200, 2026-08-31): stage 1 =
> stage-7 052f/052g device testsets 1241 pass (after fixing a REAL 052f bug
> it caught — device construction runs the all-direct fallback before
> cache.state exists; sfs config now read from device_ctx.sfs_ctx,
> translate_batched_resident.jl _alldirect_geometry_fallback!); stage 2 =
> p34 36/36 (forward + REVERSE lists bit-exact vs host oracle, step-472
> geometry, 3 configs); stage 3 = p53 8/8 (device reverse pipeline vs host
> oracle: far dev 1.7e-15, near dev 2.6e-16, total at exactly the host P=8
> expansion error).** REMAINING: end-to-end s038v spot-check with
> PANEL_WAKE_FMM=1 + XVERIFY (needs a FLOWPanel/FLOWVPM silo sync — do NOT
> touch the silos serving running jobs), perf numbers, then default-on
> decision (Ryan).

Ryan approved implement-NOW (2026-08-31). Target: replace the dense
`pass1` leg (`wake_to_rotor_panels` 0.127 + `wake_to_probes` 0.100 s/step
at np≈209k on H200; ~0.5 s/step projected at 4-rotor) with a cross-pass
FMM leg, env-gated, dense fallback stays. Assessment context:
`052-handoff-prompt-2026-08-29o.md:71-96`; verified code map in
`052-handoff-prompt-2026-08-31s.md` (spot-checked this session).

## 0. Settled inputs (not revisited here)

- Forward cross pass (panels→particles) is the template end-to-end:
  producers (`refresh_cross_producers!` cross_stencil_cuda.jl:359), Stage
  B B2M+M2M (:760), Stage C M2L (:956), Stage D L2L+L2B (:987), Stage E
  near (:1170 `apply_cross_near!`), FLOWPanel gate/fallback/xverify
  chain (`FLOWPanel_gpu_influence.jl:1254-1345`).
- Operator tables are probed from production host operators with unit
  vectors (`cross_m2m_operators` cross_stencil_host.jl:124,
  `cross_m2l_operators` :214, `cross_l2l_operators` :308), currently
  `lhv=Val(false)` phi-only, `D = 2H`, `H=(P+1)(P+2)/2`.
- Route generation (`_cross_generate_level!` :305) iterates EXPLICIT
  source-side nodes and looks up targets in a DENSE `d_node_at`
  (currently particle-side only, scattered at :383-390). Stencil tables
  (`CrossStencilTables`, push/near offsets, class masks) are pure
  geometry — direction-agnostic.
- Leak-safe capacity pattern, `needs_rebuild` frozen-box discipline,
  fallback-before-any-output-write discipline: all carried unchanged.

## 1. Decisions

**(a) Particle multipoles: device vortex B2M on the cross grid**
(NOT radix-multipole reuse). Justification:
- `_cuda_b2m_vortex_leaf_nodes_kernel!` (translate_batched_cuda.jl:1249,
  phi+chi, dispatched for `Point{Vortex}` :1343-1357) is block-per-cell
  over exactly the cross occupancy's leaf shape (`d_node_ranges`,
  `d_node_centers`, `d_body_node`); the only adaptation is `d_perm`
  indirection (cross occupancies sort logically, not physically) — the
  same adaptation `_cross_panel_b2m_kernel!` already made.
- Radix reuse is cheaper at runtime but needs radix-tree↔cross-grid node
  mapping AND breaks whenever the radix root box ≠ frozen cross box
  (multipole recentering = extra M2M translations, new machinery). The
  B2M cost is O(np·H) — small next to the 0.227 s/step dense target.

**(b) Dual-channel (Lamb-Helmholtz phi+chi) through the cross pass.**
Vortex sources REQUIRE the chi channel (velocity = ∇φ + curl
contribution). Verified this session: forward-pass state is phi-only
(`D=(P+1)(P+2)` rows, `device_cross_expansion_state`
cross_stencil_cuda.jl:738). Design: keep phi and chi as SEPARATE
`D × max_nodes` matrices (matching the radix layout — the shared
evaluator `_resident_local_eval_flat(local_phi, local_chi, …, lhv)`
already takes them separately; forward pass passes `(locals, locals,
Val(false))`, cross_stencil_cuda.jl:890/:1016). Operator tables become
2×2 channel blocks `ops[row, col, ch_out, ch_in, …]` probed with
`lhv=Val(true)` unit vectors over both channels (the χ→φ mixing lands in
the off-diagonal block automatically; χ→χ diagonal; φ→χ block is zero).
Reverse-leg state structs are NEW (`DeviceCrossReverseExpansionState`
etc.) — do not widen the forward structs; the forward pass stays
phi-only and untouched.

**(c) U-only by default at panel control points** (matches forward-pass
semantics; the dense leg's J path stays dense). `U+J` behind
`PANEL_WAKE_HESSIAN_TO_PARTICLES=true` is a later stage — needs the
hessian evaluator `_resident_local_eval_flat_hessian`
(translate_batched_cuda.jl:3428), out of scope for the first landing.

**(d) Env gates**: `PANEL_WAKE_FMM` (default off) +
`PANEL_WAKE_FMM_DEVICE` (default on within) + tunables
`PANEL_WAKE_FMM_XQ/_XELL/_XP/_XRG` frozen per-entry via the
`_cross_config()` pattern (gpu_influence.jl:885-900), + debug-only
`PANEL_WAKE_FMM_XVERIFY` mirroring `PANEL_INFLUENCE_FMM_XVERIFY`.

## 2. Open risk to retire EARLY: LH level-rescaling covariance

Stage C stores ONE reference-level (L=2) M2L table per offset class and
rescales deeper levels separably (`cross_m2l_level_scales`
cross_stencil_host.jl:277: `scale2[row,L]·pow2lvl[L]·scale2[col,L]`,
exact for Laplace phi by degree counting). The χ→φ mixing term carries a
different power of the translation length (the chi-to-phi transform is
degree-shifting), so the SAME separable factor may be wrong for the
off-diagonal block. **Stage-0 unit test (host, cheap):** probe the LH
M2L at L=2 and L=3 for a handful of offset classes; check block-wise
whether `ops_L3 == rescale(ops_L2)` per channel block. Outcomes:
- Holds (possibly with a block-specific degree shift): keep ref-level
  tables, per-block scale tables.
- Fails: fall back to per-level LH tables for the off-diagonal block
  only (memory ×(ell_x−1) on ONE block), or all blocks if simpler.
Memory note: LH tables are 4× phi tables per slot (2×2 blocks); at P=6,
q=12 ref-level ≈ 4 × 44 MB ≈ 175 MB — fine on H200/GH200; `_XP` for the
reverse leg is tunable independently anyway.

## 3. Implementation stages

**Stage R0 — LH operator builders + covariance test (host, CPU-testable).**
`cross_m2m_operators_lh`, `cross_m2l_operators_lh`,
`cross_l2l_operators_lh` in cross_stencil_host.jl (probe with
`lhv=Val(true)`, 2×2 channel blocks); the §2 covariance unit test; host
oracle for a tiny particles→panels problem (extend the host cross oracle
prototypes under `prototypes/052d_cross_stencil/`). Exit: block
structure + rescaling law confirmed against direct host FMM on random
vortex clouds.

**Stage R1 — panel-side dense `node_at` + reversed route generation
(pure plumbing, CPU-testable via the existing host oracle pattern).**
- Add a second dense occupancy vector `d_node_at_panels` to
  `DeviceCrossProducerContext` (same `level_base` linearization; scatter
  from `ctx.panels` mirroring :383-390). `ell_x <= 8` guard already
  covers it.
- Parameterize `_cross_generate_level!` on (source occupancy, node_at):
  forward call sites pass `(ctx.particles-side d_node_at, ctx.panels)`;
  reverse passes `(d_node_at_panels, ctx.particles)`. Reverse lists get
  their own capacity-tracked route/block buffers (grow-only, leak-safe
  pattern) + counts `n_routes_rev`, `n_blocks_rev`, `n_demoted_rev`.
- Reverse route sweep runs in the SAME `refresh_cross_producers!` call
  (one keys/levels/geometry build feeds both directions), behind a
  `build_reverse::Bool` context flag so the forward-only user pays
  nothing.
- Class masks/offsets are reused as-is: a reverse route (src=particle
  node, tgt=panel node) with push offset o has the identical geometry
  as a forward route with offset o — the mask tables are per-offset,
  not per-direction.
Exit: host-prototype route parity — reverse lists from the device path
match a brute-force host enumeration on random two-cloud fixtures
(mirror `cross_stencil_test`'s census style).

**Stage R2 — Stage B reverse: vortex B2M on cross grid + LH M2M.**
Perm-indirected variant of `_cuda_b2m_vortex_leaf_nodes_kernel!` writing
into `(d_multipoles_phi, d_multipoles_chi)` over `ctx.particles` leaf
nodes; upward `_cross_m2m_kernel!` generalized to 2×2 block ops (or two
kernel launches per block pair — pick whichever keeps the kernel dumb).
Input: the wake particle device buffer (positions + Γ + σ) in whatever
layout FLOWVPM already ships to `_gpu_direct_batch!` — confirm rows at
wiring time (Stage R5).
Exit: device multipoles at leaf+ancestors match host `body_to_multipole!`
+ M2M oracle on fixtures (rtol ~1e-12).

**Stage R3 — Stage C/D reverse: LH M2L + L2L + L2B at panel centers.**
`_cross_m2l_kernel!`/`_cross_l2l_kernel!` generalized to channel-block
ops per §1(b)/§2; L2B via the existing `_cross_l2b_kernel!` passing
`(d_locals_phi, d_locals_chi, Val(true))` — it already threads `lhv`
(:890). Targets: panel control points; `ctx.panels.d_body_node` is
already populated when the forward Stage B ran this step — DO NOT assume
it (reverse leg may run with forward leg disabled): populate it
unconditionally in the reverse path (idempotent kernel, cheap).
Output: 4×n_panels `d_out_rev` (potential+U) in ORIGINAL panel column
order, single accumulate discipline.
Exit: end-to-end far-field parity vs host oracle on fixtures.

**Stage R4 — Stage E reverse near: vortex→point direct blocks.**
New `_cross_near_vortex_kernel!` shaped like `_cross_near_kernel!`
(:1042, block-per-route, shared-mem source tile) but tiling PARTICLES
(position+Γ+σ rows) against panel-CP targets, math from
`_vortex_pair_ug` (translate_batched_cuda.jl:2669); regularized (`REG`)
per the wake's kernel (gaussianerf σ) — confirm which regularization the
dense `_gpu_direct_batch!` pass1 leg applies and match it exactly, else
xverify will flag the near field.
Exit: near-block parity vs dense on fixtures.

**Stage R5 — FLOWPanel wiring + xverify.**
- Entry: the `pass1` branch of `_gpu_rect_influence!`
  (gpu_influence.jl:618-798). NOTE :744 — `fmm_bodies` is `!pass1 && …`,
  so today FMM never fires for pass1; the reverse gate slots in there.
- Build `_CrossReverseEntry` per body mirroring `_cross_entry!` :1097;
  reuse the SAME producer context as the forward entry when both legs
  are on (one grid, both directions) — context lookup keyed as today.
- Fallback discipline: all fallible work BEFORE the single accumulate
  into the solver RHS (`_gpu_add_result!` :490 path →
  `boundary_condition!` solver.jl:2748/:2707; probes :525/:540);
  `_gpu_route_fallback!` before any output write; partial accumulation
  mid-loop is a hard error.
- xverify: host mirror + relU print + `PANEL_FMM_DUMP_DIR`-style dump
  hooks; consumer script extension of `p39_relU_attribution.jl`.
Exit: xverify relU ≤ forward-pass levels on the s038v restart fixture;
step-time delta measured.

**Stage R6 — GPU validation (bundle, no separate queue wait).**
Bundle into the next GPU job that goes up anyway, together with the
still-pending device testsets (stage-7 `cuda_radix_interface_test.jl`
update from 31s): R1 route parity on device, R2/R3/R4 fixture parity,
end-to-end xverify spot-check, perf numbers.

## 4. Robustness must-carries

- Frozen-box `needs_rebuild` semantics: reverse lists rebuilt (or
  emptied) under the same signal; never partial-refresh.
- Grow-only device buffers (052-leak fix pattern); capacity assertions
  with the `what` tag ("reverse-route", "reverse-block").
- Zero-route/zero-block early-outs EVERYWHERE (052f/052g lesson: the
  empty case must publish zeroed counts and skip launches — write the
  early-return FIRST, test it explicitly with an empty-side fixture:
  zero particles, zero panels, all-near, all-far).
- Int32 overflow guards mirrored from the forward pass.
- `n_skipped`-style counters for any body the reverse B2M can't handle;
  nonzero ⇒ route fallback, never silent drop.

## 5. Sequencing & effort (2-4 sessions incl. GH200 validation)

R0+R1 this session (CPU-testable; R0 retires the one real theory risk).
R2+R3 next (device kernels, still fixture-testable on any CUDA box).
R4+R5 after, R6 bundled with whatever GPU job goes up. Perf target:
reverse leg ≪ 0.227 s/step dense at np≈209k; go/no-go on xverify relU
parity with the forward pass.

## 6. Out of scope (explicit)

- Hessian channel (`PANEL_WAKE_HESSIAN_TO_PARTICLES`) — later stage.
- Radix-multipole reuse seam — rejected in §1(a); revisit only if B2M
  shows up in profiles.
- Host (non-CUDA) production path for the reverse leg — dense fallback
  covers it (host cross oracle exists for testing only).
- ground panels (`wake_to_ground_panels`) — same machinery applies, but
  first landing targets rotor panels + probes only if the wiring splits
  cleanly; otherwise all pass1 rect targets at once (decide at R5).
