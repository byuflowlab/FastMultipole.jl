# 052d handoff (session 2026-08-29l): relU root cause narrowed to the NEAR/direct path — far field is exact

## Prompt for the next agent

You are continuing the 052d small-np relU investigation. The attribution
question from handoff 2026-08-29k is ANSWERED and a long chain of eliminations
(do NOT redo any of them — see "Eliminated" below) has narrowed the ~9e-4
small-np relU to a single component: **the production NEAR/direct evaluation
differs from the certified near mirror by ~2e-3 at just-shed particles, while
the production FAR field is exact to 5.7e-16.** Both production routes (device
cross near kernel AND the host fmm! direct path) share the same near deviation
almost digit-for-digit at several particles, so this is a shared close-pair
semantic, not a device bug per se. Your job: identify the exact near-pair
difference, then discuss the fix with Ryan. Read memory `orc-cluster-access.md`
before ssh (bash -lc for slurm; auth expires — ask Ryan to run `! ssh orc echo ok`).

User guidance in force: investigate autonomously ("go ahead and investigate"),
GPU jobs for THIS investigation are authorized (pattern: eng/eng H200, short
time limits; combine stages); full-sim sizing and the xverify-gate decision
remain deferred to discussion. User is intermittently reachable — surface
findings as you go. House rules: 4 threads max locally; delegate runs to
julia-test-runner; verify claimed numbers via verifier agent before reporting.

## Where the evidence lives

- Dumps (LOCAL, in-repo): `MATRIX_OPERATOR_REFACTOR/data/052d_relU_dumps/`
  (np=3544/12776/28389 from job 13509236: positions, hostU, deviceU,
  srcmat 17xS, wakemat 8xS, cent 3xS — all f64 col-major bins) and
  `MATRIX_OPERATOR_REFACTOR/data/052d_relU_dumps_5e/` (job 13510533:
  same-step np=3544 EVAL-TIME captures `ateval_np3544_{srcmat,wakemat,cent,
  positions,farU}.bin` + meta with the true production box and counters, plus
  the post-eval dump set). Also on ORC: `~/FastMultipole-052-h200/
  relU_dumps_13509236/` and `relU_dumps_5e_13510533/`.
- Diagnostic scripts (all in `MATRIX_OPERATOR_REFACTOR/prototypes/052d_cross_stencil/`,
  all run with `JULIA_NUM_THREADS=4 DUMPDIR=<dumpdir> julia
  --project=/Users/ryan/Dropbox/research/projects/FLOWPanel.jl <script>`):
  p39 (attribution), p40 (box scan), p41 (per-target forensics), p42 (column
  fits), p43 (semantics variants), p44 (wake basis fits), p45/p46 (routing
  fits), p47 (GPU replay, runs on ORC), p48 (input-delta fit), p49
  (far/near split — THE decisive one).
- Production box at np=3544 (from ateval meta): x_min =
  (-0.18023760190087498, -0.17793772306714212, -0.17793772306714215),
  h0 = 0.17793772306714215, n_routes=4268, n_demoted=0, n_blocks=682.

## Established facts (verified; do not re-derive)

1. Oracles p34–p38 ALL GREEN (job 13509236); ulp fix + p36/p37 host-ref fixes
   recertified. relU trajectory deterministic across runs (9.058e-4 at
   np=3544 in both 13509236 and 13510533).
2. p39 attribution (verifier-reproduced): at np=3544 dev_vs_dense=1.643e-3,
   hfmm_vs_dense=1.495e-3, dev_vs_hfmm=9.058e-4 (=cluster relU ✓);
   proto_vs_dense=5.3e-5. Deviations of dev and hfmm vs dense are ~84%
   correlated; at several just-shed pids (3536/3537 twins etc.) dev≈hfmm to
   5e-6 while both are ~2-3e-3 from dense.
3. Error is carried by the just-shed particles (highest pids, twin pairs across
   blades, d_TE≈0.007–0.02); top-10 targets carry 50–79% of ||dev-dns||².
4. Device kernels are exact: an all-pairs device `_rect_panel_pair` sum equals
   the CPU dense to 1e-13 (job 13510464 stage B).
5. A FRESH device cross replay on the dumped np=3544 inputs matches dense to
   5.3e-5 (≤2.5e-5 at all 13 worst pids) and does NOT reproduce the dumped
   deviceU; insensitive to box shifts/scale (stage C).
6. Eval-time inputs ≡ dump-time inputs BITWISE (job 13510533: srcmat/wakemat/
   cent/positions maxabsdiff = 0.0). No state mutation within the step.
7. **p49 far/near split at the exact production box: production farU ≡ proto
   far to 5.7e-16 (n_m2l=4268 matches production n_routes); production near
   (deviceU − farU) differs from proto near by 2.15e-3 relL2, and per-target
   |dnear| ≈ |dev−dns| exactly. At pids 3532–3537, |hfm−dns| equals |dnear|
   to 4 digits — the host direct path shares the SAME near semantic.**

## Eliminated (each empirically, scripts above)

- Frozen/shifted/scaled root boxes (p40, p47-C): proto/replay error stays
  3–6e-5 under every legal box; the true production box is now known anyway.
- Single omitted/scaled source column (p42); per-station wake strength shifts,
  TE bound filaments, outer free-wake filaments (p44); Da/mu/kernel/core
  semantic variants (p43 — baseline recipe optimal); whole-cell far-routing
  misassignment (p45/p46); per-panel strength/ruling input deltas (p48).
- Affine attached-wake add-on (`_add_affine_attached_velocity!`): applies only
  to body controlpoints and Kutta runtime wake probes, not the pfield.
- d_out staleness across steps: `finish_cross_locals!` does
  `fill!(ls.d_out, 0)` every step (cross_stencil_cuda.jl:992).
- Kernel math and inputs (facts 4–6).

## The open question (start here)

Same kernel (`_rect_panel_pair` reg 4), same inputs, same box — yet production
near ≠ proto near by 2e-3, and the HOST direct path (FLOWPanel `_induced` /
`_induced_wake` via the fmm! compat layer) agrees with the production DEVICE
near, not with proto/dense. Two families of explanation remain:

A. **Near pair-set difference**: production's near BLOCK list (682 blocks,
   (tnode,snode) pairs from `refresh_cross_producers!`) covers a different
   pair set than the proto near shell (`near_offsets` at leaf). p34 certified
   list parity on snapshots, but the production occupancy/epoch state could
   differ. Decisive test: `FM.download_cross_lists(ctx)` exists precisely for
   host-as-oracle bit-compares — instrument `_cross_run_body!` (the 5e eval
   dump hook is already in FLOWPanel_gpu_influence.jl, uncommitted) to also
   dump the block/route lists at the threshold step, then compare the block
   set against the proto shell pair-by-pair, and compute the direct sum of
   the symmetric difference — it should equal dnear exactly.
   IMPORTANT WRINKLE: any pure device-list theory must ALSO explain why the
   host fmm shares the deviation (fact 7). If the block sets match, think
   harder about family B.

B. **Shared close-pair kernel semantic**: both production DIRECT paths (host
   `_induced` and device `_rect_panel_pair` reg 4 — ported from it in 052d
   step 3) treat very-close pairs (just-shed particles vs their TE panels/
   arms) with some branch my dense mirror exercises differently — BUT note
   fact 4 makes this hard: the dense reference uses the same `_rect_panel_pair`.
   For the host side, check `_panel_fmm_evaluate!` config
   (`_panel_fmm_p()/_panel_fmm_theta()/_panel_fmm_leaf()` env defaults) and
   whether its direct near set at these probes could coincide with the device
   block set rather than the full near shell.
   Also worth checking: the `radius_inflation` MAC guard
   (FLOWPanel_abstractbody.jl:1194) — host-only, affects near/far membership
   in the HOST tree, cannot affect the device… so a shared-membership story
   needs something else. Keep p49's per-target equality (host dev near
   deviations equal to 4 digits at pids 3532–3537 but NOT at 3538+) in mind:
   partial, target-dependent sharing.

Suggested plan: (1) rerun the 5e-style short sim with a list dump
(download_cross_lists) at np=3500 (extend the existing eval-dump hook; job
script `MATRIX_OPERATOR_REFACTOR/scripts/fp052d_step5e_evaldump.sh` is the
template — NREVS=0.05, timeout 25m, finishes in ~15 min); (2) locally
bit-compare block/route lists vs the proto shell (adapt p49); (3) compute the
direct field of the pair-set difference and match against dnear; (4) if lists
match, pivot to family B with per-pair host/device/dense triple comparison at
pid 3536's near set. Then report and discuss the fix.

## Uncommitted state (do not lose)

- FLOWPanel (`/Users/ryan/Dropbox/research/projects/FLOWPanel.jl`, branch
  `fastmultipole`): `src/FLOWPanel_gpu_influence.jl` has (a) the maybe_dump
  hook from session k, (b) NEW `_PANEL_CROSS_EVAL_DUMPED`/
  `_panel_cross_dump_hit` + eval-time dump inside `_cross_run_body!`
  (srcmat/wakemat/cent/positions/box/counters + farU after
  finish_cross_locals!). Staged on ORC (`~/FLOWPanel-052-h200/src/`).
- FastMultipole (branch `flowpanel-20260817`): session-k fixes
  (translate_batched_cuda.jl ulp tolerance, cross_stencil_cuda.jl:997 guard)
  plus new prototypes p40–p49 and scripts step5d/step5e. Nothing committed
  anywhere; Ryan decides what/when to commit.
- ORC jobs this session: 13510386 (failed, API name), 13510464 (replay,
  SUCCESS), 13510533 (eval dump, SUCCESS). All on eng/eng H200.

## Notebook

Substantial results to offer for the notebook (oracle recert; attribution
table; the elimination chain; far-exact/near-differs finding). Ask Ryan how
much detail before drafting via notebook-drafter; NEVER write without
approval.
