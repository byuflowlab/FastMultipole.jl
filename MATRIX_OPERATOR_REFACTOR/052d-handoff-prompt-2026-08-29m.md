# 052d handoff (session 2026-08-29m): relU ROOT-CAUSED and CLOSED — next step: switch production to LineGauss

## Prompt for the next agent

You are continuing task 052d. The small-np relU investigation is FINISHED —
root cause found, verified, and the fix decided by Ryan: **switch production
to LineGauss filament regularization.** Your job is to implement that switch,
re-run the xverify short sim to confirm the relU drops to cross-pass
truncation (~5e-5), and report. Do NOT redo any diagnostics (see "Established"
below — the chain is complete and verified). Read memory `orc-cluster-access.md`
before ssh (bash -lc for slurm; auth expires — ask Ryan to run `! ssh orc echo ok`).

House rules: 4 threads max locally; delegate runs to julia-test-runner;
verify claimed numbers via verifier agent before reporting; GPU jobs for this
task are authorized (short limits; combine stages; see queue tactics below).

## The root cause (verified 2026-08-29, sessions k–m)

The 9.058e-4 device-vs-host relU at np=3544 is NOT a bug in the device cross
pass. It is a modeling gap:

- Production runs the DEFAULT filament regularization: **Gaussian (reg 3)**.
  `FILAMENT_REGULARIZATION = Ref(GaussianRegularization)`
  (FLOWPanel_elements_fmm.jl:958); nothing in the production env sets
  `FLOWPANEL_FILAMENT_REG`. 052d step 3 ARMED LineGauss (reg 4) in the device
  kernel but never SELECTED it — `_gpu_filament_reg()` = enum+1 = 3.
- The entire prior diagnostic chain (dense mirror, p39–p49) hardcoded REG=4,
  miscalibrating "truth". The phantom "~2e-3 near deviation" of handoff -l was
  purely this: with reg 3 the host replay of the production block lists
  matches the device near field to relL2 = 2.2e-14 (p50).
- With the CORRECT reg-3 dense: hfmm_vs_dense = 1.697e-6 (host fmm! is nearly
  exact — its `radius_inflation` guard covers the Gaussian deviation region),
  dev_vs_dense = 9.058e-4 = dev_vs_hfmm exactly (p39 REG=3). The whole relU
  is the device FAR field's miss.
- Mechanism (p51): GaussianRegularization's deviation from the singular
  kernel decays with distance to the infinite LINE, not the SEGMENT (the
  documented "open along-line error channel", elements_fmm.jl ~line 930).
  Just-shed particles sit on the extensions of their TE wake-arm lines; the
  nearest far-classified geometry is 26–37 mm (26σ+) away, yet the reg-3 far
  truth differs from singular by up to 2e-3 there. The cross pass's far field
  (singular multipoles) cannot represent this; no R_guard value can fix it
  (the channel extends arbitrarily far). Production farU already matches the
  LINEGAUSS far truth to 1e-6–2e-5 at the worst pids — LineGauss closes the
  channel by construction, so the fix makes the device far field exact.

## Established facts (do NOT re-derive)

1. Production near block lists ≡ proto near shell: ZERO set/multiplicity
   differences over all 3544 targets (p50 on job 13510897 dumps).
2. Device near kernel ≡ host direct on the exact production pairs, reg 3:
   relL2 2.2e-14, per-target ~1e-15 at all just-shed pids.
3. p39 REG=3 (5f dumps): dev_vs_dns 9.058e-4 | hfmm_vs_dns 1.697e-6 |
   dev_vs_hfmm 9.058e-4 | proto_vs_dns 9.249e-4 | proto_vs_dev 5.694e-4.
4. p51 at worst pids (3532–3544): |farU−far4| ≤ 2.1e-5, |farU−far3| up to
   2.0e-3 and ≈ |far3−far4|; zero far columns within 6 mm; mindist 26–37 mm.
   Old "partial sharing" puzzle resolved: pids 3532–3537 have ~1e-6 far error
   (their old deviation vs reg-4 dense was the shared Gaussian NEAR
   semantic); 3538+ carry the along-line FAR error, which the host does not
   share.
5. Wake arms extend up to 8.3 mm from the TE (median 4.6 mm) but are binned
   by panel centroid; leaf cell = 11.1 mm; all core sizes = 0.001.
6. Body: RigidWakeBody{Union{ConstantSource,VortexRing}} → wake kernel
   VortexRing → wake_tag=3 (NOT ConstantDoublet). ns=36752, np=3544 step.

## The fix to implement (Ryan's decision 2026-08-29)

Switch production to LineGauss. Selection mechanisms (pick with Ryan's
workflow in mind; env is cleanest for frozen drivers):

- **Env (preferred):** `FLOWPANEL_FILAMENT_REG=linegauss` — read at package
  load in FLOWPanel.jl:134 (`__init__` block), calls
  `set_filament_regularization!(:linegauss)`. Add it to the job env (e.g. the
  sbatch line or `FM052_PRODUCTION_ENV` in
  `~/FLOWVPM-052-h200/scripts/fm052_common.sh` on ORC — NOTE that file is
  labeled "immutable production environment"; prefer adding the env var in
  the 052d job scripts rather than editing the shared array, and ask Ryan
  before changing shared production defaults).
- Or in-code: `pnl.set_filament_regularization!(:linegauss)` in the driver.

Alignment is automatic once selected: `_gpu_filament_reg()` returns 4
(device near kernel arms/rings), host `_induced`/`_induced_wake` and
`radius_inflation` read the same global, and the reg-4 kernel is already
armed and certified in `_rect_panel_pair` (052d step 3, commit 1f34e3c3).

Validation plan:
1. Re-run the 5e-style xverify short sim WITH the env var (template
   `MATRIX_OPERATOR_REFACTOR/scripts/fp052d_step5e_evaldump.sh`, NREVS=0.05,
   timeout 25m, ~15 min runtime; add `FLOWPANEL_FILAMENT_REG=linegauss` to
   the env line). Expect the printed panel_cross_xverify relU trajectory to
   drop from ~9e-4 to ~5e-5 (cross-pass truncation) at np≈3544.
2. Optionally re-run p39/p50 with REG=4 on the new dumps — both mirrors are
   now env-parameterized (`REG=4` is their default) — expect dev_vs_dns ~5e-5
   and near replay ~1e-14.
3. Verify numbers via verifier agent, report to Ryan, then discuss with him:
   whether/where to make LineGauss the pinned production default, full-sim
   sizing, and the xverify-gate decision (both still deferred to discussion).
   Note the physics change: LineGauss is the exact blob-line convolution of
   the segment kernel with the FLOWVPM Gaussian core (documented in
   elements_fmm.jl ~line 975); cost 4 erf + 1 exp per edge vs 1 expm1.

## Where the evidence lives

- Dumps: `MATRIX_OPERATOR_REFACTOR/data/052d_relU_dumps_5f/` (job 13510897:
  ateval_np3544_{srcmat,wakemat,cent,positions,farU,meta} + NEW
  ateval_np3544_lists_* producer route/block lists + occupancies, plus
  dump_np3544_{deviceU,hostU,...}). Also on ORC:
  `~/FastMultipole-052-h200/relU_dumps_5e_13510897/`. Older sets: dumps/ and
  dumps_5e/ (jobs 13509236, 13510533).
- Diagnostics in `MATRIX_OPERATOR_REFACTOR/prototypes/052d_cross_stencil/`:
  p50_pairset_compare.jl (list replay + pair-set bit-compare; REG env),
  p51_far_error_attribution.jl (far reg-3 vs reg-4 truth + geometry scan),
  p39 (attribution; REG env). Run with `JULIA_NUM_THREADS=4 DUMPDIR=<dir>
  [REG=3] julia --project=/Users/ryan/Dropbox/research/projects/FLOWPanel.jl
  <script>`. Logs from this session in the session scratchpad (ephemeral).
- List-dump format: `ateval_np*_lists_meta.txt` lines are either
  `name eltype dimsxdims` (raw col-major .bin) or `key = value`; reader is
  `read_lists` in p50/p51 (round-trip tested).

## Queue tactics (learned this session)

eng/eng H200 was backed up (7 h ETA); `--partition=m13h --qos=gpu` reached
the 32-card H200 pool in 40 min. Submit to both, `scancel` the loser;
`sbatch --test-only` gives ETAs. Memory `orc-cluster-access.md` updated.

## Uncommitted state (do not lose; Ryan decides what/when to commit)

- FLOWPanel (branch `fastmultipole`): `src/FLOWPanel_gpu_influence.jl` has
  (a) session-k maybe_dump hook, (b) eval-time dump in `_cross_run_body!`,
  (c) NEW session-m list dump (`download_cross_lists` → _lists_*.bin +
  _lists_meta.txt, inside the `hit != 0` block). Staged on ORC
  (`~/FLOWPanel-052-h200/src/`, rsynced with --checksum).
- FastMultipole (branch `flowpanel-20260817`): session-k fixes
  (translate_batched_cuda.jl ulp tolerance, cross_stencil_cuda.jl:997 guard),
  prototypes p40–p51, scripts step5d/5e, REG env edits to p39/p50.
- ORC jobs session m: 13510897 (m13h/gpu, SUCCESS — the 5f dump run),
  13510874 (eng, cancelled duplicate).

## Notebook

PENDING RYAN'S APPROVAL: a substantial entry is warranted (reg
misattribution + exoneration of the device near path + along-line mechanism
+ p39/p50/p51 tables + the LineGauss decision). Ask Ryan how much detail,
draft via notebook-drafter, and NEVER write to the notebook without his
approval.
