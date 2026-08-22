# Phase P handoff — 2026-08-22 context reset (item 048 focus)

Supersedes `handoff-phaseP-2026-08-21.md` for item 048; other items unchanged
from that file. Working dirs: this repo (branch `flowpanel-20260817`) and
`../FLOWVPM.jl` (branch `flowpanel`). Cluster: `ssh orc`, trees
`~/FLOWVPM-046` + `~/FastMultipole-046`, env `~/fm048env` (Julia 1.11.7
pinned — 1.12 crashes device JIT; CUDA.jl 6.3.0; normal H200 QoS only, no
`--qos=test`). Sync/submit driver: `FLOWVPM.jl/scripts/cuda_048_submit.sh`.
Preserve unrelated dirty-worktree changes.

## Item status

- 046, 047: approved and closed.
- 049: local remediation APPROVED (fresh review); pending H200 acceptance and
  the user's residency-mode choice after a TRUE same-job A/B. Do not launch
  049's acceptance until 048 is resolved and approved. Never auto-select the
  residency mode (no 15% threshold rule).
- 048: all 47 failures of job 13298230 are ROOT-CAUSED as test-methodology
  issues — no production-code defect found. Full analysis is recorded in
  `048-impl-gpu-sfs-enablement.md` §"Device-run failure analysis and
  resolution (2026-08-22)". Summary:
  1. "Replay J defect" (j~0.11) = statics accumulation measurement artifact.
     FLOWVPM `_reset_particles` preserves statics' U/J; every UJ delivery
     accumulates into all targets; nothing consumes statics' U/J. 3 statics
     × 7 extra evaluations = sqrt(3/20000)-per-call linear growth = 0.10984
     exactly. CUDA-graph replay is accuracy-faithful (diag jobs 13299959 and
     13302646; artifacts + sha256 in `data/gpu_sfs_enablement/`, scripts
     `FLOWVPM.jl/scripts/fm048_replay_diag{,2}.jl`, `fm048_diag_submit.sh`).
  2. Strict delivered-E gates (5e-4 F64/1e-3 F32) were transplanted from the
     host-matrix regime (n=1500, ell=2, near_radius2=20) to n=2e4 derived
     shell where E is J-error-bound (E/J 2.05 cube / 14.1 wake) — moved back
     to their valid regime (new strict testset) + tuning sweep for
     production selection.
  3. Allocation gates asserted at the wrong layer: host ~100 KB = CUDA.jl
     launch bookkeeping (~25-30 GPU ops/step, fixed, not n-scaled); device
     272-400 B = CUDA.jl accumulate!/maximum library scratch
     (`translate_batched_cuda.jl:229,:6682`,
     `translate_batched_resident.jl:2085`); SFS adds zero device alloc;
     warm `run_cuda_radix_lifecycle!` (graph replay) is ~zero-alloc.

## User decisions in force (2026-08-22)

- Tuning sweep to find settings that are efficient AND pass accuracy: gate
  is strict 5e-4 F64 delivered E **on the p018 production field**; cube grid
  runs too and the cube-vs-p018 discrepancy must be reported explicitly.
- Sweep rides in the SAME H200 job as the corrected acceptance testset.
- Pareto-frontier configs (plus P=4/derived-q baselines) run on p018.
- rho_t candidates stay 4.211 (velocity/U) and 4.789 (Jacobian/J); never
  silently change the production default. User picks final settings and
  residency mode from presented results.

## Local changes staged (uncommitted, parse-checked)

FLOWVPM.jl:
- `test/runtests_gpu_fmm.jl`: `fmm034_uj_errors(...; skip=())` + statics
  convention comment.
- `test/runtests_gpu_fmm_device.jl`: SFS testset U/J parity uses
  `skip=static_indices` (2 sites); allocation restructure (consts
  FMM048_HOST_WRAPPER_BAND=160_000 / _SFS=192_000,
  FMM048_DEVICE_SCRATCH_BAND=512, FMM048_HOST_ALLOC_BUDGET=4096 for the
  lifecycle layer; no-growth + sfs-adds-none checks; lifecycle-layer
  assertions via `ffmm.run_cuda_radix_lifecycle!(state)`); strict gate
  removed from n=2e4 loop (J-bound `e_gate` kept, e_sfs recorded); new
  replay gates (err_replay u/j <= 1.5x first-call; replay-vs-body parity via
  runtime `set_radix_setting!(:CUDA_GRAPH_LIFECYCLE,false)` flip); NEW
  appended testset "strict tail-budget operating point" (n=1500, P4/P8 x
  F64/F32 x rho 4.211/4.789, ell=2, near_radius2=20, gates 5e-4/1e-3 on
  first AND warmed calls).
- `scripts/fm048_tuning_sweep.jl` (NEW): cube n=2e4 F64 grid P{4,6,8} x
  rho{4.211,4.789} x q{derived,14,17,20} (q caps at 20 — supported rigid
  set), accuracy measured on first (uncaptured) AND warmed (replayed) calls
  vs exact direct reference (built-in replay-drift signal), warmed-median
  timings, n_direct; frontier + baselines; F32 spot checks; p018 arms with
  the strict gate + discrepancy report; CSV.
- `scripts/cuda_048_run.sh`: stage 4b runs the sweep with provenance
  hashing (stages: 1 FM device tests, 2 runtests_gpu, 3 coupling tests,
  4 A/B matrix, 4b sweep, 5 047 lock check).
- `scripts/fm048_replay_diag.jl`, `fm048_replay_diag2.jl`,
  `fm048_diag_submit.sh`: diagnostics (already run; keep for provenance).

FastMultipole:
- `MATRIX_OPERATOR_REFACTOR/048-impl-gpu-sfs-enablement.md`: resolution
  section added before the Verdict.
- `data/gpu_sfs_enablement/`: vpm048diag-13299959.out
  (sha256 70d3a9f5...), vpm048diag-13302646.out (dcf5398b...),
  fm048_diag_13299959.log (f15ba382...), fm048_diag2_13302646.log
  (61b605b6...), vpm048diag-13302371.out (autotune-flag crash, superseded).

## Defects FIXED and reviews DONE (2026-08-22, second session)

The known open defect (D1) plus two more found by the fresh-context full
review (report received intact after reset) were all FIXED, parse-checked,
and then VERIFIED by a second adversarial fresh-context review — no open
defects remain in the staged changes:

1. **D1 (fixed)** — replay-vs-body parity in `runtests_gpu_fmm_device.jl`
   now compares DELIVERED particle U/J (`Array(gpu.particles)[uj_rows,
   active_indices]`, global order, static columns excluded) after the
   replayed call vs after a graph-off call, gates 1e-10 F64 / 1e-4 F32.
   The `:CUDA_GRAPH_LIFECYCLE` flip is inside try/finally and restores the
   SAVED prior value via `ffmm.radix_setting(:CUDA_GRAPH_LIFECYCLE)` (note:
   the getter is `radix_setting`, not `get_radix_setting`). Verified: the
   flip is `:runtime`-class and cannot trip `verify_locked_radix_settings`;
   expected parity ~1e-13 F64 / ~1e-6-1e-5 F32 (F32 margin ~10x — watch it).
2. **D2 (fixed)** — `fm048_tuning_sweep.jl:load_snapshot` now zeroes SFS
   rows (`A[vpm.SFS_INDEX, :] .= 0.0`) at load: the p018 snapshot carries
   LIVE SFS in every column (|max| ~ 3.6e5, zero statics — empirically
   confirmed) and `Estr_direct!` accumulates, so the reference would have
   been contaminated and every p018 arm would have spuriously FAILED the
   strict gate.
3. **D3 (fixed)** — the `e_replay` measure was statics-diluted into
   vacuousness (identical static sentinels dominated the denominator). Now
   computed on active-column SFS deltas: `fmm048_relrms((S_replay .-
   S_before)[:, active_indices], Sref_delta[:, active_indices])` —
   `sfs=true` resets active SFS every call, so the delta is per-call E.
4. Guards added: `graph_live` logged in the strict tail-budget testset
   (records whether the n=1500 point actually replays a graph; not gated);
   frontier-cap comment aligned with code (keeps most-accurate configs).

Review verdicts: everything else in the change set (both repos) was
verified ready — helper APIs, scoping, run/submit scripts, strict-testset
feasibility, doc claims. Non-blocking residuals noted: host `e_fmm <= 5e-4`
gate at `runtests_gpu_fmm.jl:615` is bounded by legacy-FMM error (fine
today); F32 replay-parity margin ~10x.

## Job IN FLIGHT

**H200 job 13302961** submitted 2026-08-22 ~11:07 via
`bash scripts/cuda_048_submit.sh` (rsynced both trees incl. all fixes,
env unchanged, p018_710 snapshot shipped to `~/FLOWVPM-046/data/fm048/`).
Started RUNNING ~11:09; expect ~1.5-2.5 h through stages 1-4b + 047 lock
check. Output: `~/FLOWVPM-046/vpm048-13302961.out` (+ sweep CSV and
provenance hashes per `cuda_048_run.sh`). The user said THEY will announce
when it finishes — do not poll/monitor unprompted.

## Next steps (in order)

1. When the user says job 13302961 is done: check `sacct` state, retrieve
   ALL artifacts (job .out, sweep CSV, any stage logs) + sha256 into
   `data/gpu_sfs_enablement/`; update the 048 doc with results; verdict
   against the calibration expectations below. If a stage failed, root-
   cause before touching any gate (do not weaken delivered-accuracy
   evidence to make suites pass).
2. Present the sweep Pareto/p018 results + cube-vs-p018 discrepancy; the
   USER picks production SFS settings — never auto-select.
3. If stage 3 passed and 048 is approved: proceed to 049's H200 acceptance
   + true same-job stage-4 residency A/B; user picks residency mode.
4. Offer (do not write unprompted) a notebook entry per user CLAUDE.md.

## Expectations for the H200 run (calibration, not gates)

- n=2e4 cases: first-call and replayed u ~ 3.7e-4, j ~ 2.0e-3 (cube P4
  F64); e_kernel ~ 1e-15 F64; e_sfs ~ 4.3e-3 cube P4 (J-bound, recorded).
- Strict testset (n=1500/q=20): expect 9e-5-4.1e-4 F64 (host matrix
  passed there; device mechanical parity ~1e-15).
- Wrapper allocs: host ~(100480, 122704), device (384, 384) per
  job 13302646. Lifecycle: host << 4096, device 0 expected — if device != 0
  on the graph path, investigate before loosening anything.
- Sweep p018 arms: production P=4/derived-q rows will likely FAIL the
  strict gate (wake-like E/J ~ 14); that is the point — the frontier shows
  what settings/cost would pass. Do not weaken the gate; report.

## Update — 2026-08-22 third session (048 CLOSED, 049 ready)

- Job 13302961 FAILED at stage 4 (eltype-narrowing bug in
  `fm048_ab_benchmark.jl` specs vector, first job with p018 arms; stages
  1–3 all passed and matched calibration). Fixed (explicit Union eltype),
  resubmitted as **job 13303399: COMPLETED, all stages 1–5 green**.
  Artifacts + sha256s in `data/gpu_sfs_enablement/` (see
  `sha256_13303399.txt`); results recorded in the 048 doc.
- **048 COMPLETE and approved.** User selected production SFS settings
  (D14): **P=6, rho_t=4.789, derived q** (p018 e_sfs 1.46e-4, 3.4x gate
  margin, +3% cost). Implemented as coupling defaults in
  `FLOWVPM_fmm_radix.jl` (`expansion_order` 4→6,
  `_PARTITIONED_RHO_T_DEFAULT` 3.668→4.789); default-assertion tests in
  `runtests_gpu_fmm.jl` updated. Regression coverage of the new defaults
  rides in the 049 job.
- 049: harness `fm049_rotor_verify.jl` retargeted so the residency A/B,
  budget, and profile stages run at the production point (`P_PROD=6`,
  `RHO_PROD=4.789`, snaps 710:719, single rho — halves that stage);
  acceptance matrix (P4/P8 x F32/F64 x both rhos) unchanged. Parse-checked.
  `fm049_submit.sh` validates snapshots+manifest and is ready — submission
  pending (permission classifier blocks agent-side ssh submits; user runs
  `bash scripts/fm049_submit.sh` from the FLOWVPM.jl root). User picks the
  residency mode from the presented A/B — no auto-selection.
