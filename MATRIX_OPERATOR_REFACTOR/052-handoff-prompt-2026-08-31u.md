# 052 handoff (session 2026-08-31u): gpu40 blowup forensics done — NEXT: root-cause the step-995 ignition

## Prompt for the next agent

You are continuing task 052 after a context reset. Session 2026-08-31u
validated the 052h reverse leg in production geometry (spot-check job
13518964: relU ~2e-5, timing a wash at small wake), merged the gpu40
data dirs (`.todelete` retired), and ran blowup forensics on the gpu40
run: **the wake ignites abruptly at step 995–997, not 1058**. Your
mission, set by Ryan: **find the exact cause of the ignition.**
First actions, in order:

1. **Re-arm monitors** (recipes below; they died at reset). Probe:
   `ssh orc 'bash -lc "source /etc/profile; echo; bash ~/st052_probe.sh"'`
   — ids are 13518861 (LG) and 13518479 (cpu40-r3).
2. **Babysit cpu40-r3 13518479** — now the ONLY running job and the
   KEY control (host-only, clean at step 733). Watch its wake health
   around steps 985–1060: if it passes gpu40's ignition point cleanly,
   the GPU stack (or a GPU-config difference) becomes the prime
   suspect. (LG 13518861 was KILLED on Ryan's ask at ~09:2x MDT after
   we found it had blown at ~516 — see §Control results; its data
   through step ~1049 remains in the 052 silo for analysis. Do not
   re-arm the LG log monitor.)
3. Investigate the gpu40 ignition itself (leads below).

## The gpu40 blowup: established facts (this session, from monitors)

Stitched monitors (1476 steps each) live locally at
`~/Dropbox/research/projects/FLOWPanel.jl/plans/sigma_vpm_illustrations_20260827/gpu40_monitors/`
(monitor02 force, monitor03 bound circulation, monitor04 wake health).
Never read them raw — script summaries only. Timeline (monitor04):

| step | event |
|---|---|
| ≤994 | healthy: max_u 17–33, Γ/σ² 3e3–2e4, min-σ ratio 0.10–0.12; n_particles ~193k. Transient max_u spikes ~858–875 (to 400+) recovered fully. |
| 995–997 | **ignition**: max_u 25→55→132→1094 over 3 steps; Γ/σ² ×10. σ still normal — velocity/vorticity spike comes FIRST. |
| 998–1004 | min-σ ratio collapses 0.111→0.021 (step 998); max_u hits 4.0e4 at 1004. σ_max grows (reached 0.2066 by 1060). |
| 1058–1060 | integrated forces finally deviate (CFx −0.073→−0.093 at 1058; monitor02); σ_max trips the FMM near-set adequacy gate at step 1060 → the original crash (ArgumentError, translate_batched_resident.jl:2104). ~60-step lag = ignition was localized away from the rotor. |
| 1062–1070 | wake self-destructs: n_particles 168k→16k (particles flung out), max_dtZ 0.08→3000+. |
| 1060–1475 | restart (13518480, with 052f/052g demotion → all-direct, ~26 s/step) ran on the contaminated state; entire segment physically garbage. CT 218.171 ± 4.27 / Phase-2e non-convergence readouts are meaningless. |

Crash-chain context: restart 13512297 died at 1060 on the adequacy
gate (σ_max=0.2066, ρ_t·σ_max=0.9896 > gap 0.5545, "row 032a");
13513892 died on the zero-M2L cache TypeError (fixed by 052g);
13518480 ran through. "052g production-validated" = mechanically only.

## Control results (late session 2026-08-31u — REFRAMES the mission)

- **LG 13518861 (LineGauss, GPU, 052 stack, fresh start): ALSO blew
  up, at step ~490–516.** Wake health (its monitors:
  `~/FLOWPanel-052-h200/data/scr_p019_s038v_gpu40/monitors/` — same
  run name as gpu40 but inside the 052 silo): healthy to ~450 (max_u
  ~20, Γ/σ² ~1.4e3), max_u 66 / Γ/σ² 1e5 at 500, u=2e4 at 550; first
  force excursion at step 516; forces garbage from then on (CFx −166
  at 900, −68 at 1044). Distinct behaviors vs gpu40: min-σ ratio
  pinned at 1.000 (σ dynamics totally different under this config; no
  σ growth → no adequacy-gate trip, no demotion — zero "Falling back"
  lines), and it wanders in a semi-blown state (u drops back to ~70–
  220 by 600–650) instead of self-destructing. Steps slowed 3.9→12–13
  s/step by ~step 900 (unexplained by demotion). **So LineGauss does
  NOT fix the instability — it ignites even earlier.**
- **cpu40-r3 13518479 (host-only): CLEAN at step 733** (CFx −0.074,
  max_u 24, Γ/σ² 1.3e3, σ-ratio 0.133) — already well past LG's
  ignition step. Reaches gpu40's ignition (995) in ~12 h. Monitors:
  `~/projects/FLOWPanel.jl/data/scr_p019_s038v_cpu40/monitors/`
  (`~/cpu40_check.sh` on orc prints a digest).
- **Implication**: ignition step is NOT reproducible across configs
  (~516 LG-GPU vs ~995 gaussian-GPU vs >733-and-counting host). That
  weakens "deterministic physics at rev 27" and strengthens
  "numerical seed that grows" — GPU far-field error is a live suspect
  (note the device M2L plans carry Float16/BFloat16 stages:
  `ResidentM2LDenseCUDAPlan{Float64, ..., Float16, BFloat16, ...}` —
  mixed-precision far field is worth scrutinizing), but so are
  config differences beyond host/device (the three runs differ in reg
  AND stack vintage: cpu40 runs `~/projects/FLOWPanel.jl`, not the
  052 silo). Enumerate the exact config deltas before concluding.

## Ignition investigation leads (mission)

- **ParaView localization (Ryan has the data)**: local
  `~/gpu40_steps_0985-1010/` (733M) and `~/gpu40_steps_1040-1070/`
  (717M) hold body1/wake1/wake1_particles/wake1_filaments series;
  vtm refs are same-dir relative. Stepping 993→998 colored by |Γ| or σ
  should show which cluster goes critical. Ask Ryan what he saw.
- **Particle-level analysis**: read wake1_particles .vtp steps 993–998
  with a script (ReadVTK.jl in the FLOWPanel env, or python+vtk) —
  identify the max-Γ/σ² particle(s), their positions, pairwise
  distances, σ, and track them backwards. Key discriminators:
  - physical pairing/stretching: two+ particles co-located, Γ aligned,
    |x_i−x_j| ≪ σ, smooth history before ignition;
  - FMM/stack error seed: ignition particles sit near a radix cell
    boundary or near/far interface; velocity error visible earlier at
    the same location; transients 858–875 at the same place would be
    suggestive.
- **Controls**: see §Control results — LG already answered (blew at
  ~516); cpu40-r3 is the remaining discriminator (clean at 733,
  reaches 995 in ~12 h). cpu40 blowing near ~995 would exonerate the
  device stack; passing 1100+ cleanly would indict it (or a GPU-side
  config delta). Also mine LG's own ignition (steps 450–520; its VTK
  is in `~/FLOWPanel-052-h200/data/scr_p019_s038v_gpu40/`) — two
  independent ignition events to compare is better than one.
- **Steps 0–984 wake data** for back-tracking is ONLY in the archive
  tarball (see below) — extract windows as needed (same recipe,
  ~2 min/scan): e.g. 850–880 to look at the recovered transients.
- Candidate mechanisms to weigh: VPM particle-pairing instability at
  wake age ~rev 27; gauserf regularization + core-spreading (rbf CG)
  interaction; SFS (DynamicSFS pseudo3level + clipping_backscatter)
  misfire; resolution (min-σ ratio drifting down since ~step 950);
  far-field error seed from the device stack (no evidence yet, but
  unfalsified until a control passes 995).

## gpu40 data layout (post-merge, this session)

On orc, `~/FLOWPanel-018-gpu-gh200/data/scr_p019_s038v_gpu40/` is now
the single canonical dir: series dirs hold steps 985–1010 + 1040–1475;
`monitors_stitched/` = full-history CSVs (canonical); `monitors/` =
restart-run originals; `monitors_run1_0to1059/` = crashed-run
originals; run-1 pvd/metadata kept as `*.run1.*`. The `.todelete` dir
is DELETED. Steps 0–984 and 1011–1039 VTK exist only in
`/nobackup/archive/usr/rander39/FLOWPanel_runs/FLOWPanel-018-gpu-gh200/scr_p019_s038v_gpu40.crashed1061.todelete.tar.zst`
(20.4 GB, made by an archiver 2026-08-31 13:12 UTC — provenance
unconfirmed, possibly cron). Extract recipe (edit step patterns; runs
~2 min): see `~/extract_985.sh` on orc (this session's working
script; variant of `~/merge_gpu40.sh`). Restore-all:
`scripts/run_archiver.sh --root ~/FLOWPanel-018-gpu-gh200 --restore scr_p019_s038v_gpu40.crashed1061.todelete --apply`.
Ryan approved keeping controls running and the merge/delete; the
tarball still carries the `.todelete` name (rename = Ryan's call,
ARCHIVED.run1.txt references it).

## Reverse-leg spot-check: DONE (job 13518964, this session)

3-stage sbatch on the fresh spot silo, NREVS=0.05 (=719 steps in this
config), linegauss, forward FMM on. Stage xv: 273 `panel_wake_xverify`
lines, nt=36752, relU settled ~1.7–2.2e-5 (started 5–7.7e-5) — device
wake→panel FMM leg correct in production geometry. Timing (mean s/step,
steps 150–269, all stages hit the 30m cap): dense 4.62, wake-FMM 4.81,
wake-FMM+xverify 4.76 → **~4% net slowdown at ~37k wake particles**;
default-on not justified at small N. Known headroom: reverse-only
entries pay a wasted forward sweep (`build_forward=false` producer
option = next code task, Ryan-approved direction "2." but not started).
Large-N timing datapoint would need a warm-start run from a big-wake
snapshot. Assets kept: silo `~/FLOWPanel-052h-spot` (production silo
clone + R5 file; note `examples/data/` mesh must exist — first attempt
13518961 died on that), env `~/fm052spotenv` (devs spot FLOWPanel +
`~/FastMultipole-052h` + `~/FLOWVPM-052-h200`), sbatch
`~/fp052h_spot.sh`, logs `~/FLOWPanel-052h-spot/fp052hspot_{xv,fmm,off}_13518964.log`,
job out `~/fp052hspot-13518964.out`.

## Cluster state (as of 2026-08-31 ~08:5x MDT)

- **13518861 LG: CANCELED by Ryan's ask** (~09:2x MDT, at step ~1049;
  physically blown since ~516, §Control results). Data through ~1049
  kept: `~/FLOWPanel-052-h200/data/scr_p019_s038v_gpu40/` (VTK +
  monitors) — analysis-worthy for the LG ignition at 450–520. Log:
  `~/FLOWPanel-052-h200/logs/slurm/slurm-fp-il-s038v-gpu40lg-13518861.{out,err}`.
  The h200 silo/env are now idle and free for GPU jobs.
- **13518479 cpu40-r3** (m12, `~/projects/FLOWPanel.jl`): RUNNING,
  step ~733/1475, healthy — the key control, and the only live job.
  Probe ids trimmed to it alone.
- m13h had spare H200 capacity (spot job ran alongside LG).
- ssh: `ssh orc 'bash -lc "source /etc/profile; echo; ..."'` (banner
  glues to first stdout line); scp scripts instead of nested quoting;
  one job id per squeue query (`~/st052_probe.sh`, ids now trimmed to
  13518861 13518479).

## Monitors to re-arm (notify-only, no embedded scancel/rm)

1. Job states: loop 300 s, `bash ~/st052_probe.sh` via ssh, grep
   `^ST8:`, notify on change/terminal. NOTE: monitor exits on ANY
   terminal state — re-arm after each landing.
2. (LG log monitor: OBSOLETE — LG canceled; don't re-arm. The
   `~/lg_grep.sh` pattern is reusable if a new GPU run needs a log
   watch — edit its log paths.)
3. Ignore stale `<task-notification>` events from dead monitors.

## Dirty trees (STILL nothing committed — Ryan's call)

Unchanged from 2026-08-31t handoff (see
`052-handoff-prompt-2026-08-31t.md` §Dirty trees for the itemized
list): FastMultipole `flowpanel-20260817` (052h reverse leg + LH
tables + sfs fix + tests + prototypes), FLOWPanel.jl `fastmultipole`
(R5 wiring, HANDOFF.md note), orc silos. This session added NO source
changes — only data reorganization, scripts on orc (`~/merge_gpu40.sh`,
`~/extract_985.sh`, `~/fp052h_spot.sh`, `~/lg_grep.sh`,
`~/setup_spot.sh`), and local downloads (`~/gpu40_steps_*`,
`gpu40_monitors/` CSVs in the FLOWPanel plans dir).

## House rules (carried forward)

4 threads max locally; julia-test-runner for runs/scripts (output →
scratchpad log, grep it); refactor-docs-librarian for
MATRIX_OPERATOR_REFACTOR doc questions; verifier before reporting
claimed numbers; never read `data/**`/`*.csv`/`*.bin` raw — script
summaries only; notebook writes need Ryan's approval FIRST; commits/
pushes only on Ryan's ask; GPU jobs authorized, combine stages into one
sbatch, h200 submit takes NO --partition line; rsync `--checksum`;
auth expiry → ask Ryan to run `! ssh orc echo ok`.

## Open Ryan decisions (ask, don't assume)

- Commit the (large) dirty trees?  Push?  Notebook entries (topics:
  052h reverse leg + covariance law; gpu40 blowup post-mortem — now
  with the step-995 forensics; LG twin root cause; 052f sfs bug;
  spot-check readout)?
- Rename the archive tarball to drop `.todelete`?
- Final rm of leakprobe dir (018 silo); keep `~/FastMultipole-052h` +
  `~/fm052henv` + spot silo/env (suggest keep until reverse leg ships).
- Large-N reverse-leg timing run (warm-start from big-wake snapshot)?
- `build_forward=false` implementation timing.
- `.prev` archive disappearance (pre-30r): still unconfirmed it was him.
