# 052 handoff (session 2026-08-31w): blowup mechanism FORMALIZED + verified; plain-Pedrizzetti LineGauss run launched — NEXT: babysit runs, interpret outcomes

## Prompt for the next agent

You are continuing task 052 after a context reset. Session 2026-08-31w
(this one) completed Ryan's formalization mission: the blowup mechanism
is now a quantitative, verifier-confirmed model (§Formalized mechanism),
all prior puzzles are resolved (§Resolved puzzles), and a new
experiment — the LG twin with PLAIN Pedrizzetti relaxation — is running
on eng (§Live runs). First actions: (1) re-arm the three monitors
(§Monitors); (2) read §Formalized mechanism — do NOT redo the
forensics OR the formalization; (3) when run outcomes land, interpret
them against the model's predictions (§Predictions). Prior context:
`052-handoff-prompt-2026-08-31v.md` (forensics session).

## Formalized mechanism (this session; verifier-confirmed)

Runs' actual config (from metadata.toml on orc + silo source):
rVPM formulation (f=0, g=1/5), **plain unguarded Euler** integration
(NOT RK3; chosen deliberately in FLOWPanel_wake.jl:1972-1983 so
CoreSpreading runs; `sigma_guard` EXISTS in silo FLOWVPM but was never
passed — default off), transposed=true stretching, relaxation
`relax_correctedpedrizzetti` rlxf=0.3 EVERY step,
viscous=CoreSpreading ν=1.4334e-5 β=1e9 (reset disabled),
SFS_Cd_twolevel_nobackscatter = DynamicSFS(pseudo3level,
clippings=(clipping_backscatter,)), kernel gauserf, **dt=3.086e-4**
(not 2.4e-4). Per step, Z = (1/5)·ΓᵀJΓ/|Γ|²:

$$\Gamma \leftarrow \Gamma + \Delta t\,(J^{\top}\Gamma - 3Z\Gamma - C\,\mathrm{SFS}\,\sigma^3/\zeta(0)), \quad \sigma \leftarrow \sigma(1-\Delta t Z), \quad \Gamma \leftarrow \mathrm{relax\ toward\ }\hat\omega\ (|\Gamma|\ \mathrm{preserved})$$

Data fits (scripts pz_analysis.py / pair_ode.py, gpu40 idx 102340 +
partner 179085 over steps 850–1010; LG idx 160932 over 450–520):

1. **σ-law confirmed**: Δlnσ vs −dt·Z fits corr 0.95 (gpu40) / 0.998
   (LG), slope 0.88 / 0.99; residual = CoreSpreading ν·dt/σ² (0.005–
   0.02/step), which is 2–5× too weak to stop the stretch-shrink.
2. **Growth law**: λ = d(ln|Γ|)/dt ≈ **0.22·s1** (largest eigenvalue of
   symmetrized J) in BOTH runs (slope 0.22 both; gpu40 corr 0.949).
   |Γ| grows even at steps where the particle's own parallel strain is
   NEGATIVE (gpu40 992–993: λ_par −121/−450 vs λ_obs +642/+1125) — the
   every-step relaxation re-aligns Γ toward ω (ratchet), defeating the
   self-limiting misalignment real vorticity has. 0.22 ≈ 0.55×(1−3g).
3. **Pair feedback closed**: over 988–995 the local strain IS the
   antiparallel partner's Biot–Savart strain: s1 = 1.35·|Γp|/(4πd³)
   (median ratio exactly 1.35, locked), d frozen at 4.1e-3 (d/σ≈8),
   |Γ|≈|Γp|. Closed ODE dΓ/dt = 0.22·1.35·Γ²/(4πd³) ⇒ finite-time
   blowup t*−t = 4πd³/(0.30·Γ); forecast from step 988 → ignition
   996.3; observed 996. Two-phase history: background braid strain
   (s1≈300–800) drives exponential growth until ~985; partner
   induction (Γ²-term) takes over → finite-time singularity.
4. **Physics mismatches quantified**: overlap σ≥h violated (h/σ 4→10);
   no damping channel for Γ (CoreSpreading is σ-only, SFS clipped &
   negligible at patient zero — forcing ratio ≤1e-2, C often 0; no
   PSE; no reconnection); relaxation ratchet breaks tube invariant
   σ|Γ|^½ (gpu40 ×4 over 850→993, cv 2.8 through ignition; LG ×245);
   dt·s1 crosses 1 at ~994 and dt·Z hits 0.83 at 996 (Euler σ update
   next to its documented sign-flip boundary) — consequence, not cause.
5. **Fix levers identified**: sigma_guard/σ-floor (in silo code,
   unused), overlap enforcement (FLOWVPM opt-in SigmaShrinkTrigger /
   ZTrigger / merging), reduced/altered relaxation, PSE, smaller dt.

## Resolved puzzles (do not re-investigate)

- "Implied dt≈2.4e-4" (session v): real dt=3.086e-4; factor 0.78 is
  the rVPM −3ZΓ term (parallel component scaled by 1−3g=0.4).
- "Classical JΓ fits better": FLOWVPM stores J COLUMN-major
  (J[i,j]=∂U_i/∂x_j, l=3(col−1)+row; check W1=J[6]−J[8]); the .vtp
  velocity_gradient array read into numpy row-major is Jᵀ. Data
  actually fits the CONFIGURED transposed scheme (cos 0.93–0.97 near
  ignition). ΓᵀJΓ is transpose-invariant so growth-rate fits were
  never affected.
- **vol=0 in .vtp**: only PSE sets vol (FLOWVPM_viscous.jl:352 is
  PSE's, not CoreSpreading's); nothing in this pipeline populates it;
  vol is INERT (SFS scaling uses σ³ directly).
- Huge saved SFS values: the .vtp SFS array is the RAW model term;
  the dΓ/dt contribution is ×Cσ³/ζ(0) (ζ(0)=(2π)^{-3/2}); tiny.
- Viscosity was ON (Ryan asked): CoreSpreading ν=1.4334e-5 in gpu40,
  LG, cpu40 alike; scr_p019_s038 (no v) was the inviscid variant.

## Live runs (as of 2026-08-31 ~17:05 MDT)

1. **13542825 `fp-il-s038v-lgpz-e`** RUNNING on eng-1-1 (H200, 64c,
   192G, 24 h, qos eng — preempted 4 standby jobs; eng starts
   immediately with preemption privileges; m13h 64c would have waited
   13 h). Case `scr_p019_s038v_lgpz` = LG twin EXCEPT
   RELAX_SCHEME=pedrizzetti (plain, magnitude-DAMPING) — one-variable
   test of the relaxation ratchet. Runs from ~/FLOWPanel-052-h200
   silo; data → `~/FLOWPanel-052-h200/data/scr_p019_s038v_lgpz/`
   (contains stray step-0 body files from a canceled 6-min m13h
   attempt 13542734, RHPC_KEEP_PREV=true — harmless). Banner knob
   validated ("relaxation scheme: relax_pedrizzetti" printed by the
   canceled twin); confirm the eng job's banner + metadata.toml
   `[relaxation] type = "FLOWVPM.relax_pedrizzetti"` once written.
2. **13518479 cpu40-r3** RUNNING (m12, ~/projects/FLOWPanel.jl), step
   ~790 at 16:50 MDT, healthy (u=24). Crosses gpu40's ignition step
   ~995 overnight; 12 h wall from ~08:45 MDT start — may TIMEOUT
   before finishing; either blowup or clean passage is consistent
   with the model (chaotic onset). If it hits the wall, Ryan may want
   an r4 restart.

### Predictions to check against outcomes

- lgpz (plain Pedrizzetti): model says plain relaxation both bleeds
  |Γ| (non-normalized rotation shrinks magnitude) and weakens the
  alignment ratchet → expect later/no ignition, or if it still
  ignites, λ vs s1 slope should drop below 0.22. Compare wake-health
  + rerun pz-style analysis on any new patient zero.
- cpu40 ignition at ANY step would further confirm
  backend-independence (physics, not GPU).

## Session changes (uncommitted, on orc silos)

- `~/FLOWPanel-052-h200/examples/rotor_hover_pressure_comparison.jl`:
  RELAX_SCHEME env knob (default correctedpedrizzetti = unchanged);
  prints "relaxation scheme: ...".
- `~/FLOWPanel-052-h200/examples/run_p018_screen_hpc.slurm.sh`: new
  case `scr_p019_s038v_lgpz` (line ~132) = s038v_gpu40 arm +
  RELAX_SCHEME=pedrizzetti.
- **gh200 silo synced**: rsync'd src/examples/scripts of
  FastMultipole/FLOWVPM/FLOWPanel-052-h200 → -052-gh200 (was stale,
  pre-052d). Known gh200 bug: dispatcher line ~180 julia fallback
  resolves x86 ~/.juliaup/bin/julia on ARM → "Exec format error"
  (killed test job 13542733 on mgh-1-2 in 5 s). Fix = make dispatcher
  honor $P018_JULIA. Ryan says prefer **mgh-1-1** over mgh-1-2 for
  any future mgh run.
- `~/patch_lgpz.py` on orc home (idempotent patch script; also in
  session scratchpad).
- Local: memory file `052-gpu40-ignition-root-cause.md` updated with
  formalization; NOTHING else local changed. Dirty trees from prior
  sessions unchanged (FastMultipole flowpanel-20260817, FLOWPanel.jl
  fastmultipole, orc silos) — no commits/pushes without Ryan's ask.

## Data assets & scripts

Unchanged from v-handoff §Data assets (gpu40/LG/cpu40 windows local;
never read data/CSVs raw; python3+vtk local only). This session's
scratchpad (`/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/be9d64f3-e7d6-47cd-9ebe-e791813f68a2/scratchpad/`):
- `pz_analysis.py` — extracts one particle's history from a .vtp
  series (args: label idx partner_idx dt outprefix dirs...) and fits
  growth/σ/vector-prediction laws; writes `<prefix>_hist.npz` +
  prints table. Logs: gpu40_pz.log, lg_pz.log; npz: gpu40_hist.npz,
  lg_hist.npz.
- `pair_ode.py` — pair strain closure + finite-time forecast from
  gpu40_hist.npz.
- `patch_lgpz.py` — the silo patch (already applied).
DURABLE COPY (survives scratchpad wipe): all of the above copied to
`MATRIX_OPERATOR_REFACTOR/prototypes/052_ignition_formalization/`
(untracked; commit only on Ryan's ask).

## Monitors to re-arm (die at context reset)

1. lgpz-eng watch: poll 600 s over ssh orc; `sacct -j 13542825 -X -n
   -o State%20`; grep log
   `~/FLOWPanel-052-h200/logs/slurm/slurm-fp-il-s038v-lgpz-e-13542825.out`
   for "relaxation scheme" (report once), metadata.toml relaxation
   type (once), ERROR/GATE/CUDA/NaN flags, step heartbeat ~2 h; exit
   on terminal state (incl. PREEMPTED — eng qos should not be
   preempted but cover it).
2. Job-state watch 13518479 (cpu40-r3): loop 300 s,
   `ssh orc 'bash -lc "source /etc/profile; echo; bash ~/st052_probe.sh"'`,
   grep ^ST8:, emit on change, exit+notify on terminal/empty. (Probe
   script only knows 13518479; watch 13542825 via monitor 1.)
3. cpu40 wake-health babysitter: loop 900 s,
   `ssh orc 'bash -lc "source /etc/profile; echo; bash ~/cpu40_check.sh"'`,
   parse `wh last: step N u=... g=... sr=...`; ALERT u>60; heartbeat
   every 8th poll; exit when step>1100.
4. Ignore stale `<task-notification>` events from dead monitors.

## Cluster facts learned this session

- eng partition (qos eng, 8 H200) grants PREEMPTION over standby —
  starts immediately even at 64c/192G. m13h free-GPU count ≠
  startable (CPU/mem are the binding constraint). GH200 mgh needs
  `-C arm`; use mgh-1-1 per Ryan. slurm-availability skill CSV:
  ~/.claude/slurm/orc_availability.csv.
- LG production submit pattern (reference):
  `sbatch --job-name=... -p eng --qos=eng --gres=gpu:h200:1 -c 64
  --mem=192G -t 24:00:00 --export=ALL,SCR_GPU_RESERVE_GIB=16,
  FLOWPANEL_FILAMENT_REG=linegauss,RHPC_KEEP_PREV=true
  examples/run_p018_screen_gpu052.slurm.sh h200 <case>` from
  ~/FLOWPanel-052-h200.

## House rules (carried forward)

4 threads max locally; julia-test-runner for runs/scripts (output →
scratchpad log, grep it); refactor-docs-librarian for
MATRIX_OPERATOR_REFACTOR doc questions; verifier before reporting
claimed numbers; never read `data/**`/`*.csv`/`*.bin` raw; notebook
writes need Ryan's approval FIRST (offer notebook-drafter draft);
commits/pushes only on Ryan's ask; GPU jobs authorized; scp scripts
to orc instead of nested ssh quoting; rsync --checksum; auth expiry →
ask Ryan `! ssh orc echo ok`.

## Open Ryan decisions (ask, don't assume)

- Notebook entry for forensics + formalization (offered, not yet
  answered) — use notebook-drafter, ask verbosity.
- Commit dirty trees / push? Rename archive tarball (drop
  `.todelete`)? cpu40 r4 restart if walltime kills it before 995?
- Fix gh200 dispatcher julia resolution + launch gh200 copy on
  mgh-1-1?
- Mitigation experiments after lgpz reads out (σ floor, overlap
  enforcement, split/merge triggers, SFS tuning)?
- Pull body1/wake1 series 850–995 for ParaView (offered in v).
