# 052 handoff (session 2026-08-31v): ignition root-caused empirically — NEXT: formalize the blowup mechanism and its physics mismatch

## Prompt for the next agent

You are continuing task 052 after a context reset. Session 2026-08-31v
ran the gpu40/LG ignition forensics to completion. The empirical answer
is in (§Established mechanism): both blowups are a smooth, self-
consistent vortex-stretching runaway on a single small-σ "patient zero"
particle; the FMM velocity at both ignition sites was verified correct
by direct O(N) summation, exonerating the GPU far field. Ryan's next
mission: **formalize exactly what causes the blowup, and where the
mismatch with a physical system is** — i.e. turn the empirical chain
into a quantitative model (growth-rate law, feedback-loop closure,
instability criterion) and identify precisely which step of the VPM
discretization departs from Navier–Stokes physics (candidates in
§Formalization leads). First actions: (1) re-arm the two monitors
(§Monitors); (2) read §Established mechanism and §Data assets — do NOT
redo the forensics; (3) start the formalization.

## Established mechanism (this session; key numbers verifier-confirmed)

Timeline recap: gpu40 (Vatistas filament reg, GPU) ignites step
995–997; LG twin (LineGauss filament reg, GPU) ignites ~490–516;
cpu40-r3 (host-only) clean at 741+ at last check.

1. **Patient zero, gpu40 = particle idx 102340** (index is stable in
   the .vtp series 850→998). Braid-region particle ~1.5R off-axis, 2R
   downstream (rotor tip R≈0.118, body1 bounds x[-0.009,0.002]
   y[±0.024] z[±0.118]; ignition site pos≈[0.228,−0.170,0.043] at
   993). History: |Γ| 4.3e-5 (s850) → 4.0e-3 (s993) → 0.39 (s998),
   smooth quasi-exponential, accelerating (~0.02–0.03 ln/step at
   850–950; ~0.15 at 980–993; ~0.8–1.6 at 994–996). σ SHRINKS
   1.17e-3 (850) → 4.9e-4 (993) ≈ global min. |∇u| at particle:
   500–1000 s⁻¹ persistently from 850, →9e3 (993). Antiparallel
   partner idx 179085 (Γ-align −0.5..−0.7, separation ≈ constant
   4.1e-3 ≈ 8σ across 990–995) ignites one step behind. max|u|:
   24.6 (994) → 54.6 (995) → 132.5 (996) → 1094 (997).
   Γ/σ² of 102340: 2.0e4 (994) → 4.6e4 → 1.8e5 → 2.7e5 → 4.1e5 (998).
   No discrete seed event anywhere in 850–993; the recovered global
   transients at 858–875 were elsewhere (this particle's |u| dipped
   then).
2. **Patient zero, LG = particle idx 160932**: same signature, root-
   vortex region (~0.13R off-axis; pos [0.192,0.013,0.000] at s490).
   σ≈0.9–1.4e-3 vs local neighbors ~5e-3; |Γ| 4e-4 (450) → 2.2e-2
   (490) → 37 (517). |∇u| ~800–5000 throughout.
3. **Γ-updates are clean**: for both patient zeros, ΔΓ ≈ dt·(∇u)·Γ
   with cos(ΔΓ, JΓ)≈0.83–0.99 and a CONSTANT implied dt≈2.3–2.4e-4
   across the whole runaway (classical scheme, not transpose — JΓ
   fits better than JᵀΓ). Nothing injects error into Γ; anomalies can
   only enter via the velocity field.
4. **Velocity field verified correct at ignition sites**: direct O(N)
   Biot–Savart with the Gaussian/Winckelmans erf kernel
   (g(ρ)=erf(ρ/√2)−ρ√(2/π)e^{−ρ²/2}, u=−1/4π Σ g/r³ (x−x_p)×Γ_p)
   reproduces the saved velocities: gpu40 s993 idx 102340 rel err
   3.2e-4, idx 179085 3.5e-4 (baseline particles ~2e-3); LG s490 idx
   160932 rel err 1.4e-3 (and LG matches the GAUSS kernel, not
   singular — particle-particle reg is gaussian-erf in both runs).
   Residual |u_saved − u_direct| decays with rotor distance (median
   0.8 at x≈0 → 0.004 at x>0.35): missing panel/bound contribution,
   NOT solver error. → GPU mixed-precision far field EXONERATED at
   the ignition sites.
5. **Instability is systemic**: exponential-grower census (|Γ| ratio
   over ~45 steps, index-continuity filtered): gpu40 950→993:
   ×2+:2953, ×5+:493, ×10+:162; cpu40 700→745: ×2+:2211, ×5+:364,
   ×10+:121 — same background dynamics in the CLEAN run. Difference:
   cpu40 has no compounding leader (max Γ/σ² bounded ~1.2–1.5e3 at
   600–745, leader changes identity; gpu40's leader is fixed and
   reaches 2e4 pre-ignition).
6. **Config deltas** (agent-enumerated by file diffing; silos aren't
   git repos): gpu40 vs LG differ ONLY in filament regularization
   (Vatistas vs LineGauss); particle σ, SFS (DynamicSFS
   pseudo3level + clipping_backscatter), dt, NT, mesh, RPM, overlap,
   Float64 config all identical; FastMultipole CUDA source
   byte-identical between the two GPU silos. cpu40 differs in
   backend/code path only (its FastMultipole lacks 052c/052g guards —
   only matters if it trips the adequacy gate). Onset-step scatter
   (516/995/>745) = chaos + filament-kernel difference, not a GPU
   seed. LG monitor's "min-σ ratio pinned at 1.000" is a monitor
   definition quirk — actual LG σ min is 4.3e-4, percentiles at s480:
   p1 2e-3, median 4.5e-3.

Interpretation delivered to Ryan: physical/resolution VPM
pairing-stretching instability with a stretch→σ-shrink→stronger-
induction→more-strain feedback loop; onset chaotic and config-
sensitive; the FMM adequacy-gate crash at 1060 was a downstream
symptom. Memory saved:
`memory/052-gpu40-ignition-root-cause.md`.

## Formalization leads (the new mission)

Turn the above into (a) a quantitative mechanism and (b) a precise
statement of the physics mismatch. Suggested threads:

- **Growth-rate law**: fit λ(t)=d(ln|Γ|)/dt for patient zero(s) vs
  local strain (largest eigenvalue of symmetrized saved
  velocity_gradient) — is λ ≈ s₁ (material-line stretching), and when
  does it decouple (feedback onset)? All data needed is local.
- **σ-shrink law**: FLOWVPM tube model conserves volume ⇒ σ ~
  1/√stretch; test σ(t)·|Γ(t)|^α for the α the code actually enforces
  (check FLOWVPM source for the σ update under stretching + rbf CG /
  core spreading; source on orc silos, or local FLOWVPM.jl checkout if
  present). This closes the feedback loop analytically.
- **Feedback-closure model**: 2-particle antiparallel pair, separation
  d, cores σ≪d: mutual strain ~ Γ/(4π d³)·f, dΓ/dt = strain·Γ, σ ~
  Γ^{-1/2} ⇒ ODE for Γ(t) — finite-time blowup criterion; compare
  with observed d≈4.1e-3 const, Γ(t), and the observed transition
  from λ~500 to λ~5000 around s980. Where does d stay constant vs
  physical pair (which would move)? That constancy (990–995) is itself
  diagnostic.
- **Physics mismatch candidates** (where VPM departs from NS): (i)
  regularized-BS with σ ≪ inter-particle spacing = effectively
  singular interactions the true flow (with viscosity + core
  structure) wouldn't have — quantify: local spacing h vs σ along
  patient-zero history (h/σ grows past ~8); (ii) missing viscous
  diffusion at the collapsing scale: compare stretching rate vs
  viscous decay rate ν/σ² with the run's ν (get from config/agent);
  (iii) tube-model σ-shrink without a resolution floor — a real
  vortex tube thins but NS caps vorticity growth via viscosity;
  (iv) SFS model: SFS·Γ/|Γ|² values at patient zero were huge
  (±1e7–8e7) and sign-flipping — check units/scaling (SFS array
  is raw, needs C coeff and vol; saved vol=0 in the .vtp — why?)
  and whether clipping_backscatter zeroed the SFS dissipation
  exactly where it was needed; (v) discrete dt: dt·λ reached ~1.6
  at 996 — integration leaves the stability region of the stretching
  ODE around there, but that is AFTER ignition, not the cause.
- **Literature anchors**: VPM "spurious vortex pairing"/particle
  disorder instabilities (Winckelmans; Cottet & Koumoutsakos), the
  known cure = remeshing/spatial adaptation (this stack doesn't
  remesh), core-spreading validity limits, and Alvarez & Ning's
  rVPM papers (FLOWVPM's basis — its "relaxation" and tube model).
  Formalize which assumption fails: particle overlap condition
  σ ≥ h everywhere is the standard convergence requirement — these
  runs violate it locally by ~8×.
- Deliverable shape Ryan will likely want: a derivation +
  quantitative comparison written up (notebook entry needs his
  approval first — offer a draft via notebook-drafter), possibly
  feeding a mitigation decision (σ floor, overlap enforcement,
  merging, SFS tuning).

## Data assets (analysis-ready, all local unless noted)

- `~/gpu40_steps_0985-1010/` (733M) + `~/gpu40_steps_1040-1070/`
  (717M): full series (body1/wake1/particles/filaments), ParaView-
  ready (vtm refs same-dir relative).
- `~/gpu40_particles_hist/`: particles ONLY, steps 850–880 every step
  + 885–980 every 5 (bare .vtp, no pvd/body). Also staged on orc at
  `~/gpu40_particles_hist/`. Steps <850 and other 881–984 remain only
  in the tarball (recipe: `~/extract_hist.sh` on orc, edit patterns,
  ~2 min; tarball
  `/nobackup/archive/usr/rander39/FLOWPanel_runs/FLOWPanel-018-gpu-gh200/scr_p019_s038v_gpu40.crashed1061.todelete.tar.zst`).
- `~/lg_steps_450-520/particles/`: LG particles, steps 450–475 every
  5 + 480–520 every step. Full LG data (all series, through ~1049)
  on orc: `~/FLOWPanel-052-h200/data/scr_p019_s038v_gpu40/`.
- `~/cpu40_snaps/`: cpu40 particles at 600/650/700/745.
- gpu40 stitched monitor CSVs (full 1476-step history):
  `~/Dropbox/research/projects/FLOWPanel.jl/plans/sigma_vpm_illustrations_20260827/gpu40_monitors/`.
- .vtp arrays (float32 in file): gamma(3), sigma, vol (=0!),
  circulation, velocity(3), vorticity(3) (=0!), C(3), SFS(3),
  velocity_gradient(9, row-major 3×3). Index into series is stable
  across steps (verified 850→998 and 450→520). NEVER read
  data/CSVs raw — python3+vtk scripts (vtk 9.6.2 + numpy + scipy
  installed locally; NOT on orc python).
- Scripts (session scratchpad, REGENERATE if scratchpad wiped —
  logic documented here): `ignite_gpu40.py` (per-step hot-particle
  locator; args: dir, pattern, comma-list of steps),
  `track_back.py` (nearest-position backtrack; NOTE index-tracking
  beats position-tracking for fast particles), direct-sum checker
  (see kernel formula in §Established mechanism pt 4; guard r²<1e-24;
  float64). Verifier logs: verify_claim{1,2}.log same dir.
  Scratchpad: `/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/b98de049-616d-484e-b2b8-c5aab5970d0c/scratchpad/`.

## Monitors to re-arm (they die at context reset)

1. Job-state watch: loop 300 s over
   `ssh orc 'bash -lc "source /etc/profile; echo; bash ~/st052_probe.sh"'`,
   grep `^ST8:`, emit on change; exit+notify on terminal state or
   empty probe (job left queue). Probe ids trimmed to 13518479 only.
2. cpu40 wake-health babysitter: loop 900 s over
   `ssh orc 'bash -lc "source /etc/profile; echo; bash ~/cpu40_check.sh"'`,
   parse the `wh last: step N u=... g=... sr=...` line; ALERT if
   u>60; heartbeat every 8th poll; exit when step>1100 (= passed
   gpu40's ignition point cleanly). Last known: step 741, u=24,
   CFx −0.074, sr 0.133, healthy (~15:00 MDT). It reaches step 995
   roughly 2026-09-01 morning. Either outcome is consistent with the
   established mechanism (chaos ⇒ different realization); a cpu40
   ignition at any step would further confirm physics.
3. Ignore stale `<task-notification>` events from dead monitors.

## Cluster state (as of 2026-08-31 ~15:00 MDT)

- 13518479 cpu40-r3 RUNNING (m12, `~/projects/FLOWPanel.jl`), step
  ~741/1475, healthy, the only live job. 12 h walltime — check
  whether it can even reach 995 before its limit (r3 = third
  restart; if it dies at the wall, Ryan may want an r4 restart).
- LG 13518861 CANCELED (Ryan's ask, last session); data kept.
- h200 silo/env idle and free for GPU jobs (`~/FLOWPanel-052-h200`,
  `~/fm052henv`); spot assets `~/FLOWPanel-052h-spot`, `~/fm052spotenv`.
- ssh: `ssh orc 'bash -lc "source /etc/profile; echo; ..."'`; scp
  scripts instead of nested quoting; rsync `--checksum` for re-syncs;
  auth expiry → ask Ryan to run `! ssh orc echo ok`.

## Dirty trees (STILL nothing committed — Ryan's call)

Unchanged from 2026-08-31t handoff §Dirty trees (FastMultipole
`flowpanel-20260817`: 052h reverse leg + LH tables + sfs fix + tests
+ prototypes; FLOWPanel.jl `fastmultipole`: R5 wiring; orc silos).
Sessions u and v added NO source changes — only data
downloads/extraction, orc scripts (`~/extract_hist.sh` new this
session), and memory file `052-gpu40-ignition-root-cause.md`.

## House rules (carried forward)

4 threads max locally; julia-test-runner for runs/scripts (output →
scratchpad log, grep it); refactor-docs-librarian for
MATRIX_OPERATOR_REFACTOR doc questions; verifier before reporting
claimed numbers; never read `data/**`/`*.csv`/`*.bin` raw; notebook
writes need Ryan's approval FIRST (offer notebook-drafter draft);
commits/pushes only on Ryan's ask; GPU jobs authorized, combine
stages into one sbatch, h200 submit takes NO --partition line;
delegate big exploration to subagents (haiku mechanical / sonnet
conceptual / opus cross-file).

## Open Ryan decisions (ask, don't assume)

- Commit the dirty trees? Push? Notebook entries (new topic: gpu40/LG
  ignition forensics + the formalization once done)?
- Rename archive tarball to drop `.todelete`?
- cpu40-r3 walltime: restart r4 if it hits the 12 h wall before 995?
- Pull matching body1/wake1 series for steps 850–995 for ParaView
  visualization of patient zero (Ryan asked where files are; offered)?
- Silo cleanup, large-N reverse-leg timing, `build_forward=false`
  implementation (unchanged from prior handoffs).
