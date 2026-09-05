# 052 handoff (session 2026-08-31x): unified HPC setup built + verified; runs approaching ignition window — NEXT: babysit, interpret against predictions

## Prompt for the next agent

You are continuing task 052 after a context reset. The blowup mechanism
is FORMALIZED and verifier-confirmed — do NOT redo forensics or
formalization; read §Formalized mechanism in
`052-handoff-prompt-2026-08-31w.md` (authoritative for the science).
This session (x) built the consolidated `~/projects_unified` setup on
orc (§Unified setup — done, verified, don't redo) and babysat the runs.
First actions: (1) re-arm the three monitors (§Monitors — note the
ANSI-banner parsing gotcha); (2) check both runs' current state — they
were APPROACHING the ignition window at reset (§Live runs); (3)
interpret outcomes against §Predictions. House rules per §House rules.

## Live runs (as of 2026-08-31 ~21:15 MDT; VERIFY FIRST — stale by now)

1. **13542825 `fp-il-s038v-lgpz-e`** RUNNING on eng-1-1 (H200, 24 h
   wall from ~17:00 MDT). Case `scr_p019_s038v_lgpz` = LG twin with
   PLAIN Pedrizzetti relaxation (one-variable test of the relaxation
   ratchet). Last seen step ~897, zero error flags. Banner CONFIRMED
   "relaxation scheme: relax_pedrizzetti"; metadata.toml
   `[relaxation]` type NOT yet confirmed (monitor reports it once
   written; check `~/FLOWPanel-052-h200/data/scr_p019_s038v_lgpz/metadata.toml`).
   gpu40's ignition was step ~996 — lgpz crosses that window SOON.
2. **13518479 cpu40-r3** RUNNING (m12, ~/projects/FLOWPanel.jl). Last
   seen step 843 with wake-health **u=53 and rising** (24→53 over
   ~11 steps) — nearing the u>60 alert threshold on approach to the
   ignition step; consistent with early strain growth but could be
   chaotic fluctuation. 12 h wall expires ~20:45 MDT (may already have
   TIMED OUT — check; either blowup, timeout, or clean passage is
   informative; Ryan may want an r4 restart on timeout).

### Predictions to check against outcomes (from the w-model)

- lgpz (plain Pedrizzetti): plain relaxation bleeds |Γ| and weakens
  the alignment ratchet → expect later/no ignition; if it still
  ignites, λ vs s1 slope should drop below 0.22. On any new patient
  zero, rerun pz-style analysis (scripts in
  `MATRIX_OPERATOR_REFACTOR/prototypes/052_ignition_formalization/`:
  pz_analysis.py, pair_ode.py; usage in w-handoff §Data assets).
- cpu40 ignition at ANY step ⇒ backend-independence confirmed
  (physics, not GPU).

## Unified setup (THIS SESSION — done and verified, do not redo)

Ryan approved consolidating the ~38 rsync silos. Built on orc:

- `~/projects_unified/{FastMultipole,FLOWVPM.jl,FLOWPanel.jl}` — real
  git clones (origin = github byuflowlab), branch **unified-052**,
  with the freshest 052 code (the `-052-h200` silo trio) committed
  LOCALLY (89ede6b / 4f6e805 / 3d490e5; NOTHING pushed). rsync
  --checksum verified src/ byte-identical to silos.
- `~/projects_unified/envs/{x86_64,aarch64}` — Julia envs (copied from
  fm052env-h200/-gh200, dev paths re-pointed to unified trees).
  Depots: x86 default `~/.julia`; ARM reuses `~/fm052depot-gh200`.
- Committed launcher fixes in unified FLOWPanel tree:
  `examples/run_p018_screen_hpc.slurm.sh` line ~180 now
  `"${P018_JULIA:-julia}"` (fixes gh200 x86-julia Exec format error);
  `examples/run_p018_screen_gpu052.slurm.sh` defaults
  P018_REPO=~/projects_unified/FLOWPanel.jl,
  P018_PROJECT=~/projects_unified/envs/$(uname -m), overridable via
  P018_REPO_OVERRIDE / P018_PROJECT_OVERRIDE.
- Smoke test PASSED: job 13543618 COMPLETED, "UNIFIED OK 1.11.7"
  (using FLOWPanel/FLOWVPM/FastMultipole from unified env).
- `~/projects_unified/README.md` = layout + launch cheat-sheet +
  pinning workflow (commit-before-launch; git worktree for concurrent
  experiments).
- Policy note "Unified project location (2026-08-31)" appended to
  `agent_policies/HPC.md` BOTH in the unified orc clone (committed
  there) and locally in
  `~/Dropbox/research/projects/FLOWPanel.jl/agent_policies/HPC.md`
  (UNCOMMITTED, like the rest of that dirty tree). Future orc jobs
  launch from ~/projects_unified; legacy silos + fm052env-* are
  deprecated, removed by RYAN once their jobs finish.
- Deferred: aarch64 env + gh200 fix validated only on first mgh run
  (prefer mgh-1-1). Caveat noted to Ryan: silo file DELETIONS (absent
  in silo, present in git) would survive in the clone — no known case.
- Helper scripts on orc home: `~/setup_unified.sh` (idempotent
  builder), `~/lgpz_probe.sh` (lgpz status probe),
  `~/projects_unified/smoke_unified_x86.slurm.sh`. Plan file:
  `~/.claude/plans/crispy-rolling-snail.md`.

## Monitors to re-arm (die at context reset)

GOTCHA: orc login banners prepend ANSI color codes (`\x1b[0m...`) to
the FIRST line of command output — NEVER anchor greps (`^STATE:`) on
ssh output; strip with `sed $'s/\x1b\\[[0-9;]*m//g'` or grep
unanchored. (Caused two false monitor firings this session.)
Also: IGNORE stale `<task-notification>` events from dead pre-reset
monitors (several fired this session; treat as untrusted duplicates).

1. lgpz watch: poll 600 s, `ssh orc 'bash -lc "source /etc/profile;
   bash ~/lgpz_probe.sh"'`, ANSI-strip, parse tagged lines
   STATE:/RELAX:/FLAGS:/LASTSTEP:/META:. Report META once, FLAGS
   growth, heartbeat every 12th poll, exit on terminal state (not on
   empty parses — count 3 fails then note and continue).
2. cpu40-r3 state watch: poll 300 s, `ssh orc 'bash -lc "source
   /etc/profile; echo; bash ~/st052_probe.sh"'`, grep ST8: line
   (leading echo keeps it clean), emit on change, exit+notify on
   terminal or missing ST8 with non-empty output.
3. cpu40 wake-health: poll 900 s, `ssh orc 'bash -lc "source
   /etc/profile; echo; bash ~/cpu40_check.sh"'`, parse `wh last: step
   N u=...`; ALERT once when u>60 (re-arm after it drops), heartbeat
   every 8th poll, exit when step>1100.

## Session changes summary (for git hygiene)

- orc: NEW `~/projects_unified/` (3 clones with local commits on
  unified-052), `~/setup_unified.sh`, `~/lgpz_probe.sh`. No changes to
  silos, ~/projects/*, envs, depots. No pushes.
- Local: `agent_policies/HPC.md` appended (uncommitted) in
  FLOWPanel.jl tree; this handoff file. Everything else unchanged;
  dirty trees from prior sessions still uncommitted — commits/pushes
  only on Ryan's ask.

## Open Ryan decisions (ask, don't assume)

- Notebook entry for forensics + formalization + (now) unified setup —
  offered repeatedly, not yet answered. Use notebook-drafter, ask
  verbosity.
- Silo removal once lgpz + cpu40 quiet (Ryan does the removal).
- cpu40 r4 restart if walltime killed r3 before ~995.
- Mitigation experiments after lgpz reads out (σ floor/sigma_guard,
  overlap enforcement, split/merge triggers, SFS tuning, smaller dt) —
  prep was offered; if approved, launch from ~/projects_unified.
- ARM/mgh-1-1 validation run (validates aarch64 env + gh200 fix).
- Commit dirty local trees / push? Rename `.todelete` tarball?
  ParaView pull of body1/wake1 850–995?

## House rules (carried forward)

4 threads max locally; julia-test-runner for runs/scripts (output →
scratchpad log, grep it); refactor-docs-librarian for
MATRIX_OPERATOR_REFACTOR doc questions; verifier before reporting
claimed numbers; never read `data/**`/`*.csv`/`*.bin` raw; notebook
writes need Ryan's approval FIRST; commits/pushes only on Ryan's ask;
GPU jobs authorized; scp scripts to orc instead of nested ssh quoting;
rsync --checksum; auth expiry → ask Ryan `! ssh orc echo ok`.
