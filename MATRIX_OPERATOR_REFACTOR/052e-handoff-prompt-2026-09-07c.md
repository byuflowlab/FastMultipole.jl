# 052e handoff — 2026-09-07c (context reset; supersedes 052e-handoff-prompt-2026-09-07b.md)

## Immediate state: Tier 0B is DONE and RULED

- **052e.2a stage 1 PASS — ruled by Ryan 2026-09-07** ("accept the pass").
  Ruling, tables, telemetry, and provenance:
  `052e2a-tier0b-results-2026-09-07.md` (status line = RULED).
- **Run history:** run 1 (PID 51333) INVALIDATED — harness defect: the
  system-under-test body's `controlpoints`/`normals` were never populated
  (FLOWPanel fills them lazily in solve(); harness only did it for the
  oracle body). Root cause + fix recorded in
  `052e2a-tier0b-run1-invalidation-2026-09-07.md`; run-1 artifacts in
  `data/052e2a-tier0b/run1-invalid-harness/`. Fix: `make_body` now calls
  `pnl.calc_normals!`/`pnl.calc_controlpoints!`. Full clean relaunch as
  run 2 (PID 53886) per protocol — no partial reuse.
- **Run 2 headline:** E_q(finest) 0.17% (C1, AOA 0°) / 0.57% (C2, 7°) vs
  20% kill threshold; monotone refinement, orders → ~1.4–1.5; λ machine-
  zero (C1) / smooth ~-3.5e-4 mesh-converging (C2); Hodge trends halve
  per level; linearity exact; production influence! wake route matched
  the Tier 0A oracle to ~1e-10 (potential) / ~2e-15 (velocity) on its
  first exercise. Kill rule NOT triggered.
- **Two formal gate FAILs were adjudicated as artifacts** under the PASS
  ruling in `052e2a-tier0b-gate-supersession-2026-09-07.md` (prereg
  `052e2a-tier0b-preregistration-2026-09-07.md` stays LOCKED/unedited):
  B3/C1 = monotonicity clause on machine-zero flux (symmetric case,
  vacuous); B5/C2 = gauge defect 1.44e-12 vs non-N-scaled 1e-12 gate at
  N=19,384 (roundoff accumulation; future tiers should use an N-aware
  bound).
- **Post-hoc max-error finding (recorded in supersession note):** C2/L4
  E_inf=2.27% concentrates entirely on tip-cap sliver panels at y=±b/2
  (elliptic-loading wake-edge singularity meets worst-conditioned
  panels); interior max 0.69%; converging order ~1.3. Judged
  "about right" for constant-panel collocation.

## Next task: 052e.2b (implicit-Householder reduction)

Per `052e-accuracy-plan-v2-draft-2026-09-05.md` §5 / Tier 0B-R and
`052e-theory-velocity-to-potential-trace.md` §3.1 (theory note ACCEPTED
2026-09-07, status line at top):

1. Pull the .2b requirements from accuracy plan §5 (Tier 0B-R parity
   gates) via the refactor-docs-librarian agent — do not re-read the
   plan wholesale.
2. Draft a Tier 0B-R preregistration (dated file, Tier 0B pattern:
   locked fixture, gates, kill rule) and present to Ryan to LOCK before
   any registered values exist. Natural parity baseline: run-2 gates.txt
   values (`data/052e2a-tier0b/gates.txt`, script_sha256=270deb379068).
3. Harness can extend `scripts/tier0b_052e2a_bordered_formulation.jl`
   (run-2 version, includes the calc_normals!/calc_controlpoints! fix)
   or a new script; smoke-validate mechanically (TIER0B_SMOKE=1 pattern,
   exit status only) before any registered run; launch registered runs
   nohup-detached, log into `data/` (run-2 launch pattern:
   `julia --project=../FLOWPanel.jl --threads=4`, ~15 min).
4. Scheduling (ratified, accuracy plan §5): stages 2–4
   (filament/particles) run as a .2a continuation addendum and may
   parallel .2b; stage 5 folds into 052e.3.

## Not yet done / open

- **Notebook entry for Tier 0B: offered, NO REPLY yet.** Offer again
  (with verbosity options) — approval required before writing anything.
- **Nothing committed.** All 052e work (harness fix, invalidation note,
  results, supersession, this handoff) is uncommitted; Ryan has not
  asked for a commit. If asked: 052e files ONLY.
- 052e.1 regression completion after 052b closes; 052e.6 global gauge
  design study; promotion limited to gauge-invariant outputs.

## Repository/worktree cautions (unchanged)

- FastMultipole worktree heavily dirty with unrelated higher-derivative
  work + untracked `HIGHER_DERIVATIVES/`. Touch only 052e files; never
  clean/stage/commit unrelated changes.
- `../FLOWPanel.jl` dirty; `src/FLOWPanel_formulation.jl` +
  `test/formulation_test.jl` carry uncommitted 052e.1 work — preserve.
  Read its `AGENTS.md`, `CLAUDE.md`, `agent_policies/WORKFLOW.md`
  (+`TESTING.md` before testing) before FLOWPanel code work. ≤4 threads
  locally.
- Laptop runs = formulation-proof tier, not official campaigns; official
  acceptance runs need committed, tagged campaign worktrees (global
  policy).
- Token policy: never read `data/**` CSVs directly — summarize by
  script; gates.txt and log tails carry the registered values.
