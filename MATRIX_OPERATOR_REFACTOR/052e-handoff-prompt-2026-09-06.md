# 052e handoff — 2026-09-06 (context reset + machine restart)

## Session summary (2026-09-05 → 06)

- **Issue 1 (campaign tags) RULED: option (c)** — `campaign/052-expint-20260905`
  tags stay orc-local until unified-052 is ready to publish; revisit at merge
  time. No bundle insurance requested.
- **052e REWORKED and ADOPTED.** Ryan ruled the item must measure how
  accurately the hybrid computes the wake-induced potential, against directly
  verifiable references. First draft (field-point potential, far-field
  zero-at-infinity anchor) was corrected after source verification: the code
  is a body-local Neumann-to-Dirichlet Green-trace solve `(I-B)q = Sσ` from
  velocity-only particle data (subtraction extraction, per-body
  `:area_mean`/`:lsq` gauges, `recompute_interval` lag; NO particle scalar-
  potential path exists, so zero-at-infinity is currently unimplementable).
  Governing docs, all on FastMultipole `flowpanel-20260817`:
  - `052e-theory-velocity-to-potential-trace.md` (theory note — pending
    formal acceptance under 052e.0)
  - `052e-accuracy-plan-v2-draft-2026-09-05.md` (tiers 0A/0B/1/1.5/2 +
    Stage B; gates PROPOSED, ratified per subitem before it runs)
  - `052e-impl-hybrid-wake-potential-experimental.md` (objective, work plan,
    subitem structure)
  - v1 plan `052e-accuracy-plan-draft-2026-09-03.md` marked superseded
    (retained for the Stage B matrix).
- **Rulings recorded in the docs:** Tier-1 disjointness satisfied by the
  rigid TE Kutta wake-panel buffer (particles start downstream of it; still
  report leakage bounds near limits). Promotion limited to gauge-invariant
  outputs (circulation, exterior velocity, integrated loads); absolute Cp /
  unsteady Bernoulli / acoustics blocked pending 052e.6.
- **Subitem split adopted:** 052e.0 theory+Tier 0A → .1 code health (hybrid
  hard unpaired-edge init error + negative regression; host regressions
  after 052b closes) → .2 Tier 0B oracle (owns kill rule, continue/retire
  ruling) → .3 Tiers 1+1.5 → .4 Tier 2 → .5 Stage B + promotion; .6 global
  gauge recovery design study (parallel). Chain .0→.2→{.3,.4}→.5.

## Next actions (in order)

1. 052e.0: Ryan formally accepts the theory note; then Tier 0A gate values
   ratified + run (host script, laptop-scale).
2. 052e.1 hardening half (init error + negative regression) can start any
   time; regressions half waits on 052b.
3. One-time code check before Tier 1 fixture build: verify the plan's claim
   that "the ordinary body operator and downstream shedding apply the
   equivalent strength difference implicitly" for HybridWakePotential.

## Still open from the 2026-09-05 handoff

- Issue 3 — 052b rulings: 1r hover-window gate policy; Kutta :jump contract;
  2r operator-mismatch investigation (may need a run → wt052 + new tag).
- Issue 4 — 053 defaults rows (`053-defaults-enumeration-draft-2026-09-03.md`).
- Issue 5 — storage 618G vs 400G cap (home 447G/2T; trim candidates = VTK
  series in the two 29G 1080-step run dirs).
- Notebook entry for 2026-09-05 (trial-2 PASS + expint ruling + silo
  retirement) — offered, needs Ryan approval + verbosity choice; now should
  also cover the 2026-09-06 052e rework.

## Mode reminder

Discussion mode for the open issues: bring one at a time with facts + a
recommendation; no runs, deletions, pushes, or ledger edits until Ryan rules.
052c remains CLOSED; wt052 launcher is the only 052c launch path.
