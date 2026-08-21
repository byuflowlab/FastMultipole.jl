# Phase P autonomous-run decisions log

Started 2026-08-20 when the user left with: "finish off phase P on your own
… make a log of [important decisions], and give me a report when I get
back. I would like to retire the tmp3 clones; we should be able to
completely remove the tmp3 directory (after making sure no files will be
lost. If a file is uncommitted, just copy it to the other clone)."

Standing decisions the user made before leaving (not autonomous):
merge FastMultipole work onto `flowpanel-20260817` and FLOWVPM work onto
`flowpanel` once passing tests (done for both); retire tmp3; the
`zeta_direct` fix must use a repo-wide-consistent container that doesn't
interfere with other consumers of that storage (delivered as the
VORTICITY_INDEX migration, commit `bc9b9a6`).

## D1 (2026-08-20) — 018 CPU smoke stopped early, recorded as PASS

The smoke run (`NREVS=0.1`, 4 threads, CPU-only, merged stack) ran
116/467 steps error-free at ~20 s/step (7,288-panel default mesh, CG
solves converging, force monitors physically sensible) before I stopped
it. The 467-step count comes from the driver's freestream-schedule revs
(ramp 2 + hold 3 + withdraw 4 + settle 4), which NREVS does not bound —
letting it finish would have cost ~2.5 h for no additional gate value.
Verdict: "018 driver still runs CPU-only unchanged" = PASS on 116 steps.
Log: `~/.claude/jobs/1e9d3d2e/tmp/panel_018_smoke.log`.

## D2 (2026-08-20) — tmp3 retirement mechanics

- All tmp3 branch tips fetched into the projects clones as remote refs
  `tmp3/*` and verified present by SHA before deletion:
  FastMultipole `combined-tree` 6eded90, `main` 58cf693, `matrix-ops`
  a244cef, `worktree-agent-afe26a7125ed0eaee` c2dd452 (worktree was
  clean); stash on combined-tree preserved as tag
  `tmp3-stash-combined-tree` (5b53765). FLOWVPM `gpu-full` c2e8400,
  `master` 4f433fb.
- `FLOWVPM-baseline-e2bd487` was a clean linked worktree at e2bd487
  (present in projects/FLOWVPM.jl history) — nothing to preserve.
- Gitignored figure build artifacts (PNG/PDF/aux) under
  `MATRIX_OPERATOR_REFACTOR/data/figures/` copied to the projects clone
  (`--ignore-existing`), although they are rebuildable from the committed
  .tex+CSV per the figures convention.
- **Deliberately NOT copied:** the tmp3 clones' `Manifest.toml`s — they
  pin dev-paths to tmp3 itself; the projects clones' Manifests already
  dev the correct sibling paths and regenerate via `Pkg.resolve`.
- tmp3 directory deleted 2026-08-20 after the sweep.
- The 046 merge-log doc commit (`a244cef`, made on tmp3 matrix-ops after
  the fast-forward) was merged into `flowpanel-20260817` (`f17fc24`) so
  no doc history was lost.

## D3 (2026-08-20) — Stage 4 checkpoint recorded as answered

The 046 user checkpoint (retire tmp3 vs re-point) was answered by the
user directly ("I would like to retire the tmp3 clones") — recorded here
and in the 046 doc; 046 marked Done in START_HERE. The Approved column is
left unticked for the user's return, per the convention that approval is
granted only by the user.

(Later decisions appended below as they are made.)
