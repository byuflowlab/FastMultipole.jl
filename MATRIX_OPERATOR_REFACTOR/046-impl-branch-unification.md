# 046 Impl: Branch Unification (the merges)

## Status and entry gate

**Completed `2026-08-20`; review remediation recorded `2026-08-21`.**

Entry gate: `042` complete and approved (granted `2026-08-20`). First row of
the Production Integration Phase; front-loaded by design so that every
subsequent GPU diff (`047`–`052`) lands directly on the branches FLOWPanel
consumes, instead of being built on the tmp3 clone and then pushed through a
174-commit merge later. `047` and `048` both block only on this row.

**User checkpoint (do not decide unilaterally):** final repo layout after
the merges — retire the tmp3 clones vs re-point them at the unified
branches.

## Motivation

User direction (`2026-08-20`): the Production Integration Phase's objective
is to use the matrix-ops machinery to accelerate the FLOWPanel item-018
campaign. That requires merging FastMultipole `matrix-ops` (tmp3) into
`flowpanel-20260817` (projects clone), and FLOWVPM `gpu-full` (tmp3) into
`flowpanel` (projects clone) — the branches FLOWPanel's Manifest dev-paths
actually resolve to. Doing the merges first retires the riskiest task of the
phase before any new code is written on top of it.

## Objective

Both merges completed on the projects-side branches with no merge-caused test
regressions, the 018 driver still running through its CPU-only path, and the
user's repo-layout decision recorded. A dirty-tree failure is not called a
green suite: it must be isolated from the merge and followed by a green run
before approval.

## Method

### Stage 0 — safety protocol (applies to BOTH merges)

- Tag/backup both sides of each merge before touching anything
  (e.g. `pre-046-<branch>` tags in each repo, plus a bundle or clone backup).
- Perform each merge on a **scratch branch**; fast-forward the real branch
  only after the Stage 3 test gate passes.
- tmp3 and projects clones are object-disjoint (no shared objects), so each
  merge starts by `git fetch`-ing one clone into the other as a remote.

### Stage 1 — FastMultipole merge (the big one)

Fetch tmp3 `matrix-ops` (`6b166eb`, 174 commits ahead of the 2026-06-12
merge-base `58cf693`) into `projects/FastMultipole` and merge into
`flowpanel-20260817` (HEAD `645cc96`, 9 commits ahead of main@2026-08-17).
Both sides touch fmm!/tree internals. Conflict hotspots: the
flowpanel-20260817 side's **FmmPlan, NearfieldInfluenceCache (dense
nearfield as packed BLAS matvecs), cached-nearfield tune=true, autotune
perturbation, FastGaussSeidel colored sweeps, lu caching** vs matrix-ops'
refactored fmm!/tree/nearfield internals. Enumerate the 9 flowpanel-side
commits (`git log main..flowpanel-20260817`) and resolve each intentionally;
FmmPlan/nearfield-cache layers are exactly what a GPU nearfield would also
own, so resolution here is design work, not just textual conflict handling.

### Stage 2 — FLOWVPM merge

Fetch tmp3 `gpu-full` (clean tree; 23 ahead of merge-base `e2bd487`) into
`projects/FLOWVPM.jl` and merge into `flowpanel` (HEAD `16f8ef7` "lots of
work" 2026-08-20, 10 ahead of master@`76d46ed`). Disjoint lineages — a real
merge, not a fast-forward. **flowpanel's `9fd25e6` "Allow Estr_fmm! to
select source and target FMM systems" is load-bearing for 018** — preserve
its semantics. Respect FLOWVPM CLAUDE.md constraints (explicit include
order; no Particle struct; forked CPU-loop vs broadcast hot paths — do NOT
unify).

### Stage 3 — test gate + smoke

- FastMultipole, FLOWVPM, and FLOWPanel test suites show no merge-caused
  regressions post-merge (FLOWVPM
  incl. `runtests_gpu_fmm.jl` Part A host-side; device parts on the cluster
  if convenient, else recorded as deferred to `047`/`048` gates).
- The 018 driver (`examples/rotor_hover_pressure_comparison.jl`) still runs
  CPU-only unchanged — a short smoke (a few steps), not a campaign.
- Update FLOWPanel's Manifest dev-paths if needed (they point to
  `../FLOWVPM.jl` and `../FastMultipole` relative to projects/).

### Stage 4 — user checkpoint

Present the resulting topology and ask: retire tmp3 clones, or re-point them
at the unified branches. Record the decision here.

## Gates and verdict

- Both merges landed on the real branches only after green test gates on the
  scratch branches.
- 018 CPU smoke executes the unchanged CPU-only driver path without errors.
  Physical-sanity output is a smoke gate, not numerical pre/post parity unless
  a durable pre-merge reference is recorded.
- User checkpoint on repo layout answered and recorded.

## Artifacts

- Merge commits + pre-merge tags in both projects-side repos.
- A merge log section appended to this doc: fetch/merge commands, the
  complete projects-side commit list, conflicts encountered and how each was resolved, test
  outcomes.

## Verification

- `git log` shows both merge commits with the expected parents; pre-merge
  tags exist on both sides of each merge.
- All three test suites green on the merged branches, or any dirty-tree failure
  is explicitly isolated from the merge and followed by a green verification;
  018 CPU smoke executes the CPU-only path without errors.

## Recorded context (2026-08-20 staging)

Facts established at staging — do not re-derive.

**Branch topology (the merge task):**

- FLOWPanel Manifest dev-paths → `../FLOWVPM.jl` and `../FastMultipole`
  **relative to projects/** — i.e. `projects/FLOWVPM.jl` (checked out on
  `flowpanel`, HEAD `16f8ef7` "lots of work" 2026-08-20, 10 ahead of
  master@`76d46ed`; includes `9fd25e6` "Allow Estr_fmm! to select source and
  target FMM systems") and `projects/FastMultipole` (checked out on
  **`flowpanel-20260817`**, HEAD `645cc96` 2026-08-20; 9 commits ahead of
  main@2026-08-17: **FmmPlan, NearfieldInfluenceCache (dense nearfield as
  packed BLAS matvecs), cached-nearfield tune=true, autotune perturbation,
  FastGaussSeidel colored sweeps, lu caching**).
- tmp3/FastMultipole `matrix-ops` (`6b166eb`) has merge-base with main at
  `58cf693` **2026-06-12, 174 commits ahead** → the FastMultipole merge is
  the big one; both sides touch fmm! internals, and flowpanel-20260817's
  FmmPlan/nearfield-cache layers are exactly what a GPU nearfield would also
  own.
- tmp3 and projects clones are object-disjoint (no shared objects) → merging
  requires fetching one into the other.
- FLOWVPM merge: tmp3 `gpu-full` (clean tree; 23 ahead of merge-base
  `e2bd487`; no local `flowpanel` branch, only `origin/flowpanel`, last
  commit `a950790` "relaxation filter" 2026-06-30, 30 ahead / 10 behind vs
  gpu-full at merge-base `2dc7f05` 2025-09-12) vs projects `flowpanel` (10
  ahead of `76d46ed`) — disjoint lineages, real merge, not a fast-forward.

**FLOWVPM constraints (CLAUDE.md, echo here):** explicit include order in
src/FLOWVPM.jl; no Particle struct — 46-row dense matrix + index constants;
CPU/GPU switch is `pfield.particles isa Array` (`useGPU` vestigial);
hot-path physics deliberately forked CPU-loop vs broadcast (do NOT unify
without benchmarks — 4–10× regressions seen); radix coupling supports only
`gaussianerf`, autotune off, rbf/sfs fail loudly; `gpu-full` pinned to dev
FastMultipole `matrix-ops` (registry 2.0.4 lacks shrink/recenter kwargs).

**FLOWVPM tests:** `test/runtests.jl` → singlevortexring + leapfrog (CPU,
slow — minutes), `runtests_gpu.jl` (gated CUDA.functional),
`runtests_gpu_fmm.jl` (429 lines; Part A host radix, self-skips; Part B
device, hard-required under `FASTMULTIPOLE_REQUIRE_CUDA_TESTS=1`). 5 files,
12 testsets. H200 drivers: `scripts/cuda_034_{run,submit}.sh`,
`cuda_034_refcheck.jl`.

**FLOWPanel location + driver:** `/Users/ryan/Dropbox/research/projects/FLOWPanel.jl`
(user-corrected; NOT under tmp3), branch `fastmultipole`, dirty data/. Item
018 = `BRAINSTORM/INDEX.md:74`,
`018_dji9443_hover_convergence_campaign.md` (LIVE; Phase 16 opened
2026-08-14). Driver `examples/rotor_hover_pressure_comparison.jl` (1501
lines, all knobs via env vars); launcher
`examples/run_dji9443_hover_ct_hpc.slurm.sh` (`p018_*` case matrix at
:355-410).

## Merge log (2026-08-20 execution)

Reconstructed commands (the tmp3 directories were retired after SHA
verification, so their former absolute paths are intentionally not retained):

```sh
# FastMultipole projects clone
git remote add tmp3 <retired-tmp3-FastMultipole-path>
git fetch tmp3 matrix-ops
git tag pre-046-flowpanel-20260817 d7145440
git tag pre-046-matrix-ops 3505fb44
git switch -c merge-046-fastmultipole d7145440
git merge --no-ff 3505fb44
# merge commit 6c881834; flowpanel-20260817 fast-forwarded after tests

# FLOWVPM projects clone
git remote add tmp3 <retired-tmp3-FLOWVPM-path>
git fetch tmp3 gpu-full
git tag pre-046-flowpanel 16f8ef76
git tag pre-046-gpu-full c2e84009
git switch -c merge-046-flowvpm 16f8ef76
git merge --no-ff c2e84009
# merge commit 903771dd; flowpanel fast-forwarded after tests
```

The staged “9 commits” grew before execution. The complete projects-side
FastMultipole first-parent list merged against matrix-ops was:

1. `5adde3bd` callback kwarg for FastGaussSeidel;
2. `64e06ca7` FGS multithread race fix;
3. `a9b734ad` LU caching;
4. `b3cf4ad9` reusable `FmmPlan`;
5. `752c5259` colored parallel FGS sweeps;
6. `72d3f3d8` `NearfieldInfluenceCache`;
7. `02f071c2` nearfield-cache integration;
8. `0ef4e83d` cache estimate/build-time guard;
9. `204188ae` cached-nearfield `tune=true`;
10. `1ec0af9d` cache-aware leaf-size retune;
11. `645cc96c` autotune perturbation mode;
12. `eea944d0` rigid `transform_tree!`;
13. `087bf4aa` rigid `transform_plan!`;
14. `d7145440` rigid `transform_solver!` and staleness fix.

Pre-merge safety: tags `pre-046-matrix-ops` (tmp3 FastMultipole `c4c61ce`→`3505fb4`),
`pre-046-flowpanel-20260817` (`d714544`), `pre-046-gpu-full` (tmp3 FLOWVPM),
`pre-046-flowpanel` (`16f8ef7`). Cross-clone `tmp3` remotes added + fetched
(clones are object-disjoint as recorded).

Pre-merge commits to tmp3 `matrix-ops`: `c4c61ce` (previously-uncommitted
041e fused-nearfield U CSR surface + test — same hazard as the 026 refactor
surface) and `3505fb4` (untracked 041b–041k artifacts + phase staging docs),
so the merge carries the full approved surface.

**FastMultipole:** `matrix-ops` (176 commits) merged into
`flowpanel-20260817` (12 ahead, incl. 3 new transform_* commits beyond the
staged snapshot). Only 2 conflicts: `src/FastMultipole.jl` export/loader
block (union of both sides) and `CLAUDE.md` (one-blank-line delta; took
matrix-ops). The fmm!/tree/FmmPlan hotspots auto-merged textually. Full test
suite green (562k+ assertions, both sides' testsets; only benign macOS
no-CUDA notices). Untracked `FUTURE_IMPROVEMENTS.md` set aside as
`FUTURE_IMPROVEMENTS.local.md`. Fast-forwarded `flowpanel-20260817` →
`6c88183` per user direction (2026-08-20).

**FLOWVPM:** `gpu-full` merged into `flowpanel`. 11 conflict hunks / 6
files; resolution posture = flowpanel's newer physics/API as base, gpu-full
GPU machinery grafted on (details in merge commit `903771d`). Notables:
UJ_fmm keeps flowpanel's vorticity-via-extra_outputs CPU path with gpu-full's
CuArray dispatch to UJ_fmm_gpu! in front; legacy `nearfield_device`
forwarding removed (documented nearfield-dropping hazard); include order =
fmm, fmm_radix, merging, splitting. First test run: 58/66 — all 8 failures
one root cause: gpu-full-added functions still using pre-migration
J[1:3]-as-zeta storage. Follow-up commit `bc9b9a6` completed the
dedicated-vorticity (VORTICITY_INDEX 13:15) migration repo-wide per user
direction (zeta_direct_multithreaded, _corespreading_reset_broadcast!, ext
gpu_zeta_direct!, gpu parity test); audited that all remaining J[1:3]
consumers legitimately want the velocity gradient (relaxation curl,
stretching, get_W2/W3, h5 writer, Estr). Suite then fully green.
Fast-forwarded `flowpanel` → `bc9b9a6` per user direction. Untracked
`CLAUDE.md` set aside as `CLAUDE.local.md` (differs from merged one).

**FLOWPanel suite vs merged stack:** 16/18; the 2 failures
(`radius_inflation formulas`) are in the user's uncommitted FLOWPanel WIP
(FilamentRegularization enum machinery, dirty tree; test file drifted
mid-run) — not merge-caused. GPU-device test tiers (Part B) deferred to the
cluster (no CUDA on this machine); they gate `047`/`048` work anyway.

Open: 018 CPU smoke result; Stage 4 user checkpoint (retire tmp3 vs
re-point).

## Stage 4 resolution + retirement record (2026-08-20)

User checkpoint answered by the user directly: **retire the tmp3 clones**.
Executed same day: all tmp3 branch tips fetched into the projects clones as
`tmp3/*` remote refs and SHA-verified before deletion (FastMultipole
combined-tree/main/matrix-ops/worktree-agent + stash as tag
`tmp3-stash-combined-tree`; FLOWVPM gpu-full/master); the
`FLOWVPM-baseline-e2bd487` linked worktree was clean and its commit is in
history; gitignored figure artifacts copied; tmp3 Manifests deliberately not
copied (they pinned dev-paths to tmp3). tmp3 directory removed. 018 CPU
smoke: 116/467 steps error-free on the merged stack before deliberate stop
(the 467 = freestream-schedule revs; ~20 s/step, sensible CF/CM monitors) —
gate PASS. Details: `decisions-phaseP-2026-08-20.md` D1–D3.

## Review remediation verification (2026-08-21)

- Merge topology rechecked: FastMultipole `6c881834` has parents
  `d7145440`/`3505fb44`; FLOWVPM `903771dd` has parents
  `16f8ef76`/`c2e84009`. All four `pre-046-*` tags exist and point into the
  corresponding histories; `9fd25e6` remains an ancestor of unified FLOWVPM.
- The historical FLOWPanel result remains recorded honestly as 16/18 rather
  than “green.” Its two radius-inflation failures were in concurrent WIP.
  After that WIP stabilized, the same merged stack was exercised locally on
  Julia 1.12.5: all package-test groups through the solver, instrumentation,
  wake, postprocess, simulation, Kutta, warmstart, and replay surfaces passed;
  two missing direct test dependencies (`Random`, then `TOML`) were exposed and
  declared; the remaining example/analytical tail then passed. The tracked
  Gmsh topology suite passes 28/28 after the GeoIO removal follow-up.
- The 018 smoke claim is intentionally limited to the preserved evidence: 116
  error-free CPU-only steps with converged solves and physically sensible
  monitors before deliberate termination. No pre-merge numeric reference was
  preserved, so this document no longer claims numerical pre/post parity.
- The merge-log artifact now contains the reproducible fetch/merge skeleton
  and the complete actual side-commit list. No merge or user-WIP history was
  rewritten during remediation.
