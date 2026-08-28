# Handoff prompt — 052d cross-pass design review (2026-08-28)

Continue task 052d — FMM for panel→particle influence (FastMultipole
repo). Context: the host-FMM probe FAILED its 0.6 s/step gate (15.8
s/step measured, 99.8% near-field direct, root cause = core-size radius
inflation flooring source-tree subdivision); Ryan chose **Route B**: a
device cross-pass on the shared radix grid, reusing the shipped radix
stencil machinery, tolerance **1e-4 relative velocity error**. All
history, evidence, decisions, and the current design live in
`MATRIX_OPERATOR_REFACTOR/052d-plan-2026-08-26.md` — read, in order:
"Consolidated findings — 2026-08-28", "Decisions — 2026-08-28 (Ryan)",
the "Addendum 2026-08-28" on reusing the shipped stencil machinery
(including the independent cross-pass leaf level `ell_x`), and "Next
(updated 2026-08-28)". Supporting docs (skim as needed, delegate deep
questions to the refactor-docs-librarian agent):
`052d-host-profile-2026-08-28.md` (+`-REVIEW-`),
`052d-prototype-report-2026-08-28.md` (+`-REVIEW-`), prototype code in
`MATRIX_OPERATOR_REFACTOR/prototypes/052d_shared_radix/` (provably
correct adaptive dual-tree lists — now demoted to validation oracle).
Real step-472 geometry (241,986 particle positions, 36,752 panel
centroids + connectivity) is extracted under the previous session's
scratchpad `snapshot472/` dir with a `SNAPSHOT_INDEX.md` — check
`/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/*/scratchpad/snapshot472/`;
if evicted, re-extract per the profile report's provenance section
(files on orc: `~/FLOWPanel-052/data/fm052d_gpu_1080/…472.vtp/.vtu`).
CAUTION from review: the SNAPSHOT_INDEX "validated route" snippet omits
`calc_controlpoints!` and reproduces a degenerate body — see
`052d-host-profile-REVIEW-2026-08-28.md`.

Your job, three phases, in order:

**Phase 1 — review the machinery that already exists in the code**
(use subagents for bulk reading; return `file:line`-indexed findings):
the radix stencil layer in `src/containers.jl` (~line 300–460:
`RadixSeparationPolicy`, `ConstantPAnalyticStencil` oracle,
`HierarchicalRigidStencil` with near_radius2 `q`, level schedules,
task-025 exact-once proof conditions, task-028 operating points), the
`RadixFMMCache` build and the device radix FMM pipeline (search
`src/translate_batched_cuda.jl`, ~9k lines — where keys/occupancy are
computed and to WHAT DEPTH, how M2L routes are enumerated, how
near-field/direct lists launch, where B2M/L2L/L2B device kernels live,
what assumes targets ≡ sources), host panel B2M in
`src/bodytomultipole.jl`, and the device panel dense kernel
(`direct_rectangular!`, CUDA methods ~line 9100+). Deliverable: a
concise inventory of what the cross pass can reuse verbatim, what
needs a variant, and what does not exist.

**Phase 2 — review the plan for significant improvements**: read the
Phase 2b-revised design + addendum critically against the Phase-1
inventory. Look for: simplifications the existing code makes possible
that the plan missed; wrong assumptions (e.g. key depth, occupancy
representation, policy interfaces); cheaper alternatives to the
two-occupancy stencil kernel; whether the separate-downward-pass
decision still stands given the actual device local-expansion layout;
anything about the 1e-4 target the shipped q/P data already answers.
Propose concrete plan amendments with evidence.

**Phase 3 — verify/identify what to prototype** for
different-sources-vs-different-targets on a shared radix grid,
including DIFFERENT LEVELS (`ell_x` decoupled from the self-pass
`ell`): enumerate the candidate approaches (two-occupancy same-level
stencil at fixed `ell_x`; cross-level variants; q/P/ell_x operating
points; anything better you find in Phase 2), and for each say what
question only a prototype can answer vs what is already proven (the
existing prototype's pair-partition checker and the shipped exact-once
proof cover same-level coverage; the open items include: velocity
accuracy at 1e-4 for q ∈ {3,5,12} × P ∈ {4,6,8} × ell_x sweep against
dense at production shape, the MAC exact-tie convention, B2M cost at
36,752 panels, two-occupancy route enumeration cost). End with a
prioritized prototype list (smallest set that de-risks the device
port), sized (small/moderate/heavy), for Ryan's approval — do NOT
start implementing without his go.

House rules: ≤4 local threads; delegate test/script runs to
julia-test-runner (convention:
`JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:/Users/ryan/.julia
JULIA_NUM_THREADS=4 julia --project=<proj> script.jl > log 2>&1`);
../FLOWPanel.jl is a live worktree — read-only unless told otherwise;
no commits, no cluster job submissions without authorization (orc
reads are fine; needs `bash -lc` for slurm, prints banner noise);
never read raw CSV/data files — script summaries only; redirect long
output to scratchpad logs; known Julia 1.12.5 JIT segfault flake in
dense panel evaluation — rerun or `--check-bounds=yes`. Append
findings/decisions to the 052d plan doc as prior sessions did. A
lab-notebook entry for the whole 052d arc is pending Ryan's approval —
offer, don't write.
