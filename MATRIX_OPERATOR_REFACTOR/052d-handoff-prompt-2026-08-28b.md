# Handoff prompt — 052d prototypes P3.1–P3.3 (2026-08-28, post design review)

Continue task 052d — device cross-pass FMM for panel→particle influence
(FastMultipole repo). The design review is COMPLETE and all decisions are
settled; your job is to EXECUTE the approved prototype sequence
P3.1 → P3.2 → P3.3 (Ryan authorized starting them in the handoff
instruction). The device port itself still needs a separate go after
P3.2's numbers land.

## Required reading, in order (all in `MATRIX_OPERATOR_REFACTOR/`)

`052d-plan-2026-08-26.md` — read these sections (skip the rest unless
needed; delegate deep doc questions to the refactor-docs-librarian agent):
1. "Consolidated findings — 2026-08-28" and "Decisions — 2026-08-28
   (Ryan)" + its Addendum (Route B, tolerance 1e-4 relU ceiling, shipped
   rigid-stencil reuse, independent cross-pass leaf level `ell_x`).
2. "Cross-pass design review — 2026-08-28" — the Phase-1 `file:line`
   inventory of reusable machinery, amendments A1–A7, and the P3.1–P3.3
   prototype definitions you are executing.
3. "Decision — 2026-08-28 (Ryan, later ruling): device-native producers"
   and "Decision audit … R1–R6" + the two "Ruling(s)" sections after it —
   final settled state: device-native list-gen and panel B2M in
   production (host versions = validation oracles + the P3.2 vehicle);
   grid sharing default with panel-containment assert and union-box
   fallback; cross-M2L classes = the rigid stencil's level-invariant
   offset set via the existing per-class operator machinery; invoke from
   the PANEL_INFLUENCE_FMM seam; uniform-q cross schedule v1; reverse
   leg (particles→panels) in scope, sequenced LAST after machinery
   matures.

Supporting docs (skim as needed): `052d-host-profile-2026-08-28.md`,
`052d-prototype-report-2026-08-28.md`, both `-REVIEW-` files. Prototype
oracle code: `MATRIX_OPERATOR_REFACTOR/prototypes/052d_shared_radix/`
(`SharedRadix.jl` grid/tree, `validate.jl:24-40` `coverage_counts`
pair-partition checker, `check_production.jl`, `production_run.jl`).

## Key facts you'd otherwise re-derive

- Geometry: 36,752 tri panels × 241,986 particles (step-472 snapshot),
  8.89e9 dense pairs. Snapshot binaries + `SNAPSHOT_INDEX.md` under
  `/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/*/scratchpad/snapshot472/`
  (CAUTION: the index's "validated route" snippet omits
  `calc_controlpoints!` and reproduces a degenerate body — see
  `052d-host-profile-REVIEW-2026-08-28.md`). If evicted, re-extract per
  the host-profile provenance section (orc:
  `~/FLOWPanel-052/data/fm052d_gpu_1080/…472.vtp/.vtu`).
- Reusable stencil layer (occupancy-free, builds from `(h0, ell,
  ell_axes, q)`): `RigidHierarchicalTables(q)` ctor
  `src/interaction_list_batched.jl:153-206`, `rigid_stencil_epsilon`
  `:106-150`, classifier gate `:464-525`, grid-free bound `:647-665`.
  Supported q values `src/containers.jl:339`. Host panel B2M:
  `src/bodytomultipole.jl:645-869` (tri/quad × Source/Dipole/
  SourceDipole/Vortex). Single-occupancy list builders to variant:
  `build_hierarchical_routes_window!` `:578-615`,
  `build_hierarchical_direct_pairs!` `:617-642`.
- Accuracy landscape: shipped self-pass data does NOT answer 1e-4
  (task 032: q=12/P=4 → 1.088e-3 VELOCITY on the regularized
  self-field, `containers.jl:334-338`), but panel sources are benign
  (host probe ~5e-5 relU at p=4/θ≈0.4–0.5). Expect the frontier near
  q=3/P∈{6,8} or q∈{5,12}/P∈{3,4}. The leg is U-only in production.
- MAC ties: DISSOLVED for production (integer `|o|² ≤ q` classification);
  the exact-≤ tie convention lives only in the θ-MAC validation oracle.
- `ell_x > ell` is the likely optimum; device keying kernel is
  depth-parametric (`translate_batched_cuda.jl:138-159`) so this is fine
  for the port, but P3.1 must deliver the `ell_x` verdict.

## The work

**P3.1 (small) — list-statistics sweep, no expansions.** Two-occupancy
rigid-stencil lists on the shared grid at snapshot shape: occupied
panel/particle cell counts, M2L route counts, near-field pair counts,
per-offset-class census, over `ell_x` (self-`ell` .. +4ish) ×
q ∈ {3,5,12}, uniform schedule. Run the prototype `coverage_counts`
checker on EVERY config (exact-once certification — the task-025 proof
does not cover two-occupancy lists). Output: cost-model table, narrowed
(q, `ell_x`) grid, `ell_x` vs `ell` verdict.

**P3.2 (moderate) — host accuracy/cost harness at production shape.**
Panel B2M per `ell_x`-cell (TIME IT at 36,752 panels — open item),
cross-M2L via host translate ops, L2L + U-only L2B, velocity relRMS vs
a dense reference on a ~5–10k-particle sample, over the narrowed grid ×
P ∈ {3,4,6,8}, using the production mixed panel kernels (this covers
the φ/χ channel check). Output: the (q, P, `ell_x`) operating point vs
the 1e-4 ceiling WITH margin per config (Ryan may tighten below 1e-4 if
the 36-step locked-gate fingerprint demands — report margins so he can
make that call), + measured host-side costs.

**P3.3 (small, paper-only) — coverage note + device-interface memo.**
Half-page uniform-q two-occupancy exact-once argument; device producer
interfaces (what arrays the cross pass consumes; host-as-oracle parity
checks: list bit-compare, B2M coefficient parity); the `ell_x > ell`
data-path (keying + non-counting sort, cross permutation
gather/scatter).

Then report to Ryan with the recommended operating point and ask for
the device-port go. Do NOT start the device port without it.

## House rules

≤4 local threads; delegate test/script runs to julia-test-runner
(convention: `JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:/Users/ryan/.julia
JULIA_NUM_THREADS=4 julia --project=<proj> script.jl > log 2>&1`);
../FLOWPanel.jl is a live worktree — read-only unless told otherwise; no
commits, no cluster submissions without authorization (orc reads fine;
`bash -lc` for slurm, banner noise); never read raw CSV/bin data — script
summaries only; long output → scratchpad logs; known Julia 1.12.5 JIT
segfault flake in dense panel evaluation — rerun or `--check-bounds=yes`.
Append findings to the 052d plan doc as prior sessions did. A
lab-notebook entry for the whole 052d arc is still pending Ryan's
approval — offer, don't write. New prototype code goes in
`MATRIX_OPERATOR_REFACTOR/prototypes/` (e.g. a `052d_cross_stencil/`
sibling dir), not in `src/`.
