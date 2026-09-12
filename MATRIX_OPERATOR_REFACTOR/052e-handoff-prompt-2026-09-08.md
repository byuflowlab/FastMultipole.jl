# 052e handoff — 2026-09-08 (context reset; supersedes 052e-handoff-prompt-2026-09-07d.md)

## Immediate state: production integration of the 052e.2b reduction is DONE

Tier 0B-R (052e.2b, implicit-Householder reduction) closed with an ADOPT
ruling (Ryan 2026-09-07); see `052e2b-tier0br-results-2026-09-07.md` and the
2026-09-07d handoff for the run record. On 2026-09-07 the consequence task was
completed: the reduction is now the production dense `:area_mean` Green-solve
route in FLOWPanel, bordered retained as debug/reference.

### What changed in `../FLOWPanel.jl` (ALL UNCOMMITTED, layered on the
### preserved uncommitted 052e.1 work — do not separate them casually)

- `src/FLOWPanel_formulation.jl`:
  - New `GreenHouseholderState <: AbstractGreenState` +
    `_build_green_householder_state(body)`: assemble A = I−B in place,
    reflector v with cancellation-avoiding sign s (H â = s e_N), two-sided
    transform H A Hᵀ via two generic rank-one broadcast updates (NOT
    `BLAS.ger!`, to keep AD-compatible eltypes working), `lu!` of the leading
    (N−1)×(N−1) view, λ from the omitted row. Allocation-free
    `_green_solve_q!(gs::GreenHouseholderState, Ssigma)` + 4-arg forwarder.
  - `_build_green_solve_state(body, :area_mean)` → Householder state.
    Bordered LU retained under gauge `:area_mean_bordered` (GreenSolveState
    gauge field/branches renamed accordingly); `:lsq` unchanged.
  - `GreenReconstruction` and `HybridWakePotential` constructors accept
    `:area_mean_bordered` (requires `green_solver=nothing` in
    GreenReconstruction). `TraceCorrected` (deprecated) untouched.
  - New `_green_lambda(gs)` accessor (Householder → `gs.lambda[]`; bordered →
    `sol_b[end]`; `:lsq` → 0); `_green_diagnostics!` now uses it.
  - Docstrings + file header updated.
- `test/runtests_unit_green_householder.jl` (NEW, included in
  `test/runtests.jl` after `runtests_unit_solver.jl`): parity vs bordered at
  τ(N)=1e3·√N·eps (trace + λ), gauge defect ≤ τ_g=1e2·√N·eps, full-coordinate
  residual ≤ 1e-10 vs independently assembled dense B, incompatible-RHS
  parity, bit-identical state reuse, route selection/validation. Fixture:
  `make_dirichlet_diamond_body(nspan=12)` from `test/test_helpers.jl`,
  synthetic smooth RHS (no wake needed). PASSED 13/13 (2026-09-07).
- `test/formulation_test.jl` (052e.1 file): Stage 5 updated to the new state
  (`_green_lambda` instead of `sol_b[end]`; asserts
  `gs isa GreenHouseholderState`) and extended with a Householder-vs-bordered
  parity check. Full standalone script PASSED, all 10 stages (2026-09-07),
  including Stage 7 Krylov/FGS/FMM route agreement end-to-end.
- `agent_policies/TESTING.md`: one verification-matrix entry for the new test
  file.

### Registered-artifact caution

`MATRIX_OPERATOR_REFACTOR/scripts/tier0br_052e2b_householder_parity.jl` is
frozen (sha256-registered in gates.txt) and calls
`_build_green_solve_state(body, :area_mean)` expecting the BORDERED state
(reads `gs.sol_b[end]`). It no longer runs against updated FLOWPanel. Do NOT
edit it (that invalidates its registered sha); any future re-run derives a new
dated harness substituting `:area_mean_bordered`.

FLOWPanel examples passing `GREEN_GAUGE` env symbols
(`rotor_hover_pressure_comparison.jl`, `rotor_hover_ground_effect.jl`) need no
changes; they now transparently get the Householder route and can select
`:area_mean_bordered` for debugging.

## Next task (Ryan chose no successor yet — ask, or default to #1)

1. **052e.2a continuation addendum, stages 2–4** (filament/particles),
   registered in the accuracy plan §5. Stage 5 folds into 052e.3.
2. Later: 052e.3 hybrid fixture; 052e.1 regression completion after 052b
   closes; 052e.6 gauge design study.

## Not yet done / open

- **Notebook entries: offered THREE times (Tier 0B, Tier 0B-R, and on
  2026-09-07 also the production integration), NO REPLY.** Offer again with
  verbosity options; approval required before writing anything.
- **Nothing committed, in either repo.** All 052e work (Tier 0B
  results/supersession, .2b prereg/harness/results/adjudication, handoffs,
  and the FLOWPanel integration above) is uncommitted; Ryan has not asked for
  a commit. If asked: 052e files ONLY (FastMultipole side), and on the
  FLOWPanel side the formulation/test/TESTING.md changes above (which include
  the pre-existing 052e.1 edits — flag that entanglement when committing).

## Repository/worktree cautions (unchanged)

- FastMultipole worktree (branch flowpanel-20260817) heavily dirty with
  unrelated higher-derivative work + untracked `HIGHER_DERIVATIVES/`. Touch
  only 052e files; never clean/stage/commit unrelated changes.
- `../FLOWPanel.jl` dirty: besides the 052e files above, unrelated BRAINSTORM
  018/021/026 files are modified/untracked. Leave them alone.
- Before touching FLOWPanel: read its `AGENTS.md`, `CLAUDE.md`,
  `agent_policies/WORKFLOW.md` (+`TESTING.md` before testing). ≤4 threads
  locally.
- Laptop runs = formulation-proof tier, not official campaigns; official
  acceptance runs need committed, tagged campaign worktrees.
- Token policy: never read `data/**` CSVs directly — summarize by script;
  gates.txt and log tails carry registered values. Registered runs launch
  nohup-detached (`julia --project=../FLOWPanel.jl --threads=4`), logs into
  `data/`.
- Any new registered run needs its own dated, LOCKED prereg before launch
  (Ryan locks; no registered values before lock).
