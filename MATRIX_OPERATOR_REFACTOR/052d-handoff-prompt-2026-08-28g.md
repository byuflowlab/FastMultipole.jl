# Handoff — Step 3 (LineGauss device arms) DONE; Step 4 sub-plan drafted, D1–D2 to settle with Ryan (2026-08-28g)

Context: the 2026-08-28 session-4 (handoff `052d-handoff-prompt-2026-08-28f.md`)
completed Step 3 and drafted the Step-4 sub-plan. Records:
`052d-plan-2026-08-26.md` § "Step-3 — LineGauss device arms in the
rectangular kernel (2026-08-28, session 4)" and the Step-4 sub-plan doc
`052d-step4-subplan-2026-08-28.md` — read BOTH first. Do not re-derive
anything summarized there.

## How to work this handoff (process contract)

Attack steps ONE AT A TIME, in order. For each: develop your own lower-level
plan, execute, validate, append findings to `052d-plan-2026-08-26.md`, then
EITHER hand off via a fresh `052d-handoff-prompt-<date><letter>.md` OR
continue if context is slim. Delegate: test/script runs → `julia-test-runner`;
doc questions → `refactor-docs-librarian`; bulk exploration → Explore
subagents. Never pull raw logs/CSVs into context — script summaries only.

## State (do not re-establish)

- **Step 3 COMPLETE, validated, UNCOMMITTED** (awaiting Ryan's diff review).
  FastMultipole branch `flowpanel-20260817`, working-tree changes:
  - `src/direct_rectangular.jl`: `_rect_lg_*` helper block +
    `_rect_linegauss_gradient` (faithful transcription of FLOWPanel commit
    `8b07f96` elements_fmm.jl:973-1152, FIXED axis guard ĥ² < 1e-7; one
    deviation: incremental Float64 binomial, exact for m ≤ 12, GPU-safe);
    `REG == 4` arms in `_rect_bound_vortex_velocity/_gradient`;
    `:linegauss` → code 4 → `Val(4)` (was THROW); codes ≥ 5 still throw.
    erf backend = the file's vendored fdlibm `_rect_erf` (== FLOWVPM
    `custom_erf64` bit-for-bit) — host and device agree exactly.
  - `test/direct_rectangular_test.jl`: `:linegauss` added to gate-4b and
    both FLOWPanel family-parity loops (famtol 1e-7 for linegauss only —
    erf-backend floor). 67/67 PASS.
  - `prototypes/052d_compact_kernel/k03_flowpanel_port.jl`: P5 flipped from
    THROW-assert to code-4 routing assert.
  - NEW `prototypes/052d_compact_kernel/k04_rect_linegauss.jl`: Step-3
    parity harness, ALL PASS (26 checks): vs FLOWPanel port 5.3e-10 worst;
    vs prototype 4.441e-8 vel / 1.599e-8 grad (= the known k03 erf ×
    q̃-conditioning floor — no transcription error). Endpoint contract exact.
  - No CUDA-file edits were needed (host + device share `_rect_reg_val` and
    the pair math). GPU compile/parity deferred to the Step-5 sbatch.
- **Step-4 sub-plan DRAFTED**: `052d-step4-subplan-2026-08-28.md` — Route B
  in stages A–G (A: productionize CrossStencil two-occupancy list builder +
  guard schedule q_L into src/; B: host panel B2M + coefficient upload;
  C: cross-M2L device kernel reusing per-class operator tables,
  translate_batched_cuda.jl:3560-4667 — THE heavy item; D: L2L + U-only L2B
  through perm_x; E: block-sparse near field wrapping `_rect_panel_pair`;
  F: PANEL_INFLUENCE_FMM seam dispatch in FLOWPanel_gpu_influence.jl;
  G: host-as-oracle harnesses). No Step-4 code written.
- Operating point (certified, p32g): q=12 / ell_x=5 / P=6 / Rg=6 mm.
  The p33 memo's R_guard=0.06 m is the SUPERSEDED Gaussian-kernel value.
- Lab-notebook entry for the 052d arc: STILL PENDING Ryan approval — offer,
  don't write.

## Steps

**Step 0 — settle D1–D2 with Ryan (he has said he'll decide these; ask
first thing, referencing sub-plan §1).**
- D1: producer location. Handoff-f said host-built lists + host B2M with
  per-step upload; the p33 memo (§2, device-native ruling) says
  keying/sort/occupancy/routes/B2M on device. Recommendation on the table:
  HOST producers for v1 (validated prototype, 0.015–0.05 s/step fits the
  0.6 s gate; device-native = later optimization).
- D2: MAC-tie convention. Recommendation: ≤ everywhere (near iff |o|² ≤ q —
  what the uniform-q classifier already computes and what amendment A1 pins
  in the θ-MAC oracle; guard demote strict-<).
- D3 (env-flag naming for validation mode) defaults to the sub-plan proposal
  unless Ryan objects. Also offer him the Step-3 diff for review/commit —
  commit scope: the four files above (FastMultipole only; FLOWPanel
  untouched this session).

**Step 1 — execute the sub-plan stages A–B (host side, CPU-verifiable).**
Update the sub-plan doc first with Ryan's D1/D2 rulings. Stage A:
`CrossInteractionLists` in src/ from
`prototypes/052d_cross_stencil/CrossStencil.jl` (CrossGrid / LevelCells /
UniformQTables / sweep_config) + guard classifier (per-level integer
thresholds t_L from q_L = (Rg/w_L + √3)², exact-once per memo §1 with
q → q_L; currently only in the p32e harness) + perm_x (segmented sort to
ell_x within ell-cells) + upload-shaped POD outputs. Tests: count identity
Σ|A||B| = ns·nt, brute per-pair coverage subset, bit-compare vs untouched
prototype. Stage B: host B2M per cross leaf at P=6 into a
FlatCoefficientBuffer-shaped array + upload path; log the per-step cost vs
the 0.6 s gate.

**Step 2 — stages C–E (device kernels).** Cross-M2L (C) is the long pole:
shakedown-read the per-class table builder for single-occupancy assumptions
BEFORE committing to the reuse claim. Then D (downward, existing kernels) and
E (block-sparse near field, ~0.039 s target at step-472 shape).

**Step 3 — stages F–G.** Seam dispatch (FLOWPanel edits minimal), oracle
harnesses (list bit-compare exact; B2M parity; end-to-end sampled dense at
step-472, relRMS ≤ 7.2e-6 + margin). CPU parts run locally; GPU parts
accumulate into the Step-5 sbatch (former handoff-f Step 5: device parity +
36-step gate A/B + timing — needs Ryan's authorization to submit).

**Step 4 — multi-rotor tail (052b checklist)** — unchanged from handoff-f
Step 6: `052b-impl-multirotor-ige-gpu.md` sections D–F; re-tune ell_x/Rg
headroom at 4-rotor scale.

## Key facts you'd otherwise re-derive

- Run pattern (from FastMultipole repo root):
  `JULIA_DEPOT_PATH=/private/tmp/flowpanel-052b-depot:/Users/ryan/.julia
  JULIA_NUM_THREADS=4 julia --project=../FLOWPanel.jl <script> > log 2>&1`.
  Julia 1.12.5 JIT segfault flake in dense panel eval — rerun once.
- Snapshot472 binaries + solved sigma/gamma:
  `/private/tmp/claude-502/-Users-ryan-Dropbox-research-projects-FastMultipole/1a9c539a-6d87-44ee-87f7-d4e1c17d2793/scratchpad/snapshot472/`
  (VERIFY — /private/tmp is volatile; re-export recipe in
  `052d-handoff-prompt-2026-08-28c.md`). rc = core_size = 1e-3 m; 36,752
  tri panels × 241,986 particles; dense 8.89e9 pairs @ 2.695e9 pairs/s.
- Step-3 validation logs (session-4 scratchpad, volatile): k04.log,
  direct_rect_test.log — k04 is rerunnable in ~1 min if gone.
- Two-occupancy exact-once proof + producer interfaces + ell_x>ell path:
  `052d-p33-coverage-and-device-interface-2026-08-28.md` (201 lines, read
  directly; but its §3b R_guard value is superseded, see State).
- Seam: `FLOWPanel.jl/src/FLOWPanel_gpu_influence.jl` (env
  `PANEL_INFLUENCE_FMM`, `_gpu_route_*` counters/fallbacks, `pack_panels!`
  17-row buffer). Hot-loop contract: NEVER read `FILAMENT_REGULARIZATION[]`
  per edge — `Val(family)` barrier (+34-49% regression otherwise).
- Known pre-existing wrinkle (flagged, unfixed): semi-infinite wake
  `_U_boundvortex_gradient` tracks the active family while its velocity
  sibling hardcodes Vatistas (elements_fmm ~:1552 vs ~:1608); LineGauss
  z2→−∞ limit derived in the session-2 plan section if ever scoped.

## House rules

≤4 local threads; delegate runs to julia-test-runner; no cluster submissions
without Ryan's authorization; no commits without Ryan seeing the diff;
FLOWPanel edits minimal and scoped; never read raw CSV/bin — script
summaries; long output → scratchpad logs; append findings to
`052d-plan-2026-08-26.md`; math per user-CLAUDE.md Math Syntax ($$ blocks);
notebook entries need Ryan's approval; new prototype code stays in
`MATRIX_OPERATOR_REFACTOR/prototypes/`.
