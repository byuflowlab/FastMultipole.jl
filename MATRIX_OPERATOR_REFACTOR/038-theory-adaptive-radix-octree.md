# 038 Theory: 2:1-Balanced Adaptive Radix Octree

## Status and Entry Gate

**DONE `2026-08-14` (overnight campaign lead agent); first clear-context
review returned CHANGES REQUIRED (19:27 MDT decision-log entry); the
blocker is fixed (sticky demotion, see Review Correction below); awaiting
fresh clear-context re-approval.**

Entry gate: `035` must complete its profiling campaign and `036` (Integration
Phase milestone review) and `037` (rectangular grid) must be complete and
approved. Before derivation begins, confirm from the `035`/`037` measurement
record that multi-scale density (dense clusters plus diffuse regions — e.g.
wake rollup, `CoreSpreading`-grown σ) is a binding cost that the uniform-depth
grid cannot serve, or obtain an explicit user waiver to proceed without that
evidence. Record the confirming evidence or waiver here.

**Entry gate: MET on evidence (adjudicated `2026-08-14 19:04 MDT`, decision
log entry of that time).** `037b`'s rotor-wake campaign measured multi-scale
density as a binding cost of the uniform-depth grid: 7.16x at `n=1e6`
(238.7 → 33.3 ms at pinned `ℓ8`) and 1.81x at `n=1e5` from depth selection
alone; occupancy contrast (max/mean bodies per occupied cell 5–7× at every
level vs wake ≤1.9×; top-1% densest cells hold ~6% of bodies,
`data/rotor_wake/rotor_case_stats.csv`); the uniform path's `ℓ≤8` cap
plausibly still binds at `1e6`. Nuance recorded for honesty: the `σ_max`
geometry-gate mechanism does NOT bind on the rotor (thin tip cores keep
`σ_max` small) — the binding mechanism is density contrast; the per-cell σ
gate is derived as a correctness/generality deliverable, not the measured
win mechanism. No waiver was needed.

## Completion Notes (2026-08-14)

Artifacts delivered:

- `theory/adaptive-radix-octree.md` — the full derivation, items 1–7 of the
  Objective: §1 construction as sort/scan/compact over full-depth Morton
  keys (occupied linear octree, prefix property, frontier sweep, 2:1
  balance sweep with termination proof); §2 near predicates (same-level
  `N_q` offsets; mixed-level finer-lattice clamp rule), the dual-tree
  recursion with invariants, U/V/W/X definitions, V-class admissibility
  (**`025` level-scaled operator tables reused unchanged — no new
  tables**), W/X level structure (classic one-level proposition proven for
  the complete-tree limit, occupancy-pruned deeper entries identified and
  measured), pipeline placement, flag/scan/compact generation; §3
  exact-once theorem (partition-invariant proof, predicate-independent —
  which is what makes the σ-gate demotion free) + uniform-limit `025`
  parity proposition; §4 M2T/S2L in the compressed complex basis
  (Gumerov-normalized irregular harmonics, production-consistent P2M/P2L
  mirror rules, LH channels at `P_chi = P_phi + 1` per `008h`, exact
  M2L-composition verification oracles for 040/041, error bounds inside
  the same-level V budget via the `008d` bound, `c > 2` shown for every
  admissible offset at both radii); §5 per-cell σ gate (per-node
  `σ_max` upward sweep, gated far predicate with demotion + correctness
  theorem implying the `031a` §5.1 contract, split-veto variant, global
  throw eliminated); §6 cost/capacity model (work terms, list bounds,
  node capacity `≤ β_bal(1 + 8·ℓ_max·⌈n/(K_max+1)⌉)`, explicit
  `RadixFMMCache` capacity box, counted multi-scale evidence); §7
  refresh/rebuild/recenter! policy (frozen-leaf-set refresh with cheap
  validity reductions incl. unmatched-key and σ-margin triggers,
  hysteresis `K_hi = 2K_max`, epoch semantics); §9 explicit implementation
  contract for 039–041.
- `scripts/adaptive_octree_verify.jl` — stdlib-only, single-threaded,
  deterministic validation (byte-identical reruns verified).
- `data/adaptive_octree/{exact_once_coverage,sigma_gate_contract,
  cost_model_counts,constant_p_bound_consistency,s2l_m2t_convergence}.csv`
  + `summary.txt`.

## Review Correction (2026-08-14, after the 19:27 MDT CHANGES REQUIRED)

The first clear-context review found the §2.4 "no new operator tables"
claim FALSE as originally derived: the §5 gate's demote-and-descend rule
allowed descendants of a demoted (geometrically separated) pair to
re-admit to V, emitting V pairs with separated parents — offsets outside
the `025` phase-table set (reviewer measured up to Chebyshev reach 11 on
the one-fat-core fields; the original script only checked V classes on
UNGATED lists, masking it). Coverage and accuracy were unaffected.

Fix adopted (option a, **sticky demotion** — overnight decision, user
ratification item recorded in the decision log): a demoted pair's entire
descendant pair set terminates in U; V/W/X are emitted only on
never-demoted paths, restoring Invariant 2 geometrically and making the
§2.4 table claim hold unconditionally with the gate active. Theory
§2.4/§5.2/§5.3/§5.4 reconciled; the former "descent helps" re-admission
claim replaced by a cost-locality argument; the table-free M2T/S2L
re-admission variant (option b) is recorded in §5.4 as a user-ratifiable
optimization for `039`/`040`. Script now checks V-class/phase-table
membership and W/X structure on GATED lists (all pass), and the
uniform-limit parity check additionally runs exact-once painting on the
parity tree (reviewer's non-blocking note). Sticky over-cost quantified
in `sigma_gate_contract.csv` (gated vs ungated direct pairs: worst 3.5×
at q=3 with 2-decade heterogeneous σ; ≤3% on q=12 heterogeneous).

The verification claims below are as corrected (gated-list V-class checks
included); data regenerated, ALL CHECKS PASS, byte-identical rerun
re-confirmed.

Verification status — ALL CHECKS PASS:

- exact-once ordered-pair coverage (brute-force `n²` painting, `n = 3000`)
  on 5 distributions (uniform, multiscale 30×/100×, rotor-like filament,
  adversarial two-cluster) × `q ∈ {3, 12}` × `K_max ∈ {16, 64}` ×
  balanced/unbalanced — 40 configurations, zero violations;
- 2:1 balance fixed point + property; V-class admissibility (separated
  offset, near parent, Chebyshev reach ≤ 3/7) on every emitted V pair, on
  both ungated AND σ-gated lists (review correction);
- uniform-limit parity vs an independent `025` first-separated-ancestor
  implementation, both radii — exact list equality, W/X empty;
- per-cell σ gate: `031a` cutoff-coverage contract (every pair with
  `r ≤ ρ_t σ_src` lands in U) holds with zero violations across uniform /
  heterogeneous (2-decade) / one-fat-core σ fields, with sticky demotion
  active, coverage still exact-once, and V classes still in-table;
- M2T/S2L scalar convergence to the analytic potential with
  self-contained Gumerov-form harmonics at `P = 4/8/12` (P=4 coverage
  invariant): relative errors ~1e-8 / ~1e-14 / ~1e-16, confirming the
  documented sign/conjugation rules numerically;
- capacity bounds respected in every configuration; `008d` bound
  monotonicity at `P = 4` and `P = 8`.

Counted cost-model headline (equal worst-cell population `max|A|`, the GPU
fat-cell metric): multiscale100 `q=3` **7.8×** less modeled work than the
uniform grid (`q=12`: **20.7×**), filament `q=3` **5.1×**; uniform-cube
non-regression is *exact row equality* (adaptive collapses to the uniform
grid's own depth).

Open items / notes for the approval agent:

- The E2-subsumption observation (per-cell gate naturally replaces the
  global adequacy mechanism) is logged in the overnight decision log; the
  held E2 disposition item remains open for the user per standing rule.
- W/X entries deeper than one level under occupancy pruning are a
  deliberate, proven-correct deviation from the classic complete-tree
  statement; capacity formulas do not assume the one-level property.
- LH vortex S2L channel content is defined structurally (production B2M
  strength-to-channel map on irregular harmonics) with the exact
  M2L∘P2M point-source oracle mandated as a `040` parity test; no
  theory-level numerical LH check was run (stdlib scope).

Scoped derivation row: `theory/`, `scripts/`, `data/` artifacts only; does not
reopen the Theory Phase hard gate and does not re-block any completed row.

## Clear-Context Re-Approval (2026-08-14 19:38 MDT)

**APPROVED** by a fresh clear-context re-approval agent (no prior campaign
context), after the sticky-demotion correction (`5f57db4`, following
`04f6c3c`).

Evidence checked: START_HERE protocol §6 + 038 row; this task file
including the Review Correction block; `theory/adaptive-radix-octree.md`
in full; `scripts/adaptive_octree_verify.jl` in full; all 5 CSVs +
`summary.txt`; decision-log entries 19:04/19:22/19:27/19:33 MDT. Re-ran
the script (`--threads=1`, ~10 s): ALL CHECKS PASS and the regenerated
artifacts are byte-identical (shasum before/after) to the committed data.
`git show --stat 5f57db4`: `MATRIX_OPERATOR_REFACTOR/` only, no `src/`.

Blocker verification (the point of this re-review):

- `build_lists` sticky semantics confirmed in code: the lineage bit `dem`
  propagates through every descent, `near = dem || isnear(...)` forces
  demoted lineages to terminate in U, and V/W/X emission requires
  `!near` — hence a never-demoted path, on which every descended ancestor
  pair satisfied the *geometric* near predicate. Invariant 2 is restored
  geometrically; §2.4's "025 tables reused unchanged" now holds
  unconditionally with the gate active, and §5.2/§5.3 give the matching
  proof (emission ⇒ the pair itself passed the gate, which also yields
  the §5.3 cutoff-coverage contrapositive).
- `check_v_classes` (separated offset + parent offset in `N_q` +
  Chebyshev reach ≤ 3/7 — exactly phase-table membership) now runs on
  GATED lists (`vok_g`, folded into `allpass`):
  `v_classes_ok_gated = true` on all 18 σ-gate rows, alongside
  `contract_bad = 0` and exact-once `= 0`. `wx_max_leveldiff_gated`
  recorded. The reviewer's measured out-of-table pairs (reach ≤ 11) are
  gone.
- Nothing else broke: exact-once passes on all 40 configurations (both
  radii, balanced/unbalanced) and on every gated configuration;
  uniform-limit parity passes with the newly added exact-once painting
  on the parity tree (first review's noted item, addressed); σ-contract
  zero violations; capacity bounds hold; P=4/P=8 bound monotonicity and
  M2T/S2L convergence (P=4/8/12) unchanged.
- Quantitative claims re-derived from the CSVs: sticky over-cost worst
  3.51× (filament q=3 heterogeneous, 993224/283296), one-fat-core
  1.25/1.36/2.18× at q=3, ≤2.9% on all q=12 heterogeneous rows —
  matching the task file and 19:33 decision-log entry (no remaining
  overstatement). Cost headlines re-verified at equal `max|A|`:
  7.78× (multiscale100 q=3), 20.7× (q=12), 5.12× (filament q=3);
  uniform-cube non-regression is exact row equality
  (`K=64 ↔ ℓ=2`, `K=16/32 ↔ ℓ=3`, both radii).
- Option (b) (table-free M2T/S2L re-admission) properly recorded in
  §5.4 + decision log as a user-ratification item with quantified
  stakes; the sticky-demotion adoption itself is flagged for user
  ratification. E2 disposition stays open per standing rule.

NOTED (non-blocking, fix on next touch of the theory file): §2.4's
displayed phase identity pairs `o = c_B - c_A` (A = target, i.e.
`S - T`) with `u = c_B mod 2` (source phase); `025` defines
`p = fld(u+o,2)` for `o = T - S` with source phase, so as literally
written the identity misses the parent offset when parent phases differ
(e.g. target phase 1, source phase 0). Set-level membership — the claim
that matters — is orientation-independent (`N_q = -N_q`, 025's symmetry
note) and the script verifies parent offsets directly from the tree, so
no proof or datum is affected; either flip the offset to `c_A - c_B` or
use the target phase. Also noted: the script's `balance!` is the
O(leaves²) reference form — fine for validation; `039` should implement
the §1.4 Sundar-style sweep as the theory already specifies.

## Motivation

The uniform radix grid is occupancy-compacted (compute scales with occupied
cells) but rigidly uniform: one global leaf width `h`, one depth `ell`, no
leaf-population bound. Consequences established by the 2026-08-06 sparsity
review:

- One locally large σ forces a globally shallow tree via the geometric gate
  `g_min·h_leaf > ρ_t·σ_max` (`translate_batched_resident.jl` gate), even
  where the field is fine.
- A single fat cell costs `O(K²)` inside one warp in the nearfield pair
  kernel and serializes B2M in one thread; clustering hurts through load
  imbalance before it hurts through cell counts.
- The legacy octree solves both adaptively but is host-only and per-pair
  MAC-driven, incompatible with translation-invariant operator tables.

The enabling fact: the `025` level-scaling law (one operator table serves
every level) plus per-`(level, offset)` class batching over occupied nodes is
exactly the structure a 2:1-balanced adaptive octree needs. With 2:1 balance,
same-level M2L (the V-list) remains a finite translation-invariant
offset-class set — identical batching, identical level-scaled tables. This is
the PVFMM / ExaFMM-T design, proven on GPUs at scale.

## Objective

Derive the complete theory for an adaptive Morton octree on the radix path
that preserves the batched, translation-invariant, device-resident operator
machinery:

1. **Tree definition and construction.** Leaves are the deepest cells with
   population ≤ a split threshold `K_max` (and depth ≤ `ell_max`); top-down
   split on Morton key prefixes over the sorted body stream; 2:1 balance
   sweep. All steps must be expressible as sort/scan/compact primitives
   (the device feasibility argument, not device code).
2. **Interaction lists.** U (adjacent leaves, possibly different levels →
   direct), V (same-level well-separated children of neighbors → M2L by
   offset class, reusing the `025` stencil and scaling law unchanged),
   W (coarse leaf vs finer non-adjacent descendants → M2T), X (dual of W →
   S2L). Precise definitions on Morton keys, plus generation algorithms in
   flag/scan/compact form.
3. **Exact-once coverage proof.** Every body pair is covered exactly once by
   the union of U/V/W/X plus self-interaction, for any 2:1-balanced leaf set,
   at both supported near radii (`|o|² ≤ 12` and classic `|o|² ≤ 3`). Verify
   computationally on test trees including adversarial (highly clustered)
   distributions.
4. **M2T and S2L operators.** Formulate in the compressed complex basis
   consistent with `005`/`007` conventions and the Lamb-Helmholtz channel
   (`003`, `008h`): M2T evaluates a multipole expansion directly at target
   bodies; S2L accumulates sources directly into a local expansion. Derive
   error bounds consistent with the constant-`P` stencil error model (`008d`,
   `025`) so W/X interactions respect the same accuracy target as V.
5. **Per-cell geometry gate.** Replace the global `σ_max` gate with a
   per-leaf admissibility rule (leaf width vs the σ of the bodies it holds
   and its U-list partners), and prove it implies the phase accuracy gate
   under the regularized-kernel contract from `031a`.
6. **Cost and capacity model.** Expected list sizes and work vs `K_max`,
   depth ceiling, and occupancy statistics for the two phase test cases plus
   a synthetic multi-scale case (e.g. cube + embedded dense cluster at
   10–100× local density); capacity formulas suitable for the
   `RadixFMMCache` no-realloc contract (max leaves, max list lengths as
   functions of `max_n_bodies`, `K_max`, `ell_max`).
7. **Refresh/rebuild policy.** When body motion invalidates the leaf set;
   what an in-place refresh can reuse (cf. the occupancy-epoch window cache)
   vs a full rebuild; interaction with `recenter!`.

## Deliverables

- `theory/adaptive-radix-octree.md` covering items 1–7 with derivations.
- A stdlib-only validation script under `scripts/` exercising the exact-once
  proof and the per-cell gate on the test distributions.
- Generated evidence tables under `data/adaptive_octree/`.

## Constraints

- Reuse approved conventions: `025` stencil and scaling law, `007` buffer
  layout, `008h` χ-at-`P+1` rule, `031a` regularized-nearfield contract.
- The V-list path must require **no new operator tables** beyond the existing
  level-scaled set; if any deviation is unavoidable, quantify it and stop for
  user discussion.
- No production `src/` changes in this row.

## Acceptance

Approved when the exact-once proof is verified computationally on clustered
and uniform test trees, the M2T/S2L error bounds are consistent with the
constant-`P` model, the cost model quantifies the expected win on the
multi-scale case (and the expected non-regression on the cube/wake cases),
and the capacity formulas are explicit enough for `039`–`041` to implement
against without re-derivation.
