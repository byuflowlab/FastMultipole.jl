# Overnight Campaign Decision Log — 2026-08-14 (7pm start)

Coordination/reporting doc for the user-authorized overnight campaign
(brief: `overnight-brief-2026-08-14.md`). Rows 038 → 042 of
`START_HERE.md`, each with a background lead agent followed by a fresh
clear-context approval agent. Timestamped entries: decisions, blockers,
verdicts, and anything the user must ratify.

Standing invariants in force all night: NO production default changes;
no lab-notebook writes; velocity RMS ≤ 1e-3 gate; P=4 test coverage;
capacity / zero-recurring-allocation / counter / graph-capture contracts;
pre-registration committed BEFORE job submission; same-job anchors;
critical-path pricing. Held user decisions stay held (037g staging, rotor
auto-depth, E2→038 acknowledgment, notebook entry).

---

## 2026-08-14 19:00 MDT — Campaign start

- Orchestrator read the brief and `START_HERE.md` in full.
- Confirmed row status: 037a–037f all Done + Approved (037c closed by
  pointer per user direction 2026-08-14). **038 is the first unblocked
  row** — matches the brief's expectation.
- Plan of record: 038 (theory) → 039 (host construction) → 040 (host
  lifecycle) → 041 (CUDA) → 041a (figures) → 042 (Milestone Review), each
  gated by a fresh clear-context approval agent before the next row
  starts. 042 follows the Milestone Review protocol (reads
  `../MATRIX_OPERATOR_REFACTOR.md`).
- 038 entry-gate note carried from the brief and the START_HERE 038 row:
  multi-scale-density evidence must bind (037b's rotor-wake case supplies
  or refutes it) or an explicit user waiver is required. The lead agent's
  first job is to adjudicate this gate from the recorded 037b/035/037
  evidence; if the evidence refutes and no waiver exists, that is a
  user-only decision → log it, park 038, and look for independent work
  (per brief rule 7).
- E2 subsumption rule in force (brief rule 5): if 038's per-cell σ gate
  naturally subsumes the E2 mechanism, proceed in-scope and log it; the
  held E2 disposition item stays open regardless.
- Launching the 038 lead agent (background) next.

## 2026-08-14 19:04 MDT — 038 entry gate: MET (lead agent adjudication)

- Evidence examined: `037b` Final Report §6 and the "038 multi-scale-density
  evidence" section (task file lines ~285–298, 392–395), backed by
  `data/rotor_wake/rotor_case_stats.csv` and screen job 13170509 of record.
- Findings: multi-scale density is a MEASURED BINDING COST on the realistic
  rotor wake — uniform-grid auto-selected geometry costs 7.16x at n=1e6
  (238.7 → 33.3 ms at pinned ℓ8) and 1.81x at n=1e5; occupancy contrast
  (max/mean bodies per occupied cell 5–7× at every level vs wake ≤1.9×;
  top-1% densest cells hold ~6% of bodies) is the mechanism; the uniform
  path's ℓ≤8 cap plausibly still binds at 1e6 (nearfield ~70% of the ℓ8
  eval). `037b` itself recorded "038 entry gate: MET".
- Nuance logged for honesty: the σ_max *geometry-gate* mechanism named in
  the 038 motivation does NOT bind on the rotor (thin young tip cores keep
  σ_max small); the binding mechanism is density contrast (fat-cell
  nearfield concentration + depth mis-selection). The 038 task gate asks
  for "multi-scale density ... a binding cost that the uniform-depth grid
  cannot serve" — that is squarely met. The per-cell σ gate deliverable
  remains in scope as a correctness/generality requirement
  (`CoreSpreading`-grown σ), not as the measured win mechanism; the
  derivation will record this framing.
- VERDICT: gate MET on evidence; no waiver needed. Proceeding to derivation.

## 2026-08-14 19:22 MDT — 038 derivation complete (lead agent)

Decisions of record made during the derivation (no user input available;
all in-scope per the brief):

1. **Mixed-level near predicate defined on the finer cell's lattice**
   (per-axis clamp of the finer coordinate to the coarser cell's tile
   interval, then the `N_q` test). Chosen because (a) it reduces exactly
   to the 025 same-level rule, (b) it is integer-exact and O(1), and
   (c) "far" then means every body in the coarse leaf sits at same-level
   `025` far geometry relative to the finer cell — which makes the
   M2T/S2L error bounds *inherit* the `008d` constant-P V-budget with
   margin instead of needing a new error model.
2. **Exact-once proof is predicate-independent** (partition-invariant
   dual-tree recursion). This is deliberate: it makes the per-cell σ
   gate implementable as pure demotion (treat an inadmissible far pair
   as near) with zero impact on coverage — proven, not assumed.
3. **Per-cell σ gate = source-side per-node `σ_max` folded into the far
   predicate + demotion, with an optional construction-time split veto**
   (veto population splits, never balance splits). This *eliminates* the
   global `g_min·h_leaf > ρ_t·σ_max` throw: any σ field is admissible,
   worst case degrading locally to regularized-everywhere direct.
   **E2 subsumption note (per brief rule 5):** this per-cell
   admissibility mechanism appears to naturally subsume the E2
   mechanism; proceeding was authorized, and the held E2 disposition
   item remains OPEN for the user — nothing here closes it.
4. **Honest weakening of the classic W/X one-level claim:** with
   occupancy-pruned children (our radix ethos), W/X partners can be >1
   level finer even at q=3 with 2:1 balance (empty-sibling pruning
   breaks the textbook complete-tree argument). Proven for the
   complete-tree limit, measured otherwise (max level diff 3 across all
   test cases, mass overwhelmingly at one level); capacity formulas do
   not assume the one-level property. Balance is retained as a
   performance/regularity device; correctness never depends on it.
5. **S2L/M2T defined by production-mirror rules with exact
   M2L-composition oracles** (S2L = M2L∘P2M of a zero-size source cell —
   finite point multipole, hence exact; M2T = trivial-evaluation of an
   M2L to a zero-size target). Scalar sign/conjugation rule
   (`L_n^m += -(-1)^(n+m) conj(S_n^m(x_s - c_A))`) confirmed numerically
   (P=4/8/12 → ~1e-8/1e-14/1e-16 relative). LH channel content defined
   structurally per 003/008e/008h (χ at P+1 incl. neighbor row); the LH
   numerical parity check is mandated as a 040 test (out of stdlib
   scope here) — flagged in the task file for the approval agent.
6. **V-list reuses the `025` tables unchanged** — verified structurally
   (every emitted V pair proven inside the phase-table class set,
   asserted computationally on every configuration). The task-file "no
   new operator tables" constraint is met; no user stop needed.

Verification: `scripts/adaptive_octree_verify.jl` (stdlib-only, 1 thread,
~40 s local) — ALL CHECKS PASS; byte-identical rerun confirmed. Evidence
in `data/adaptive_octree/` (5 CSVs + summary). Counted cost model at equal
worst-cell population: 7.8×/20.7× (multiscale100 q=3/q=12), 5.1×
(filament q=3) less modeled work than the uniform grid; exact row-equality
non-regression on the uniform cube.

Bookkeeping: 038 task file updated (gate record + completion notes),
START_HERE row 038 marked Done (Approved left blank), committing on
`matrix-ops` next. Row is ready for clear-context approval.

## 2026-08-14 19:27 MDT — 038 clear-context review: CHANGES REQUIRED

Reviewer: clear-context approval agent (no prior campaign context).
Verdict: **CHANGES REQUIRED — 038 not approved; Approved checkbox left
blank.**

Evidence checked: START_HERE (038 row + phase preamble + protocol §6),
the 038 task file (incl. entry-gate record — adjudication accepted:
037b density-contrast evidence is on point, the σ-gate honesty nuance is
properly recorded), `theory/adaptive-radix-octree.md` in full,
`scripts/adaptive_octree_verify.jl` in full, all 5 CSVs + summary.
Re-ran the script (`--threads=1`): ALL CHECKS PASS, byte-identical rerun
confirmed. Commit `04f6c3c` surface verified: `MATRIX_OPERATOR_REFACTOR/`
only, no production `src/`. Headline cost ratios independently recomputed
from `cost_model_counts.csv`: 8.68e7/1.116e7 = 7.8×, 3.578e8/1.731e7 =
20.7×, 2.946e7/5.752e6 = 5.1×, at equal `max|A|` as stated; uniform-cube
non-regression row equality confirmed. Exact-once proof (§3.2), mixed-level
finer-lattice near predicate (§2.1, incl. the every-tile-separated
property §4.3 relies on), σ-gate correctness theorem (§5.3), `008h` χ at
P+1 usage (§4.1–4.3), and the c>2 exclusion (§4.4) all check out.

**REQUIRED CHANGE (correctness of the `025` table-reuse claim under the
§5 σ gate).** §2.4 claims every emitted V offset lies inside the finite
`025` phase-table set ("no new operator tables"), proven via Invariant 2
(parent nearness). With the §5 demotion gate active, "near" includes
demoted-but-geometrically-separated pairs, and §5.3's "descent helps"
re-admission then emits V pairs whose parents are geometrically separated
— outside the `V_push` phase-table set. Reviewer measured it on the
script's own σ-gate configurations (K_max=32, ρ_t=4.789): e.g. uniform
q=3 one_fat: 2712 V pairs beyond Chebyshev reach 3 (worst reach 7), 2971
with separated parents; filament q=3 one_fat: worst reach 11; even q=12
one_fat cases emit 932–963 V pairs with separated parents (i.e. not in
the 1253-entry push table) while within reach 7. The script masked this
by running `check_v_classes` only on ungated lists. Exact-once coverage
and the σ contract are unaffected (predicate-independent), and the error
bound at larger ‖o‖ is smaller, so this is a table-membership/constraint
violation, not an accuracy bug — but the task constraint says any
deviation from "no new tables" must be quantified and stopped for user
discussion. Fix options for the owner (pick + prove + re-verify):
(a) sticky demotion — a demoted pair descends to U only (matches the
`build_lists` docstring as written; weakens §5.3 "descent helps" and its
cost claim); (b) re-admit demoted-descendant far pairs only as W/X-style
body-mediated M2T/S2L (table-free; §4.3 bound already covers the
geometry); (c) allow out-of-table V classes and stop for user discussion
per the constraint. Required in all cases: make §2.4/§5.3 consistent
with the chosen rule, run the V-class/phase-table membership check on
GATED lists in the script, regenerate data, and correct the overstated
"every emitted V pair proven inside the phase-table class set" claims
(this log's earlier entry, task-file completion notes).

Noted (non-blocking): uniform-limit parity compares `Set(L.V)` — a
duplicate V emission would pass the set equality (exact-once painting is
not run on the parity tree); consider painting there too when touching
the script.

## 2026-08-14 19:33 MDT — 038 review correction: sticky demotion (lead agent)

Correction to the 19:04/19:22 entries above, fixing the 19:27 review
blocker. The reviewer was right: the original demote-and-descend gate
allowed descendants of a demoted (geometrically separated) pair to
re-admit to V, emitting V pairs with separated parents — outside the 025
phase-table set — so the 19:22 entry's claim 6 ("every emitted V pair
proven inside the phase-table class set") was OVERSTATED: it had been
verified on ungated lists only. Coverage and accuracy were never
affected.

**Decision (overnight, user unavailable): option (a), sticky demotion.**
A demoted pair's entire descendant pair set terminates in U (one lineage
bit; no V/W/X below a demotion). Rationale: (i) restores Invariant 2
*geometrically*, making the no-new-tables claim unconditional — the
cleanest theory; (ii) matches the build_lists docstring semantics as
originally written; (iii) simplest proofs and 039–041 contract (no
internal-node M2T/S2L semantics, no extra capacity term); (iv) the
over-cost is local to fat-σ neighborhoods that the 031a contract forces
(mostly) direct anyway, and the split veto bounds it.

**Item for user ratification tomorrow — recorded alternative (option b):**
re-admit demoted-descendant far pairs as table-free body-mediated M2T/S2L
entries (the §4.3 bound already covers that geometry). Recovers far-field
acceleration for small-σ sources inside demoted subtrees at the price of
internal-node-partner M2T/S2L (per-body cost ∝ subtree size), a larger
W/X capacity term, and a more intricate lineage rule. Measured stakes
(sigma_gate_contract.csv, gated vs ungated direct pairs, n=3000): worst
case 3.5× more direct pairs under sticky demotion (q=3, 2-decade
heterogeneous σ; filament 993k vs 283k); one-fat-core 1.2–2.2× at q=3;
≤3% on all q=12 heterogeneous rows. If real σ fields look like the
2-decade case at q=3, option (b) is worth implementing in 039/040; it
fits inside this row's theorems (recorded in theory §5.4) and needs no
re-derivation.

Changes made: theory §2.4/§2.7/§3.2/§5.2/§5.3/§5.4/§8/§9 reconciled
("descent helps" replaced by the cost-locality argument; option (b)
recorded in §5.4); build_lists made sticky; V-class/phase-table + W/X
checks now run on GATED lists (all pass); uniform-limit parity now also
runs exact-once painting on the parity tree (reviewer's non-blocking
note); data regenerated; ALL CHECKS PASS; byte-identical rerun
re-confirmed; task file carries a Review Correction block. Committing as
`038: review corrections` next. Row ready for fresh re-approval.

## 2026-08-14 19:38 MDT — 038 clear-context re-approval: APPROVED

Reviewer: fresh clear-context re-approval agent (no prior campaign
context). Verdict: **APPROVED** — 038 Approved checkbox marked in
START_HERE; committed as `038: clear-context re-approval`.

Blocker verified fixed: `build_lists` sticky semantics confirmed in code
(lineage bit propagates every descent; `near = dem || isnear`; V/W/X
require `!near`, hence never-demoted paths with geometric parent
nearness — Invariant 2 restored); `check_v_classes` = exact phase-table
membership (separated + parent in N_q + Chebyshev reach) now runs on
GATED lists and passes on all 18 σ-gate rows (`v_classes_ok_gated=true`,
`contract_bad=0`, exact-once 0). Script re-run (`--threads=1`, ~10 s):
ALL CHECKS PASS, artifacts byte-identical to committed (shasum). Nothing
else broke: 40-config exact-once, uniform-limit parity (now with
exact-once painting — first review's noted item addressed), σ contract,
capacity bounds, P=4/8 monotonicity, M2T/S2L P=4/8/12 all pass.
Quantitative claims re-derived from CSVs and match the task file / 19:33
entry exactly (3.51× worst sticky over-cost; ≤2.9% q=12; 7.78×/20.7×/
5.12× cost headlines; exact uniform non-regression). `5f57db4` surface:
`MATRIX_OPERATOR_REFACTOR/` only. Option (b) and the sticky-demotion
adoption remain user-ratification items; E2 disposition stays open.

NOTED (non-blocking, recorded in the task-file approval block): §2.4's
displayed identity `p = fld(u+o,2)` pairs `o = c_B - c_A` (S−T) with the
source phase, but 025 defines it for `o = T − S`; as literally written it
misses the parent offset when parent phases differ. Set-level membership
(the actual claim) is orientation-independent and the script checks true
tree parents, so nothing downstream is affected — fix the orientation on
the next touch of the theory file (e.g. during 039). Also noted: script
`balance!` is the O(leaves²) reference form; `039` implements the §1.4
Sundar-style sweep per the theory.
