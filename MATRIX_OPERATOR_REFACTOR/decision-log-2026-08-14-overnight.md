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
