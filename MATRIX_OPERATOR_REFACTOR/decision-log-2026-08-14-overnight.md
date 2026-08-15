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

## 2026-08-14 19:40 MDT — 039 lead agent start + theory orientation fix

- 039 lead agent started (row 038 Done+Approved as of 19:38 MDT). Read
  START_HERE (protocol + placement rules), this log, the 039 task file, the
  full `theory/adaptive-radix-octree.md`, and the full reference oracle
  `scripts/adaptive_octree_verify.jl`. Production-surface mapping delegated
  to an Explore subagent.
- Carry-over item DONE: fixed the theory §2.4 orientation slip flagged in
  the 038 re-approval. The displayed identity now pairs `o = c_A - c_B`
  (= T − S, the 025 convention) with the source phase `u = c_B mod 2`,
  with a derivation line (`c_B = 2p_B + u ⇒ p_A − p_B = fld(u+o,2)`) and
  an explicit note that set-level membership was always
  orientation-independent (N_q = −N_q). No downstream claim changes.

## 2026-08-14 20:10 MDT — 039 implementation decisions (lead agent)

Production surfaces mapped (Explore subagent), theory + oracle read in full.
Implementation decisions of record (overnight, user unavailable; in-scope):

1. **Placement & shape.** New structs `AdaptiveTreePolicy` /
   `AdaptiveRadixTree{TF}` / `AdaptiveInteractionLists` in
   `src/containers.jl`; construction/balance/σ-sweep in
   `src/tree_batched.jl`; DTR list build in
   `src/interaction_list_batched.jl` (per START_HERE placement rules).
   Opt-in on `RadixFMMCache` via a new `adaptive=` keyword (3 new
   trailing `Any` fields, `nothing` by default; device ctor passes
   nothings; `recenter!` forwards the policy). With `adaptive === nothing`
   (default) behavior is bit-identical to before; when set,
   `update_radix_state!` refreshes the adaptive tree+lists AFTER the
   uniform structures — the uniform lifecycle still executes unchanged
   (040 wires consumption). NO production default changed.
2. **V-list format parity.** V stream uses the exact production arrays
   (`route_levels/route_offsets/route_targets/route_sources/route_class`)
   with the production class numbering
   `(L - first_m2l_level)*noffsets + k` over
   `RigidHierarchicalTables(q).push_offsets` and
   `_hierarchical_class_metadata` (first_m2l_level=2, cubic), PLUS a CSR
   `class_starts` from an in-place counting sort (class-partitioned,
   level-major then canonical offset order — the layout the windowed
   resident M2L strategies consume). No new operator tables (theory §2.4).
3. **U/W/X endpoints are flat adaptive node indices** (not the uniform
   path's leaf-cell indices): adaptive leaves live at multiple levels, so
   the `direct_sources = node - leaf_base` convention cannot apply; 040
   consumes node body ranges directly. Logged as a deliberate format
   decision.
4. **DFS pair stack replaces the §2.7 frontier** on the host: capacity
   63·(2·ell_max+1)+1 (≈2.7k at ell_max=21) instead of a cap_V-sized
   frontier — an improvement on the theory's capacity table; the
   flag/scan/compact frontier form remains the 041 device shape.
5. **Balance sweep**: §1.4 Sundar-style implemented with per-round sorted
   leaf-interval-start tables + binary-search matching, deepest-first,
   fixed-point with a guard; each leaf emits its 8 touching parent-level
   cells (exactly 2 per axis) instead of 26 neighbors — same set, fewer
   lookups. O(leaves·log leaves) per round, replacing the oracle's
   O(leaves²) reference form as mandated.
6. **Constant q only** on the adaptive path (both q=3 and q=12 supported
   and tested); the production per-level radius schedule is a recorded
   deferral (theory treats constant q; scheduled radii would need
   per-level transition tables in the DTR near predicate).
7. **Capacities**: theory §6.4 formulas with hard occupancy caps
   (node ≤ (ell_max+1)·n+1; U/W/X sized per-leaf with leaf_cap ≤ maxn;
   V = push_max·min(node_cap, 4·maxn)) + per-policy explicit overrides;
   overflow is a loud AssertionError, never a realloc. The unmodified
   §6.4 products (e.g. push_max·node_cap) are memory-infeasible at
   n=1e6 — measured-to-capacity ratios recorded for 040 tightening.
8. **Sticky-demotion invariant enforced at emission**: every V pair is
   checked against the 025 phase-table membership
   (`level_class_of[phase(source), k, L+1] != 0`) and THROWS on
   violation — the 038 review's gated-list check is now a runtime
   invariant, not just a test.
9. **Cubic-only cache integration** (rectangular ell_axes throws with a
   deferral message); host-only (device=true + adaptive throws; 041).

**Deviation from theory (quantified): §5.4 split veto defaults OFF.**
The literal §5.4 rule (veto population split of a cell when
g_min·Δ_{ℓ+1} < ρ_t·σ_max(cell)) keys on the cell's own subtree σ_max, so
one fat-σ body vetoes every ancestor split up to the root: measured on the
one-fat-core multiscale field (n=1500, σ=3e-4 + one 0.15, ρ_t=4.789):
q=3 veto ON → 1 leaf, all-direct (U=1 self pair; the global-gate pathology
reborn as a performance collapse); q=12 (g_min=√5) veto ON → 411 vs 424
leaves (mild local coarsening, as intended). Veto OFF: sticky demotion
alone gives the theory's cost locality (dem=116/183 at q=3/12, exact-once
and 031a contract PASS). Decision: `split_veto=false` default, veto
retained as an option for spatially smooth σ fields; documented in the
policy docstring + theory §5.4 implementation note. **User ratification
item**: veto default and whether §5.4 should be re-derived with a
locality-limited veto (e.g. σ quantile or population-bounded).

Verified so far (local, 1 thread): exact-once brute force on uniform +
multiscale100 (balanced/unbalanced, q=3/12, σ-gated and ungated) all 0
bad; 031a cutoff contract 0 bad; zero-allocation standalone refresh
(update=0 B, lists=0 B); cache-integrated step adds ~128 B constant
dynamic-dispatch overhead on top of a PRE-EXISTING ~30 KB/step
update_radix_state! baseline (measured without adaptive; not introduced
by 039; repo's own step-allocation gates are <512 KB bounds).

## 2026-08-14 20:18 MDT — 039 tests green + cost-measurement pre-registration

- New test file `test/adaptive_octree_test.jl` wired into runtests.jl
  (after `radix_trimming_test.jl`). All pass locally (1 thread):
  construction invariants 2879, balance 6, exact-once brute force 57,790
  (uniform/multiscale/filament × 2 seeds × K_max 8/32 × q 3/12, balanced
  + unbalanced), σ gate 26 (sticky demotion, 031a contract, gated V-class
  membership), uniform-limit parity 48 (q ∈ {3,5,12}, ell ∈ {2,3}, exact
  production route-set + class-id + direct-pair equality at P=4),
  zero-allocation refresh 6 (0 bytes standalone tree+lists), cache opt-in
  18 (bit-identical uniform results with the policy armed, guards,
  recenter! preservation).
- Regression: the six related host radix test files re-run in one session
  — 114,015 pass, 0 fail.
- **Pre-registration (committed before submission):**
  `scripts/fm039_construction_cost.jl` measures adaptive vs uniform
  construction/refresh cost per the protocol in its header: cases
  unitcube / wake (033 helical wake cylinder positions) / multiscale100;
  n ∈ {1e5, 1e6}; adaptive K_max ∈ {32,64,128} at ell_max=10, q=5,
  balance on, veto off; uniform baseline RadixFMMCache default policy at
  ell ∈ {5,6}; cold = alloc+first build, warm = median of 5 in-place
  refreshes; same-job anchors, single thread, CPU node; counts + measured
  /capacity ratios recorded. Output CSV
  `data/fm039_construction_cost.csv` (cluster run is the measurement of
  record; a local n=2e4 smoke validated the script only).

## 2026-08-14 20:20 MDT — 039 cost job submitted

- Implementation + tests + pre-registration committed as `fd58aea` BEFORE
  submission (protocol honored).
- Cluster job **13178905** (`fm039cost`, CPU node, 1 Julia thread,
  OPENBLAS=1, 64G, 3h wall) submitted from `~/FastMultipole-039` (rsync
  snapshot of the fd58aea working tree; Julia 1.12.6 module; package
  loads verified on login node). Output `~/fm039cost_13178905.out`;
  CSV of record `~/FastMultipole-039/MATRIX_OPERATOR_REFACTOR/data/
  fm039_construction_cost.csv` — to be pulled back and committed.
- sacct will be checked on every resume until a terminal state is
  verified (monitors have missed terminal states before).

## 2026-08-14 20:32 MDT — 039 DONE (lead agent completion)

- Cluster job **13178905** terminal: COMPLETED 00:05:20 ExitCode 0:0
  (sacct re-verified by the lead agent per the standing rule). CSV of
  record pulled to `data/fm039_construction_cost.csv` and committed.
- Measurement headlines (n=1e6, warm refresh = tree+lists median-of-5,
  same-job anchors): uniform-limit sanity — unitcube adaptive K=64
  reproduces uniform ell=5 structure to 3 digits (37,363 vs 37,390 nodes,
  identical u_pairs/V) at 2.3x refresh cost (1.22 vs 0.53 s). Bounded
  worst cell where uniform cannot: wake adaptive K=128 popmax 128 vs 1231
  (ell=5) / 182 (ell=6), direct body-pair work 28.8x / 4.4x lower;
  multiscale100 K=128 popmax 128 vs 2442/346, u_pairs 15.9x / 2.6x lower,
  refresh 1.24 s vs 2.22 s (ell=6). The 038 counted mechanism reproduced
  at production scale on the host. Worst adaptive refresh 4.35 s
  (unitcube K=32, over-fine); cold construction <= 5.3 s. Capacity ratios
  peak node 0.83 / U 0.60 / V 0.91 (wake K=128 — V margin thin; 040
  tightening note).
- Task file updated with completion notes + results; START_HERE 039 row
  marked **Done** (Approved left blank). Committing as `039: close-out`.
- Row 039 is ready for clear-context approval. User-ratification items
  carried: split-veto default OFF (quantified deviation), 038 option (b)
  M2T/S2L re-admission (not implemented), E2 disposition (open).

## 2026-08-14 20:37 MDT — 039 clear-context approval: APPROVED

Reviewer: clear-context approval agent (no prior campaign context).
Verdict: **APPROVED** — 039 Approved checkbox marked in START_HERE;
committed as `039: clear-context approval`.

Evidence: read START_HERE protocol + 039 row, the 039 task file, the full
theory doc, and the complete `fd58aea`/`44a803f` diffs (src, test, script,
CSV). Re-ran `test/adaptive_octree_test.jl` locally (1 thread): 60,773
pass / 0 fail, per-testset counts matching the completion notes exactly.
Hand-verified: sticky-demotion semantics (lineage bit, gate on !near only,
V/W/X on never-demoted paths — theory §5.2/§2.4 as re-approved), the
emission-time 025 phase-table membership throw (T−S offset + source
phase), integer-exact source-side σ-gap test, the 8-cell balance emission
set, DFS/pair stack capacity sufficiency, CSR counting-sort indexing, and
σ-from-buffers ordinal-order consistency. Uniform-limit parity test is
genuine (production `build_hierarchical_routes_window!` reference, tuple
equality incl. class ids, duplicate-exclusion, P=4). No production default
changes (bit-identical uniform fmm! asserted; guards throw). Measurement
of record: pre-registration in `fd58aea` (20:18) precedes job 13178905
submission (20:20); same-job anchors; every headline number re-derived
from the CSV and exact (28.8x/4.4x wake, 15.9x/2.6x multiscale, 2.3x
uniform-limit refresh cost, capacity peaks 0.83/0.60/0.91). Deviations
accepted as logged (veto default OFF, capped capacities, host DFS stack);
option (b) correctly not implemented; user-ratification items unchanged
(veto default, option (b), E2).

NOTED (non-blocking, recorded in the task-file approval block): (1) theory
§9 still says "split-veto default-on" for 040 — stale vs the §5.4 note;
fix on next theory touch. (2) the K_max population test's
`|| n_balance_splits > 0` escape is weaker than needed. (3) W/X duality
is correctly asserted only ungated (source-side σ is asymmetric).

## 2026-08-14 20:39 MDT — 040 lead agent start

- 040 lead agent started (039 Done+Approved 20:37). Read START_HERE
  (protocol + 040 row), this log, the 040 task file, the full theory doc
  `theory/adaptive-radix-octree.md`, and the 039 surfaces in full
  (containers adaptive types, `tree_batched.jl` adaptive section,
  `interaction_list_batched.jl` DTR + CSR, `test/adaptive_octree_test.jl`).
  Host resident-lifecycle surface mapping delegated to an Explore subagent.
- Binding constraints re-confirmed from the brief: sticky demotion is the
  approved gate; option (b) NOT implemented; split veto stays default OFF;
  theory §9 stale "split-veto default-on" to be fixed on this touch; LH
  vortex S2L numerical parity (M2L∘P2M oracle) is due in this row; V-list
  M2L consumes the existing resident strategies/tables UNCHANGED; host-only;
  no production default changes; zero per-step allocation on the adaptive
  path; P=4+P=8, Float64+Float32, velocity RMS ≤ 1e-3 on cube/wake/
  multi-scale; uniform-limit lifecycle parity.
- Carry-over item DONE: theory §9 stale "split-veto default-on" corrected to
  default OFF (039 deviation, pending ratification), with an explicit
  correction note on the 040 touch.

## 2026-08-14 20:55 MDT — 040 design decisions (lead agent)

Full surface map complete (Explore subagent + direct reads of the resident
pipeline). Design of record, chosen for maximal reuse of the UNCHANGED
resident strategies/operator tables and minimal invasion:

1. **Adaptive lifecycle = a second capacity-sized `DeviceResidentRadixState`**
   (host arrays) assembled from the 039 tree+lists, stored in a NEW
   `RadixFMMCache.adaptive_state::Any` field (struct
   `AdaptiveResidentLifecycle` in containers.jl). Its `grid` is a genuine
   `DeviceRadixGrid` mirror of the adaptive node table (level-major layout
   matches by 039 construction): `node_centers`/`node_keys`/`perm`/
   `body_system`/`body_index` alias the tree; Int mirrors are refreshed for
   `node_levels`/`parent_index` (tree stores Int32); leaves are presented as
   "cells" (`cell_ranges`/`cell_centers`/`leaf_to_node` over `leaf_index`).
   The adaptive path packs its OWN sorted `source_bodies`/`output` (the
   adaptive full-depth sort differs from the uniform ℓ-depth sort; sort
   unification stays an open 039 item, cost priced in 041a).
2. **Stage reuse.** B2M = `_host_b2m_kernel!`/`_host_b2m_vortex_kernel!`
   verbatim over adaptive leaves. M2M/L2L = the existing
   `_resident_capacity_group` + `_refresh_resident_stage_groups!` +
   `_resident_stage_group_apply!` machinery verbatim (adaptive parent-child
   radius at child level L is the same `sqrt(3)·h0/2^L` the per-level groups
   bake in; every non-root node has a parent; empty levels have count 0).
   ONE new launcher `_launch_adaptive_m2m!` loops the groups WITHOUT the
   uniform `_zero_resident_nonleaf_multipoles!` prefix zeroing — the uniform
   prefix rule (nonleaf = first n_nodes−n_cells columns) would zero coarse
   LEAVES on an adaptive tree; B2M's full-buffer fill makes zeroing
   unnecessary. L2L reuses `_launch_resident_l2l!` unchanged.
3. **V-list M2L consumes the existing plans UNCHANGED** via a window driver
   that mirrors `_launch_hierarchical_resident_m2l!`: the 039 CSR route
   stream is copied window-by-window into `state.route_*` + `plan.route_class`
   (global class ids match the plan's `effective_offsets` ordering exactly —
   both are `_hierarchical_class_metadata` order), then the UNCHANGED
   `_refresh_dense_m2l_routes!` / `_refresh_precomputed_y_m2l_routes!` /
   `_launch_hierarchical_concat_window!` launchers run with
   `clear_locals=false`. No new operator tables; windows may split a class
   (each window is an independent accumulate). Window capacity
   `min(v_capacity, 32768)` bounds the concat slabs.
4. **M2T/S2L kernels** (translate_batched.jl per the task file): per-pair
   irregular-harmonic evaluation using the validated legacy
   `irregular_harmonics!` into a preallocated scratch (host single-threaded),
   with the legacy `evaluate_multipole` (M2T: φ deg-shift + χ same-degree,
   `008h` χ at P+1) and `test/bodytolocal.jl` `body_to_local_point!` (S2L:
   scalar `L += −(−1)^{n+m} q conj(S)`, vortex verbatim, χ filled through
   P_active) formulas ported to the flat resident layout. Sign audit:
   resident scalar output = −legacy and resident scalar M/L = −legacy, so the
   legacy formulas port VERBATIM for u, gradient, and hessian; exact
   M2L-composition oracles (theory §4) gate this in tests. M2T includes the
   n = P term of the gradient (the legacy `n < P` gate is a legacy truncation
   quirk; the M2L-composition oracle includes n = P, so M2T must too).
5. **U-direct** = existing `_host_direct_pairs_functor_kernel!` (inside the
   reused `_launch_host_l2b!`) over adaptive leaf slots: U node endpoints are
   mapped through a refreshed `leaf_slot_of` to leaf-cell indices.
6. **fmm! wiring**: with `adaptive=` armed, the host branch runs the adaptive
   lifecycle INSTEAD of the uniform one (039's "uniform bit-identical with
   policy armed" was an explicit stopgap — "rows 040+ wire consumption").
   `finalize_radix_output!` works verbatim on the adaptive state (it carries
   the tree's perm metadata). The 039 opt-in test's bit-identity assertion is
   updated to an accuracy assertion; logged as the intended 040 semantics
   change of the OPT-IN path only. Production defaults untouched.
7. **Gate policy**: with adaptive armed, the global
   `_direct_kernel_geometry_gate!` throw is SKIPPED (the §5 per-cell sticky
   demotion is the contract enforcer). New construction-time guards:
   regularized kernels on an adaptive cache REQUIRE the per-cell gate armed
   (`rho_t >= _gate_reach_rho(kernel)`, `sigma_row > 0`); `TwoPassVortex`/
   `PartitionedVortex` + adaptive throw (deferral — the twopass deficit sweep
   enumerates the uniform lattice); `hessian=true` + LH + adaptive throws
   (the legacy LH multipole hessian is marked broken upstream; W-path LH
   hessian is a recorded deferral — scalar hessian IS supported).
8. **Memory note (logged, not fixed tonight)**: workspace `max_batch` and the
   M2M/L2L stage slabs scale with the max per-level node count (same behavior
   as the uniform host cache); a chunked group apply is a 041 tightening
   candidate.

## 2026-08-14 21:29 MDT — 040 implementation green locally (lead agent)

Implementation complete per the 20:55 design; all local verification green.

- **Sign corrections found by the dev oracles** (the design's "legacy formulas
  port verbatim" audit was wrong in one premise): the resident scalar pipeline
  carries NO legacy strength negation anywhere (resident direct u = +q/4πr),
  so (a) M2T returns +u/4π (the legacy evaluate_multipole's −u flip removed;
  gradient/hessian unchanged) and (b) scalar S2L is
  `L += +(−1)^{n+m} q conj(S)` (theory §4.2's leading minus is the legacy
  convention; the resident rule drops it). Both locked machine-exact by the
  M2L-composition oracles. Vortex ports needed no changes.
- **Theory §4.2 "exact per-channel oracle" nuance (measured)**: the
  P2M→M2L composition matches the vortex S2L machine-exactly on φ (all rows)
  and χ degrees ≤ P, but the χ TOP row (P_active = P+1, the 008h neighbor
  row) differs at O(1): the M2L's top row carries its own truncated LH row-up
  mixing while S2L projects it exactly. A physical probe (analytic
  Biot-Savart at leaf-scale offsets, P=4) shows the two representations are
  EQUALLY accurate (S2L marginally better: rel err 3.4e-4 vs 4.5e-4 at
  0.1-offset scale); the difference is truncation-tail content, not error.
  The parity test therefore asserts machine parity on φ + χ(≤P), evaluated
  parity + an independent analytic Biot-Savart anchor for the top row. Noted
  for a future theory-touch caveat on §4.2's "exact" wording.
- **Local verification**: new `test/adaptive_lifecycle_test.jl` (wired into
  runtests after the 039 file) — all pass, 1 thread:
  scalar accuracy 40 (cube/filament/multiscale × P=4,8 × F64,F32, vel rel
  RMS ≤ 1e-3, W/X asserted nonempty on multi-level cases; measured e.g.
  multiscale P=4 F64 3.9e-4, P=8 6.5e-6, F32 P=8 1.8e-5); LH vortex accuracy
  12 (cube+multiscale × P × TF, e.g. P=4 5.0e-4, P=8 1.1e-5); uniform-limit
  lifecycle parity 16 (matched rigid policy, ≤1e-12·scale — measured
  2.4e-17, machine-exact); M2T oracles 128 (φ+χ+scalar hessian, both TF,
  P=4/8); S2L oracles 144 incl. the 038-mandated LH vortex parity;
  RegularizedVortex vs erf-based regularized direct through the adaptive U
  list + armed per-cell gate 3; zero-allocation 11 (typed refresh = 0 B,
  typed run = 1520 B constant Val() dispatch at the shared nearfield-mode
  barrier ≤ 4096 gate, warm fmm! 28.8 KB < 512 KB repo gate — the
  pre-existing ~30 KB update_radix_state! baseline, NOT inherited by the
  adaptive stages themselves); counters all zero; guards 3.
- 039 opt-in test updated for the 040 semantics (bit-identity stopgap →
  accuracy agreement ≤ 2e-3 combined truncation) — all 039 testsets pass.
- **Regression**: 7 related host radix test files in one session:
  150,279 pass / 0 fail.
- Pre-registration `scripts/fm040_lifecycle_cost.jl` (protocol in header:
  Gravitational P=4 F64 q=5, n ∈ {1e5,1e6}, adaptive K ∈ {64,128} ell_max=10
  with 039 capacity overrides vs uniform ell ∈ {5,6}, warm medians of 5,
  same-job anchors, 2000-target sampled-direct velocity rel RMS). Local
  n=2e4 smoke validated the script only (all rel RMS 1.7e-4–4.8e-4; local
  CSV deleted; cluster run is the measurement of record). Committing BEFORE
  submission next.

## 2026-08-14 21:34 MDT — 040 cost job submitted

- Implementation + tests + pre-registration committed as `118174d` BEFORE
  submission (protocol honored).
- Cluster job **13179268** (`fm040cost`, CPU node, 1 Julia thread,
  OPENBLAS=1, 64G, 3h wall) submitted from `~/FastMultipole-040` (rsync
  snapshot of the 118174d working tree). Output `~/fm040cost_13179268.out`;
  CSV of record `~/FastMultipole-040/MATRIX_OPERATOR_REFACTOR/data/
  fm040_lifecycle_cost.csv` — to be pulled back and committed.
- sacct will be checked on every resume until a terminal state is verified.
- Carry-over finding on the pre-existing ~30 KB/step `update_radix_state!`
  allocation (brief: "do not let the adaptive path inherit it"): measured
  warm at n=1500/ell=3 — plain uniform cache 29,984 B/step; adaptive-armed
  cache 26,912 B/step (slightly LOWER; the global geometry gate is skipped).
  The adaptive-specific refresh is exactly 0 B and the adaptive lifecycle
  run is a 1,520 B constant, so the adaptive path does NOT inherit the
  baseline — it is entirely the uniform refresh, which still runs when the
  policy is armed (the known double-refresh open item, priced in 041a).
## 2026-08-14 22:37 MDT — 040 cost job 13179268 CANCELLED + protocol-neutral resubmission

- Full local `Pkg.test()` (all suites incl. the new lifecycle tests): PASSED
  (exit 0), logged here for the record.
- Job 13179268 diagnosed at ~1h elapsed: julia at 99% CPU (healthy,
  computing) but ZERO output — stdout block-buffered AND the script only
  wrote its CSV at sweep end, so the projected n=1e6 host-step cost
  (~minutes/step × 12 configs) risked a 3h-wall TIMEOUT losing every
  completed row. Decision: `scancel 13179268` (verified), amend the script
  to (re)write the CSV after EVERY row + flush stdout per row, raise the
  wall to 12h. The measurement protocol (cases, params, medians, anchors,
  accuracy sampling) is UNCHANGED — amendment recorded in the script header
  and committed BEFORE resubmission. Also fixed a small-n-only sampling bug
  found in the smoke (dedup could undershoot nsample; now an exact
  shuffle-sample — no effect at n ≥ 1e5).
- Amendment committed as `48b5ac6`; snapshot re-rsynced; resubmitted as
  cluster job **13179323** (`fm040cost`, 12h wall, CPU node, 1 Julia thread,
  64G; submit script now at `~/fm040_cost.sh` outside the repo dir so rsync
  cannot delete it). Output `~/fm040cost_13179323.out`; CSV of record
  `~/FastMultipole-040/MATRIX_OPERATOR_REFACTOR/data/fm040_lifecycle_cost.csv`.
- sacct will be checked on every resume until a terminal state is verified.

## 2026-08-15 10:07 MDT — 040 DONE (lead agent completion)

- Cluster job **13179323** terminal: **COMPLETED 02:38:19 ExitCode 0:0**
  (sacct re-verified by the lead agent per the standing rule; superseded job
  13179268 confirmed CANCELLED). CSV of record pulled to
  `data/fm040_lifecycle_cost.csv` and committed — 24/24 rows ok.
- Local full `Pkg.test()` confirmed PASSED ("Testing FastMultipole tests
  passed", exit 0) — includes the new `adaptive_lifecycle_test.jl` and the
  updated 039 suite.
- **Measurement headlines** (warm fmm! median-of-5, same-job anchors, P=4
  F64 q=5, 2000-target sampled-direct accuracy): every row inside the 1e-3
  gate (adaptive 3.87e-4–6.58e-4; uniform 2.30e-4–5.09e-4). At n=1e6,
  adaptive K=64 vs BEST uniform depth: wake **18.31 s vs 44.37 s (2.42×)**,
  multiscale100 **18.62 s vs 60.94 s (3.27×)**; vs the auto-scale ℓ=5:
  14.4× / 9.5×. Uniform-cube non-regression: 18.90 s vs 17.67 s (1.07×
  slower; structurally the same partition — 32,686 vs 32,710 leaves, equal
  U pairs to 5 digits — with the gap dominated by the known double refresh:
  adaptive t_update 1.79 s vs 0.51 s). The 038 mechanism reproduced
  end-to-end: popmax = K_max everywhere; wake U body pairs 9.39e8 vs
  4.36e10 (ℓ=5, 46×) / 6.69e9 (ℓ=6, 7.1×). Honest negative: wake at n=1e5
  is the one case where best-uniform (ℓ=6, 1.39 s) beats adaptive K=64
  (1.75 s); K < 64 (outside the pre-registered sweep) would likely close
  it — recorded for 041a.
- Task file updated (completion notes + measurement section); START_HERE 040
  row marked **Done** (Approved left blank). Committing as `040 DONE:
  close-out` next.
- Row 040 is ready for clear-context approval. User-ratification items
  carried: split-veto default OFF, 038 option (b) not implemented, E2
  disposition open, plus the 040 deferrals (TwoPass/Partitioned kernels on
  adaptive, W-list M2T LH hessian, rectangular domains, per-level radius
  schedules) and the theory §4.2 top-row-caveat note.
- 039-approval noted item (2) DONE: the K_max population test's
  `|| n_balance_splits > 0` escape removed — with the veto off, a balance
  split opens a leaf that already had pop ≤ K_max, so children inherit the
  bound; the test now asserts `pop <= K_max` unconditionally below the depth
  cap (039 suite re-run: 60,774 pass). Item (3) preserved as-is (W/X duality
  asserted only on ungated lists — source-side σ asymmetry is correct).

## 2026-08-15 10:14 MDT — 040 clear-context approval: APPROVED

Reviewer: clear-context approval agent (no prior campaign context).
Verdict: **APPROVED** — 040 Approved checkbox marked in START_HERE;
committed as `040: clear-context approval`.

Evidence: read START_HERE protocol + 040 row, the 040 task file, the full
`118174d`/`48b5ac6`/`630f3f3` diffs, theory §2.6/§4/§5/§9, the script and
CSV of record. Re-ran `test/adaptive_lifecycle_test.jl` locally (1
thread): 357 pass / 0 fail, exit 0, per-testset counts matching the
completion notes. Hand-verified: the no-new-tables claim (window driver
mirrors the uniform hierarchical driver call-for-call; CSR `vstage_class`
and the workspace plans share `_hierarchical_class_metadata` numbering;
route_levels/route_offsets unread by plan launchers; machine-exact
uniform-limit parity at ell=3 locks multi-level class ids); the M2M
no-prefix-zeroing argument (B2M `fill!`s the whole buffer); the resident
sign findings (resident P2M carries `+(−1)^{n+m} q`; the independent
analytic Biot-Savart anchor excludes a shared convention error); the
§4.2 χ-top-row caveat (sound — truncated LH row-up mixing vs exact
projection, equal truncation order, properly tested); gate replacement
semantics (global throw skipped only when adaptive armed; regularized
kernels require the armed σ gate at construction); exact-once consumption
(each of U/V/W/X consumed in exactly one accumulate-only stage over the
039-proven lists). Measurement protocol honored: pre-registration in
`118174d` before job 13179268; the `48b5ac6` amendment committed+logged
before resubmission 13179323 and is protocol-neutral (incremental CSV,
flush, wall; the sampling fix also removed a low-index draw bias —
strictly an improvement, noted); every headline recomputed from the CSV
and exact (2.423×/3.273×/1.07×, 46×/20× U pairs, accuracy range, honest
wake n=1e5 negative). Production defaults unchanged; option (b) not
implemented; split veto OFF preserved; theory §9 fixed as mandated.

NOTED (recorded in the task-file approval block): (1) "best uniform" =
best of the pre-registered ℓ∈{5,6} only — 041a should widen the depth
sweep before publishable claims; (2) the amendment's sampling change is
slightly understated as "no effect at n ≥ 1e5"; (3) adaptive S2L
body-type coverage is a runtime throw — consider a construction guard in
041; (4) user-ratification items carried unchanged (veto OFF, option (b),
E2, 040 deferrals, cube "agreed tolerance", §4.2 wording caveat).

## 2026-08-15 10:17 MDT — 041 lead agent start

- 041 lead agent started (040 Done+Approved 2026-08-15 10:14). Read
  START_HERE (protocol + 041 row + placement rules), this log in full, and
  the 041 task file. Binding items re-confirmed: sticky demotion approved,
  option (b) NOT implemented, split veto default OFF, no production default
  changes, 023 counter/zero-alloc/graph-capture contracts, resident sign
  conventions (+(-1)^(n+m) q; chi at P+1; theory §4.2 top-row caveat),
  V-list M2L consumes UNCHANGED plans/tables.
- Carry-over items in this lane: (a) 040-approval note (3) — adaptive S2L
  body-type construction-time guard; (b) 040 priced levers: double refresh
  elimination, sort unification, stage-slab memory chunking (implement if
  they fit, else price+log).
- Surface mapping of the CUDA lifecycle (7,739-line translate_batched_cuda.jl)
  and the 039/040 adaptive host surfaces delegated to Explore subagents.

## 2026-08-15 10:26 MDT — 041 design of record (lead agent)

Design decisions of record (overnight/user-absent; all in-scope):

1. **Branch, don't append (kills the device double refresh).** The device
   adaptive path branches inside `update_cuda_radix_state!`: with the
   adaptive policy armed on a device cache, the UNIFORM grid update,
   uniform route/direct generation, and `node_at` scatter do NOT run —
   the adaptive refresh (device tree build + DTR lists + leaf-cell
   presentation) replaces them. This lands the 040 "double refresh
   elimination" lever on the device path by construction. The adaptive
   path also uses ONLY the full-depth Morton sort (single sort — the
   "sort unification" lever lands on device too). Stage-slab chunking:
   price + log only (unchanged from host).
2. **Full device construction + DTR** in a new `src/tree_batched_cuda.jl`
   (placement rule 2/4), included from `translate_batched_cuda.jl`:
   theory §1.2 as a host-orchestrated per-level frontier of small kernels
   (map/scan/compact + per-split 8-way binary-search child runs), §1.4
   balance rounds (emit ≤8 parent keys, device sort/unique, interval
   match, split), finalize = one (level, full-depth-shifted-key) device
   sort into the level-major node table, σ upward sweep per level. All
   data device-resident; host sees only 4-byte scalars (existing
   pinned-scalar pattern). Same resulting topology as the host tree
   (split rule is local; finalize order is a unique total order) —
   asserted by structural parity tests.
3. **DTR lists on device** as the theory §2.7 frontier: classify kernel
   (three-clamp near test, source-side σ gate, sticky lineage bit),
   scan/compact per class (deterministic, no atomic cursors for V), CSR
   class partition via stable sort by global class id. U/W/X grouped;
   U node pairs mapped to leaf slots on device. The emission-time 025
   phase-table membership invariant is enforced as a device-side flag +
   loud host assert (mirror of 039's throw).
4. **Occupancy lookup:** the adaptive path needs NO dense Σ8^L table —
   occupancy is resolved by the tree's own child links + sorted-key
   binary search (parent/balance/range lookups). The uniform device
   path's `node_at` table and its ℓ≤8 cap are left UNTOUCHED this row:
   swapping binary search into the shipped `_cuda_hier_*_flags/compact`
   kernels risks the 028/029 record for zero measured need (deep uniform
   grids are exactly what the adaptive path replaces). RECORDED DECISION
   per the task: the cap does not lift for the uniform path in 041; the
   adaptive device path has no ℓ cap up to `RADIX_GRID_MAX_ELL`(21).
   Evidence + revisit note go in the completion report.
5. **V-list M2L**: device mirror of the 040 host window driver — walk the
   device-resident CSR stream in windows, D2D-copy targets/sources/class
   into `state.route_*`/`plan.route_class`, dispatch the UNCHANGED
   dense/precomputed-y/concat plan launchers with `clear_locals=false`.
   No new operator tables; construction-only operator/route uploads.
6. **M2T/S2L device kernels** in `translate_batched_cuda.jl`: ports of
   the 040 host kernels (resident signs: +(-1)^(n+m) q, chi at P+1),
   thread-per-body (M2T) / block-per-pair with shared accumulation or
   atomic adds (S2L), irregular harmonics computed thread-locally with
   compile-time P sizing. Ordered on the main stream (M2T after L2B).
   NEW construction-time guard for unsupported S2L body types (040
   approval note (3)) on BOTH host and device caches.
7. **Epoch caching + graph capture**: adaptive leaf-key-set epoch check
   reuses the `_cuda_keys_differ_kernel!` pattern; on an unchanged epoch
   the DTR list build and stage-group refresh are SKIPPED (the CSR
   stream is the window cache — it is device-resident and static within
   an epoch). Graph capture of the adaptive lifecycle body follows the
   uniform warm-up/epoch pattern; if capture proves unstable on H200 it
   ships OFF for adaptive with the deviation priced + logged.
8. **Contracts**: zero per-step allocation (all capacities from the 039
   policy formulas/overrides; loud asserts); counters —
   route_uploads/operator_uploads constant after construction,
   expansion_host_copies == 0; binned/split nearfield contexts and the
   symmetric path stay OFF on adaptive (guards); TwoPass/Partitioned +
   adaptive device throws (mirror of host).
9. **Tests**: new `test/cuda_radix_adaptive_test.jl` (standard
   FASTMULTIPOLE_REQUIRE_CUDA_TESTS gating), wired into runtests.jl +
   test/cuda/runtests.jl: structural parity (device tree/node table/
   U/V/W/X/CSR set-equality vs host 039), lifecycle parity vs host
   adaptive (P=4+P=8, F64+F32, LH vortex), velocity RMS <= 1e-3 vs
   direct on cube/wake-filament/multiscale, counter/zero-alloc tests,
   uniform-device non-regression (existing suites re-run).
10. **Measurement**: pre-registered `scripts/fm041_cuda_cost.jl` — H200,
    cube/wake/multiscale x n in {1e5, 1e6}, adaptive K in {64,128} vs
    uniform ell in {5,6}, per-stage breakdown + refresh + memory,
    Float64 + Float32, same-job anchors, warm medians; CSV of record
    `data/fm041_cuda_cost.csv`. Cluster env: reuse the fm034env
    local-toolkit CUDA environment with a fresh `~/FastMultipole-041`
    snapshot (dev path repoint).


## 2026-08-15 10:54 MDT — 041 implementation surface complete (lead agent)

- Implemented per the design of record:
  - `src/tree_batched_cuda.jl` (NEW, ~1050 lines): device adaptive
    construction (Phase A K_max frontier split via sorted-key binary-search
    child occupancy; Phase B 2:1 balance as Jacobi rounds over the leaf key
    set — proven to terminate at the same unique balance closure as the
    host's deepest-first sweep, so structural parity is exact; Phase C
    level-major finalize reusing the per-level ancestor-compaction /
    parent-child binary-search patterns), per-node sigma sweep, DTR frontier
    (deterministic scan-ordered emission, sticky demotion, device 025
    phase-table membership flag), deterministic (class, index)-keyed CSR
    partition, U slot mapping, occupancy-epoch snapshot/compare over the
    adaptive leaf set (epoch fast path refreshes node/cell ranges only).
  - `src/translate_batched_cuda.jl` adaptive section: device M2T/S2L kernels
    (thread-local irregular harmonics reusing the HOST `irregular_harmonics!`
    + `_resident_multipole_eval_flat*` as device functions — sign conventions
    shared by construction; vortex S2L verbatim port; atomic accumulation),
    adaptive M2L driver over the UNCHANGED plans (dense CUDA family:
    per-level in-place applies of CSR segments, zero copies; precomputed-y/
    concat: D2D windows mirroring the 040 host driver), lifecycle body with
    nearfield overlap + graph capture (uniform warm-up/epoch pattern,
    dense-fused-only eligibility, exact mirror), cache build/step/update
    plumbing (BRANCH design: uniform grid/route refresh does NOT run on the
    adaptive device path — double-refresh + sort-unification levers land).
  - `src/containers.jl`: `DeviceAdaptiveCUDAContext` (Any-typed device
    fields per the established convention).
  - `src/translate_batched_resident.jl`: device throw removed; NEW
    construction guards — S2L body-type (host+device, the 040 approval
    item) and device+split_veto (device limitation, veto pending
    ratification anyway).
  - `_cuda_hier_dense_apply_routes!` hctx parameter annotation relaxed
    (duck-typed; only first_m2l_level + scales read) so the adaptive context
    can drive the unchanged per-level dense applies. No other change to the
    uniform path.
- Local: all 4 touched files parse; package loads; host adaptive suites
  re-run green (exit 0, incl. guards) — host path unbroken.
- Cluster: `~/fm041env` created from fm034env (local-toolkit CUDA prefs,
  dev path repointed to `~/FastMultipole-041`).
- Tests (`test/cuda_radix_adaptive_test.jl`) + pre-registered measurement
  script (`scripts/fm041_cuda_cost.jl`) delegated to a fork subagent with
  full context; runtests wiring + sbatch included.

## 2026-08-15 11:01 MDT — 041 committed + bring-up job submitted

- Implementation + tests + pre-registered measurement committed as 69ac964
  BEFORE any submission (protocol honored). Snapshot rsynced to
  ~/FastMultipole-041; env ~/fm041env (fm034env clone, dev path repointed).
- Cluster job **13180171** (fm041b, GPU partition m13l/m13h, 1 GPU, 2h wall)
  submitted: bring-up only — CUDA load preflight + test/cuda_radix_adaptive_test.jl.
  Output ~/fm041b_13180171.out. The full pre-registered measurement job
  (scripts/fm041_cuda_cost.jl, committed in 69ac964) submits only after the
  tests are green. sacct will be verified on every resume.

## 2026-08-15 11:09 MDT — 041 bring-up iteration 1

- Job **13180171** FAILED (00:04:52): slurm placed it on m13l-1-2 (L40S).
  The fm034env CUDA stack's pkgimage caches are CPU-target-specific and the
  local-toolkit discovery failed during re-precompile on that node — the
  proven environment (job 13170768) runs on the m13h H200 partition, which
  the measurements require anyway. Fix: '#SBATCH -p m13h' added.
- Resubmitted as job **13180196** (fm041b, m13h, 1 GPU, 2h).

## 2026-08-15 11:13 MDT — 041 bring-up iteration 2

- Job **13180196** FAILED (00:00:40, m13h-1-1): same CUDA.jl load failure.
  ROOT CAUSE found via the proven in-repo GPU submit script
  (scripts/cuda_035_run.sh): all working GPU jobs pin
  'module load cuda julia/1.11.7-6bmogfl' (julia 1.12.6 segfaults in host
  LLVM JIT — job 13058191 — and has no CUDA pkgimage caches, so it
  re-precompiled the CUDA stack broken) and pin the GPU as
  '--gpus=h200:1'. My scripts used 'module load julia cuda' (1.12.6).
- Both the bring-up and the measurement submit scripts fixed to the proven
  module line + h200 gres. Resubmitted bring-up as job **13180197**.

## 2026-08-15 11:24 MDT — 041 bring-up iteration 3

- Job **13180197** FAILED but was major progress: CUDA loads under
  julia/1.11.7, FastMultipole compiles on H200, and every testset reached
  execution. Single failure mode: DeviceAdaptiveCUDAContext MethodError —
  the allocator passed 91 args to the 93-field struct (missing the fb2/fdem2
  DTR ping-pong frontier buffers). Fixed, arity mechanically verified 93/93,
  committed, resynced. Resubmitted bring-up as job **13180198**.

## 2026-08-15 11:34 MDT — 041 bring-up iteration 4 (major pass)

- Job **13180198**: 563/576 PASS on H200 — ALL 24 structural-parity configs
  (exact node-table equality incl. parent/child links, level_offsets,
  n_balance_splits — the Jacobi≡deepest-first closure argument HOLDS on
  hardware; U/W/X set equality; V CSR multiset + class_starts), both σ-gate
  parity sets, and cube lifecycle parity (P=4/8 × F64/F32). The 13 errors
  share ONE root cause: the M2T/S2L kernels' per-thread MArray harmonics
  escaped to device heap (exactly 448 B/thread at P=4) and exhausted the
  device-malloc heap; the KernelException then poisoned every later testset
  (filament/multiscale lifecycle, strategies, LH multiscale, counters).
- Fix: preallocated per-thread irregular-harmonic scratch slab
  (2 × 65536 slots × NH, 69 MB worst case at P=8 F64) with a fixed 512×128
  grid-stride kernel shape; MArray removed. Committed; resynced.
- Resubmitted bring-up as job **13180242**.
