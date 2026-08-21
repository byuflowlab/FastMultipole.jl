# Overnight Campaign Brief — 2026-08-14, 7pm start (user-authorized)

Scheduled autonomous campaign: work rows 038 through 042 of
`MATRIX_OPERATOR_REFACTOR/START_HERE.md`. The user is unavailable until
tomorrow — solve problems with best judgement, never wait on them, and log
every important decision.

Context for a fresh session: rows 037a–037f are all Done + clear-context
Approved (037c closed by pointer). 037f shipped the `:fp32` F64 default
(user-approved). The next unblocked row is 038. Recent campaign convention
examples live in the 037b/037e/037f task files.

## Ground rules

1. FIRST: create `MATRIX_OPERATOR_REFACTOR/decision-log-2026-08-14-overnight.md`
   (a coordination/reporting doc, not a task file) and append timestamped
   entries throughout: every nontrivial decision, blocker + resolution,
   verdict, and anything the user must ratify later. This is the primary
   deliverable for the user's return alongside the work itself.
2. Row order and gating per START_HERE.md: 038 (theory) → 039 (host
   construction) → 040 (host lifecycle) → 041 (CUDA) → 041a (figures) → 042
   (milestone review). Each row: execute via a background lead agent (with
   internal Explore scouts / parallel file-disjoint workers), then a FRESH
   clear-context approval subagent before the next row starts (protocol in
   START_HERE.md §Routine task protocol item 6). If an approval returns
   required changes, route fixes to the lead and launch a fresh re-approval
   agent. 042 is a Milestone Review: its agent follows the Milestone Review
   protocol (reads ../MATRIX_OPERATOR_REFACTOR.md etc.).
3. HPC: ssh alias `orc` (BYU rc), expected open. H200/CPU-node jobs
   unrestricted; local Mac work stays single-threaded and light (smoke/parse
   checks only). Leads must verify cluster job terminal states with sacct on
   every resume (monitors have repeatedly missed terminal states); the
   orchestrator should also arm a fallback sacct watch (background bash
   until-loop, 5-min poll) for every submitted job batch.
4. Standing invariants no overnight authority can override: NO production
   default changes (record evidence + recommendation in the decision log
   instead); no lab-notebook writes; velocity RMS ≤ 1e-3 gate; P=4 test
   coverage; capacity/zero-recurring-allocation/counter/graph-capture
   contracts; pre-registration committed BEFORE job submission; same-job
   anchors for every comparison; critical-path pricing (overlapped critical
   path, not stage sums).
5. Held user decisions stay held: 037g staging (VIC row funded by 037d),
   rotor auto-depth, E2→038 acknowledgment, notebook entry. If 038's
   derivation naturally subsumes the E2 mechanism via the per-cell σ gate,
   proceed (in-scope for 038) and log it; do not close the held E2
   disposition item.
6. Realism: 038–042 may not all complete in one night. Never skip or thin an
   approval gate to go faster. If unfinished by morning, leave a clean
   handoff: decision log current, in-flight jobs listed with IDs, next
   actions stated. Quality over completion.
7. If a hard blocker requires a genuine user-only decision, log it, park that
   path, and continue any independent work available.

## Start

Read START_HERE.md, confirm 038 is the first unblocked row, then launch the
038 lead agent.
