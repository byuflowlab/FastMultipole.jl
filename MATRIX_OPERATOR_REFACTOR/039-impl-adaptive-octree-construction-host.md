# 039 Impl: Adaptive Octree Construction and Interaction Lists (Host)

## Status and Entry Gate

**DONE 2026-08-14 (overnight lead agent); pending clear-context approval.**

Entry gate: `038` complete and approved — satisfied 2026-08-14 19:38 MDT
(commits `04f6c3c`/`5f57db4`/`86c4951`).

## Objective

Implement the `038` adaptive tree and its U/V/W/X interaction lists on the
host as the reference path: Morton-prefix top-down splitting with population
threshold `K_max` and depth cap, 2:1 balance sweep, compact leaf/node
metadata, and list generation — all structured as the sort/scan/compact
primitives that `041` will mirror on device.

## Scope and placement

- New tree code extends `tree_batched.jl`; list construction extends
  `interaction_list_batched.jl`; new structs go in `containers.jl`
  (per the Implementation Code Placement rules).
- The uniform-depth radix grid remains the default and is untouched; the
  adaptive tree is an opt-in constructor/policy on `RadixFMMCache`.
- No CUDA code in this row (`041`); no M2T/S2L operator execution in this
  row (`040`) — this row builds the tree and the lists and proves them
  correct.

## High-level deliverables

- Adaptive leaf construction from sorted Morton keys: split-while
  `count > K_max`, depth ≤ `ell_max`, then 2:1 balance; occupied-ancestor
  node levels compatible with the existing `level_offsets` convention.
- U/V/W/X list generation per `038`, with V lists emitted as
  `(level, offset)` classes identical in format to the existing hierarchical
  route stream (so `040` can feed them to the unchanged M2L strategies).
- Per-cell geometry gate from `038` item 5 replacing the global `σ_max` gate
  on the adaptive path.
- Capacity plumbing per the `038` cost model: all list/leaf buffers sized at
  construction from `max_n_bodies`/`K_max`/`ell_max`, zero per-step
  allocation on refresh.
- Refresh path: re-sort, re-split, re-balance, and regenerate lists in place
  within capacity, following the `023` invariant contract.

## Verification

- Exact-once coverage: brute-force pair enumeration vs U∪V∪W∪X∪self on
  randomized uniform, wake-like, and clustered multi-scale distributions
  (multiple seeds, multiple `K_max`), both near radii.
- 2:1 balance invariant asserted structurally.
- V-list class format parity: on a uniform distribution with
  `K_max` chosen so all leaves land at one level, the adaptive V lists must
  reproduce the existing hierarchical route set exactly.
- Tests include `P=4` (standing rule) and run in the standard suite.

## Acceptance

Lists proven exact-once on all test distributions, uniform-limit parity with
the existing hierarchical routes, zero per-step allocation on refresh, and
construction cost measured and recorded vs the uniform grid on the two phase
cases plus the multi-scale case.

## Completion Notes (2026-08-14, overnight lead agent)

### What shipped and where

- **`src/containers.jl`** — new types (per the Implementation Code Placement
  rules): `AdaptiveTreePolicy` (K_max, ell_max, constant `near_radius2`,
  balance/split_veto toggles, `rho_t`/`sigma_row` σ-gate plumbing,
  `beta_balance`, explicit capacity overrides), `AdaptiveRadixTree{TF}`
  (capacity-sized SoA node pool + final **level-major, Morton-sorted-within-
  level** node table matching the uniform `level_offsets` convention, per-body
  sort machinery, per-node subtree `sigma_max`), `AdaptiveInteractionLists`
  (V routes in the production class format + CSR class partition, U/W/X,
  DTR stack, staging + counting-sort scratch). `RadixFMMCache` gains three
  trailing opt-in fields (`adaptive`/`adaptive_tree`/`adaptive_lists`,
  all `nothing` by default).
- **`src/tree_batched.jl`** — construction: `AdaptiveRadixTree(systems; ...)`,
  `update_adaptive_tree!` (fixed root cube/capacities; keys at `ell_max`,
  in-place LSD sort, DFS top-down `K_max` split with optional §5.4 veto,
  §1.4 **Sundar-style 2:1 balance sweep** (per-round sorted leaf-interval
  tables + binary-search matching, deepest-first, fixed point with guard;
  8 touching parent-level cells per leaf — provably 2 per axis — instead of
  the 26-neighbor form), level-major finalize by per-level counting/key
  sort, upward `sigma_max` sweep), `adaptive_is_leaf`/`adaptive_node_range`
  accessors, and the cache refresh hook `_refresh_adaptive_radix!`.
- **`src/interaction_list_batched.jl`** — `AdaptiveInteractionLists(tree)`
  and `build_adaptive_interaction_lists!`: theory §2.2 dual-tree recursion
  with the §5.2 **sticky** per-cell σ demotion gate (integer-exact
  finest-lattice AABB gap; source-side σ), mixed-level near predicate
  (§2.1 per-axis clamp), U/W/X emission with structural invariant throws,
  V emission with **runtime 025 phase-table membership enforcement**
  (`level_class_of[phase(source), k, L+1] != 0`, else throw), and the
  in-place counting sort producing the class-partitioned route stream.
- **`src/translate_batched_resident.jl`** — `RadixFMMCache(...; adaptive=)`
  keyword: host-only + cubic-only guards, σ-row validation, construction of
  the adaptive structures, `update_radix_state!` refresh hook (runs AFTER
  the unchanged uniform refresh), `recenter!` forwards the policy.
  **No production default changes**: `adaptive === nothing` is bit-identical
  to before (asserted by test).
- **`src/translate_batched_cuda.jl`** — device constructor passes the three
  `nothing`s (adaptive is host-only until row 041; `device=true` + adaptive
  throws).
- **`src/FastMultipole.jl`** — exports.
- **`test/adaptive_octree_test.jl`** (wired into `test/runtests.jl` after
  `radix_trimming_test.jl`).
- **`MATRIX_OPERATOR_REFACTOR/theory/adaptive-radix-octree.md`** — §2.4
  orientation slip fixed on this touch (per the 038 re-approval note):
  the phase identity now pairs `o = c_A - c_B` (= T − S, the 025
  convention) with the source phase, with the one-line derivation; §5.4
  gains the split-veto implementation note (below).
- **`MATRIX_OPERATOR_REFACTOR/scripts/fm039_construction_cost.jl`** +
  **`data/fm039_construction_cost.csv`** — pre-registered acceptance
  measurement (protocol in the script header; cluster CPU job of record).

### Format decisions (for row 040)

- V routes: the five production arrays (`route_levels/route_offsets/
  route_targets/route_sources/route_class`) with the production numbering
  `(L - 2) * noffsets + k` over `RigidHierarchicalTables(q).push_offsets`
  and `_hierarchical_class_metadata(tables, ell_max, 2)`
  (`effective_offsets[c] = 2^(ell_max - L) * o`, leaf reference width at
  `ell_max`) — byte-compatible with the existing windowed consumers — plus
  a CSR `class_starts` so 040 can feed whole classes or windows without
  re-sorting. No new operator tables (theory §2.4).
- U/W/X endpoints are **flat adaptive node indices** (leaves live at
  multiple levels; the uniform path's `direct_sources = node - leaf_base`
  cell convention cannot represent them). Body ranges come from
  `node_lo`/`node_hi` (subtree ranges into `perm`).
- Constant near radius only (`q ∈` supported set; both 3 and 12 tested);
  the production per-level schedule is a recorded deferral.
- Host DFS pair stack (capacity `64·(2·ell_max+2)`) replaces the theory
  §2.7 frontier on the host; the flag/scan/compact frontier remains the
  041 device shape.

### Verification (all local, single thread, commands exact)

- `julia --project=. --threads=1 -e 'using Test; include("test/adaptive_octree_test.jl")'`
  — **60,773 pass / 0 fail**: construction invariants (partition, K_max,
  depth cap, level-major layout, parent/child/subtree consistency, trivial
  and coincident-body edge cases); 2:1 balance asserted structurally;
  **exact-once brute-force painting** over all ordered body pairs on
  uniform / wake-like filament / clustered multi-scale fields × 2 seeds ×
  K_max ∈ {8,32} × **q ∈ {3,12}**, balanced AND unbalanced (57,790
  assertions, all zero-defect); independent V-class re-checks (separated,
  geometric parent nearness, Chebyshev reach, phase membership, class ids)
  on gated AND ungated lists; W/X structure + ungated W/X duality; σ-gate
  suite (sticky demotion engaged, 031a cutoff contract `r ≤ ρ_t σ_src ⇒ U`
  brute-forced to zero violations, coverage exact under the gate, veto-ON
  variant); **uniform-limit parity** (jittered complete grids, `K_max=1`):
  adaptive V set == production `build_hierarchical_routes_window!` set
  (level, offset, coords, AND class id) and adaptive U == production
  direct pairs, exactly, for q ∈ {3,5,12} × ell ∈ {2,3} at **P=4**;
  zero-allocation refresh (0 bytes for `update_adaptive_tree!` and
  `build_adaptive_interaction_lists!`, gated and ungated, after body
  motion); cache opt-in (P=4 construction, bit-identical uniform fmm!
  output with the policy armed, σ pulled from packed row 4, guards,
  `recenter!` preservation, <1 KB constant dynamic-dispatch overhead per
  step through the cache hook).
- Regression: `radix_grid_clustering / radix_interaction_list /
  hierarchical_m2l_host / radix_fmm_integration / radix_trimming /
  radix_fmm_timestepping` re-run in one session — **114,015 pass / 0
  fail**.

### Deviations from theory (quantified; logged in the decision log)

1. **§5.4 split veto defaults OFF** (`AdaptiveTreePolicy(split_veto=false)`).
   The literal veto keys on the cell's own subtree `σ_max`, so one fat-σ
   body vetoes every ancestor split: measured one-fat-core field (n=1500,
   σ=3e-4 + one 0.15, ρ_t=4.789) collapses to a **single root leaf** at
   q=3 (g_min=1) — global direct, the pathology §5 exists to remove —
   while q=12 (g_min=√5) only coarsens locally (411 vs 424 leaves).
   Sticky demotion alone preserves cost locality (116/183 demotions at
   q=3/12; exact-once + contract intact). Veto remains available for
   spatially smooth σ fields. Implementation note added to theory §5.4;
   **user ratification item**.
2. **Capacity formulas**: §6.4 with hard occupancy caps
   (`node ≤ (ell_max+1)·n+1`; U/W/X per-leaf factors × `min(node_cap, n)`;
   `V = push_max · min(node_cap, 4n)`) + explicit per-policy overrides;
   overflow throws. The raw §6.4 products are memory-infeasible at n=1e6;
   measured-to-capacity ratios are recorded in the cost CSV for 040
   tightening.
3. DFS stack in place of the §2.7 frontier (host only; improvement, not a
   weakening — the device form is unchanged for 041).

### Construction cost (acceptance measurement)

Pre-registered protocol committed at `fd58aea` before submission; cluster
CPU job **13178905** (`~/FastMultipole-039`, Julia 1.12.6, 1 thread,
same-job anchors). Cases: unitcube, helical-wake-cylinder positions (033),
multiscale100; n ∈ {1e5, 1e6}; adaptive K_max ∈ {32,64,128} (ell_max=10,
q=5, balance on) vs uniform `RadixFMMCache` default policy at ell ∈ {5,6}.
Results in `data/fm039_construction_cost.csv`:

Job 13178905 COMPLETED 00:05:20, ExitCode 0:0 (sacct-verified). Headline
rows at n = 1e6 (warm refresh = tree + lists, median of 5; `u_pairs` =
direct body-pair work; popmax = worst leaf population):

| case | config | refresh (ms) | popmax | u_pairs | V routes |
| --- | --- | --- | --- | --- | --- |
| unitcube | adaptive K=64 | 1222 | 64 | 1.84e9 | 1.23e7 |
| unitcube | uniform ell=5 | 534 | 61 | 1.84e9 | 1.23e7 |
| wake | adaptive K=128 | 1502 | 128 | 1.51e9 | 1.45e7 |
| wake | uniform ell=5 | 311 | 1231 | 4.36e10 | 2.72e5 |
| wake | uniform ell=6 | 377 | 182 | 6.69e9 | 2.65e6 |
| multiscale100 | adaptive K=128 | 1239 | 128 | 1.79e9 | 1.25e7 |
| multiscale100 | uniform ell=5 | 464 | 2442 | 2.85e10 | 1.23e7 |
| multiscale100 | uniform ell=6 | 2220 | 346 | 4.72e9 | 8.70e7 |

Reading: (i) **uniform-limit sanity** — on the unit cube, adaptive K=64
reproduces the uniform ell=5 structure almost exactly (37,363 vs 37,390
nodes; identical direct pairs and routes to 3 digits), at 2.3x the refresh
cost (1.22 s vs 0.53 s) — the price of pool build + balance + finalize +
DTR vs closed-form windowed emission at equal structure. (ii) **Bounded
worst cell where the uniform grid cannot** — on the wake/multiscale cases
the adaptive tree holds popmax at K_max by construction while the uniform
grid pays fat cells (1231/2442 at ell=5; 182/346 at ell=6) or explodes its
route count going deeper: at n=1e6 the adaptive K=128 direct body-pair
work is 28.8x/4.4x (wake) and 15.9x/2.6x (multiscale) below uniform
ell=5/ell=6, at comparable or lower refresh cost than ell=6 (1.24 s vs
2.22 s on multiscale). This is the counted 038 mechanism reproduced at
production scale on the host. (iii) Worst-case adaptive refresh at n=1e6
is 4.35 s (unitcube K=32, a deliberately over-fine setting); cold
construction <= 5.3 s everywhere. (iv) Measured-to-capacity ratios peak
at node 0.83, U 0.60, V 0.91 (wake K=128) — the shipped formulas held
with margin but the V margin is thin at wake-like density contrast;
recorded for 040 tightening.

### Open items

- User ratification: split-veto default (deviation 1); option (b)
  table-free M2T/S2L re-admission (038 carry-over, not implemented per
  standing instruction); E2 disposition (still open).
- 040: consume the lists (M2T/S2L kernels + pipeline), unify the body sort
  with the uniform path, tighten capacities from the recorded ratios,
  rectangular domains, per-level radius schedule.
- Pre-existing (not 039): `update_radix_state!` allocates ~30 KB/step on
  the uniform path with or without the adaptive opt-in (repo's own gates
  are <512 KB bounds); noted for a future cleanup row.

## Clear-Context Approval (2026-08-14 20:37 MDT)

**Verdict: APPROVED.** Reviewer: clear-context approval agent (no prior
campaign context), per START_HERE protocol §6.

Evidence checked:

- Read in full: START_HERE (protocol, 039 row, phase preamble), this task
  file, `theory/adaptive-radix-octree.md`, the decision log
  (`decision-log-2026-08-14-overnight.md`, 038/039 entries), the complete
  `fd58aea` diffs of `src/containers.jl`, `src/tree_batched.jl`,
  `src/interaction_list_batched.jl`, `src/translate_batched_resident.jl`,
  `src/translate_batched_cuda.jl`, `src/FastMultipole.jl`,
  `test/adaptive_octree_test.jl`, `scripts/fm039_construction_cost.jl`,
  `data/fm039_construction_cost.csv`.
- Re-ran locally (`--threads=1`): `test/adaptive_octree_test.jl` — 60,773
  pass / 0 fail, exit 0, per-testset counts matching the completion notes
  exactly (2879 construction + 6 balance + 57,790 exact-once + 26 σ-gate +
  48 uniform-limit parity + 6 zero-alloc + 18 cache opt-in).
- Correctness spot-proofs done by hand: sticky-demotion semantics
  (`near = dem || isnear`; gate only on `!near`; lineage bit on every
  push; V/W/X only on `!near` ⇒ never-demoted paths — matches theory §5.2
  and restores Invariant 2 geometrically); emission-time 025 phase-table
  membership throw uses `o = c_A − c_B` (T−S) with the SOURCE phase, per
  the corrected §2.4; σ-gap test is integer-exact
  (`delta_min² · g² < (ρ_t σ_max(src))²`, source side only, matching
  031a); the 8-per-leaf balance emission set ({fld(c−1,2), fld(c−1,2)+1}
  per axis) is exactly the touching parent-cell set; DFS split-stack
  (8·(ell_max+2)+8 ≥ 7·ell_max+8) and pair-stack (64·(2·ell_max+2) ≥
  63·(2·ell_max+1)+1) capacities are sufficient; counting-sort CSR
  indexing verified. σ-from-buffers ordering verified consistent:
  `source_to_buffer!` packs ordinally, and `_radix_fill_body_data!`
  enumerates the same global ordinal order (the sorted copy goes to
  `state.source_bodies`, not the buffers).
- Coverage-test genuineness confirmed: brute-force UInt8 painting over
  all n² ordered pairs, 3 distributions × 2 seeds × K_max {8,32} ×
  q {3,12}, balanced AND unbalanced, plus gated configs and the parity
  tree; the gated V-class re-check (`_adt_v_classes_ok`) independently
  re-verifies geometric parent nearness + phase membership on σ-gated
  lists (the 038 review's blocker, now double-enforced: runtime throw +
  test).
- Uniform-limit parity confirmed genuine: reference set built from the
  production `build_hierarchical_routes_window!` emitter and the
  production direct pairs on a cache with complete occupancy; tuple
  equality includes level, offset, both coords, and class id; duplicate
  emission excluded by set-cardinality checks; P=4.
- No-default-change confirmed: `adaptive === nothing` default; three
  trailing fields; refresh hook is a no-op without the policy;
  bit-identical uniform `fmm!` output asserted by test; device ctor
  passes nothings; device/rectangular guards throw.
- Measurement methodology verified: pre-registration (script + protocol
  header) is in `fd58aea`, committed 20:18 before the 20:20 job
  submission (decision log); job 13178905 COMPLETED 0:0; same-job
  anchors (adaptive + uniform in one process, 1 thread). Every headline
  claim re-derived from the CSV and exact: 28.8x = 4.360e10/1.515e9 and
  4.4x (wake, ell=5/6), 15.9x/2.6x (multiscale), unitcube K=64 parity
  37,363 vs 37,390 nodes at 1222 vs 534 ms, worst refresh 4348 ms, cold
  ≤ 5.25 s, capacity-ratio peaks 0.83/0.60/0.91 (wake K=128).
- Minimal invasiveness: `git show --stat` — the two commits touch only
  the listed files; existing `interaction_list_batched.jl`/
  `tree_batched.jl` code is appended-to, not modified; the resident-path
  diff is keyword + guards + hook only.
- Deviations accepted as logged: split-veto default OFF (quantified
  root-leaf collapse; user-ratification item stands), capped §6.4
  capacity formulas with overrides + loud overflow (raw products
  measured memory-infeasible at n=1e6), host DFS stack for §2.7
  (device frontier form reserved for 041). Option (b) correctly NOT
  implemented.

NOTED (non-blocking):

1. Theory §9 still says 040 runs "split-veto default-on for population
   splits", contradicting the §5.4 implementation note (default OFF,
   pending ratification). Fix §9 on the next theory touch so the 040
   lead is not misdirected; the decision log and this task file are
   unambiguous in the meantime.
2. The construction-invariants population check
   (`pop <= K_max || tree.n_balance_splits > 0`) is weaker than needed:
   with the veto off, every leaf above level ell_max satisfies
   `pop <= K_max` unconditionally (balance splits only shrink
   populations), so the escape clause could mask a future split bug.
   Tighten when next touching the test.
3. `wset == xset` W/X duality is asserted only on ungated lists —
   correct as-is (source-side σ makes the gated predicate asymmetric),
   recorded here so no one "fixes" it into the gated suite.
