# 025 Theory Hierarchical Rigid M2L Stencil

## Objective

Derive the source-major, phase-indexed, level-invariant M2L stencil that lets the radix
path do genuine multi-level (node-to-node) M2L, and prove it covers every body pair
exactly once. Derive the level-scaling law that lets one operator table serve every level.
Parameterize the derivation by a single near radius so the same construction yields both
the task-`024b` `theta = 0.5` stencil and the classic FMM `27 / 189` stencil, giving `026`
and `027` a like-for-like cost/accuracy comparison.

This row derives; it does not modify `src/`.

## Motivating Finding

The clear-context review of `024b` established that the radix path's far field is
**single-level**. `build_radix_routes!` (`src/interaction_list_batched.jl:410`) stamps
`route_levels[n_routes] = ell` — a constant — and both endpoints go through
`leaf_to_node[...]`, so every M2L is leaf-node to leaf-node. Verified at runtime
(`n=20000`): `unique(route_levels) == [ell]`, every route target inside the leaf node
range, and `routes ~ 0.97 * C^2` for `C` occupied leaf cells.

`ParentNeighborM2L` is not an exception: it walks levels but expands each ancestor
interaction back down to all descendant leaf pairs through the `Dict`-based
`_radix_ancestor_leaf_map` (`src/interaction_list_batched.jl:206`), and
`_flatten_radix_node_routes` (`src/translate_batched_cuda.jl:708`) maps those leaves
through `leaf_to_node`. No genuine node-to-node multi-level M2L exists in the repo, and
`RadixFMMCache` cannot select `ParentNeighborM2L` at all — it requires a policy with a
`.config` field (`src/translate_batched_resident.jl:761-764`).

Consequences: `M2M` and `L2L` run every step but feed nothing; M2L cost is `O(C^2)`, so
with direct cost `~ |near| * n * (n/C)` the optimum is `C ~ n^(2/3)` and the total is
`O(n^(4/3))`. The measured GPU exponent from `n=1e4` to `1e6` is `1.46`. The
`min(n_classes, n_cells) * n_cells` route reservation is tight, not pessimistic (measured
within 4% of actual), so no allocation change can help.

## Dependencies

- `008d-theory-dynamic-p-error-m2l-integration.md` (constant-`P` stencil bound)
- `008f-theory-radix-sort-clustering.md` (uniform-grid cell geometry)
- `008g-theory-radix-interaction-list.md` (coverage argument for the flat stencil)
- `021-impl-constant-p-stencil-and-interaction-list.md` (shipped stencil and policies)
- `024b-impl-cpu-gpu-scaling-benchmark.md` (the ell-scaled epsilon and the finding above)

## Required Reading

- `START_HERE.md`
- The dependency task files above
- `theory/024b-mac-stencil-compatibility.md` and its review
- `src/interaction_list_batched.jl` (bound, classification, traversal)

## Artifacts or Production Surface

- `MATRIX_OPERATOR_REFACTOR/theory/hierarchical-rigid-m2l-stencil.md`
- `MATRIX_OPERATOR_REFACTOR/scripts/hierarchical_rigid_stencil_verify.jl`
- `MATRIX_OPERATOR_REFACTOR/data/hierarchical_rigid_stencil/`

No production `src/` changes belong to this task.

## Deliverables

Derive and numerically verify the following. Preliminary values below were computed during
planning and must be reproduced by the verifier, not assumed.

1. **Level-invariance of the near set.** `constant_p_stencil_bound`
   (`src/interaction_list_batched.jl:8`) depends on `ell` only through
   `cell_half_width = h0 / 2^ell`, so `024b`'s `stencil_epsilon(ell) ∝ 2^ell` fixes one
   integer-lattice cutoff at every depth. At `theta = 0.5` the near set is
   `N = {o : |o|^2 <= 12}`, `|N| = 179`, identical at all levels.

2. **A single near-radius parameter spans both stencils.** For integer offsets,
   `{|o|_inf <= 1}` is *exactly* `{|o|^2 <= 3}` (verify). The classic FMM non-touching
   criterion is therefore the `near_radius2 = 3` member of the same family, and
   `near_radius2 = 12` is `024b`'s `theta = 0.5`. Record the correspondence
   `near_radius2 = floor(3 / theta^2)` and the admissible `theta` interval for each.

3. **Source-major (push) V-list.** For source `S` with phase `u = S .& 1`, offset
   `o = T - S`, the parent offset is `p_k = fld(u_k + o_k, 2)`. Define
   `V_push(u) = {o : o not in N and p(u,o) in N}`. Prove equivalence to the pull-indexed
   form under `(S,T,u,o) -> (T,S,(u+o) mod 2, -o)`. Preliminary counts to reproduce:

   | `near_radius2` | equivalent | `\|N\|` | `\|V_push(u)\|` (all 8 phases) | union | `max\|o\|_inf` |
   | --- | --- | --- | --- | --- | --- |
   | 3 | classic FMM / `theta in (0.866, 1]` | 27 | 189 | 316 | 3 |
   | 12 | `024b` `theta = 0.5` | 179 | 1253 | 1740 | 7 |

   The union column is the number of distinct `(level, offset)` operator classes per level.

4. **No level-1 or level-2 special case.** `p = fld(u + o, 2)` equals the geometric
   parent offset `fld(T,2) - fld(S,2)` exactly, and for in-box `S, T in [0, 2^L)^3` each
   `fld(., 2)` lies in `[0, 2^(L-1))`, so the tight in-box bound is
   `|p|_inf <= 2^(L-1) - 1` — not the looser formula-range bound `2^(L-1)`. At `L = 2`
   this gives `|p|_inf <= 1`, hence `|p|^2 <= 3`, so every separated in-box level-2
   offset has a near parent for **both** `near_radius2 = 12` and `near_radius2 = 3`, and
   no level-2 branch is needed for either radius. Verify both cases numerically
   (preliminary: all 1312 separated in-box level-2 offsets qualify at
   `near_radius2 = 12`; the expected result at `near_radius2 = 3` is that all qualify as
   well, via `|p|^2 <= 3`). Note the boundary robustness: at `near_radius2 = 3` the
   membership `|p|^2 = 3` sits exactly on the near cutoff, but it holds structurally in
   the shipped bound — `|o|^2 <= 3` gives `c <= 2`, where `constant_p_stencil_bound`
   returns `Inf`, so those offsets are rejected (near) at every `epsilon`, independent of
   the inclusive-vs-strict `epsilon` comparison.

5. **Exact-once coverage.** For an occupied leaf pair let `o_L` be the offset of their
   level-`L` ancestors and `L* = min{L >= 2 : o_L not in N}`. Show `L*` is well defined
   whenever the pair is not near at the leaf level, that the pair is emitted at `L*` and at
   no other level (below `L*` the offset is near; above `L*` the parent is already
   separated), and that leaf-near pairs go to direct exactly once. Uniqueness above `L*`
   requires an explicit **downward monotonicity lemma**: if the parent offset is not in
   `N`, the child offset is not in `N` — minimality of `L*` alone gives only
   `o_{L*} not in N`, not `o_L not in N` for all `L > L*`. Prove it via the
   per-component bound `|o_k| >= 2 |p_k| - 1` (preliminary: the minimum child `|o|^2`
   over all non-near parents is `34` at `near_radius2 = 12` and `9` at
   `near_radius2 = 3`, both strictly separated). This is the analytic
   form of `_assert_complete_body_coverage` (`test/radix_interaction_list_test.jl`).
   State the finite-box caveat: offsets leaving the domain are dropped by the bounds test,
   which removes pairs rather than duplicating them.

6. **Level-scaling law.** The M2L operator for offset `o` at level `L` has
   `r_L = |o| * w_L` with `w_L = 2*h0/2^L`, while `theta` and `phi` depend only on the
   *direction* of `o` and are level-independent. Establish whether
   `K(s*r0, theta, phi) = s^(-1) * Lambda(s) * K(r0, theta, phi) * Lambda(s)` with
   `Lambda(s)` degree-diagonal, pin the exponents, and verify numerically to ~1e-13
   relative in Float64 over a sample of offsets covering `theta in {0, pi/2, pi}` and
   generic directions. If it holds, operator tables in `026`/`027` stay indexed **by offset
   only** (<= 1740 matrices) instead of by `(level, offset)`; levels differ by `s = 2^(+/-k)`
   so the scaling is exact in binary floating point. If it fails, say so plainly — dense
   storage then reverts to `(ell-1) * union * D^2` and `DenseTranslationM2L` likely falls
   out of scope at `ell >= 6`.

7. **Cost model and the honest constant.** Asymptotics improve from `O(n^(4/3))` to `O(n)`.
   But at `theta = 0.5` the constants are ~6.6x the classic figures in *both* lists
   (1253 vs 189, 179 vs 27), because the cutoff radius is `sqrt(12) ~ 3.46` cells instead
   of `2`. Since total cost balances as `sqrt(|N| * |V|)`, re-tuning leaf occupancy does
   not recover it. Record:
   - the class/route reduction versus the shipped flat stencil, **stating the occupancy
     basis for every ratio** (preliminary planning figures: classes 64x / 396x / 2579x
     and routes 23x / 183x / 1465x at `ell = 5/6/7`, computed at the `024b` benchmark
     occupancies — record which `n` — since they count nonempty classes/actual routes;
     the closed-form dense-lattice class ratios,
     `((2^(ell+1)-1)^3 - 179) / ((ell-1) * 1740)`, are exactly 35.9x / 235.4x / 1588.2x
     and serve as the occupancy-independent anchor);
   - the hierarchical-vs-flat crossover, `C^2 > |V| * (8/7) * C`, i.e. `C ~ 1430` occupied
     cells at `near_radius2 = 12` and `C ~ 216` at `near_radius2 = 3` — below which the
     shipped flat path is genuinely cheaper and must remain selectable;
   - the accuracy price of the classic stencil under the same bound. Preliminary: the
     smallest accepted classic offset has `|o|^2 = 4`, giving a required epsilon
     `238.2x` larger than the `theta = 0.5` value at every level (level-invariant, as both
     scale as `2^ell`). Note that `|o|^2 = 3` yields `c = 2` exactly, where the analytic
     bound diverges, so the classic near set has no finite upper epsilon endpoint.

## Verification

The verifier enumerates offsets directly and records, for `near_radius2 in (3, 12)` and
`ell in 2:7`: `|N|`, per-phase `|V_push(u)|`, union size, `max|o|_inf`, `min|o|^2`, the
push/pull equivalence check, the level-1/level-2 branch check (both radii), the downward
monotonicity check (enumerate all non-near parents and confirm the minimum child `|o|^2`
exceeds `near_radius2`), and the required epsilon endpoints. It also runs a brute-force exact-once coverage check on fully occupied `ell=3`
and `ell=4` grids and on a boundary-truncated grid, comparing against a pairwise partition
computed independently. All counts written to CSV under
`MATRIX_OPERATOR_REFACTOR/data/hierarchical_rigid_stencil/`.

Record commands and result summaries in a `Verification Notes` section.

## Implementation Notes

- Added `theory/hierarchical-rigid-m2l-stencil.md`. It derives the integer near
  family and theta intervals, source-major phase table and pull bijection,
  level-1/2 behavior, exact-once first-separated-ancestor proof, scalar and
  asymmetric Lamb–Helmholtz scaling diagonals, and the linear cost model.
- Added `scripts/hierarchical_rigid_stencil_verify.jl`, a deterministic verifier
  that uses the production `build_dense_m2l_operator` implementation for its
  scalar and Lamb–Helmholtz checks. No expected matrix entries are embedded.
- Added five generated CSVs plus `verification_summary.md` under
  `data/hierarchical_rigid_stencil/`.
- The preliminary 024b occupancy-dependent reduction ratios were audited rather
  than promoted to measurements. Published 024b timing CSVs have no cell, class,
  or route fields. The new evidence reconstructs exact per-level occupied-cell
  counts from the campaign seed/domain and labels route reductions as
  uniform-without-replacement expectations, not measured telemetry. Exact
  fully-occupied dense-lattice anchors are reported separately.
- No public API, production type, or `src/` file was changed for this item.

## Verification Notes

Verified with Julia 1.12.5:

```sh
julia --project MATRIX_OPERATOR_REFACTOR/scripts/hierarchical_rigid_stencil_verify.jl
shasum -a 256 MATRIX_OPERATOR_REFACTOR/data/hierarchical_rigid_stencil/*
julia --project MATRIX_OPERATOR_REFACTOR/scripts/hierarchical_rigid_stencil_verify.jl
shasum -a 256 MATRIX_OPERATOR_REFACTOR/data/hierarchical_rigid_stencil/*
```

Both verifier runs passed and the two checksum manifests were byte-identical.
The generated evidence records:

- 27/189/316 and 179/1253/1740 near/phase/union counts at every depth 2–7;
- push/pull equivalence for all eight phases;
- all 3096 (`q=3`) and 752 (`q=12`) separated ordered level-2 cell pairs,
  plus the requested stronger 1312-case `q=12` bounded-offset/phase audit;
- downward-monotonicity minima 9 and 34;
- exact-once coverage on two full grids and sparse/boundary-truncated cases for
  both radii, with zero missing, duplicate, or wrong-kind pairs;
- 32 production dense-operator scaling cases (scalar and Lamb–Helmholtz,
  axial/equatorial/generic offsets, four binary scale factors), all within
  relative tolerance `1e-13`;
- epsilon endpoints, cost crossovers, dense-lattice anchors, and the
  explicitly labeled 024b occupancy audit.

`git diff -- src` was captured before and after this work and remained unchanged:
the repository already contained unrelated dirty `src/` changes, but item 025
introduced no additional `src/` diff.

## Approval Notes

Approved `2026-07-28` by a clear-context agent (not the completing agent), per the
`START_HERE.md` protocol. Reviewed: this task file, the theory derivation
(`theory/hierarchical-rigid-m2l-stencil.md`), the verifier
(`scripts/hierarchical_rigid_stencil_verify.jl`), all six generated files under
`data/hierarchical_rigid_stencil/`, and the production
`constant_p_stencil_bound` (`src/interaction_list_batched.jl:8`) that the
epsilon-endpoint derivation mirrors.

Independent verification performed during review:

1. Reran the verifier (`julia --project MATRIX_OPERATOR_REFACTOR/scripts/hierarchical_rigid_stencil_verify.jl`,
   Julia 1.12.5): PASS, and all six output files were byte-identical
   (sha256 diff) to the committed data — reproducibility confirmed on a third run.
2. Wrote an independent brute-force enumeration (separate code, no shared
   helpers) reproducing near/phase/union counts 27/189/316 and 179/1253/1740,
   `max|o|_inf` 3 and 7, downward-monotonicity minima 9 and 34, the
   `|o|_inf<=1 ⟺ |o|^2<=3` equivalence, and crossovers 216 / 1432.
3. Hand-checked the push/pull bijection algebra (`fld(v-o,2) = -p(u,o)` via
   `v-o = u-2p`), the tight in-box parent bound `2^(L-1)-1`, the scalar
   `s^-(n+m+1)` entrywise homogeneity behind the `Lambda(s)` factorization, and
   that the script's epsilon formula matches the shipped
   `constant_p_stencil_bound` (analytic factor, unit strength) exactly.

Assessment against the review priorities: (1) all seven deliverables are met as
stated, including both near radii, the level-2 audits for both, and the
`(level, offset)`-free operator-table conclusion; (2) correctness is proved
analytically and verified numerically, with the downward-monotonicity lemma
supplied explicitly as required; (3) the cost model, crossovers, and the honest
6.63x constant are recorded; (4) the verifier is deterministic, uses the
production dense-operator builder for scaling (32 cases, rtol 1e-13, LH
asymmetric `n_chi±1` diagonals — a genuine refinement beyond the task's ask),
and embeds no expected matrix entries; (5) no `src/` change was introduced;
(6) artifacts are concise and readable.

Notable and correct judgment call: the preliminary occupancy-dependent
reduction ratios (64x/396x/2579x classes, 23x/183x/1465x routes) were audited
and *not* promoted — published `024b` CSVs contain no cell/class/route fields,
so the data reports exact seed-24025 occupancy reconstructions with route
comparisons explicitly labeled `routes_uniform_expectation_not_measured`, and
occupancy-independent dense-lattice anchors (35.9x/235.4x/1588.2x) as the firm
figures. `026`/`027` must record actual per-level routes/class occupancies, as
already noted.

Minor, non-blocking observations (no change requested): the `campaign_audit`
rows reuse the dense-lattice class ratio rather than an occupancy-conditioned
nonempty-class count (an exact count needs an O(C^2) offset-set enumeration;
`026` will measure the real thing), and the epsilon helper restates the
production bound formula instead of calling it (verified to match exactly).

Row `026` is unblocked.
