# Independent review — 052d shared-radix dual-tree prototype (2026-08-28)

Skeptical review of `052d-prototype-report-2026-08-28.md` and
`prototypes/052d_shared_radix/` against the "Phase 2b-revised design"
section of `052d-plan-2026-08-26.md` (lines 413–493). Method: close code
reading, independent re-runs of both validation scripts, six new
adversarial test families (`reviewer_tests.jl`, added to the prototype
dir; original files untouched), and an independent audit of the cost-model
arithmetic. All runs: 4 threads, host laptop.

## Verdict summary

| claim | verdict |
|---|---|
| 1. Pair-partition correctness | **CONFIRMED** |
| 2. MAC validity + exact-tie subtlety | **CONFIRMED** |
| 3. Determinism under permutation | **CONFIRMED** (small configs only — see caveat) |
| 4. Device cost model / gate conclusion | **CONFIRMED-WITH-CAVEATS** |

## Claim 1 — pair-partition correctness: CONFIRMED

**Code reading.** The traversal (`SharedRadix.jl:231-262`) maintains the
invariant that the stack always holds a disjoint cover of
points(src\_root) × points(tgt\_root): each pop either emits the pair (M2L
or leaf-leaf near) or replaces exactly one side by its children, and
`build_tree` (`SharedRadix.jl:165-198`) constructs children as contiguous
runs of the sorted code array that cover the parent's range exactly (empty
children are never materialized, and hold no points anyway). Every branch
of the if/elseif ladder is covered: leaf-leaf → near; T leaf, S not →
descend S; neither leaf → descend the shallower (ties → S); S leaf, T not
→ falls through to descend T. Termination is guaranteed because each
descent strictly deepens one cell and oversized cells at `MAX_LEVEL`
become leaves. The cross-level rule is therefore a genuine partition — no
pair reachable by two paths, no pair droppable, including leaves at very
different levels. I found no hole.

**The checker is itself sound.** `coverage_counts` (`validate.jl:24-40`)
increments a full ns × nt matrix over `S.range × T.range` for every list
pair — this IS subtree membership (a `Cell.range` equals its full
subtree's points by construction), so double-counting cannot hide.
I additionally validated the validator (test T4): corrupting the lists by
(a) dropping a near pair, (b) duplicating an M2L pair, and (c) replacing
an M2L pair's source cell by its parent (the classic overlapping-subtree
double-count) — the checker flagged all three (misses > 0 / dups > 0).

**Re-runs.** `validate.jl` re-run: exit 0, output byte-identical to the
committed `validate.log` (15/15 PASS, zero miss/dup/macbad, det ok).
`check_production.jl` re-run against the step-472 snapshot: all three
configs reproduce `covered=8,893,469,472` (= 36,752 × 241,986 exactly —
I recomputed the product independently), `macbad=0`, `bad_samples=0/400`,
exit 0 — matching the committed `check_production.log`.

**Adversarial tests (all PASS, `reviewer_tests.jl` / rerun log
`/tmp/rerun_reviewer.log`).**

- T1: 9³ lattice with points exactly on cell-boundary coordinates,
  src=tgt, θ ∈ {0.4, 0.5, 0.6, 1.0}; and points placed exactly on
  fine-lattice cell corners of the implied grid. Zero miss/dup/violation.
- T3: maximum level disparity — a 1e-15-extent source cloud (forces the
  MAX\_LEVEL oversized-leaf path) inside a 2000-wide target cloud, both
  role orders; single-point source; empty source set. All clean.
- T5: identical source/target sets containing 200 exact internal
  duplicates, offset to coordinates ~1e6 (FP center-arithmetic stress),
  θ ∈ {0.3, 0.5}. All clean.

**Residual caveat (minor).** At production scale the exhaustive matrix is
replaced by count-identity + 400 sampled targets (~0.17% of targets).
A compensating miss+dup of exactly equal pair counts that also avoided all
400 samples could in principle evade this; given the exhaustive small-suite
coverage of the same code paths and my corruption tests, I consider the
risk negligible, but the report's "provably-correct" phrasing slightly
overstates what the production-scale check alone establishes.

## Claim 2 — MAC validity and exact ties: CONFIRMED

`exact_mac_leq` (`SharedRadix.jl:278-290`) is genuinely exact: integer
center offsets in fine-half-width units (odd-integer lattice), Int128
throughout (magnitudes ≤ ~2^47 here, far from overflow), rational θ via
`rationalize` (0.4 → 2/5, 0.5 → 1/2, 0.6 → 3/5 — exact). No floating
point anywhere in the test `3 q² s² ≤ p² |Δ|²`. One asymmetry to be aware
of: the traversal accepts with FP strict `<` at the Float64 θ while
validation checks exact `≤` at the rational θ; a pair strictly between
the two thresholds could in principle be FP-accepted yet exact-rejected —
but on this grid distinct integer MAC quantities differ by ≥1 part in
~5e13, far above Float64 rounding, so the only reachable disagreements
are exact ties, which the non-strict definition covers. Empirically
(test T6): zero strict violations over the full production M2L lists at
leaf 32 and 256, θ = 0.6.

**The tie subtlety is real, not manufactured.** My unit test constructs
the Δlevel = 3, s = 9, |Δ|² = 675 = 3·15² configuration at θ = 3/5:
3·25·81 = 9·675 = 6075 exactly — a genuine tie; one cell nearer fails,
one farther passes. Ties occur in the wild: 64–713 exact ties among
accepted pairs in my lattice configs, 9 in a random cluster-in-wide
config at θ = 0.6, and **489 / 135 ties in the actual production lists**
(leaf 32 / 256, θ = 0.6). The report's demand for an explicit device tie
convention is well-founded — this is not a hypothetical.

## Claim 3 — determinism: CONFIRMED (with a scope note)

Re-run reproduced `det=ok` on all 15 cases. Structurally this must hold:
Morton codes are permutation-invariant per point, the cell decomposition
depends only on the multiset of codes, and the traversal is a
deterministic function of the two cell arrays. Sort-order ties among
equal codes only permute points *within* a cell range, which the
cell-identity comparison correctly ignores. Note determinism was checked
on the small suite only, and only under point permutation (not, e.g.,
under thread count — though the only threaded loop is an elementwise code
computation, so thread-count invariance is structural too).

## Claim 4 — cost model: CONFIRMED-WITH-CAVEATS

**Arithmetic audit — reproduces exactly.** I recomputed every row of the
report's table from the logged `nearpairs` and `n_M2L`:
frac = nearpairs / 8,893,469,472; t\_near = frac × 3.3; t\_M2L =
n\_M2L × 729 / 1e12. All 12 rows match to the printed precision
(0.0065 … 0.0552 s), best 0.0020 s (299×), worst 0.0552 s (11×). The
dense baseline is applied to the correct pair count (the covered-count
identity proves the near+far lists sum to exactly the dense total, and
3.3 s is stated as measured on this same 8.9e9-pair geometry). The
near-field fractions 0.06%–1.7% are correctly computed.

**Caveats, in decreasing order of importance:**

1. **The 3.3 s dense baseline is an external, unverifiable input.** No
   log of that A100 measurement is in the prototype dir; every t\_near
   scales linearly with it. If it is right, the conclusion is safe.
2. **Weakest assumption: proportional near-field scaling.** The dense
   3.3 s kernel is one regular all-pairs sweep; the near-field is 6k–116k
   scattered leaf-pair blocks averaging only ~150–6,000 pair-interactions
   each (e.g. leaf 32/θ 0.5: 9.85e6 pairs over 65k blocks ≈ 151/block).
   Small irregular blocks are launch/bandwidth-bound, not flop-bound; the
   report's 5× sensitivity allowance is plausible for a competently
   batched kernel but not demonstrated. However, at the recommended
   operating point (leaf 32–64, θ 0.5, modeled 0.004–0.007 s) even a 50×
   efficiency penalty stays under the 0.6 s gate; only the worst corner
   (leaf 256/θ 0.4) fails beyond ~10×. The gate conclusion is robust
   where it matters.
3. **M2L flop count is an undercount but doesn't matter.** 729 flops/pair
   at p = 4 understates a real rotation-trick M2L (two Wigner rotations +
   z-translate, possibly dual φ/χ channels for the vortex-ring panels —
   plan open item (d)); even 100× more flops is ~8 ms at the largest
   list, and the report's 1 µs/pair overhead bound (≤ 0.11 s) covers
   realistic per-pair launch/gather costs. Agreed negligible.
4. **Panel B2M is asserted, not measured, and the panel count is 4× the
   design's assumption.** The plan budgeted host B2M for "~9k panels";
   reality is 36,752. The report waves this into "tens of ms" without a
   measurement; a naive host B2M at a few µs/panel could plausibly reach
   ~0.1–0.2 s/step — eroding (though not breaking) the margin, and it is
   per-step (geometry moves every step). Should be measured early in the
   device port.
5. The model covers only the panels→particles leg (the failed-gate leg) —
   consistent with the gate's definition, but the particles→panels cross
   leg adds its own (small) cost on top.

## Scope gaps — what this prototype does NOT validate

The report's "Costs NOT in the model" list is honest on the cost side
(B2M, downward pass, transfers, list upload). Gaps beyond that list:

1. **No numerical accuracy validation.** Lists are validated; no FMM was
   evaluated with them. The claim that p = 4 accuracy "carries over" rests
   on production numbers measured with a *different* list structure
   (same-level stencil, shrunk boxes). Raw-box MAC is conservative (the
   circumscribed radius overestimates content, so the θ bound holds
   a fortiori), and every accepted pair provably satisfies the MAC — so
   the classical error bound applies — but the actual p = 4 error on
   cross-level pairs at θ = 0.5 for this geometry is unmeasured. A cheap
   host check (evaluate a few hundred targets through multipoles vs
   direct) would close this before the device port.
2. **Design-consistency deviation, underplayed.** The plan's consequence 1
   claims cross-tree M2L is "the classical same-level stencil …, reusing
   the self-interaction's stencil machinery." The prototype implements an
   adaptive *cross-level* dual traversal instead (anticipated by open
   item (c), and validated here) — but that means the device port cannot
   simply reuse the existing same-level stencil kernels for list
   generation; it needs this traversal (host-side is fine per the
   measured 0.015–0.05 s) plus device handling of Δlevel ≠ 0 translation
   classes. The report's "no other design flaws surfaced" glosses this.
3. **Grid-coverage hazard for the device port.** The prototype's grid is
   the union box of sources ∪ targets; `point_code` silently *clamps*
   out-of-box points. The plan suggests using "in practice the particle
   box." If any panel ever leaves the particle box, clamping misassigns
   its cell and silently breaks the MAC bound (not the partition). The
   device port must build the root box from the union, or assert
   containment.
4. Untouched plan open items (a) lifecycle hook, (b) per-leg p choice,
   (d) φ/χ channel handling for source+vortex-ring panel B2M — all still
   open, correctly listed in the plan.

## Adversarial test results (new file: `reviewer_tests.jl`)

22/22 checks PASS (log `/tmp/rerun_reviewer.log`): boundary-lattice
partitions at four θ; exact-corner points; engineered exact-tie unit
tests (tie/near/far triplet); ties counted in production lists (489 and
135); 1e-15-vs-2000 extent disparity both role orders; single-point and
empty source; checker-corruption detection (drop/duplicate/parent-swap
all caught); duplicate-laden identical sets at 1e6 coordinate offset;
zero strict exact-MAC violations at production scale.

## Overall recommendation

**Proceed to the device port**, with three conditions:

1. Before or during the port, run the cheap host accuracy spot-check
   (gap 1) — it is the only correctness-adjacent question the prototype
   leaves open, and it is a few hours of work.
2. Measure host panel-B2M cost at the real 36,752-panel count early; it
   is the one unmodeled per-step cost with realistic potential to reach
   0.1 s+.
3. Adopt the exact-≤ tie convention in device integer arithmetic (the
   prototype proves ties are common: hundreds in real production lists).

The list-generation machinery itself is correct by every attack I could
mount, the validation methodology is sound (and survives deliberate
corruption of its inputs), and the gate margin at the recommended
operating point survives order-of-magnitude model pessimism.
