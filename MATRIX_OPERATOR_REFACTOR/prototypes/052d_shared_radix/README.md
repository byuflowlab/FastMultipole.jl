# 052d shared-radix dual-tree list prototype

Host-side Julia prototype validating the LIST-GENERATION machinery of the
Phase 2b-revised design (see `../../052d-plan-2026-08-26.md`, section
"Phase 2b-revised design — shared-radix dual-tree device FMM (2026-08-27)"):
all octrees are sparse subsets of ONE implied global radix grid; cells are
(level, Morton code) pairs; centers/half-widths come from index arithmetic
only; cross-tree M2L and near-field lists come from an adaptive dual traversal
over two sparse index sets.

Plain Julia, Base + StaticArrays only. Does not touch FastMultipole source.

## Files

- `SharedRadix.jl` — implementation module: implied global `Grid` (center +
  half-width; level-l half-width = `h/2^l`), 21-bit-per-axis Morton
  encode/decode (UInt64), on-the-fly `cell_center`/`cell_halfwidth`, sparse
  adaptive `build_tree` (Morton sort + BFS split of cells with more than
  `leaf_size` points; empty children skipped; coincident points force an
  oversized leaf at `MAX_LEVEL`), `dual_traversal` (below), and
  `exact_mac_leq` (exact integer-arithmetic MAC check).
- `validate.jl` — correctness suite on small random + pathological
  configurations: exhaustive pair-partition (every source-target point pair
  covered exactly once), exact-arithmetic MAC validity, and list determinism
  under input permutation.
- `check_production.jl` — partition checks at the real 36,752 x 241,986
  step-472 shape: exact covered-pair count identity, sampled exact per-target
  coverage, exact MAC over the full M2L list.
- `production_run.jl` — real-geometry list build (panel centroids -> particle
  cloud), leaf_size x theta sweep, and device cost model against the
  0.6 s/step gate. Falls back to a clearly-labeled synthetic stand-in if the
  snapshot directory is missing.
- `*.log` — run logs from 2026-08-28 (results quoted in
  `../../052d-prototype-report-2026-08-28.md`).

## Traversal / MAC rules

- MAC: Barba-style `r_S + r_T <= theta * dist`, `r = sqrt(3) * halfwidth(level)`
  of the RAW grid box (no shrinking, per the design), default `theta = 0.5`.
- On rejection with both cells leaves -> near-field pair; otherwise DESCEND
  THE CELL WITH THE LARGER GRID HALF-WIDTH (the shallower level; ties descend
  the source; if the larger cell is a leaf, descend the other). Accepted M2L
  pairs may therefore sit at different levels; the recursion always replaces
  one side by the disjoint cover of its children, so it partitions
  points(src) x points(tgt) exactly regardless of level heterogeneity.
- Shared-grid distances are quantized, so exact MAC boundary ties occur
  (e.g. `0.6 * 15/128 = 9/128` exactly for a Δlevel-3 diagonal pair). The FP
  traversal may classify a tie either way (harmless for the partition and for
  the theta error bound); validation therefore checks the non-strict MAC in
  exact integer arithmetic via `exact_mac_leq`.

## Run

```bash
JULIA_NUM_THREADS=4 julia --project=/Users/ryan/Dropbox/research/projects/FastMultipole validate.jl
JULIA_NUM_THREADS=4 julia --project=/Users/ryan/Dropbox/research/projects/FastMultipole check_production.jl
JULIA_NUM_THREADS=4 julia --project=/Users/ryan/Dropbox/research/projects/FastMultipole production_run.jl
```

(`check_production.jl`/`production_run.jl` read the step-472 snapshot binaries
from the scratchpad path hard-coded at the top of each script.)
