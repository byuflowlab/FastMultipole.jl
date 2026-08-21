# Radix-Sort Clustering

## Scope

This artifact specifies the clustering geometry for the new large-`N`,
GPU-oriented path. It is theory-only and does not modify production code. The
legacy recursive `Tree` remains the adaptive dynamic-`P` structure; the radix
path uses a separate future container, tentatively named `RadixGrid`.

The design target is a constant-`P` M2L pipeline with translation-invariant
interaction stencils. The radix path therefore prioritizes regular batches,
contiguous body ranges, and offset-class lookup over reproducing the recursive
topology of `Tree` and `Branch`.

## Selected Container: `RadixGrid`

`RadixGrid` is a flat uniform-grid container. It is not required to be layout-,
type-, or topology-compatible with `Tree` or `Branch`.

Required fields for a future implementation are:

- root cubic domain: center `c0`, half-width `h0`, lower corner `x_min = c0 -
  h0`, upper corner `x_max = c0 + h0`;
- grid depth `ell` and grid resolution `G = 2^ell` cells per coordinate;
- uniform cell half-width `w = h0 / G`;
- cell bounding radius `rho = w * sqrt(3)`;
- integer cell coordinates `(i, j, k)`, with each coordinate in `0:G-1`;
- Morton key for each occupied cell;
- occupied-cell table sorted by Morton key;
- sorted body permutation `perm`, such that `points[perm]` is grouped by cell;
- inverse permutation `invperm`, such that `sorted[invperm]` returns original
  order;
- contiguous body ranges into `perm` for each occupied cell;
- optional maps from key or integer coordinate to occupied-cell index.

The legacy `Tree` path remains responsible for adaptive leaf-size behavior,
dynamic expansion order, and current production error machinery. `RadixGrid`
only needs interface-level compatibility with downstream operator code:

- query source and target cells;
- query cell centers, `w`, and `rho`;
- query body ranges for a cell;
- query expansion-buffer indices for a cell;
- query integer offset classes between target/source cells.

Downstream code must not assume a recursive parent/child topology on this path.

## Root Cubic Domain

For a set of bodies, choose a cubic root domain that contains all positions. Let
`b = max(maximum(x) - minimum(x), maximum(y) - minimum(y), maximum(z) -
minimum(z)) / 2`, and choose `h0 >= b` with a small nonnegative padding if
needed. The root center is normally the midpoint of the axis-aligned bounding
box, expanded to the cubic half-width `h0`.

Degenerate domains with `b = 0` must choose a positive fallback half-width so
that quantization is well-defined. The exact fallback is an implementation
policy; it must be deterministic.

## Quantization

At depth `ell`, the grid resolution is:

```text
G = 2^ell
```

The cell width is `Delta = 2h0 / G`, and the cell half-width is:

```text
w = Delta / 2 = h0 / G
```

For coordinate `x`, define the floating grid coordinate:

```text
u = (x - x_min) / Delta
```

The integer cell coordinate is:

```text
i = clamp(floor(Int, u), 0, G - 1)
```

Equivalently, cells are half-open intervals
`[x_min + i*Delta, x_min + (i+1)*Delta)`, except that the global upper boundary
`x == x_max` is clamped into the last cell. This convention prevents a point on
the root maximum face from quantizing outside the grid.

The same rule applies independently to `x`, `y`, and `z`.

## Morton Keys

The Morton key interleaves the bits of the integer coordinates. For coordinates
`i`, `j`, and `k`, and bit index `b = 0:ell-1`, the key contains:

```text
key[3b + 0] = bit b of i
key[3b + 1] = bit b of j
key[3b + 2] = bit b of k
```

The result is a `3ell`-bit unsigned integer. A 64-bit key supports
`ell <= 21`; deeper grids require a wider key or a pair of words.

The key is an ordering and lookup identifier only. The canonical geometric
identifier is the integer coordinate triple `(i, j, k)` because it is the direct
source of centers and offset classes.

## Radix-Sort Clustering

For each body:

1. Quantize the position to integer cell coordinates.
2. Compute its Morton key.
3. Record the original body index.

Sort bodies by Morton key. Ties must preserve original-index order; a stable
radix sort satisfies this directly, and a non-stable sort must include the
original index as a secondary key. The result is `perm`, a permutation of
original body indices.

After sorting, compress equal adjacent keys into occupied-cell records:

```text
OccupiedCell = (key, coord, center, body_range, expansion_index)
```

where `body_range` is a contiguous range into `perm`. Empty cells do not appear
in the occupied-cell table.

The inverse permutation is defined by:

```text
invperm[perm[s]] = s
```

for sorted position `s`. This is the same logical contract used by the legacy
tree sort: sorted body data can be scattered back to original body order through
the inverse permutation.

## Cell Geometry

For integer coordinate `(i, j, k)`, the cell center is:

```text
center = x_min + Delta * ((i, j, k) + (1/2, 1/2, 1/2))
```

Every cell at a fixed depth has:

```text
box half-widths = (w, w, w)
rho             = w * sqrt(3)
```

For two same-depth cells with target coordinate `t` and source coordinate `s`,
the center displacement is:

```text
r_vec = Delta * (t - s)
```

The integer offset `d = t - s` is therefore sufficient to identify the
translation direction and distance class. The physical distance is:

```text
r = Delta * norm(d)
```

This is the geometric input provided to the conservative constant-`P` stencil in
task `008d`.

## Offset Classes

An offset class is the integer triple:

```text
d = target_coord - source_coord
```

For a fixed-depth uniform grid, every source/target pair with the same `d` has
the same normalized geometry:

```text
r_vec / w = 2d
r / rho  = 2 * norm(d) / sqrt(3)
```

The M2L stencil selected by `008d` can therefore be represented as a set of
accepted offsets. Queue construction becomes:

1. Iterate occupied target cells.
2. For each accepted offset `d`, compute `source_coord = target_coord - d`.
3. If that source coordinate is occupied, enqueue the target/source cell pair in
   the batch for offset `d`.

This keeps operator lookup translation-invariant: all pairs in one offset batch
share the same rotation/translation class at the selected `P`.

Nearfield and self interactions are the complement handled by direct evaluation
or other nearfield kernels, according to the policy selected by downstream
implementation tasks.

## Source/Target Grids

For source-only or target-only clustering, use the same root-domain convention
so offsets are meaningful across source and target cells. For a source/target
solve, source and target grids used by one stencil must share:

- root center;
- root half-width;
- grid depth;
- cell width;
- coordinate quantization convention.

If separate source and target domains are desired later, the pair no longer has
a single integer offset-class stencil without an additional coordinate
normalization layer. That variant is out of scope for the first radix path.

## Adaptive-Cell Variants

### Fixed-depth uniform grid

The selected first path is one fixed depth for all occupied cells. Its strengths
match the constant-`P` stencil goal:

- one cell size and one bounding radius;
- one translation-invariant offset set;
- direct coordinate-to-cell lookup;
- regular M2L batches by offset class;
- no parent/child traversal during interaction-list construction.

The main risk is occupancy imbalance: clustered data can put many bodies in a
single cell or leave most cells empty. Empty cells are cheap because the
occupied-cell table is compressed, but dense cells increase nearfield/direct
work.

### Level-by-level radix grids

A level-binned variant could build several independent uniform grids at
different depths and choose a depth by occupancy or error target. This keeps
translation-invariant stencils within each level, but introduces cross-level
interactions. Cross-level pairs require either separate stencils per level pair
or fallback nearfield/direct handling. It also fragments batches by level.

This is a plausible second step if fixed-depth occupancy is pathological, but it
is not the first implementation target.

### Fully adaptive leaf-size Morton tree

A fully adaptive radix/Morton tree is feasible: sort by Morton key, split ranges
until a leaf-size rule is satisfied, and store leaves as variable-depth Morton
prefixes. This can reproduce much of the occupancy behavior of an octree
without pointer-heavy insertion.

It weakens the main performance goal for the new path:

- mixed levels break a single translation-invariant stencil;
- queue construction must reason about level pairs and variable cell radii;
- parent/child traversal reappears in interaction construction;
- batches become smaller and more irregular;
- expansion storage and operator lookup require level-aware indexing.

Conclusion: fully adaptive leaf-size behavior is possible with radix sorting,
but it should be deferred. The first radix path should use fixed-depth uniform
grids; adaptive or level-binned variants should be revisited only if benchmarks
show that occupancy imbalance dominates runtime or memory.

## Verification Requirements

The standalone verifier checks:

- key construction by explicit bit interleaving;
- half-open quantization and upper-boundary clamping;
- stable original-index behavior for equal keys;
- compression of sorted keys into occupied-cell records and contiguous body
  ranges;
- inverse-sort round trip;
- cell center, half-width, and `rho = w * sqrt(3)`;
- offset-class consistency between integer coordinates and physical centers;
- representative point sets: grid-aligned, boundary/tie cases, clustered,
  sparse occupancy, and fixed-seed random.

The verifier does not import or mutate production FastMultipole code.
