# 008f Theory Radix-Sort Clustering

## Objective

Derive a radix-sort (Morton / Z-order) clustering of source and target bodies
suited to large `N` and GPU execution, producing **uniform-grid cells** that
support translation-invariant M2L stencils. This replaces the recursive octree
construction (`src/tree.jl`) for the new high-throughput path; the legacy octree
is retained for the dynamic-`P` path.

This task supplies the cell geometry that the constant-`P` interaction-list
stencil in `008d` operates over. It was added by the `2026-06-13` re-plan
addendum recorded in `008b`. It is a Theory Phase task: it blocks every
Implementation task under the standard hard phase gate.

## Dependencies

- `008b-implementation-replan.md`
- `007-theory-coefficient-buffer-layout.md`

## Required Reading

- `START_HERE.md`
- Approved dependency task files listed above
- Current production tree / sorting code (read-only, no edits):
  - `src/tree.jl`: recursive octree construction via pigeonhole sort
  - `src/containers.jl`: `Branch`, `Tree`, buffer layout
  - `src/interaction_list.jl`: geometric MAC and list construction

## Scope / Deliverables

- **Morton / Z-order key construction.** Defined in
  `theory/radix-sort-clustering.md`: bit interleaving of quantized integer cell
  coordinates at grid depth `ell`.
- **Radix sort of bodies by key.** Defined in
  `theory/radix-sort-clustering.md`: stable key sort, sorted body permutation,
  inverse permutation, and occupied-cell range compression.
- **Uniform-grid cell geometry.** Defined in
  `theory/radix-sort-clustering.md`: root cubic domain, depth `ell`, cell
  half-width `w`, bounding radius `rho = w * sqrt(3)`, centers, and integer
  offsets.
- **Coexistence with the legacy octree.** The chosen design is a parallel future
  container, tentatively `RadixGrid`, not a `Branch` / `Tree` compatibility
  layer. Legacy `Tree` remains the adaptive dynamic-`P` structure. `RadixGrid`
  provides only interface-level compatibility for downstream queries of cells,
  centers, body ranges, expansion indices, and offset classes.
- **Relationship to `008d`.** The uniform-grid cell geometry defines the integer
  center-to-center offsets and separation ratios over which `008d`'s conservative
  error bound and translation-invariant stencil are evaluated.
- **Adaptive-cell variants.** Fixed-depth uniform grids, level-by-level radix
  grids, and fully adaptive leaf-size Morton trees are analyzed in the theory
  artifact. The selected first path is fixed-depth uniform grids; adaptive
  variants are deferred until benchmarks justify the added irregularity.

## Artifacts or Production Surface

This is a Theory Phase task. It must not modify production code under `src/`.

Artifacts:

- `theory/radix-sort-clustering.md` — derivation and clustering specification
- `scripts/radix_sort_clustering_verify.jl` — standalone deterministic
  verification script
- `data/radix_sort_clustering/verification_summary.md` — generated summary

## Verification

Run:

```sh
julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/radix_sort_clustering_verify.jl
```

The verifier confirms key construction, half-open quantization with upper-bound
clamping, stable original-index tie behavior, occupied-cell range compression,
inverse-sort round trip, cell geometry, and offset-class consistency on
grid-aligned, boundary/tie, clustered, sparse-occupancy, and fixed-seed random
point sets.

Confirm no production `src/` code changed during this Theory task.

The generated summary is:

```text
MATRIX_OPERATOR_REFACTOR/data/radix_sort_clustering/verification_summary.md
```

Production code note: no `src/` files are intentionally changed by this task.

## Approval Notes

**Status: Approved.** Clear-context approval performed by a separate reviewing
agent (not the completing agent), reading only `START_HERE.md`, this task file,
and the three listed artifacts.

- **Hard Phase Gate — PASS.** `git status --short src/` and `git diff --stat --
  src/` are both empty; no production code changed. All three artifacts are
  untracked files under the allowed Theory dirs (`theory/`, `scripts/`, `data/`),
  and the verifier imports no production FastMultipole code.
- **Dependencies satisfied.** Blockers `008b` and `007` are both Done and
  Approved. The intentional f-before-d ordering is honored (008f precedes the
  008d row that depends on it).
- **Scope / deliverables — all present.** Morton key construction; stable radix
  sort with `perm`/`invperm` and occupied-cell range compression; uniform-grid
  geometry (`G=2^ell`, `w=h0/G`, `rho=w*sqrt(3)`, centers, integer offsets);
  `RadixGrid` coexistence with the legacy `Tree` at interface level only (no
  parent/child topology assumed downstream); explicit 008d relationship through
  integer offset classes; adaptive-variant analysis with fixed-depth uniform
  grids selected and level-binned / fully-adaptive Morton variants analyzed and
  deferred.
- **Correctness verified.** Morton samples (`(3,5,6)@ell=3 -> 427`,
  `(7,7,7) -> 511`) reproduce by hand; theory and script agree exactly on
  `Delta=2h0/G`, `w=Delta/2`, `rho=w*sqrt(3)`, and
  `center = x_min + Delta*(coord+1/2)`; offset identities `r_vec/w = 2d` and
  `r/rho = 2*norm(d)/sqrt(3)` are algebraically consistent with `r_vec = Delta*d`;
  half-open quantization with upper-face clamping is correct.
- **Verifier reproduced — PASS.** Re-ran
  `julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/radix_sort_clustering_verify.jl`;
  output `radix_sort_clustering_verify: PASS`, confirming the committed summary
  is current. All 5 point sets (grid-aligned, boundary/tie, clustered, sparse,
  fixed-seed random) pass key construction, boundary clamping, stable ties, range
  compression, inverse round trip, geometry, and offset-class checks.
- **Non-blocking note for implementation.** The verifier exercises
  quantization/clustering against a fixed root domain (center `(0,0,0)`,
  half-width `1`). The bounding-box -> cubic root-domain fitting (`h0 >= b`,
  degenerate `b=0` fallback) is specified as deterministic implementation policy
  but is not numerically exercised here; the implementation task that builds
  `RadixGrid` should add a domain-fitting check.
