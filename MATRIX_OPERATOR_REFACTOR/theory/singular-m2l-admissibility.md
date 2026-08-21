# Singular-M2L admissibility for regularized U/J

## Result

**NO-GO.** A sigma class can satisfy the regularization geometry and scalar
constant-P screen, but this row found three independent blockers:

1. the registered shipped `q=12`, combined-U/J `rho=4.252` rotor lists contain
   zero sigma-demoted routes at both 100,000 and 1,000,000 particles;
2. the proposed low-dimensional collapse map fails its registered agreement
   tolerance; and
3. the existing task-025 constant-P bound controls scalar potential, not the
   delivered U and all nine J derivatives with live phi/chi budgets.

Thus no production successor or default change is justified.

## What can be proved

For a fixed predeclared source class (c), let
(sigma_s\le\sigma_c), let (g) be the source/target AABB gap, and let
({\cal M}_P) be a singular constant-P M2L. In any common output norm,

\[
 \|F_c^\sigma-{\cal M}_P F_c^0\|
 \le \|F_c^\sigma-F_c^0\|+
      \|F_c^0-{\cal M}_P F_c^0\|.
\]

This triangle-inequality composition is additive and assumes no cancellation.
Monotonicity gives (r/\sigma_s\ge g/\sigma_c). From 031a,

\[
 E_U(\rho)=\bar g(\rho),\qquad
 E_J(\rho)=\bar g(\rho)+{\rho\over2}g'(\rho),\qquad
 \rho=g/\sigma_c.
\]

Rotation invariance makes the J envelope cover FLOWVPM's nine stored entries
`J11,J21,J31,J12,J22,J32,J13,J23,J33`. The pointwise half-budget radii are
4.211 for U and 4.789 for J.

The shipped partitioned policy is statistical instead: 3.668 is the U-only
accumulated-RMS radius and **4.252 is the combined U/J accumulated-RMS radius**.
The selector/census use 4.252, consistently with production. It is not called a
pointwise bound; it inherits 031a's incoherent-tail assumption and existing
sampled-direct validation.

Task 025 supplies the production-normalized scalar bound

\[
 B_P={1\over4\pi}{2A\over r_c(C-2)}
       \left({1\over C-1}\right)^{P+1},\qquad C=R/r_c>2,
\]

where (A) is the class absolute source-strength sum. The corresponding
Lamb--Helmholtz envelope would require measured (A_\phi,A_\chi):

\[
 B_{LH}=B_P(A_\phi)+(1+2R)B_P(A_\chi).
\]

Those live channel budgets are not present in the offline refresh data, and a
scalar-potential absolute bound cannot simply be added to the dimensionless
relative U/J tails. A valid delivered-output proof must first convert both
terms to the same absolute U/J norm and bound derivatives of the P-truncated
local expansion. This row does not invent a chi/phi ratio or claim that the
existing scalar bound supplies that missing result. Therefore the scalar
screen below is a ceiling only and `accuracy_certified=false` forces direct
fallback in every selector row.

## Numerical boundary checks

`boundary_checks.csv` tests both sides of the pointwise 4.211 U and 4.789 J
radii, P4/P8, Float32/Float64, phi/chi-labelled scalar compositions, U, and all
nine J entries. The U/J columns evaluate exact regularized and singular
formulas over deterministic aligned/transverse orientations in the named
precision. Separately, the scalar columns evaluate an exact collinear local
geometric series through P4/P8 and its task-025 remainder. Each measurement is
compared only with a like-unit bound. The two unlike quantities are
deliberately not summed; their missing delivered-output composition is a
NO-GO finding.

## Classes and exact-once ownership

The predeclared global logarithmic bins `{1,2,4,8}` assign every source body to
exactly one class. A class-filtered P2M represents a complete leaf/class, never
a target-dependent subset. For each sigma-demoted terminal U pair, the compact
oracle paints every ordered target/source body pair into either candidate M2L
or residual direct. Its disjoint sum equals the original demoted-U ownership
matrix with zero omissions and duplicates. The imported task-038 painter
separately proves whole-list U/V/W/X exact-once coverage. Accepted contributions
would accumulate into existing target locals and free-ride on L2B.

## Collapse test and map

Uniform density suggests

\[
 w/\sigma\sim K^{1/3}/\beta.
\]

The registered reduced map uses `w/sigma`, leaf-local spread, class count, and
P. `admissibility_map.csv` compares it against the independent route census for
the full synthetic sweep and matched `(K,beta)=(16,1.5),(128,3)` checks at two
counts. It records occupied fill, a domain-gradient diagnostic, and gap-bin
cardinality to test the proposed missing variables. The reduced map still fails
`max(0.02 absolute, 10% relative)`.

An exact answer requires the full strength-weighted directional AABB-gap/class
histogram. Replaying that histogram is the route census itself, not an
independent low-dimensional collapse, so this report does not relabel it as a
successful predictive map. The artifact is therefore a documented failed map
and another promotion-gate failure.

The rotor illustrates why global spread is insufficient: its global sigma
range is about 17.94x, while its 90th-percentile leaf-local spread is about
1.010 at 100k and 1.001 at 1M.

## Census and pricing

`sigma_class_m2l_census.jl` imports task-038's tree, list builder, sigma upward
pass, and painter. It loads the canonical DJI-9443 iterator verbatim from
`benchmark_033_common.jl`. Both rotor counts emit exactly the requested number
of particles and fully materialize their tree/list/class route census; the
million-particle field is never evaluated. The synthetic sweep materializes
all 36 combinations at exactly `n=100_000`:

- beta `{1.5,2,3}`;
- global spread `{1,3,10,18}`; and
- uncorrelated, Morton-leaf-band, and domain-scale sigma fields.

Demotion attribution follows the recorded `Lists.demoted` ancestry exactly.
Under `q=12,rho=4.252`, both rotor counts have zero demotions. Ordinary
positive-gap geometric U pairs are not mislabeled as demoted work.

The ceiling price includes residual direct work, class-filtered P2M,
refresh/scatter, metadata, route grouping, actual normalized offset classes,
and bounded route/operator capacity. It records proposed nearfield and complete
solve deltas separately from the selected fallback deltas. Existing 041c/041a
rates carry their uncertainty; because the accuracy certificate is absent and
the rotor material fraction is zero, all rows select direct with zero predicted
regression.

## Reproduction

```text
JULIA_NUM_THREADS=4 julia --project=. \
  MATRIX_OPERATOR_REFACTOR/scripts/sigma_class_m2l_census.jl
```

Compact checksummed artifacts are under `data/sigma_class_m2l/`.
