# Hierarchical Rigid M2L Stencil

## Geometry and the near family

At level \(L\), a radix cell has half-width \(h_0/2^L\), width
\(w_L=2h_0/2^L\), and equal-cell bounding radius
\(\rho_L=\sqrt3h_0/2^L\). For integer cell offset \(o=T-S\), the legacy
equal-cell MAC accepts precisely when

\[
 \|o\|w_L\theta>2\rho_L,\qquad\text{or}\qquad
 \|o\|^2>{3\over\theta^2}.
\]

Thus the rejected, direct-work family is level invariant:

\[
 N_q=\{o\in\mathbb Z^3:\|o\|^2\le q\},\qquad
 q=\left\lfloor{3\over\theta^2}\right\rfloor .
\]

For a fixed integer \(q\), the exact parameter interval is

\[
 \sqrt{3/(q+1)}<\theta\le\sqrt{3/q}.
\]

The two required members are \(|N_3|=27\) and \(|N_{12}|=179\).
Moreover, \(\|o\|_\infty\le1\) iff \(\|o\|^2\le3\), so \(q=3\) is exactly
the classic touching-neighbor stencil. The \(q=12\) member contains the
task-024b \(\theta=0.5\) endpoint.

## Source-major phase table

Use the repository convention \(o=T-S\). Let the source phase be
\(u=S\bmod2\). Since \(S=2\operatorname{fld}(S,2)+u\),

\[
 p=\operatorname{fld}(T,2)-\operatorname{fld}(S,2)
   =\operatorname{fld}(u+o,2)
\]

componentwise. The push list is

\[
 V_{\rm push}(u)=\{o:o\notin N_q,\ p(u,o)\in N_q\}.
\]

All eight phases have the same cardinality. Direct enumeration gives:

| \(q\) | near | each phase | phase union | maximum \(\|o\|_\infty\) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 27 | 189 | 316 | 3 |
| 12 | 179 | 1253 | 1740 | 7 |

For the pull view exchange source and target. Then \(o'=-o\), the new source
phase is \(v=(u+o)\bmod2\), and
\(\operatorname{fld}(v-o,2)=-p(u,o)\). Since \(N_q=-N_q\), membership is
unchanged. This is a bijection, proving push/pull equivalence.

## Coarse levels need no special branch

The floor identity above is exact, including negative offsets. For in-box
coordinates at level \(L\), each parent coordinate lies in
\([0,2^{L-1})\), so \(\|p\|_\infty\le2^{L-1}-1\). At level 1, the
\(2\times2\times2\) grid has \(\|o\|^2\le3\), so it has no separated pair
for either radius. At level 2, \(\|p\|_\infty\le1\), hence
\(\|p\|^2\le3\). Every separated level-2 pair therefore has a near parent
for both \(q=3\) and \(q=12\); the ordinary phase table applies without a
level-1 or level-2 exception.

The \(q=3\) boundary is structural. At \(\|p\|^2=3\), the production
constant-\(P\) bound has \(c=2\) and returns `Inf`, so the offset is near for
every finite epsilon regardless of the later inclusive `bound <= epsilon`
test.

## Exact-once ordered-pair coverage

For an ordered occupied leaf pair, let \(o_L\) be the offset of its level-\(L\)
ancestors, with levels increasing downward. If its leaf offset is near, emit it
once as direct work and nowhere in M2L.

Otherwise define the first separated ancestor

\[
 L_*=\min\{L\ge2:o_L\notin N_q\}.
\]

It exists because the leaf is separated. Its parent is near by minimality, so
\(o_{L_*}\in V_{\rm push}(u)\) and the pair is emitted at \(L_*\). It was not
emitted above \(L_*\), because those offsets are near.

It remains to exclude emission below \(L_*\). For every component of a child
offset \(o\) and parent offset \(p\),

\[
 |o_k|\ge2|p_k|-1.
\]

Minimizing the child norm over non-near parents gives
\(\min\|o\|^2=9\) for \(q=3\) and \(34\) for \(q=12\), both strictly outside
the corresponding near set. Separation is therefore downward monotone: once
the parent is separated, every descendant offset is separated. A lower level
cannot have a near parent and hence cannot re-enter a V-list. This proves
exactly-once coverage of every ordered pair.

In a finite box, a phase-table offset whose target leaves the domain is simply
dropped by the bounds lookup. That removes a nonexistent pair; it cannot create
a duplicate. The generated coverage evidence checks full depth-3 and depth-4
boxes and sparse and boundary-truncated occupancies independently.

## One operator table for every level

For the scalar Laplace M2L operator, a source coefficient of degree \(m\), a
target coefficient of degree \(n\), and the Green function's \(r^{-1}\)
homogeneity give degree \(s^{-(n+m+1)}\). With
\(\Lambda_n(s)=s^{-n}\),

\[
 K(sr)=s^{-1}\Lambda(s)K(r)\Lambda(s).
\]

Direction angles depend only on the integer offset and are level invariant.
Since adjacent levels use powers of two, the diagonal factors are exactly
representable in binary floating point.

The Lamb–Helmholtz channels have distinct physical homogeneities. In stacked
`[phi; chi]` degree-major order, define

\[
\Lambda_t(s)=\operatorname{diag}
  (s^{-n_\phi},s^{-(n_\chi+1)}),\qquad
\Lambda_s(s)=\operatorname{diag}
  (s^{-n_\phi},s^{-(n_\chi-1)}).
\]

Then the exact production law is

\[
 K_{\rm LH}(sr)=s^{-1}\Lambda_t(s)K_{\rm LH}(r)\Lambda_s(s).
\]

The shifted chi exponents account for the curl-like target channel and its
source gauge homogeneity; using the same diagonal on both sides is incorrect.
The verifier builds the production dense matrices at axial, equatorial, and
generic offsets and confirms both laws for binary scale factors at relative
tolerance \(10^{-13}\). Scalar and LH implementations can therefore store at
most 316 or 1740 offset-only matrices, rather than a separate table per level.

## Cost and accuracy constants

For \(C\) occupied leaves with mean leaf occupancy \(b=n/C\), direct work is
approximately \(|N|n b\). Hierarchical M2L has at most
\((8/7)|V|C\) routes over a full octree, hence total work is linear when \(b\)
is held constant. The shipped flat far field instead costs \(O(C^2)\);
balancing it with direct work yields \(O(n^{4/3})\).

The crossover inequality

\[
C^2>{8\over7}|V|C
\]

gives \(C>216\) for \(q=3\) and \(C>1432\) (about 1430) for \(q=12\).
The flat path is legitimately cheaper below these thresholds and should remain
selectable.

The \(q=12\) constants are not classic-FMM constants: \(1253/189=6.63\) for
the phase list and \(179/27=6.63\) for direct neighbors. At the balanced leaf
occupancy, cost scales like \(\sqrt{|N||V|}\), so retuning occupancy does not
remove this factor.

For the production analytic bound at code order 3, the first classic accepted
shell is \(\|o\|^2=4\). Its required epsilon is approximately 238.2 times the
lower compatible epsilon endpoint for the \(q=12\), \(\theta=0.5\) stencil at
every level. (The campaign selected an interior midpoint, so this is an
endpoint-to-endpoint accuracy price, not a ratio to that selected midpoint.)
The classic upper endpoint is
unbounded because the \(\|o\|^2=3\) shell has \(c=2\) and infinite bound.
The generated level CSV records the closed lower/open upper epsilon endpoints.

### Flat-versus-hierarchical audit

Every reduction must name its occupancy basis. On fully occupied dense lattices,
the exact class anchor for \(q=12\) is

\[
{(2^{L+1}-1)^3-179\over(L-1)1740},
\]

which is 35.9, 235.4, and 1588.2 at depths 5, 6, and 7. These are
occupancy-independent finite-domain comparisons.

The preliminary 64/396/2579 class and 23/183/1465 route reductions were
occupancy-dependent projections, not measurements in the published 024b CSV:
that schema contains `n` and `ell` but no cells, classes, or routes. The
verifier reconstructs exact leaf and ancestor-level occupancies for the actual seed 24025,
fixed `[-0.01,1.01]^3` domain, and campaign cases
`(n,L)=(31623,5),(316228,6),(1000000,7)`. It labels those values
`measured_leaf_occupancy_reconstructed_seed24025`. Route reductions conditioned
only on those occupancies are explicitly labeled
`routes_uniform_expectation_not_measured`; they must not be cited as campaign
telemetry. Tasks 026/027 must record actual per-level routes and class
occupancies before replacing these estimates.

## Reproducibility

Run:

```sh
julia --project MATRIX_OPERATOR_REFACTOR/scripts/hierarchical_rigid_stencil_verify.jl
```

The script writes five machine-readable CSV files and a concise summary under
`data/hierarchical_rigid_stencil/`. Running it twice must produce byte-identical
outputs.
