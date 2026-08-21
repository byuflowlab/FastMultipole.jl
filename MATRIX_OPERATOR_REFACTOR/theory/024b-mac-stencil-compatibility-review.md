# Independent Review: 024b MAC / Stencil Compatibility

## Scope

This is a clean-context review of the 024b plan, the compatibility derivation,
the verifier and its CSV output, and the production implementations of the
legacy Barba MAC and radix constant-\(P\) stencil. No benchmark implementation
or result was used to justify this gate.

## Findings

1. **Literature \(P=4\) maps to code `expansion_order=3`: confirmed.**
   The requested literature convention retains four degrees, \(0,1,2,3\).
   Production storage and operator loops treat the code value as the highest
   retained degree: for example, `initialize_expansion(P)` allocates the
   triangular coefficients through degree \(P\), and the translation/operator
   loops run over `0:P`. Therefore four retained degrees require
   `expansion_order=3`.

2. **The campaign `RadixFMMCache` uses analytic normalization and
   `source_strength=1`: confirmed.**
   With no explicit `policy`, `RadixFMMCache` constructs
   `ConstantPStencilConfig(P, TF(stencil_epsilon); lamb_helmholtz=LH)`.
   `ConstantPStencilConfig` defaults its positional `source_strength` to
   `one(epsilon)` and its `normalization` keyword to `:analytic`. The analytic
   specialization applies a factor of one, whereas `:production` would apply
   \(1/(4\pi)\). The gravitational campaign system reports
   `has_vector_potential=false`, so the scalar \(B_\phi\) expression reviewed
   here is the one used; no Lamb--Helmholtz \(B_\chi\) term is added.

3. **\(h_0=0.51\) follows from the fixed 1.02-wide box: confirmed.**
   For explicit `bounds=(x_min, box_size)`, the production cache sets
   `h0 = box_size / 2`. Thus `box_size=1.02` gives \(h_0=0.51\). Production then
   uses cell half-width \(h_0/2^\ell\), cell width \(2h_0/2^\ell\), and bounding
   radius
   \[
   \rho=\sqrt{3}\,h_0/2^\ell,
   \]
   exactly as stated in the derivation.

4. **Every offset in every campaign finite domain has zero predicate
   mismatches: confirmed.**
   Production classifies the full Cartesian domain
   \[
   [-(2^\ell-1),\,2^\ell-1]^3\cap\mathbb Z^3,
   \]
   and the verifier enumerates that identical domain with identical inclusive
   integer endpoints. I independently enumerated each domain and compared the
   strict legacy squared predicate \( \|o\|^2\theta^2>3 \) with the production
   analytic-bound predicate \(B(o)\le\epsilon_\ell\). The reproduced results
   match the CSV:

   | `ell` | total offsets | accepted | rejected | nearest rejected \( \|o\|^2 \) | nearest accepted \( \|o\|^2 \) | mismatches |
   | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | 2 | 343 | 164 | 179 | 12 | 13 | 0 |
   | 3 | 3,375 | 3,196 | 179 | 12 | 13 | 0 |
   | 4 | 29,791 | 29,612 | 179 | 12 | 13 | 0 |
   | 5 | 250,047 | 249,868 | 179 | 12 | 13 | 0 |
   | 6 | 2,048,383 | 2,048,204 | 179 | 12 | 13 | 0 |
   | 7 | 16,581,375 | 16,581,196 | 179 | 12 | 13 | 0 |

5. **Scaling \(\epsilon\) proportionally to \(2^\ell\) preserves the cutoff:
   confirmed.**
   At fixed integer offset, \(c=2\|o\|/\sqrt{3}\) is independent of depth, while
   the bound is proportional to \(1/\rho\), hence to \(2^\ell\). Scaling
   \(\epsilon_\ell\) by the same factor leaves `B <= epsilon` invariant in
   integer-offset space. The constant rejected count and the identical
   norm-squared boundary at all six depths provide the finite-domain check of
   that conclusion.

## Boundary and interval audit

The legacy implementation accepts only when
\[
R^2\theta^2>(\rho_s+\rho_t)^2.
\]
For equal radii this is the strict condition \(R\theta>2\rho\). With radix cell
offset \(o\), \(R/\rho=2\|o\|/\sqrt{3}\), so at \(\theta=0.5\) the shell
\(\|o\|^2=12\) is rejected and the next representable shell,
\(\|o\|^2=13\), is accepted.

The radix implementation accepts non-strictly, `bound <= epsilon`. Since the
bound decreases with separation over the admissible region, the exact
compatible interval is therefore
\[
B_{13}\le\epsilon<B_{12}.
\]
The verifier implements this as `lower <= epsilon < upper`, so the lower
endpoint is correctly closed and the upper endpoint correctly open. Its chosen
midpoints are strictly interior and reproduce the full-precision endpoints in
the CSV. Rounding those midpoints to Float32 also remains safely inside the
same intervals.

The continuous substitution is algebraically correct:
\[
B=\frac{2A}{\rho(c-2)}\left(\frac{1}{c-1}\right)^{P_{\rm code}+1},
\qquad c=\frac{2}{\theta},
\]
gives
\[
\epsilon_{\rm compat}
=\frac{A}{\rho}\frac{\theta}{1-\theta}
\left(\frac{\theta}{2-\theta}\right)^{P_{\rm code}+1}.
\]
For \(P_{\rm code}=3\), \(\theta=0.5\), \(A=1\), and \(h_0=0.51\), this reduces
to \(2^\ell/(81\cdot0.51\sqrt{3})\), as documented.

The review also agrees with the stated limitation: this establishes an exact
same-level, equal-radius geometric classification, not equality of adaptive
global errors.

## Result

**Result: Approved**
