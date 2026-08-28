# 052d — Line-convolved Gaussian filament kernel ("LineGauss")

**Date:** 2026-08-28. **Task:** derive the velocity of a straight vortex
segment with a Gaussian (FLOWVPM erf-blob) core — the physically-consistent
object the shipped `GaussianRegularization` approximates — and show its
deviation from the singular kernel decays with distance to the **segment**
(transverse AND along-line), unlike the shipped families whose deviation is a
function of the infinite-line distance $h$ only.

## 1. Setup

The FLOWVPM Gaussian blob has vorticity kernel and induced-velocity
regularization function

$$
\zeta_\sigma(r) = \frac{1}{(2\pi)^{3/2}\sigma^3} e^{-r^2/2\sigma^2}, \qquad
g(t) = \operatorname{erf}\!\left(\frac{t}{\sqrt{2}}\right)
     - \sqrt{\frac{2}{\pi}}\, t\, e^{-t^2/2},
$$

i.e. a particle of strength $\boldsymbol\alpha$ induces
$\mathbf{u} = -\frac{1}{4\pi} g(|\mathbf r|/\sigma)\,
\mathbf{r}\times\boldsymbol\alpha / |\mathbf r|^3$. Note $g$ is **odd** when
extended to signed argument, and $g'(t) = \sqrt{2/\pi}\, t^2 e^{-t^2/2}$.

A straight segment $P_1 \to P_2$ of length $L$, circulation $\Gamma$, with a
Gaussian core of radius $\sigma$ induces, by superposition of blob kernels,

$$
\mathbf{u}(\mathbf x) = \frac{\Gamma}{4\pi} \int_0^L
\frac{g(|\mathbf x - \mathbf s(l)|/\sigma)}
     {|\mathbf x - \mathbf s(l)|^3}\;
\hat{\mathbf t} \times (\mathbf x - \mathbf s(l))\, dl .
$$

For a closed polygonal ring this is also the velocity of the corresponding
Gaussian-convolved vortex line: endpoint contributions cancel between
adjacent edges.  An isolated open segment is not by itself divergence-free;
the often-stated vorticity $\Gamma\hat{\mathbf t}\int_0^L\zeta_\sigma\,dl$
has endpoint divergence.  For an open edge, the precise interpretation of
the formula above is therefore **Gaussian convolution of the segment
velocity**, not a standalone physically complete vorticity distribution.

Segment coordinates: $\hat{\mathbf t} = (P_2 - P_1)/L$; axial coordinates
$z_1 = \hat{\mathbf t}\cdot(\mathbf x - P_1)$,
$z_2 = z_1 - L$; perpendicular distance to the line $h$, radial unit vector
$\hat{\mathbf n}$, binormal $\hat{\mathbf b} = \hat{\mathbf t} \times
\hat{\mathbf n}$; endpoint distances $R_1 = \sqrt{z_1^2 + h^2}$,
$R_2 = \sqrt{z_2^2 + h^2}$. Along the segment,
$\hat{\mathbf t} \times (\mathbf x - \mathbf s) = h\, \hat{\mathbf b}$
(constant), so with $\zeta = z$-offset from the perpendicular foot,

$$
\mathbf u = \frac{\Gamma\, h}{4\pi}\, \hat{\mathbf b}\; J, \qquad
J = \int_{z_2}^{z_1} \frac{g(R/\sigma)}{R^3}\, d\zeta, \qquad
R = \sqrt{\zeta^2 + h^2}.
$$

## 2. Closed form

**Claim.** The integrand has the elementary antiderivative

$$
F(\zeta) = \frac{1}{h^2}\left[\frac{\zeta\, g(R/\sigma)}{R}
- e^{-h^2/2\sigma^2}\, g(\zeta/\sigma)\right].
$$

**Proof.** Differentiate, using $dR/d\zeta = \zeta/R$,
$\frac{d}{d\zeta}(\zeta/R) = h^2/R^3$, and
$g'(t) = \sqrt{2/\pi}\,t^2 e^{-t^2/2}$:

$$
\frac{d}{d\zeta}\left[\frac{\zeta g(R/\sigma)}{R}\right]
= \frac{h^2\, g(R/\sigma)}{R^3}
+ \sqrt{\frac{2}{\pi}}\,\frac{\zeta^2}{\sigma^3}\, e^{-R^2/2\sigma^2},
$$

$$
\frac{d}{d\zeta}\left[e^{-h^2/2\sigma^2} g(\zeta/\sigma)\right]
= e^{-h^2/2\sigma^2} \sqrt{\frac{2}{\pi}}\, \frac{\zeta^2}{\sigma^3}\,
  e^{-\zeta^2/2\sigma^2}
= \sqrt{\frac{2}{\pi}}\,\frac{\zeta^2}{\sigma^3}\, e^{-R^2/2\sigma^2},
$$

since the 3-D Gaussian factorizes, $e^{-R^2/2\sigma^2} =
e^{-h^2/2\sigma^2} e^{-\zeta^2/2\sigma^2}$ — this separability is what makes
the closed form exist. The two exponential terms cancel, leaving
$F'(\zeta) = g(R/\sigma)/R^3$. $\square$

(Route to discovery: integrate $J$ by parts against the singular
antiderivative $\zeta/(h^2 R)$; the remainder integral
$\int g'(R/\sigma)\,\zeta^2/(\sigma R^2)\, d\zeta \propto
\int \zeta^2 e^{-R^2/2\sigma^2} d\zeta$ is elementary.)

Hence, let $N$ denote the bracket difference.  Expanding every $g$ shows that all
endpoint Gaussian terms cancel exactly, giving the numerically preferable
form

$$
\boxed{\;
\mathbf u = \frac{\Gamma N}{4\pi h}\,\hat{\mathbf b},\qquad
N = \frac{z_1}{R_1}\operatorname{erf}\frac{R_1}{\sqrt2\sigma}
-\frac{z_2}{R_2}\operatorname{erf}\frac{R_2}{\sqrt2\sigma}
-e^{-h^2/2\sigma^2}\left(
 \operatorname{erf}\frac{z_1}{\sqrt2\sigma}
-\operatorname{erf}\frac{z_2}{\sqrt2\sigma}\right)
\;}
$$

Cost for velocity: **4 erf + 1 exp per edge**.  The unreduced expression
appears to require five exponentials; four cancel algebraically.

## 3. Code-quantity (D-form) mapping

With FLOWPanel's per-edge quantities $\mathbf r_1 = P_1 - \mathbf x$,
$\mathbf r_2 = P_2 - \mathbf x$, $\mathbf c = \mathbf r_1 \times \mathbf r_2$,
$\mathbf s = \mathbf r_1 - \mathbf r_2$, $A = |\mathbf c|^2$,
$B = |\mathbf s|^2 = L^2$, $q = \mathbf s\cdot(\hat{\mathbf r}_1 -
\hat{\mathbf r}_2)$: one checks $\mathbf c = hL\,\hat{\mathbf b}$,
$q = L\,(z_1/R_1 - z_2/R_2) \equiv L\tilde q > 0$ (strictly, off-axis), and
$\hat{\mathbf t} = -\mathbf s/\sqrt B$, $z_1 = -\hat{\mathbf t}\cdot\mathbf
r_1$, $h^2 = A/B$. The singular kernel is $\mathbf u_{\rm sing} = \mathbf c\,
q/(4\pi A)$, and the LineGauss kernel is the **scalar modulation**

$$
\mathbf u = \frac{\mathbf c\, q}{4\pi A}\; W, \qquad
W = \frac{N}{\tilde q}, \qquad
N = \frac{z_1}{R_1}\operatorname{erf}\frac{R_1}{\sqrt2\sigma}
-\frac{z_2}{R_2}\operatorname{erf}\frac{R_2}{\sqrt2\sigma}
-e^{-h^2/2\sigma^2}\left(\operatorname{erf}\frac{z_1}{\sqrt2\sigma}
-\operatorname{erf}\frac{z_2}{\sqrt2\sigma}\right),
$$

i.e. in the enum's $D$-form, $D = A/W$. Unlike the shipped families, $W$
depends on $(h, z_1, z_2)$, not on $h$ alone — the price of true compactness.

## 4. Limits

- **$\sigma \to 0$ (or target far from the segment):** $g \to 1$,
  $e^{-h^2/2\sigma^2} \to 0$, so $N \to \tilde q$ and $W \to 1$: the shipped
  **singular kernel exactly**, with exponentially small remainder (§6).
- **$L \to \infty$ (infinite filament):** $z_1/R_1 \to 1$, $z_2/R_2 \to -1$,
  $g(R_i/\sigma) \to 1$, $g(z_1/\sigma) - g(z_2/\sigma) \to 2$, so

$$
W \;\to\; 1 - e^{-h^2/2\sigma^2},
$$

  **exactly the shipped `GaussianRegularization` profile** — the shipped
  kernel is the infinite-line limit of the true convolution. Consequently the
  matched-core-size peak table (phase_00) is unchanged in this limit:
  $u_{\max} r_c = 0.45$, $\max|du/dh|\, r_c^2 = 0.50$ (units $\Gamma/2\pi$).
- **$h \to 0$ on the segment interior:** $N \to g(z_1/\sigma) +
  g(|z_2|/\sigma) - (g(z_1/\sigma) + g(|z_2|/\sigma)) = 0$ like $h^2$, and
  $\tilde q \to 2$, so $\mathbf u \to 0$ on the axis — correct Gaussian-tube
  physics (shipped Gaussian shares this).
- **$h \to 0$ on the line extension:** both $N$ and $\tilde q$ are $O(h^2)$
  with finite ratio; $\mathbf u \to 0$ by symmetry ($\mathbf c \to 0$). See
  §7 for the guarded series.

## 5. Velocity gradient

The field is axisymmetric about the segment line: $\mathbf u = u_\theta(h,
z)\, \hat{\mathbf b}$ with $u_\theta = \Gamma N/(4\pi h)$ (and $z \equiv
z_1$, $z_2 = z - L$). For such a field

$$
\nabla \mathbf u = \frac{\partial u_\theta}{\partial h}\,
\hat{\mathbf b}\hat{\mathbf n}^T
+ \frac{\partial u_\theta}{\partial z}\, \hat{\mathbf b}\hat{\mathbf t}^T
- \frac{u_\theta}{h}\, \hat{\mathbf n}\hat{\mathbf b}^T,
$$

(rows = velocity component, columns = differentiation direction; divergence
zero by inspection). The partials of $N$ are elementary. Writing hatted
variables in units of $\sigma$ ($\hat z_i = z_i/\sigma$ etc.) and $G =
e^{-\hat h^2/2}$:

$$
\frac{\partial N}{\partial \hat z}
= \hat h^2\left(\frac{g(\hat R_1)}{\hat R_1^3}
              - \frac{g(\hat R_2)}{\hat R_2^3}\right),
$$

$$
\frac{\partial N}{\partial \hat h}
= \hat h\left[- \frac{\hat z_1 g(\hat R_1)}{\hat R_1^3}
+ \frac{\hat z_2 g(\hat R_2)}{\hat R_2^3}
+ G\left(\operatorname{erf}\frac{\hat z_1}{\sqrt2}
-\operatorname{erf}\frac{\hat z_2}{\sqrt2}\right)\right],
$$

so with $C = \Gamma/(4\pi\sigma^2)$:

$$
\frac{\partial u_\theta}{\partial h}
= C\left(\frac{1}{\hat h}\frac{\partial N}{\partial \hat h}
       - \frac{N}{\hat h^2}\right), \qquad
\frac{\partial u_\theta}{\partial z}
= \frac{C}{\hat h}\frac{\partial N}{\partial \hat z}, \qquad
\frac{u_\theta}{h} = C\,\frac{N}{\hat h^2}.
$$

**Enum-contract consequence:** the shipped contract is $(D, \nabla D =
\kappa \nabla A)$ — a single scalar $\kappa$ multiplying $\nabla A =
2\,\mathbf s \times \mathbf c$. Because $W$ depends on $z$ as well as $h$,
$\nabla D = \nabla A / W - (A/W^2)\, \nabla W$ with $\nabla W = W_h
\hat{\mathbf n} + W_z \hat{\mathbf t}$ acquires a component along
$\hat{\mathbf t}$ that $\kappa \nabla A$ cannot represent (both $\nabla A$
and $\hat{\mathbf n}$ are $\perp$-plane vectors only when combined with the
$\hat{\mathbf b}$ row structure). Integration would extend the family
contract from one scalar $\kappa$ to two scalars $(W_h, W_z)$ (or
equivalently assemble $\nabla\mathbf u$ directly from the cylindrical form
above). The prototype implements the cylindrical assembly.

## 6. Far-field error decay (the FMM payoff)

The deviation from the singular kernel is $1 - W = (\tilde q - N)/\tilde q$
with

$$
\tilde q - N = \frac{z_1\,\bar g(R_1/\sigma)}{R_1}
             - \frac{z_2\,\bar g(R_2/\sigma)}{R_2}
             + e^{-h^2/2\sigma^2}\big(g(z_1/\sigma) - g(z_2/\sigma)\big),
\qquad \bar g \equiv 1 - g,
$$

where $\bar g(t) = \operatorname{erfc}(t/\sqrt2) + \sqrt{2/\pi}\, t\,
e^{-t^2/2} \sim \sqrt{2/\pi}\, t\, e^{-t^2/2}$. The first two terms decay as
$e^{-R_i^2/2\sigma^2}$ (endpoint distances); the third carries
$e^{-h^2/2\sigma^2}$ times $g(z_1/\sigma) - g(z_2/\sigma)$, which is $O(1)$
only when the segment's axial span straddles or nears the foot of the
perpendicular ($z_2 \lesssim 0 \lesssim z_1$, where $h$ IS the segment
distance) and is itself $\sim e^{-z_*^2/2\sigma^2}$ ($z_*$ = axial distance
to the nearer endpoint) when the target lies beyond the segment ends. In all
cases

$$
1 - W \;\lesssim\; \text{poly} \times e^{-d^2/2\sigma^2}, \qquad
d = \operatorname{dist}(\mathbf x,\ \text{segment}),
$$

closing the along-line channel by construction.  This asymptotic statement
does **not** make the infinite-line Gaussian fixed-point radius a rigorous
finite-segment relative-gradient bound: at $5.898\sigma$ the finite-edge
error can exceed $10^{-6}$.  The dense length/capsule scan in
`k01_validate.jl` instead uses separate conservative radii: $5.90\sigma$
for velocity and $6.25\sigma$ for gradient at tolerance $10^{-6}$.  The
latter scan (22 segment lengths, 181 cap angles, and 101 cylinder positions)
measured a worst relative-gradient error of $3.03\times10^{-7}$.

## 7. Numerical guards (implementation)

For small arguments, $g$ is evaluated without subtraction as

$$
g(t)=\sqrt{\frac{2}{\pi}}\left(\frac{t^3}{3}-\frac{t^5}{10}
+\frac{t^7}{56}-\frac{t^9}{432}+\cdots\right),
$$

and $\psi$ uses its corresponding odd series with $\psi(0)=0$.  On/near the
line ($\hat h^2 \ll \min(\hat z_1^2, \hat z_2^2)$), both $N$ and
$\tilde q$ suffer cancellation. Each endpoint function admits the $h$-series
$\phi(z) \equiv \dfrac{z\, g(R/\sigma)}{R} - e^{-h^2/2\sigma^2} g(z/\sigma) =
h^2 \psi(z/\sigma)/\sigma^2 + O(h^4)$ with the odd function

$$
\psi(\hat z) = \sqrt{\tfrac{2}{\pi}}\,\frac{\hat z}{2}\, e^{-\hat z^2/2}
- \frac{g(\hat z)}{2\hat z^2} + \frac{g(\hat z)}{2}
\;=\; \frac{\operatorname{erf}(\hat z/\sqrt2)}{2} - \frac{g(\hat z)}{2\hat
z^2} + \frac{g(\hat z)}{2} - \frac{\operatorname{erf}(\hat
z/\sqrt2)}{2} + \sqrt{\tfrac{2}{\pi}}\,\frac{\hat z}{2}\,e^{-\hat z^2/2},
$$

i.e. simply $\psi(\hat z) = \sqrt{2/\pi}\,(\hat z/2)e^{-\hat z^2/2} -
g(\hat z)/(2\hat z^2) + g(\hat z)/2$; likewise $\tilde q = h^2(\chi(z_1) -
\chi(z_2))/\sigma^2 + O(h^4)$ with $\chi(\hat z) = -1/(2\hat z|\hat z|)$.
The guarded branch evaluates $N/\hat h^2 = \psi(\hat z_1) - \psi(\hat z_2)$
and $\tilde q/\hat h^2 = \chi(\hat z_1) - \chi(\hat z_2)$ directly, valid for
either sign configuration.  It is activated only from the endpoint-distance
scale (there is no additive $1$ in the criterion).  Configurations wholly
inside the small-radius region integrate the $g(R)/R^3$ power series
term-by-term.  Mixed-scale endpoint cases split that integral into a core
series piece and a fixed-$z$ piece, so the axis expansion is never applied
across a nearby endpoint.  Exact endpoints retain zero velocity but use the
finite gradient limit
$\nabla\mathbf u=\Gamma M[\hat{\mathbf t}]_\times/(4\pi\sigma^2)$.
