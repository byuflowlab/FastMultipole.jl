# 052e theory — recovering a body potential trace from velocity

## Purpose

This note derives what `HybridWakePotential` can and cannot "solve" when a
wake is represented by particles that provide velocity but no scalar
potential. It is the prerequisite for the 052e accuracy campaign. The result
is a Neumann-to-Dirichlet map on each closed body, not a reconstruction of a
global wake potential.

## 1. Continuous problem

Let $D\subset\mathbb R^3$ be the interior of one closed, connected body, with
outward unit normal $n$. Suppose the sampled wake velocity $u_w$ satisfies

$$
\nabla\times u_w=0,\qquad \nabla\cdot u_w=0\quad\text{in }D,
$$

and $D$ is simply connected. Then there is a harmonic scalar potential
$\psi$ in $D$ such that

$$
u_w=\nabla\psi,\qquad \Delta\psi=0.
$$

Velocity supplies the Neumann datum

$$
g=\frac{\partial\psi}{\partial n}=u_w\cdot n
\quad\text{on }\partial D.
$$

The Neumann compatibility condition follows from incompressibility:

$$
\int_{\partial D}g\,dS
=\int_D\nabla\cdot u_w\,dV=0.
$$

This condition is necessary for an exact harmonic reconstruction. A discrete
flux defect measures incompatibility introduced by geometry, quadrature, or
the velocity representation.

The hypotheses are substantive. Singular filaments and compact vorticity
support must be disjoint from $\overline D$. For a noncompact regularization
such as a Gaussian core, exact irrotationality is lost everywhere; the method
therefore requires a quantified upper bound on vorticity leakage into $D$ and
must treat the resulting reconstruction error as regularization bias.

## 2. Boundary-integral derivation

Let the Laplace fundamental solution be

$$
G(x,y)=\frac{1}{4\pi|x-y|}.
$$

Green's third identity for $x\in D$ is

$$
\psi(x)=
\int_{\partial D}
\left[G(x,y)\frac{\partial\psi}{\partial n_y}(y)
-\psi(y)\frac{\partial G}{\partial n_y}(x,y)\right]dS_y.
$$

Define the boundary trace $q=\psi|_{\partial D}$ and use the implementation's
source-density convention

$$
\sigma=-g=-u_w\cdot n.
$$

To map the standard identity to FLOWPanel's signs, define its volume-point
operators by

$$
(S\sigma)(x)=-\int_{\partial D}G(x,y)\sigma(y)\,dS_y,
\qquad
(Dq)(x)=-\int_{\partial D}q(y)
\frac{\partial G}{\partial n_y}(x,y)\,dS_y.
$$

Let $B$ be the complete interior boundary trace of $D$, including the
one-sided self limit used by the panel implementation. The interior boundary
limit then becomes

$$
q=S\sigma+Bq,
$$

and hence

$$
\boxed{(I-B)q=S\sigma.}
$$

Thus the apparent potential "solve" is the Neumann-to-Dirichlet map for a
harmonic field inside the body. Only normal velocity is required. Tangential
velocity is not part of the solve; it can independently diagnose whether the
reconstructed trace has the expected surface gradient.

## 3. Gauge and solvability

The velocity is unchanged by $\psi\mapsto\psi+c$, so the trace is determined
only modulo a constant. Correspondingly, $I-B$ has the constant mode in its
nullspace in the continuum problem. Let

$$
a_i=A_i
$$

be the panel-area vector. The production `:area_mean` route selects
$a^Tq=0$ (equivalently $\langle q\rangle_A=0$) with the bordered system

$$
\begin{bmatrix}
I-B & a\\
a^T & 0
\end{bmatrix}
\begin{bmatrix}q\\\lambda\end{bmatrix}
=
\begin{bmatrix}S\sigma\\0\end{bmatrix}.
$$

The multiplier $\lambda$ records discrete incompatibility in the null mode;
it is telemetry, not a physical potential. The `:lsq` route instead solves
the overdetermined constrained problem

$$
\min_q\left\|
\begin{bmatrix}I-B\\a^T\end{bmatrix}q-
\begin{bmatrix}S\sigma\\0\end{bmatrix}
\right\|_2.
$$

Both routes recover the same physical object only up to body-panel
discretization and compatibility errors. For multiple disconnected
Dirichlet bodies, the construction and gauge are applied independently to
each body because each interior domain has its own constant mode.

## 4. Doublet-sheet and filament equivalence

A constant-strength open doublet sheet has no interior vorticity in the
singular continuum model; its velocity is equivalent to that of the vortex
filament along the sheet boundary. With consistent orientation, the filament
circulation has magnitude equal to the doublet strength $\mu$. This is the
classical basis for the disk/ring manufactured oracle (see the
[NASA doublet-panel reference](https://ntrs.nasa.gov/api/citations/19810008456/downloads/19810008456.pdf)).

The equivalence does not make a finite regularized particle ring identical to
the panel sheet. Segment quadrature, particle spacing/phase, and core smoothing
remain distinct approximations and must be refined independently.

## 5. What topology permits

No branch cut is needed inside a simply connected body when wake singular
support stays outside it: $u_w$ is exact there and $q$ is single-valued over
the entire body surface. A branch surface used to represent an exterior
doublet wake must be chosen not to intersect the body.

This local statement does not imply that a single-valued potential exists in
the entire exterior wake domain. Loops linking vortex filaments can have
nonzero circulation, so integrating velocity from infinity is path-dependent
unless cuts, homology classes, and circulation jumps are tracked. Therefore
velocity-only particle data do not currently define a production-capable
zero-at-infinity gauge. The body-local solve cannot supply absolute $C_p$,
unsteady pressure, or acoustic phase without a separate topology-aware global
gauge construction.

## 6. Discrete diagnostics and falsification conditions

The reconstruction should record, per body:

- flux compatibility $|\sum_i A_i\sigma_i|$ with a declared dimensional
  normalization;
- bordered multiplier $|\lambda|$ for `:area_mean`;
- the linear Green residual and area-gauge defect;
- area-mean-aligned trace error against a direct oracle;
- tangential Hodge projection defect and gauge-aligned Green/Hodge mismatch.

A small linear residual establishes only that the discrete system was solved;
it does not establish that its operators approximate the continuum map. The
primary manufactured test must therefore prescribe an external harmonic
field with a directly evaluable trace and normal velocity, then refine the
body mesh independently of the wake representation.

Failure of the analytic-normal-velocity case to improve with body-mesh
refinement falsifies the implemented Green trace map or its conventions.
Failure introduced only by filament discretization, particle regularization,
or total-minus-retained-panel extraction diagnoses those approximations
instead. If the flux defect or vorticity leakage exceeds its pre-registered
tolerance, the analytic precondition is not met and the case is invalid rather
than evidence for or against the Green identity.

## 7. Consequences for 052e

The accuracy campaign must compare reconstructed values only on closed-body
control points, align each body's trace by a constant, and test both supported
gauges. Standalone doublet solid-angle checks remain useful for kernel signs
and normalization, but wake-alone field-point potential and branch-jump tests
do not exercise `HybridWakePotential`. Promotion claims are limited to
gauge-invariant circulation, exterior velocity, and integrated loads until a
separate global gauge method is available and verified.
