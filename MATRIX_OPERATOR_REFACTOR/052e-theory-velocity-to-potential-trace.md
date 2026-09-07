# 052e theory — recovering a body potential trace from velocity

**Status: ACCEPTED by Ryan 2026-09-07** (acceptance given after review of the
area-mean closure rationale; closes the theory gate required by
`052e-accuracy-plan-v2-draft-2026-09-05.md` before Tier 0B interpretation).

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
nullspace in the continuum problem. Let $a_i=A_i$ be the panel-area vector.
The production `:area_mean` route selects $a^Tq=0$ (equivalently
$\langle q\rangle_A=0$) with the bordered system

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
it is telemetry, not a physical potential. This full area-augmented system is
the authoritative 052e formulation through the Tier 0B proof against directly
evaluated doublet-panel-wake induced potential. Keeping it unchanged during
that proof separates validation of the Green reconstruction from validation
of a different algebraic representation.

The existing `:lsq` route solves the overdetermined constrained problem

$$
\min_q\left\|
\begin{bmatrix}I-B\\a^T\end{bmatrix}q-
\begin{bmatrix}S\sigma\\0\end{bmatrix}
\right\|_2.
$$

Least squares is not part of 052e certification or its required tests; this
restriction does not require removal of the existing API. For multiple
disconnected Dirichlet bodies, the construction and gauge are applied
independently to each body because each interior domain has its own constant
mode.

### 3.1 Orthogonal reduction after the Tier 0B proof

The border is a convenient way to impose the gauge, but increasing the system
dimension is not mathematically necessary. After the bordered formulation
passes Tier 0B, replace its production representation with a square solve on
the area-mean-zero subspace

$$
\mathcal V=\{q\in\mathbb R^N:a^Tq=0\}.
$$

Write $A=I-B$, $b=S\sigma$, and normalize $\widehat a=a/\|a\|_2$.
Choose a numerically stable Householder reflector $H=I-2vv^T$ satisfying

$$
H\widehat a=s e_N,\qquad s\in\{-1,1\},
$$

with the sign chosen by the usual cancellation-avoiding Householder rule.
If $Z=H[:,1:N-1]$, then $Z^TZ=I$, $Z^Ta=0$, and every admissible trace has
the unique representation $q=Zy$. Premultiplying the bordered physical
equations by $Z^T$ eliminates the multiplier and gives

$$
\boxed{(Z^TAZ)y=Z^Tb,\qquad q=Zy.}
$$

This is an exact $(N-1)\times(N-1)$ square coordinate reduction of the
bordered equations, not a least-squares problem. It also preserves the
bordered treatment of a discretely incompatible right-hand side.

No dense $Z$ need be formed or stored. Transform $A$ once as
$\widetilde A=HAH^T$ using the reflector's rank-one form, factor its leading
$(N-1)\times(N-1)$ block, and transform each right-hand side as
$\widetilde b=Hb$. After solving for $y$, set $\widetilde q=[y;0]$ and recover
$q=H^T\widetilde q$. Because $H$ is symmetric, the forward and inverse vector
transforms use the same dot product and scaled vector update. The omitted
last equation retains the incompatibility telemetry:

$$
\lambda=\frac{s}{\|a\|_2}
\left(\widetilde b_N-\widetilde A_{N,1:N-1}y\right).
$$

Reflector construction and each vector projection cost $O(N)$. The one-time
two-sided dense transform costs $O(N^2)$, small beside the $O(N^3)$ LU
factorization; recurring triangular solves remain $O(N^2)$. Reducing the
dimension by two relative to the border saves only $O(N)$ storage. A material
peak-memory benefit is possible only if the assembled $B$ buffer is reused in
place for $A$, its transform, and the reduced LU.

Advantages of the reduced formulation are an ordinary square solve, exact
enforcement of the area gauge by coordinates, 2-norm-stable orthogonal
transforms, preservation and recovery of $\lambda$, and negligible recurring
projection cost. Its disadvantages are more complicated setup and indexing,
less transparent operator storage, and the need to prove parity with the
already validated bordered reference before adoption.

Important pitfalls are:

- Forming $P=I-aa^T/(a^Ta)$ and factoring the full $N\times N$ matrix $PAP$
  does not work: it remains singular in full coordinates.
- Projecting only the unknown, or only the right-hand side, does not reproduce
  the bordered treatment of discrete incompatibility; both trial and equation
  spaces must use $Z$.
- Replacing the bordered solve before its Tier 0B oracle passes would conflate
  formulation validation with representation validation.
- The reflector must use a cancellation-avoiding sign and normalized area
  vector; an explicitly constructed dense basis is unnecessary.

### 3.2 Alternatives considered

- **Area-augmented bordered LU:** clearest robust reference and returns
  $\lambda$ directly; costs one extra row and column and has saddle-point
  structure.
- **Householder orthogonal reduction:** exact $(N-1)$ square equivalent with
  cheap, stable projections; adds setup complexity and requires bordered
  parity proof.
- **Rank-one completion:** retains an $N\times N$ square matrix and is simple
  for exactly compatible data; can change the selected discrete result or
  obscure incompatibility telemetry when compatibility is imperfect.
- **Replace one equation or pin one potential:** simple $N\times N$ solve;
  arbitrarily discards or localizes one residual equation and can be
  mesh/order dependent.
- **Projected matrix-free Krylov/fixed-point solve:** can remove dense matrix
  storage; introduces convergence, stopping-criterion, and preconditioning
  obligations and must match the bordered incompatibility convention.
- **Constrained least squares:** tolerates inconsistency, but solves a
  different optimization problem and is explicitly excluded from 052e
  certification.

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
control points, align each body's trace by a constant, and certify only the
area-mean gauge. The full bordered solve remains authoritative through the
direct doublet-panel-wake Tier 0B proof. Only after that explicit pass may the
Householder-reduced solve be implemented; reduced-versus-bordered parity is a
separate gate before later tiers. Standalone doublet solid-angle checks remain
useful for kernel signs and normalization, but wake-alone field-point
potential and branch-jump tests do not exercise `HybridWakePotential`.
Promotion claims are limited to gauge-invariant circulation, exterior
velocity, and integrated loads until a separate global gauge method is
available and verified.
