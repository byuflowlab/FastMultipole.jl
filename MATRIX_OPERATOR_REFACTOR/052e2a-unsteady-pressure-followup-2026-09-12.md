# Unsteady pressure and forces with a Neumann body solve

2026-09-12. Follow-up opinion for Ryan. Mathematical derivations and architecture recommendations; no new numerical validation or code changes.

**Recommendation:** retain the proposed second-kind Neumann velocity solve. Recover unsteady Bernoulli pressure from its potential representation when the complete flow admits a suitable local potential. For general particle wakes, develop a consistent surface reconstruction of pressure from material acceleration, with multigrid preconditioning. For integrated rigid-body force and moment, precompute six adjoint surface solutions so that pressure need not be solved every timestep. This is a plausible efficient architecture, not a claim that the current pressure implementation is validated.

## Neumann does not mean potential-free

Neumann specifies a normal derivative boundary condition. With a source representation, the body correction potential is available directly as a single-layer potential once the density is solved. With the explicit lifting basis proposed in the preceding review, locally write

$$
\phi=\phi_\infty+\mathcal S\eta+\sum_j\Gamma_j\Phi_j+\phi_w.
$$

This expression presumes potentials for the circulation fields and wake on consistently chosen branches. Its velocity is the complete fluid velocity, not just the body-induced correction. The unsteady Bernoulli equation, neglecting gravity, is

$$
\frac p\rho+\partial_t\phi+\frac12|u|^2=C(t).
$$

Thus a Neumann boundary condition does not prevent pressure recovery. It does not, however, manufacture a missing potential for a rotational particle field. Differentiating only the source/body potential would omit wake-induced unsteady pressure.

For moving surface samples $q(X,t)=\phi(x(X,t),t)$ and surface grid velocity $w=\partial_t x$, the correct conversion is

$$
\partial_t\phi=\left.\frac{dq}{dt}\right|_X-w\cdot u.
$$

Use inertial velocity and a temporally consistent potential branch. At a moving wake cut, apparent potential jumps must not be interpreted as ordinary time derivatives. A spatially varying branch mismatch is not a harmless gauge change.

## Two ways to obtain the unsteady potential term

**Time differentiation:** evaluate the full surface potential each timestep and differentiate with a consistent second-order scheme, including the moving-point correction above. This is the simplest reference when direct panel-wake potential is available. Potential evaluation errors can be amplified by $1/\Delta t$ when their variation between steps is not smooth. Refine temporal and spatial/solver tolerances together.

**Differentiate the solve / use an acceleration-potential formulation:** if the coupled discrete body and circulation equations are $K(t)x(t)=f(t)$, then

$$
K\dot x=\dot f-\dot Kx.
$$

The same operator and preconditioner can solve for the coefficient derivatives. Differentiate the representation as well: source positions, moving surfaces, changing circulation, wake geometry, and strengths all contribute. For fixed rigid geometry in body coordinates, some operator derivatives vanish, but moving wake and kinematic terms generally do not. Differentiating a converged solve implicitly is preferable to differentiating its Krylov iteration history.

This approach avoids subtracting two noisy potential solves. It requires a consistent wake evolution RHS, differentiated Kutta closure, and special treatment of discrete shedding/remeshing events. In continuum irrotational flow, $\partial_t\phi$ is harmonic away from moving singularities; an acceleration-potential boundary problem is another route to it. Its boundary data must be obtained by differentiating moving-body impermeability, including normal rotation and spatial derivative terms. It is not generally just the body normal acceleration.

Acceleration-potential methods are established in unsteady BEM; see [Coslovich et al., Added resistance, heave and pitch for the KVLCC2 tanker](https://research.chalmers.se/publication/524241/file/524241_Fulltext.pdf), sections 2.3–2.4. This is evidence for the technique, not validation for the proposed lifting/particle discretization. For freely moving bodies, the fluid added-mass contribution must also be coupled consistently to the rigid-body acceleration equations.

## A useful correction about the gauge and forces

A potential change $\phi\mapsto\phi+c(t)$ changes the inferred pressure, at fixed Bernoulli constant, only by $-\rho\dot c(t)$. On a closed surface,

$$
\int_S n\,dS=0,\qquad
\int_S r\times n\,dS=0.
$$

Therefore a uniform pressure shift contributes neither net force nor net moment. **A temporally changing area-mean potential gauge is not intrinsically an obstacle to total closed-body unsteady loads.** My previous review's concern about an absolute potential reference should be read as a concern for local absolute pressure and globally consistent pressure comparisons, not a blanket prohibition on integrated forces.

This cancellation does not apply to uncapped surfaces, selected panel subsets, or arbitrary sectional loads. A single pressure offset per connected closed body is sufficient for total loads; independently shifting upper and lower sides would be wrong. For local physical pressure, use a pressure reference with a known physical value, rather than assuming the value at an arbitrary surface panel is fixed in time.

## Rotational particle wakes: reconstruct pressure directly on the surface

For incompressible constant-density inviscid flow, without nonconservative body forces,

$$
\nabla p=-\rho a,\qquad
a=\partial_tu+(u\cdot\nabla)u.
$$

Let $P_s=I-nn^T$, let $w$ be the surface sampling velocity, and let $v=u-w$. Impermeability gives $v\cdot n=0$. The acceleration on the exterior surface is consequently

$$
a=\left.\frac{du_s}{dt}\right|_X+(v\cdot\nabla_s)u_s,
\qquad
\nabla_s p=-\rho P_s a.
$$

This derivation is exact on smooth portions of an impermeable surface. It needs no scalar potential and no normal derivative of surface velocity. Spatial differentiation acts on the full inertial velocity vector, so curvature effects are included. Differentiating local-frame components requires the corresponding rotation terms. Sharp edges need one-sided traces and a consistent weak treatment.

Given approximate acceleration, solve the surface gradient-fitting problem

$$
\min_{p,\ \int_Sp=0}
\frac12\int_S|\nabla_sp+\rho P_sa|^2\,dS.
$$

For a closed surface its Euler–Lagrange equation is

$$
-\Delta_sp=\rho\nabla_s\cdot(P_sa).
$$

For an exact Euler solution, this recovers the actual surface pressure up to a constant. For inconsistent approximate velocity/acceleration, it returns a least-squares pressure. The residual $\nabla_sp+\rho P_sa$ is essential telemetry: a converged scalar solve does not establish that the input acceleration satisfies Euler. In particular, particle regularization, remeshing, modeled stresses, and velocity-solve errors may introduce terms missing from the inviscid equation. Include actual modeled force/stress contributions when the governing equations require them.

This method is especially attractive here because only surface loads are wanted. It avoids a volumetric pressure mesh and the interior-harmonicity assumption of Green trace reconstruction.

## Discretization and conditioning are the decisive work

Use compatible surface gradient/divergence operators. In weak form, with pressure basis functions $N_i$,

$$
L_{ij}=\int_S\nabla_sN_i\cdot\nabla_sN_j\,dS,
\qquad
b_i=-\rho\int_S\nabla_sN_i\cdot P_sa\,dS.
$$

Then $Lp=b$. This is symmetric positive semidefinite, with a constant nullspace per connected component. Remove those modes and use PCG with geometric multigrid or a suitable AMG preconditioner. The unpreconditioned surface Laplacian typically has mesh-dependent conditioning of order $h^{-2}$ on regular mesh sequences. Jacobi scaling alone is not a credible mesh-independent solution. Highly anisotropic blade meshes require appropriate coarsening and potentially line/patch smoothing.

A consistent piecewise-linear surface finite-element pressure discretization is one reasonable candidate. A panel-centered finite-volume scheme can also work if its nonorthogonal corrections and discrete differential operators are consistent. Do not assume that a positive graph Laplacian automatically approximates the desired surface PDE accurately on arbitrary triangles.

The expensive accuracy issue is reconstructing $(v\cdot\nabla_s)u_s$ from panel data. Use a stencil with genuine two-dimensional tangent-plane coverage, conditioning checks, and suitable polynomial reproduction; expand stencils or use a higher-order surface velocity representation when necessary. This does not require taking an unvalidated exterior hypersingular Hessian limit.

FLOWPanel already contains this general pressure direction. `../FLOWPanel.jl/docs/pressure_poisson.md:176` derives the moving-control-point acceleration correction, and line 492 identifies its solve as surface reconstruction. However, `../FLOWPanel.jl/agent_policies/MONITORS.md` explicitly says no formulation has passed the complete unsteady pitching-wing acceptance gate. It describes pinning, Jacobi-preconditioned CG, and unimplemented AMG/IncompleteCholesky options. The theory document also identifies tangent-plane gradient failure on anisotropic meshes near line 774. Therefore I recommend improving and validating this path, not treating the existing `PressureLaplace` output as a certified reference.

## For rigid-body force and moment, six offline solves can suffice

This follows directly from the linear pressure reconstruction. Define a pressure integration vector $c_k$ so that a force or moment component is $J_k=c_k^Tp$. For force, for example, $(c_k)_i=-\int_S N_i n_k\,dS$. Compute once, in the gauge-fixed subspace,

$$
Lz_k=c_k.
$$

Then at each timestep,

$$
J_k=z_k^Tb.
$$

There are six such vectors for three force and three moment components. Their RHS vectors are compatible because uniform pressure produces zero total force and moment on a closed surface. Use quadrature and geometry that preserve these identities discretely.

For fixed rigid geometry, store the six solutions in body coordinates; transform loads and moments to the desired frame afterward. A per-step pressure solve is then unnecessary unless the pressure distribution is requested. The acceleration/RHS evaluation remains necessary, and precomputed solve accuracy still matters. Deformation changes the operator and load vectors, while sectional loads generally require a pressure-reference convention because their constant mode need not cancel.

This is an exact discrete adjoint identity for the selected reconstruction, not a separate physical approximation. It offers a concrete performance advantage over repeatedly solving a global potential-trace problem merely to integrate six loads.

## When a full pressure Poisson solve is appropriate

If pressure away from the body or a global reference is required, the full inviscid pressure equation is

$$
\Delta p=-\rho\sum_{i,j}(\partial_i u_j)(\partial_j u_i),
\qquad
\partial_np=-\rho n\cdot a.
$$

Supply a consistent far-field/reference condition and the correct body acceleration boundary condition. The quadratic source is not confined to the vortex cores; irrotational strain contributes too. A vortex-particle discretization does not automatically provide a good quadrature of this entire source. A volume/grid/adaptive representation plus a multigrid solve, or a volume-potential and boundary-correction method, is possible but more intrusive than surface recovery. Normal acceleration at a curved slipping wall is not generally equal to body normal acceleration: even a stationary curved wall can have nonzero normal fluid acceleration.

For a broader pressure-free force alternative, finite-domain force identities can use velocity/vorticity with boundary corrections; see [Noca, Shiels and Jeon, On the evaluation of instantaneous fluid-dynamic forces on a bluff body](https://authors.library.caltech.edu/records/cqakz-aww67). A naive derivative of wake impulse alone omits body/added-mass and boundary contributions and should not be used as a universal force formula.

## Proposed decisive tests

1. An accelerating or oscillating sphere in otherwise quiescent inviscid fluid: verify analytic added mass, local pressure modulo a constant, and temporal order. This is a direct test of the unsteady term with no lifting or wake ambiguity.
2. A fixed-body analytic potential-flow case, followed by translation/rotation of coordinates: test steady convective pressure and moving-frame invariance separately.
3. A pitching lifting body with an explicitly represented panel wake: compare full-potential Bernoulli and surface-acceleration pressure on the same closed mesh, including load phase and amplitude. Refine timestep independently of mesh.
4. A controlled rotational-flow case consistent with Euler and impermeability: test pressure against an analytic/manufactured or independently resolved reference. An arbitrarily prescribed moving vortex is not automatically an Euler-consistent pressure oracle.
5. Verify six-adjoint loads equal direct integration of the reconstructed pressure, then check mesh refinement, anisotropy, and multigrid iteration counts.

The resulting recommendation is **Neumann velocity solve plus surface Euler pressure/adjoint loads for general particles**, with **full-potential Bernoulli or differentiated-potential recovery as the irrotational reference**. There is no need to give up unsteady forces merely because the body boundary condition becomes Neumann.
