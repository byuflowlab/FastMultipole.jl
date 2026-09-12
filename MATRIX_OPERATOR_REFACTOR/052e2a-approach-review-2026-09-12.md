# Mathematical assessment and matrix-free recommendation

2026-09-12. Written at Ryan's request. This assesses the method rather than executing the review prompt's prescribed workflow. No simulation or implementation changes were made. Numerical results below are reported results, not independently rerun measurements. Code observations refer to the current local FLOWPanel.jl checkout, which is not established here to be identical to the registered run.

My opinion: **Green reconstruction is a valid conditional formulation and a useful reference, but the evidence does not yet justify calling it a robust production solution for arbitrary particle wakes.** Its second-kind integral equation is a reasonable starting point for matrix-free iteration. The immediate improvement is to preserve that equation, give it a properly projected and preconditioned Krylov implementation, and use direct potential evaluation for retained panel wakes. For a longer-term architecture, I would prototype a **second-kind exterior Neumann source solve with explicit circulation/Kutta unknowns**. That could eliminate the extra trace solve and the assumption of a harmonic wake field inside the body. Its circulation closure still needs derivation and validation; it is a promising alternative, not an established replacement.

## Mathematical soundness

For a closed, simply connected body interior with a curl-free, divergence-free wake velocity, the derivation is correct. With the theory note's signs, Green's identity gives

$$
Aq=b,\qquad A=I-B,\quad b=S\sigma,\quad \sigma=-u_w\cdot n.
$$

The unknown is the interior harmonic potential trace, modulo a constant. Global wake circulation does not prevent this local construction. The assumptions and signs are explicitly stated in `052e-theory-velocity-to-potential-trace.md`, sections 1–3. Body-only assembly suppresses attached-wake contributions in `../FLOWPanel.jl/src/FLOWPanel_formulation.jl:646`; this separation is appropriate for the interior Green identity.

However, the motivating claim that “sources cannot carry the wake's circulation, therefore velocity coupling is wrong” is too broad. A source correction need not carry wake circulation: the wake and additional circulation degrees of freedom can carry it. A properly formulated Neumann method couples a vortical incident field through its normal velocity. What the results implicate is **this particular Dirichlet source substitution and its coupling to the doublet/Kutta unknowns**, not velocity-based coupling in general. Source, doublet, and boundary-condition choices must be derived together; the NASA [panel-method introduction](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19910009745.pdf) provides the relevant classical framework.

Three limitations matter:

1. **Particle vorticity leakage.** If regularized particles induce nonzero curl inside the body, normal data still define a compatible harmonic Neumann problem, but its solution need not reproduce the particle field's tangential velocity. Incompressibility and zero net flux do not prove irrotationality. Green reconstruction then produces a harmonic surrogate. Small Green residuals cannot measure this modeling error. Quantify interior curl and compare the reconstructed surface gradient with tangential incident velocity. The reported particle/panel trace difference rising from 0.0073735 to 0.016798 does not bound this error separately from different wake trajectories and strengths.
2. **Trailing-edge contact.** The harness explicitly permits approximately zero wake clearance (`scripts/addendum_052e2a_realsim_2026-09-11.jl:53`). Thus the clean separated-support assumption of the theory is not available uniformly as the mesh refines. Green identities can survive appropriate weak limits with edge singularities, but classical smooth-data convergence estimates do not automatically transfer. A wake sheet touching the boundary is not by itself proof of failure; the traces, local singularity, and limiting representation must be checked.
3. **Pressure and topology.** An independent area-mean constant at each time does not determine an absolute unsteady potential reference. Circulation agreement and Kutta–Joukowski loads do not validate unsteady pressure recovery. A constant pressure shift cancels from the net pressure force on an exactly closed surface, but that does not establish local pressure accuracy. Multiply connected interiors also require circulation/period information beyond normal data.

The capped GR body and uncapped Neumann referee impose different boundary problems. Their agreement is encouraging; their remaining gap is not a clean estimate of reconstruction error. A same-surface comparison is needed before interpreting the roughly 0.0055 Phase B circulation-gap plateau.

## Gauge treatment: correct reduction, but no conditioning guarantee

The production bordered equations are

$$
Aq+a\lambda=b,\qquad a^Tq=0.
$$

The Householder reduction to $Z^TAZ$ is algebraically exact and numerically sensible. The implementation at `../FLOWPanel.jl/src/FLOWPanel_formulation.jl:700` follows that derivation. It removes the gauge without forming a dense basis. It does **not** precondition the physical operator or remove near-null modes associated with geometry.

For an exactly rank-$N-1$ discretization, let $A\mathbf1=0$ and $\ell^TA=0$. Border invertibility requires $a^T\mathbf1\ne0$ and $\ell^Ta\ne0$. The right null vector is constant; the left null vector need not be the area vector. In particular,

$$
\lambda=\frac{\ell^Tb}{\ell^Ta}.
$$

The border removes an algebraic compatibility defect along $a$. It is not generally equivalent to correcting the physical Neumann flux before applying $S$. If $a^T\sigma\ne0$, the uniform-density correction

$$
\sigma_c=\sigma-\mathbf1\frac{a^T\sigma}{a^T\mathbf1}
$$

is the smallest correction in the area-weighted density norm, but solving with $S\sigma_c$ is a deliberate change of discretization policy. Record the defect and justify any correction; do not let the gauge silently serve as a physics repair.

This distinction exposes an actual discrepancy in the available iterative methods. The Picard route subtracts the area mean of $b+Bq$ (`../FLOWPanel.jl/src/FLOWPanel_formulation.jl:923`). Define

$$
Q=I-\mathbf1\frac{a^T}{a^T\mathbf1}.
$$

Its fixed point satisfies $q=Q(b+Bq)$, hence $Aq-b\in\operatorname{span}\{\mathbf1\}$. The border instead requires $Aq-b\in\operatorname{span}\{a\}$. These generally differ for incompatible data on unequal-area panels. This is an algebraic finding, not a demonstrated explanation of the dense-run G4 error. Compatible data can conceal the difference.

## What the convergence evidence supports

Phase B trace errors decrease from 0.21333 to 0.058507 over 1,744 to 19,384 panels; Phase A decreases from 0.19680 to 0.048951. This favors a converging discretization over a large constant formulation error, but four coupled refinements cannot establish the asymptotic limit or rule out a smaller floor.

There is a material documentation correction: **the implemented $E_q$ is relative area-weighted RMS potential-trace error**, not dy-weighted RMS and not source error. See `scripts/addendum_052e2a_realsim_2026-09-11.jl:304`. Only the circulation metric uses spanwise weights. Interpret the reported values using the implemented definition.

An empirical $N^{-0.6}$ rate translates to approximately $h^{1.2}$ only for effectively uniform two-dimensional refinement. It is plausible for low-order panels with edge effects; it is not a universal predicted rate for this graded, capped, changing-wake problem. Low-order density approximation, geometry, close evaluation, wake variation, and singular behavior can all contribute. The maximum error also decreases, reaching 0.37653 in A and 0.57989 in B. Its equality with the TE-excluded maximum means the maximum is attained outside the chosen mask; blaming only TE-adjacent panels is unsupported.

The harness's “Green residual” includes $a\lambda$ (`scripts/addendum_052e2a_realsim_2026-09-11.jl:423`). Its reported $1.22\times10^{-14}$ therefore measures solution of the bordered equations, not necessarily $Aq=b$. Both residuals and the dimensionless size of $a\lambda$ should be reported. The summary does not give enough multiplier/flux values to resolve this issue here.

I would not spend another large uniform-refinement run to reach the 1% gate. Extrapolating the observed Phase B error with exponent 0.6 requires about $5.85^{1/0.6}\approx19$ times as many panels, or roughly 370,000. This is an illustrative extrapolation, not a forecast. The gate lacked a demonstrated accuracy budget at L4; missing it establishes insufficient achieved accuracy, not a false Green identity.

The cheapest useful diagnosis is a frozen external harmonic field on independently refined body meshes. For a gauge-aligned analytic trace $q_*$, compute

$$
r_*=Aq_*-b,\qquad A(q_h-q_*)+a\lambda=-r_*.
$$

Map $r_*$ and the trace error by location, panel area, aspect ratio, and distance to the wake. Then separate projection along $a$ from the part seen by the reduced operator. This distinguishes consistency error from its amplification by the solve. Repeat with a fixed separated wake and a fixed TE-touching wake. These tests are much more diagnostic than another freely evolving wake ladder.

## Matrix-free design I recommend for the existing Green method

$A$ is a second-kind operator: on smooth closed surfaces it contains an identity contribution plus a compact integral part. This is much more favorable than inverting a first-kind single-layer operator or an unpreconditioned hypersingular operator. Nevertheless, sharp edges, near-touching faces, thin geometry, and nonsymmetric collocation can cause difficult Krylov behavior. Even second-kind Laplace systems can require substantial geometry-aware preconditioning; see [Quaife and Biros, On preconditioners for the Laplace double-layer in 2D](https://arxiv.org/abs/1308.1937). That paper supports the caution, not a performance guarantee for this 3D wing.

**Preserve the certified incompatibility convention.** Either apply $Z^TAZ$ using implicit Householder vector transforms, or use the following exact full-space completion. Set

$$
u=a/\|a\|_2,\qquad P=I-uu^T,\qquad
T=PAP+\alpha uu^T,\qquad Tq=Pb,
$$

where $\alpha$ is nonzero and comparable to the dimensionless operator's identity scale. Multiplication by $u^T$ gives $u^Tq=0$. The remaining equation gives $P(Aq-b)=0$, exactly reproducing the border, including incompatible data. Recover

$$
\lambda=\frac{a^T(b-Aq)}{a^Ta}.
$$

Each product needs one $B$ application and a few $O(N)$ vector operations. No dense $P$ is needed. In orthogonal coordinates $T$ is block diagonal with blocks $Z^TAZ$ and $\alpha$. Thus completion fixes the missing gauge eigenvalue without worsening the reduced block through saddle-point coupling. This is different from merely adding a rank-one term to unprojected $A$, which need not preserve the same incompatible-data solution. It improves the available implementation choices, not the underlying approximation order.

**Use GMRES with a Green-specific preconditioner.** The existing matrix-free bordered route normalizes the border, which is good, but explicitly ignores the supplied preconditioner (`../FLOWPanel.jl/src/FLOWPanel_formulation.jl:848`). Start with local block inverses containing same-surface neighbors and nearby opposite-side panels. Add a coarse correction for body-scale modes if iteration counts grow. A patch restricted to mesh adjacency can miss strong upper/lower-surface interactions on a thin wing. Project preconditioner input/output consistently with the gauge. Use FGMRES if preconditioner applications vary. Do not choose CG or MINRES solely because the underlying PDE is elliptic: the collocation matrix is not established to be symmetric positive definite.

**Scale in a physical norm.** With $M=\operatorname{diag}(a)$, work in $z=M^{1/2}q$ and use area-weighted residuals. Preserve equivalence by transforming the full operator, RHS, and border: $A_s=M^{1/2}AM^{-1/2}$, $b_s=M^{1/2}b$, transformed border column $M^{1/2}a$, gauge row $a^TM^{-1/2}$. Do not independently replace these by a symmetric-looking border. Weighting controls mesh-dependent Euclidean scaling; it does not prove a condition-number bound or remove physical ill-conditioning.

**Keep accurate near interactions and accelerate the far field.** Use analytic panel integrals where valid, or consistent singular/near-singular quadrature, with FMM for the far field. Do not soften the Laplace kernel merely to ease iteration: that changes the Green identity. Higher-order densities and geometry may be needed even when constant-panel integrals are evaluated exactly. Special close-evaluation quadrature can remove quadrature error but cannot by itself repair a low-order density approximation; see [Wala and Klöckner, A fast algorithm with error bounds for Quadrature by Expansion](https://www.sciencedirect.com/science/article/abs/pii/S0021999118302985).

**Set and verify an error budget.** Check the final residual with a stricter operator evaluation; keep FMM error and iterative error below the desired discretization/model error. Warm-start consecutive solves, but do not use a warm start to conceal iteration growth. The current Green Krylov routine warns and returns after nonconvergence (`../FLOWPanel.jl/src/FLOWPanel_formulation.jl:905`); production acceptance should reject or explicitly recover from an inaccurate trace solve. Measure iteration count, setup cost, total step time, memory, true residual, trace error, and circulation error on the same cases. Dense residuals do not certify any of these iterative properties.

## A potentially better formulation: solve the physical Neumann problem directly

My preferred longer-term experiment uses **unknown source density for the noncirculatory body correction, plus explicit lifting/circulation degrees of freedom**. Unlike legacy VTS, the source density is solved through the exterior boundary operator; it is not prescribed as minus the incident normal velocity and inserted into a Dirichlet system.

Using conventional positive single-layer potential $\mathcal S$, write

$$
u=U_\infty+u_w+\nabla\mathcal S\eta+
\sum_j\Gamma_j v_j.
$$

Here $v_j$ are circulation-carrying fields with an explicitly defined body/wake attachment. For outward body normals, the exterior single-layer normal derivative is

$$
\partial_n^+\mathcal S\eta=(-\tfrac12 I+K')\eta.
$$

Consequently the impermeability equations become

$$
(-\tfrac12 I+K')\eta+N_v\Gamma
=-(U_\infty+u_w-u_{\rm body})\cdot n.
$$

This is second-kind in $\eta$, uses wake velocity directly, and requires no harmonic extension of the wake inside the solid. The single-layer representation and second-kind exterior Neumann equation are established potential theory; an explicit formulation appears in [A numerical method for the solution of exterior Neumann problems for the Laplace equation in domains with corners](https://www.sciencedirect.com/science/article/pii/S0168927417300314). The proposed lifting extension here is my design recommendation, not a result established by that paper.

The important unresolved work is the lifting closure. Sources alone have zero circulation. Choose suitable independent bound/wake fields and impose a physically derived Kutta condition. For a linearized closure, the system has the schematic form

$$
\begin{bmatrix}A_N&N_v\\C_\eta&C_\Gamma\end{bmatrix}
\begin{bmatrix}\eta\\\Gamma\end{bmatrix}
=\begin{bmatrix}f\\k\end{bmatrix}.
$$

Precondition $A_N$ locally and approximate the circulation Schur complement $C_\Gamma-C_\eta A_N^{-1}N_v$. Its dimension can scale with shedding stations rather than total surface panels, although constructing a dense Schur complement may still be expensive. A nonlinear pressure Kutta condition requires its own Newton/linearization strategy. The existing doublet-jump rule cannot simply be copied to source unknowns, and the circulation basis must represent the relevant distributed wing/wake degrees of freedom.

This architecture can remove one global solve and the trace-reconstruction error altogether. It does not remove particle-core modeling error, wake/body penetration, poor mesh resolution, or unsteady-pressure recovery requirements. Its overall robustness depends on the Kutta block as well as the second-kind source block. Therefore I recommend a small prototype before committing to a rewrite.

Merely changing to the existing doublet-only Neumann route is less compelling for conditioning: normal differentiation of a double-layer potential leads to a hypersingular operator. A stable variational discretization with suitable operator/Calderón preconditioning can work well, but “Neumann” by itself does not imply good conditioning. The proposed source formulation specifically avoids that principal operator.

## Alternatives and practical ordering

| Option | Expected benefit | Cost and limitation |
|---|---|---|
| Direct panel-wake potential + particle-only GR | Removes unnecessary reconstruction error for panel contributions | Lowest implementation disruption; already represented by `HybridWakePotential` in `../FLOWPanel.jl/src/FLOWPanel_formulation.jl:1157`; particle assumptions remain |
| Higher-order densities/geometry, local refinement, accurate close quadrature | Can improve trace accuracy and, with regularity, convergence order | Moderate implementation effort; diagnose which error dominates before changing quadrature |
| Projected/completed preconditioned GR | Makes the current formulation scalable while preserving the reference solution | Additional trace solve remains; requires actual iteration/geometry tests |
| Exterior source Neumann + circulation/Kutta block | Avoids reconstruction and its interior-harmonicity assumption | Larger change; circulation basis and Kutta closure are the decisive risks |
| Existing doublet Neumann + operator preconditioning | Reuses more lifting machinery | Hypersingular discretization and preconditioner compatibility need care |
| Oversampled weighted least squares | Useful for noisy/inconsistent velocity observations | Extra operator/adjoint cost; does not cure model bias; use QR/LSQR, not normal equations that square conditioning |
| Surface tangential least squares / Hodge reconstruction | Independent curl/trace diagnostic; possible sparse reconstruction | Surface Poisson solve needs multigrid; does not by itself enforce interior harmonicity or normal data |
| Vector-potential coupling | Natural representation of vortical velocity | Does not directly supply the scalar Dirichlet trace; more attractive when paired with a Neumann body formulation |
| Different gauge or kernel softening | Gauge may improve scaling; softening may reduce sharp numerical behavior | Gauge cannot repair spatial accuracy; softening adds model bias and breaks the unmodified Laplace identity |

For the current hybrid implementation, also avoid a hidden scalability trap: `surface_hodge_trace!` constructs a dense edge-by-panel least-squares matrix (`../FLOWPanel.jl/src/FLOWPanel_formulation.jl:1120`) and is called during particle reconstruction. A matrix-free Green solve does not make that diagnostic scalable. Use a sparse incidence operator and a preconditioned surface solve, or run the diagnostic selectively.

I would proceed in this order:

1. On small fixed meshes, compare dense border, implicit Householder, and projected/completed GMRES with deliberately incompatible RHS data and unequal panel areas. This exposes the Picard convention difference and validates a matrix-free replacement cheaply.
2. Run separated harmonic and fixed near-TE oracle tests, evaluating consistency residuals and area-weighted error. Compare local refinement against improved density/quadrature before choosing an accuracy upgrade.
3. Use direct panel potential plus particle-only reconstruction, measuring leakage and tangential mismatch independently of particle/panel trajectory differences.
4. Prototype source-Neumann/circulation coupling on exactly the same closed geometry. Test zero-wake impermeability, a prescribed harmonic incident field, and a lifting case with fixed wake geometry and consistent Kutta closure. Compare against direct-potential Dirichlet results. Track iterations under refinement and reduced thickness. Reject the prototype if circulation agreement or bounded practical iteration counts cannot be obtained without expensive global preconditioning.

I would retain GR as the reference and near-term route, but would not endorse its present iterative implementations as production-ready based on this dense campaign. The most promising efficiency gain is removing the reconstruction solve through a well-conditioned physical Neumann formulation; the lowest-risk immediate gain is a preconditioned, gauge-consistent matrix-free GR plus the existing direct-panel hybrid split.
