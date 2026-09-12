# Vortex wakes, layer potentials, and harmonic reconstruction

2026-09-12. Follow-up mathematical clarification for Ryan. This qualifies the earlier approach review: lack of a physical scalar potential for the particle field does not by itself invalidate an auxiliary harmonic reconstruction used to enforce impermeability.

## The layer potential represents the body correction

For clarity take a stationary, closed, smooth body with exterior fluid domain Omega, and an incident velocity u_v that is divergence-free and has the prescribed vortex-particle vorticity. Hold circulation/wake degrees of freedom fixed for this argument. Seek

$$
u=u_v+\nabla\phi_b,
\qquad \Delta\phi_b=0\quad\text{in }\Omega,
\qquad \partial_n\phi_b=-u_v\cdot n\quad\text{on }S.
$$

Apply the appropriate decay condition at infinity. In a multiply connected fluid domain, specify additional circulation/harmonic-field degrees of freedom; a single-valued scalar correction does not determine or change them. A moving body changes the boundary datum to $(u_{\rm body}-u_v)\cdot n$.

The correction is harmonic, but the total field is not generally a scalar-potential field:

$$
\nabla\cdot u=0,\qquad \nabla\times u=\nabla\times u_v=\omega_v.
$$

Single- and double-layer potentials are representations of this harmonic correction. They do not have to represent the vortical incident field itself. At an instant this is a well-defined div-curl boundary problem, with uniqueness subject to the relevant far-field and circulation conditions. Layer densities can have representation redundancies even when the exterior velocity is unique; the BIE must remove those redundancies or constrain its densities.

For compatible no-net-flux data, the exterior correction needs no net-source monopole. In three exterior dimensions a decaying scalar solution can more generally accommodate nonzero flux via a 1/r potential; allowing that is not automatically an appropriate physical choice for an impermeable rigid body.

This is a boundary correction preserving the prescribed vorticity. With the appropriate stationary-domain function spaces and boundary conditions, it is the divergence-free special case of a Helmholtz–Hodge projection onto divergence-free, boundary-tangent velocity fields. It is not a projection of the complete velocity onto curl-free fields. For a moving wall, the target constraint is affine. Nondecaying freestreams require the usual disturbance-field/function-space qualifications before making an L2 orthogonality claim.

Vortex-particle/BEM coupling through Helmholtz–Hodge decomposition is an established approach; see [Chorin's approaches revisited: Vortex Particle Method vs Finite Volume Method](https://www.sciencedirect.com/science/article/pii/S0955799719302164). This supports the decomposition, not validation of the current FLOWPanel implementation.

## An auxiliary interior harmonic field is a different object

Given compatible wake normal velocity g on a closed body, solve inside the solid D:

$$
\Delta\psi=0\quad\text{in }D,
\qquad \partial_n\psi=g,
\qquad \int_S\psi\,dS=0.
$$

If the actual wake field is harmonic and exact in D, this recovers its potential trace up to a constant. If the actual field has interior curl, it generally does not recover its full velocity or physical potential. It matches the normal data, not arbitrary tangential data. In a suitable interior Hodge decomposition, the difference $u_v-\nabla\psi$ is divergence-free and tangent to S.

However, the latter case does not invalidate psi as an auxiliary datum for a correctly derived boundary correction. This is a substantive qualification to my earlier characterization of core leakage as necessarily introducing velocity-coupling bias.

Here is an explicit continuum construction for the wake-induced correction alone. Let Dcal be a conventional Laplace double-layer potential. Choose its density mu so that its interior trace satisfies

$$
\gamma^-\mathcal D\mu=-\psi|_S.
$$

By interior Dirichlet uniqueness,

$$
\mathcal D\mu=-\psi\quad\text{throughout }D.
$$

The double-layer normal derivative is continuous across S, so

$$
\partial_n^+\mathcal D\mu
=\partial_n^-\mathcal D\mu
=-g.
$$

Thus its exterior field is exactly the required harmonic normal-velocity correction, even though psi need not be a physical potential for the vortical wake. The normal-derivative continuity used here is a standard [double-layer jump relation](https://www.math.mcgill.ca/gantumur/math580f11/downloads/doublelayer.pdf), pp. 2–3. Signs of individual operators differ across conventions; the invariant statement is continuity of the normal derivative.

This derivation assumes regular compatible data, a suitable closed geometry, the genuine Laplace layer operator, and correct interior/exterior traces. It establishes a route to equivalence, not parity of discrete FLOWPanel routes. Source contributions, coupled bodies, attached sheets, Kutta conditions, and singular trailing-edge limits must be included consistently in the complete formulation. It must not be confused with prescribing the same trace as exterior Dirichlet data: interior and exterior Dirichlet-to-Neumann maps differ.

In particular, the difference between the reconstructed harmonic tangential velocity and the actual particle tangential velocity can falsify the claim that psi is the wake potential. It does not alone falsify a body correction that only needs the wake normal data and retains the actual incident particle velocity in the total field.

## Where a projection would discard physics

Replacing the actual incident velocity everywhere by a reconstructed gradient discards its non-gradient component. That would change vorticity and generally be wrong for a vortical wake.

Keeping the actual particle velocity and adding a harmonic correction does not do this. Likewise, using the interior harmonic reconstruction only as an auxiliary BIE datum need not do this, provided the BIE is equivalent to the desired Neumann correction.

The physical velocity remains particle velocity plus body correction, with any separately specified circulation fields. In a lifting calculation, the boundary/Kutta solve and shedding evolution must also determine the appropriate new circulation; harmonic correction of a frozen incident field alone does not supply an entire unsteady aerodynamic model.

## Pressure is still a separate question

An auxiliary harmonic psi cannot automatically be differentiated and inserted into Bernoulli as though its gradient were the actual incident velocity. In rotational flow, pressure follows Euler momentum, not a spatially global scalar Bernoulli formula for an invented total potential.

Therefore two statements can both hold:

- A Green-reconstructed auxiliary trace produces the correct instantaneous body velocity correction.
- Its time derivative is not the missing physical wake contribution to unsteady Bernoulli pressure.

Use genuine potential recovery where the relevant physical flow is irrotational with consistent branch/reference handling. Otherwise use the acceleration-based pressure reconstruction discussed in `052e2a-unsteady-pressure-followup-2026-09-12.md`.

Finally, instantaneous impermeability and correct curl do not prove dynamical Euler consistency. The particle strengths, positions, cores, and shedding must evolve consistently with the modeled equations. Nor does harmonic correction enforce viscous no-slip. These are separate modeling requirements from whether a source/doublet representation of the boundary correction exists.
