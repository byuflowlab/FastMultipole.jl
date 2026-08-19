#=##############################################################################
FmmPlan: prebuilt-plan fmm! must reproduce the allocating fmm! entry point
bitwise, across repeated calls with changed source strengths (the plan's
validity contract: frozen geometry, mutable strengths).

Motivated by FLOWPanel BRAINSTORM 021 Phase 2b: matrix-free Krylov solves
rebuild both trees + lists on every operator apply; the plan path reuses them
within a solve.
=###############################################################################

@testset "FmmPlan: prebuilt-plan fmm! equivalence" begin

    n_bodies = 2000
    expansion_order = 8
    multipole_acceptance = 0.4
    leaf_size_source = 30

    sys_plan = generate_gravitational(123, n_bodies)
    sys_ref = generate_gravitational(123, n_bodies)

    fresh_kwargs = (; expansion_order, multipole_acceptance, leaf_size_source,
                    scalar_potential=true, gradient=true, hessian=false)

    plan = FastMultipole.FmmPlan((sys_plan,), (sys_plan,); fresh_kwargs...)

    # premise guards: the case must exercise both farfield and nearfield
    @test length(plan.m2l_list) > 0
    @test length(plan.direct_list) > 0

    strengths_prev = [b.strength for b in sys_plan.bodies]
    for trial in 1:3
        # mutate strengths identically in both systems (trial > 1 proves the
        # per-call strength refresh — premise guard below checks they changed)
        if trial > 1
            for i in 1:n_bodies
                b = sys_plan.bodies[i]
                new_strength = b.strength * (1.0 + 0.5 * trial) + 0.001 * i
                sys_plan.bodies[i] = typeof(b)(b.position, b.radius, new_strength)
                r = sys_ref.bodies[i]
                sys_ref.bodies[i] = typeof(r)(r.position, r.radius, new_strength)
            end
            @test any(sys_plan.bodies[i].strength != strengths_prev[i]
                      for i in 1:n_bodies)
            strengths_prev .= [b.strength for b in sys_plan.bodies]
        end

        sys_plan.potential .= 0
        sys_ref.potential .= 0

        FastMultipole.fmm!((sys_plan,), (sys_plan,), plan)
        FastMultipole.fmm!((sys_ref,), (sys_ref,); fresh_kwargs...)

        # bitwise: identical trees/lists/order => identical arithmetic
        @test sys_plan.potential == sys_ref.potential
        @test any(!iszero, sys_plan.potential)   # non-vacuous
    end

    # geometry-change guard: mismatched body count must throw
    sys_small = generate_gravitational(7, n_bodies ÷ 2)
    @test_throws ArgumentError FastMultipole.fmm!((sys_small,), (sys_small,), plan)

end
