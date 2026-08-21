#=##############################################################################
transform_solver!: rigid-motion FastGaussSeidel reuse (FLOWPanel BRAINSTORM
021). Requires the Gravitational solver-compat overloads defined at the top
of solve_test.jl (influence!, target_influence_to_buffer!, value_to_strength!,
strength_to_value, buffer_to_system_strength!) — include after solve_test.jl,
or define them first when running standalone. `_rodrigues` comes from
transform_tree_test.jl.

FastGaussSeidel builds trees + interaction lists + dense influence matrices
once at construction; solve! refreshes source buffers and target influence
rows per call but never target POSITIONS or branch centers. Under rigid
motion every quantity in the scalar solve is invariant (relative distances
preserved), so a COLD FIXED-ITERATION solve — the suite's own repeatability
harness — must produce the same residual/strength trajectory before and
after (motion + transform_solver!), while an untransformed (stale) solver
forms far-field expansions about construction-time centers and must deviate:
that is the unsteady staleness bug this machinery fixes. Convergence-based
metrics are unusable here: the gravitational FGS case diverges to NaN when
pushed past the canonical 1e-3 tolerance, so trajectories are compared at
fixed iteration count instead.
=###############################################################################

@testset "transform_solver!: rigid-motion FGS reuse" begin

    n_bodies = 800
    seed = 20260815
    R = _rodrigues(SVector(0.2, 1.0, -0.5), 63.0 * pi / 180)
    t = SVector(0.6, -0.3, 0.4)
    fgs_kwargs = (; expansion_order=4, multipole_acceptance=0.5, leaf_size=40,
                  shrink=true, recenter=false)

    function make_system(; move::Bool)
        sys = generate_gravitational(seed, n_bodies)
        direct!(sys; scalar_potential=true, gradient=false)
        sys.potential[1, :] .*= -1.0
        if move
            for i in eachindex(sys.bodies)
                b = sys.bodies[i]
                sys.bodies[i] = typeof(b)(R * b.position + t, b.radius, b.strength)
            end
        end
        return sys
    end

    # cold fixed-iteration solve: zero strengths, fixed 6x2 iterations, no
    # threshold stopping (the same harness as the threaded-M2L repeatability
    # test) — returns the residual trajectory and final strengths
    function cold_fixed_solve!(sys, fgs)
        for i in eachindex(sys.bodies)
            b = sys.bodies[i]
            sys.bodies[i] = typeof(b)(b.position, b.radius, 0.0)
        end
        residuals = Float64[]
        FastMultipole.solve!(sys, fgs; scalar_potential=true, gradient=false,
            max_iterations=6, inner_iterations=2, tolerance=-1.0,
            reverse_pass=false, final_update=false, verbose=false,
            callback=(_, residual) -> push!(residuals, residual))
        return residuals, [b.strength for b in sys.bodies]
    end

    # baseline trajectory on the original geometry
    sys0 = make_system(move=false)
    fgs0 = FastMultipole.FastGaussSeidel((sys0,), (sys0,); fgs_kwargs...)
    @test length(fgs0.m2l_list) > 0            # premise: far field exercised
    @test length(fgs0.source_tree.leaf_index) > 1
    res0, x0 = cold_fixed_solve!(sys0, fgs0)
    @test all(isfinite, x0) && any(!iszero, x0)  # non-vacuous
    @test length(res0) == 6

    # transformed: built pre-motion, then rigidly followed — the invariant
    # solve replays the baseline trajectory (differences only from rotated
    # floating-point arithmetic)
    sys1 = make_system(move=false)
    fgs1 = FastMultipole.FastGaussSeidel((sys1,), (sys1,); fgs_kwargs...)
    for i in eachindex(sys1.bodies)
        b = sys1.bodies[i]
        sys1.bodies[i] = typeof(b)(R * b.position + t, b.radius, b.strength)
    end
    FastMultipole.transform_solver!(fgs1, (sys1,), R, t)
    res1, x1 = cold_fixed_solve!(sys1, fgs1)
    err_transformed = norm(x1 - x0) / norm(x0)
    @test err_transformed < 1e-9
    @test isapprox(res1, res0; rtol=1e-9)

    # stale: identical construction, motion NOT mirrored — far field about
    # construction-time centers must measurably corrupt the trajectory
    sys2 = make_system(move=false)
    fgs2 = FastMultipole.FastGaussSeidel((sys2,), (sys2,); fgs_kwargs...)
    for i in eachindex(sys2.bodies)
        b = sys2.bodies[i]
        sys2.bodies[i] = typeof(b)(R * b.position + t, b.radius, b.strength)
    end
    res2, x2 = cold_fixed_solve!(sys2, fgs2)
    err_stale = norm(x2 - x0) / norm(x0)
    @test err_stale > 1e-3                     # premise: staleness bites
    @test err_stale > 1e3 * err_transformed

    # gradient solve on a transformed solver refuses (dense matrices embed
    # build-time gradient rows, which do not rotate with the body)
    @test_throws ArgumentError FastMultipole.solve!(sys1, fgs1;
        scalar_potential=false, gradient=true, max_iterations=2,
        tolerance=1e-3, verbose=false)

    # an untransformed solver still accepts gradient solves (scalar_potential
    # stays on: the Gravitational influence! compat reads the potential row)
    sys3 = make_system(move=false)
    fgs3 = FastMultipole.FastGaussSeidel((sys3,), (sys3,); fgs_kwargs...)
    FastMultipole.solve!(sys3, fgs3; scalar_potential=true, gradient=true,
        max_iterations=2, tolerance=1e-3, verbose=false)
    @test all(isfinite(b.strength) for b in sys3.bodies)

end

@testset "transform_solver!: body-count mismatch" begin
    sys = generate_gravitational(71, 96)
    solver = FastMultipole.FastGaussSeidel((sys,);
        expansion_order=5, multipole_acceptance=0.4, leaf_size=12,
        cache_leaf_lu=false)
    smaller = generate_gravitational(72, 95)

    @test_throws ArgumentError FastMultipole.transform_solver!(solver,
        (smaller,), Matrix{Float64}(I, 3, 3), SVector(0.0, 0.0, 0.0))
end
