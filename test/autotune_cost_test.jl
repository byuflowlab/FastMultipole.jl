#=
`tune_fmm_perturb(...; cost=)` — the caller-supplied objective (BRAINSTORM 021,
Ryan 2026-08-24).

Everything here runs against an ANALYTIC cost surface rather than real timings,
so the test asserts descent logic, not performance, and is immune to machine
noise. That matters: the feature exists precisely because timing-based
objectives were found to be unreliable at the 15% level on non-exclusive
hardware.
=#

@testset "tune_fmm_perturb cost=" begin
    sys = generate_gravitational(123, 200)

    # unique minimum at P=10, MAC=0.5, leaf=25; leaf > 200 is "infeasible"
    # (stands in for a near-field-cache memory budget)
    calls = Ref(0)
    function surface(; expansion_order, multipole_acceptance, leaf_size_source)
        calls[] += 1
        leaf_size_source > 200 && return (Inf, false)
        t = (expansion_order - 10)^2 + 400*(multipole_acceptance - 0.5)^2 +
            0.02*(leaf_size_source - 25)^2 + 1.0
        return (t, true)
    end

    tuned, _, info = FastMultipole.tune_fmm_perturb(sys, sys;
        expansion_order=14, multipole_acceptance=0.7, leaf_size_source=100,
        reps=3, cost=surface, abandon_factor=Inf, verbose=false)

    @test tuned.expansion_order == 10
    @test isapprox(tuned.multipole_acceptance, 0.5; atol=1e-9)
    # the leaf neighbourhood is multiplicative (x/÷1.5) from 100, so 20 and 30
    # bracket the true optimum 25 and tie on this surface
    @test 19 <= tuned.leaf_size_source <= 33

    # memoization: every candidate evaluated exactly `reps` times, never twice
    @test calls[] == info.n_candidates * 3

    # an infeasible starting point must fail loudly, not descend silently
    @test_throws ErrorException FastMultipole.tune_fmm_perturb(sys, sys;
        expansion_order=14, multipole_acceptance=0.7, leaf_size_source=500,
        cost=surface, verbose=false)

    # knobs the cost function supersedes are REFUSED, never silently ignored —
    # silent ignoring is the failure class this feature was built to remove
    base = (; expansion_order=10, multipole_acceptance=0.5,
              leaf_size_source=25, cost=surface, verbose=false)
    @test_throws ArgumentError FastMultipole.tune_fmm_perturb(sys, sys;
        base..., tree_amortization=Inf)
    @test_throws ArgumentError FastMultipole.tune_fmm_perturb(sys, sys;
        base..., error_tolerance=FastMultipole.PowerAbsolutePotential(1e-3))
    @test_throws ArgumentError FastMultipole.tune_fmm_perturb(sys, sys;
        base..., scalar_potential=true)

    # early abandonment still fires, and still saves evaluations
    calls[] = 0
    _, _, i2 = FastMultipole.tune_fmm_perturb(sys, sys; expansion_order=14,
        multipole_acceptance=0.7, leaf_size_source=100, reps=5,
        cost=surface, abandon_factor=1.05, verbose=false)
    @test i2.n_abandoned > 0
    @test calls[] < i2.n_candidates * 5

    # all three accepted return shapes
    for f in ((; expansion_order, multipole_acceptance, leaf_size_source) ->
                  Float64(expansion_order),
              (; expansion_order, multipole_acceptance, leaf_size_source) ->
                  (; t=Float64(expansion_order), success=true))
        t, = FastMultipole.tune_fmm_perturb(sys, sys; expansion_order=6,
            multipole_acceptance=0.5, leaf_size_source=25, cost=f,
            verbose=false)
        @test t.expansion_order == 1
    end
    @test_throws ArgumentError FastMultipole.tune_fmm_perturb(sys, sys;
        expansion_order=10, multipole_acceptance=0.5, leaf_size_source=25,
        verbose=false, cost=(; kwargs...) -> "not a number")
end

#=
`memo=` / `on_measure=` — interruption-transparent tuning (BRAINSTORM 021,
Ryan's ruling 2026-08-24: "checkpoint every benchmark during tuning so a
re-launched job picks up where the interrupted one stopped").

The claim under test is REPLAY FAITHFULNESS: a descent resumed from a persisted
memo must walk the identical path and reach the identical winner as one that
was never interrupted, at whatever prefix it was cut. Same analytic surface as
above, so this asserts control flow only.
=#
@testset "tune_fmm_perturb memo= / on_measure=" begin
    sys = generate_gravitational(123, 200)

    calls = Ref(0)
    function surface(; expansion_order, multipole_acceptance, leaf_size_source)
        calls[] += 1
        leaf_size_source > 200 && return (Inf, false)
        t = (expansion_order - 10)^2 + 400*(multipole_acceptance - 0.5)^2 +
            0.02*(leaf_size_source - 25)^2 + 1.0
        return (t, true)
    end

    # `abandon_factor` deliberately tight: abandonment depends on the running
    # `t_best_ok`, which is exactly what a memo hit has to keep updating.
    start = (; expansion_order=14, multipole_acceptance=0.7,
               leaf_size_source=100, reps=5, cost=surface,
               abandon_factor=1.05, verbose=false)

    # --- reference: one uninterrupted descent, traced -----------------------
    trace = Tuple{Int,Float64,Any,Float64,Bool,Bool}[]
    calls[] = 0
    ref_tuned, ref_hist, ref_info = FastMultipole.tune_fmm_perturb(sys, sys;
        start..., on_measure=(P, mac, leaf, t, s, a) ->
            push!(trace, (P, mac, leaf, t, s, a)))
    ref_calls = calls[]

    # the hook fires once per freshly measured candidate — never on a memo hit
    @test length(trace) == ref_info.n_candidates
    @test allunique([(r[1], round(r[2]; digits=3), r[3]) for r in trace])

    # rebuild a memo from a prefix of the trace, exactly as the driver's
    # trace file replay does
    seed(n) = Dict{Any, @NamedTuple{t::Float64, success::Bool, abandoned::Bool}}(
        (r[1], round(r[2]; digits=3), r[3]) => (; t=r[4], success=r[5], abandoned=r[6])
        for r in trace[1:n])

    # --- full replay: zero measurements, identical answer -------------------
    calls[] = 0
    m = seed(length(trace))
    tuned, hist, info = FastMultipole.tune_fmm_perturb(sys, sys; start..., memo=m)
    @test calls[] == 0                      # every candidate came from the memo
    @test tuned == ref_tuned
    @test hist == ref_hist
    @test info.n_candidates == ref_info.n_candidates
    @test info.n_abandoned == ref_info.n_abandoned

    # --- partial replay at every cut point: replay prefix, then continue ----
    # This is the real acceptance test. If a memo hit failed to tighten
    # `t_best_ok`, the resumed run would leave the abandonment cutoff at Inf,
    # measure candidates the reference abandoned, and diverge here.
    for n in 1:length(trace)
        calls[] = 0
        t2, h2, i2 = FastMultipole.tune_fmm_perturb(sys, sys; start..., memo=seed(n))
        @test t2 == ref_tuned
        @test h2 == ref_hist
        @test i2.n_abandoned == ref_info.n_abandoned
        # replaying a prefix strictly saves work, and more prefix saves more
        @test calls[] < ref_calls
    end

    # --- the memo is shared, not copied: the caller's dict is filled in -----
    m2 = Dict{Any, @NamedTuple{t::Float64, success::Bool, abandoned::Bool}}()
    FastMultipole.tune_fmm_perturb(sys, sys; start..., memo=m2)
    @test length(m2) == ref_info.n_candidates
end
