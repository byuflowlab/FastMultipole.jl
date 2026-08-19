#=##############################################################################
NearfieldInfluenceCache: cached near-field must reproduce the kernel
near-field (rtol 1e-12 — BLAS sums in a different order than the kernel's
per-source accumulation, so bitwise equality is NOT expected across the two
paths), reuse across strength changes, guard tree identity, refuse
direct_conditioning, enforce the memory cap before allocation, and evaluate
deterministically (bitwise) at any thread count.

Motivated by FLOWPanel BRAINSTORM 021 Phase 2b: ~40% of R1 krylov_ilu solve
samples are near-field kernel calls; for frozen relative geometry those are a
linear map cacheable as packed BLAS matvecs.
=###############################################################################

@testset "NearfieldInfluenceCache: cached near-field" begin

    n_bodies = 2000
    plan_kwargs = (; expansion_order=8, multipole_acceptance=0.4,
                   leaf_size_source=30, scalar_potential=true, gradient=true,
                   hessian=false)

    sys = generate_gravitational(123, n_bodies)
    plan = FastMultipole.FmmPlan((sys,), (sys,); plan_kwargs...)

    # premise guards: the case must have a real near field AND a real far field
    @test length(plan.direct_list) > 0
    @test length(plan.m2l_list) > 0

    tt = plan.target_tree
    st = plan.source_tree
    switches = plan.derivatives_switches
    out_range = FastMultipole.output_range(switches[1])

    cache = NearfieldInfluenceCache((sys,), tt, (sys,), st, plan.direct_list, switches)
    @test cache.bytes > 0
    @test cache.build_time > 0.0

    kernel_nearfield! = function ()
        FastMultipole.reset!(tt.buffers)
        FastMultipole.nearfield_singlethread!(tt.buffers, tt.branches, (sys,),
            st.buffers, st.branches, switches, plan.direct_list)
        return deepcopy(tt.buffers[1][out_range, :])
    end
    cached_nearfield! = function (n_threads)
        FastMultipole.reset!(tt.buffers)
        nearfield_matvec!(tt.buffers, cache, st.buffers; n_threads)
        return deepcopy(tt.buffers[1][out_range, :])
    end

    #--- test 1: exactness, scalar system ---#

    ref = kernel_nearfield!()
    @test any(!iszero, ref)   # non-vacuous
    got = cached_nearfield!(1)
    @test isapprox(got, ref; rtol=1e-12)

    #--- test 7: determinism (bitwise), any thread count ---#

    got_1t_again = cached_nearfield!(1)
    @test got_1t_again == got
    got_mt = cached_nearfield!(Threads.nthreads())
    got_mt_again = cached_nearfield!(Threads.nthreads())
    @test got_mt == got_mt_again
    @test got_mt == got   # owner-partitioned accumulate ⇒ thread-count invariant

    #--- test 3: strength-change reuse (only strengths are read at eval) ---#

    old_strengths = copy(st.buffers[1][5, :])
    st.buffers[1][5, :] .= old_strengths .* 3.0 .+ 1e-3
    @test any(st.buffers[1][5, :] .!= old_strengths)   # premise: they changed
    ref2 = kernel_nearfield!()
    got2 = cached_nearfield!(Threads.nthreads())
    @test isapprox(got2, ref2; rtol=1e-12)
    @test !isapprox(got2, ref; rtol=1e-6)   # premise: new strengths changed the answer
    st.buffers[1][5, :] .= old_strengths

    #--- test 4: tree-identity guard ---#

    FastMultipole.check_cache_trees(cache, tt, st)   # correct trees: no throw
    plan2 = FastMultipole.FmmPlan((sys,), (sys,); plan_kwargs...)
    @test_throws ArgumentError FastMultipole.check_cache_trees(cache,
        plan2.target_tree, plan2.source_tree)
    # body-count guard at evaluation
    sys_small = generate_gravitational(7, n_bodies ÷ 2)
    plan_small = FastMultipole.FmmPlan((sys_small,), (sys_small,); plan_kwargs...)
    @test_throws ArgumentError nearfield_matvec!(plan_small.target_tree.buffers,
        cache, plan_small.source_tree.buffers)

    #--- test 5: direct_conditioning refusal ---#

    noop_conditioner = (source_buffer, source_system, i_source_system,
        target_buffer, i_target_system) -> nothing
    rule = DirectConditioningRule(SelfPairs(), noop_conditioner, noop_conditioner)
    @test_throws ArgumentError NearfieldInfluenceCache((sys,), tt, (sys,), st,
        plan.direct_list, switches; direct_conditioning=rule)

    #--- test 6: memory cap enforced before allocation ---#

    tiny = 64
    @test cache.bytes > tiny   # premise: estimator exceeds the cap
    @test_throws ArgumentError NearfieldInfluenceCache((sys,), tt, (sys,), st,
        plan.direct_list, switches; max_bytes=tiny)

    #--- estimator + build-time cap (estimate BEFORE building, never after) ---#

    est = estimate_nearfield_cache(tt, st, plan.direct_list, switches, (sys,))
    @test est.bytes == cache.bytes            # exact same size-pass arithmetic
    @test est.n_blocks == length(cache.entries)
    @test est.total_probe_pairs > 0
    @test est.est_build_time > 0.0            # sampled kernel time
    est_nosample = estimate_nearfield_cache(tt, st, plan.direct_list, switches,
        (sys,); sample=false)
    @test est_nosample.bytes == est.bytes
    @test isnan(est_nosample.est_build_time)
    # absurdly small max_build_time throws before probing
    tiny_time = est.est_build_time / 1e6
    @test est.est_build_time > tiny_time      # premise: estimate exceeds the cap
    @test_throws ArgumentError NearfieldInfluenceCache((sys,), tt, (sys,), st,
        plan.direct_list, switches; max_build_time=tiny_time)

end

@testset "NearfieldInfluenceCache: fmm!/FmmPlan integration" begin

    n_bodies = 2000
    plan_kwargs = (; expansion_order=8, multipole_acceptance=0.4,
                   leaf_size_source=30, scalar_potential=true, gradient=true,
                   hessian=false)

    sys_cached = generate_gravitational(123, n_bodies)
    sys_ref = generate_gravitational(123, n_bodies)
    plan_cached = FastMultipole.FmmPlan((sys_cached,), (sys_cached,); plan_kwargs...)
    plan_ref = FastMultipole.FmmPlan((sys_ref,), (sys_ref,); plan_kwargs...)

    # premise guards: real near field AND real far field
    @test length(plan_cached.direct_list) > 0
    @test length(plan_cached.m2l_list) > 0

    cache = build_nearfield_cache!(plan_cached, (sys_cached,), (sys_cached,))
    @test plan_cached.nearfield_cache[] === cache
    @test isnothing(plan_ref.nearfield_cache[])

    for trial in 1:2
        if trial > 1
            # mutate strengths identically in both systems (proves cache
            # validity across strength changes through the plan path)
            for i in 1:n_bodies
                b = sys_cached.bodies[i]
                new_strength = b.strength * 1.5 + 0.001 * i
                sys_cached.bodies[i] = typeof(b)(b.position, b.radius, new_strength)
                r = sys_ref.bodies[i]
                sys_ref.bodies[i] = typeof(r)(r.position, r.radius, new_strength)
            end
        end

        sys_cached.potential .= 0
        sys_ref.potential .= 0
        FastMultipole.fmm!((sys_cached,), (sys_cached,), plan_cached)
        FastMultipole.fmm!((sys_ref,), (sys_ref,), plan_ref)

        @test any(!iszero, sys_ref.potential)   # non-vacuous
        # rtol, not bitwise: the cached near field sums in BLAS order
        @test isapprox(sys_cached.potential, sys_ref.potential; rtol=1e-12)
    end

    # tune=true with a provided cache suggests a new leaf from cached timing
    # (user decides whether to re-cache) — see the cached-path tuning testset

end

@testset "NearfieldInfluenceCache: cached-path tuning" begin

    n_bodies = 2000
    plan_kwargs = (; expansion_order=8, multipole_acceptance=0.4,
                   leaf_size_source=30, scalar_potential=true, gradient=true,
                   hessian=false)

    #--- provided cache + tune=true: leaf suggested from cached timing ---#

    sys = generate_gravitational(321, n_bodies)
    sys_ref = generate_gravitational(321, n_bodies)
    plan = FastMultipole.FmmPlan((sys,), (sys,); plan_kwargs...)
    @test length(plan.direct_list) > 0   # premise: real near field
    @test length(plan.m2l_list) > 0      # premise: real far field
    build_nearfield_cache!(plan, (sys,), (sys,))

    sys.potential .= 0
    sys_ref.potential .= 0
    result = FastMultipole.fmm!((sys,), (sys,), plan; tune=true)
    optargs = result[1]
    # a leaf suggestion IS returned (computed from the cached per-interaction
    # timing); acting on it means new trees, so the USER decides whether to
    # rebuild the cache at the suggested leaf (Ryan 2026-08-19)
    @test all(optargs.leaf_size_source .>= 1)
    @test optargs.expansion_order == plan_kwargs.expansion_order
    @test optargs.nearfield_cache_feasible === true
    @test optargs.nearfield_cache_build_time == 0.0   # provided, not built here
    # outputs unaffected by tune=true on the cached path
    FastMultipole.fmm!((sys_ref,), (sys_ref,); plan_kwargs...)
    @test any(!iszero, sys_ref.potential)   # non-vacuous
    @test isapprox(sys.potential, sys_ref.potential; rtol=1e-12)

    #--- tune_fmm with per-trial throwaway caches (uncapped) ---#

    tune_kwargs = (; scalar_potential=true, gradient=true, hessian=false,
                   leaf_size_source=20, expansion_order=6,
                   multipole_acceptances=[0.4], verbose=false)
    sys_a = generate_gravitational(11, 800)
    tuned, _, info = FastMultipole.tune_fmm((sys_a,), (sys_a,);
        tune_kwargs..., tune_nearfield_cache=true)
    @test all(tuned.leaf_size_source .>= 1)
    @test tuned.expansion_order >= 1
    @test info.cache_capped === false   # premise: 4 GiB cap can't bind at n=800

    #--- capped run: leaf changes stop at the last cache-feasible trial ---#

    # place the cap BETWEEN the initial trial's bytes and the (measured)
    # tuned-leaf bytes so the first trial fits and the tuner's suggested move
    # does not (at this scale the tuner SHRINKS the leaf and bytes grow with
    # smaller leaves — the clamp is direction-agnostic either way)
    est_at(leaf) = begin
        p = FastMultipole.FmmPlan((sys_a,), (sys_a,);
            scalar_potential=true, gradient=true, hessian=false,
            leaf_size_source=leaf, expansion_order=6, multipole_acceptance=0.4)
        estimate_nearfield_cache(p.target_tree, p.source_tree, p.direct_list,
            p.derivatives_switches, (sys_a,); sample=false)
    end
    est0 = est_at(20)
    est_move = est_at(tuned.leaf_size_source)
    @test est_move.bytes > est0.bytes   # premise: the tuner's move costs more bytes
    max_bytes_cap = (est0.bytes + est_move.bytes) ÷ 2
    tuned_c, _, info_c = FastMultipole.tune_fmm((sys_a,), (sys_a,);
        tune_kwargs..., tune_nearfield_cache=true,
        nearfield_cache_max_bytes=max_bytes_cap)
    @test info_c.cache_capped === true   # premise: the cap actually bound
    # the returned knobs must themselves be cache-feasible under the cap
    plan_tuned = FastMultipole.FmmPlan((sys_a,), (sys_a,);
        scalar_potential=true, gradient=true, hessian=false,
        leaf_size_source=tuned_c.leaf_size_source,
        expansion_order=tuned_c.expansion_order,
        multipole_acceptance=tuned_c.multipole_acceptance)
    est_tuned = estimate_nearfield_cache(plan_tuned.target_tree,
        plan_tuned.source_tree, plan_tuned.direct_list,
        plan_tuned.derivatives_switches, (sys_a,); sample=false)
    @test est_tuned.bytes <= max_bytes_cap

    # first trial already infeasible -> loud error
    @test_throws ErrorException FastMultipole.tune_fmm((sys_a,), (sys_a,);
        tune_kwargs..., tune_nearfield_cache=true, nearfield_cache_max_bytes=64)

    #--- multi-system throwaway tuning completes (attribution smoke) ---#

    sys_b = generate_gravitational(12, 400)
    tuned_m, _, info_m = FastMultipole.tune_fmm((sys_a, sys_b), (sys_a, sys_b);
        scalar_potential=true, gradient=true, hessian=false,
        leaf_size_source=20, expansion_order=6,
        multipole_acceptances=[0.4], verbose=false,
        tune_nearfield_cache=true)
    @test all(tuned_m.leaf_size_source .>= 1)
    @test info_m.cache_capped === false

end

@testset "NearfieldInfluenceCache: standalone direct!" begin

    n_targets = 300
    n_sources = 400
    sys_t = generate_gravitational(21, n_targets)
    sys_s = generate_gravitational(22, n_sources)
    sys_t_ref = generate_gravitational(21, n_targets)
    sys_s_ref = generate_gravitational(22, n_sources)

    targets = (sys_t,)
    sources = (sys_s,)
    switch_kwargs = (; scalar_potential=true, gradient=true, hessian=false)

    # treeless cache over full index ranges
    switches = FastMultipole.DerivativesSwitch([true], [true], [false], targets)
    target_buffers = FastMultipole.allocate_buffers(targets, true, Float64, switches)
    FastMultipole.target_to_buffer!(target_buffers, targets,
        SVector{1}([1:n_targets]), switches)
    source_buffers = FastMultipole.allocate_buffers(sources, false, Float64, switches)
    FastMultipole.system_to_buffer!(source_buffers, sources)
    cache = NearfieldInfluenceCache(targets, target_buffers, sources,
        source_buffers, switches)
    @test cache.target_tree_id == 0   # treeless: no tree identity to guard

    sys_t.potential .= 0
    sys_t_ref.potential .= 0
    direct!(targets, sources; switch_kwargs..., n_threads=1,
        target_buffers, source_buffers, nearfield_cache=cache)
    direct!((sys_t_ref,), (sys_s_ref,); switch_kwargs..., n_threads=1)
    @test any(!iszero, sys_t_ref.potential)   # non-vacuous
    @test isapprox(sys_t.potential, sys_t_ref.potential; rtol=1e-12)

    # cached eval refuses conditioning rules
    noop_conditioner = (source_buffer, source_system, i_source_system,
        target_buffer, i_target_system) -> nothing
    rule = DirectConditioningRule(SelfPairs(), noop_conditioner, noop_conditioner)
    @test_throws ArgumentError direct!(targets, sources; switch_kwargs...,
        n_threads=1, target_buffers, source_buffers, nearfield_cache=cache,
        direct_conditioning=rule)

end
