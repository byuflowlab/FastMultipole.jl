#------- choose max expansion order -------#

function leaf_size_converged(optargs, leaf_size_source)
    converged = true
    for i in eachindex(optargs.leaf_size_source)
        converged = abs(optargs.leaf_size_source[i] - leaf_size_source[i]) < 0.1 * leaf_size_source[i]
    end
    return converged
end

tune_fmm(system; kwargs...) = tune_fmm(system, system; kwargs...)

function tune_fmm(target_systems, source_systems; kwargs...)
    # promote arguments to Tuples
    target_systems = to_tuple(target_systems)
    source_systems = to_tuple(source_systems)

    return tune_fmm(target_systems, source_systems; kwargs...)
end

"""
    tune_fmm(target_systems, source_systems; optargs...)

Tune the Fast Multipole Method (FMM) parameters for optimal performance on the given target and source systems, optionally subject to an error tolerance.

**Arguments**

- `target_systems::Union{Tuple,{UserDefinedSystem}}`: a user-defined system object (or a tuple of them) for which the FMM interface functions have been defined
- `source_systems::Union{Tuple,{UserDefinedSystem}}`: a user-defined system object (or a tuple of them) for which the FMM interface functions have been defined

**Keyword Arguments**

- `error_tolerance::Union{Nothing,Float64}`: the error tolerance for the FMM; if `nothing`, the FMM will simply use the `expansion_order` keyword argument to fix the expansion order
- `expansion_order::Int`: the max expansion order for the FMM; defaults to 4
- `leaf_size_source::Int`: the leaf size for the source systems; defaults to `default_leaf_size(source_systems)`
- `max_expansion_order::Int`: the maximum allowable expansion order if an error tolerance is requested; defaults to 20
- `multipole_acceptances::AbstractRange{Float64}`: a range of multipole acceptance critia to test; defaults to `range(0.3, stop=0.8, step=0.1)`
- `lamb_helmholtz::Bool`: whether to use the Lamb-Hellmholtz decomposition; defaults to `false`
- `verbose::Bool`: whether to print progress information; defaults to `true`
- `kwargs...`: additional keyword arguments to pass to the `fmm!` function

**Returns**

- `tuned_params::NamedTuple`: a named tuple containing the best parameters found during tuning, which can be used in subsequent `fmm!` calls by splatting it as a keyword argument:`:
  
    - `leaf_size_source::Int`: the optimal leaf size for the source systems
    - `expansion_order::Int`: the optimal expansion order for the FMM
    - `multipole_acceptance::Float64`: the optimal multipole acceptance criterion

- `cache::Tuple`: a tuple containing the cache used during tuning, which can be reused for subsequent `fmm!` calls by splatting it as a keyword argument

"""
function tune_fmm(target_systems::Tuple, source_systems::Tuple;
    error_tolerance=nothing,
    expansion_order=4, leaf_size_source=default_leaf_size(source_systems),
    max_expansion_order=20, # max_iter=10,
    multipole_acceptances=range(0.3, stop=0.8, step=0.1),
    tune_nearfield_cache::Bool=false,
    nearfield_cache_max_bytes::Integer=NEARFIELD_CACHE_DEFAULT_MAX_BYTES,
    nearfield_cache_max_build_time::Real=Inf,
    verbose=true, kwargs...
)

    if verbose
        println("\n#======= Begin FastMultipole.tune_fmm() =======#")
    end

    #--- save best parameters ---#

    t_fmm_best = Inf
    expansion_order_best = max_expansion_order
    leaf_size_source_best = leaf_size_source
    multipole_acceptance_best = multipole_acceptances[1]
    original_max_expansion_order = max_expansion_order

    #--- preallocate cache ---#

    # kwargs must be forwarded so the cache is allocated with the same target
    # buffer layout (scalar_potential/gradient/hessian) as the tuning calls below
    t_fmm = @elapsed _, cache, _ = fmm!(target_systems, source_systems;
                       expansion_order=1, leaf_size_source,
                       nearfield=false, farfield=false, self_induced=false,
                       kwargs...,
                       tune=true, update_target_systems=false
                      )

    #--- cached-near-field tuning state (see tune_nearfield_cache) ---#

    # "stop growth at the cap" (Ryan 2026-08-19): when a trial's throwaway
    # cache exceeds nearfield_cache_max_bytes or _max_build_time, clamp
    # leaf_size_source to the last cache-feasible trial and keep tuning there.
    last_feasible_leaf = nothing
    cache_capped = false

    # one tuning fmm! call; under tune_nearfield_cache this re-runs at the
    # clamped leaf when the trial is infeasible, and excludes the throwaway
    # cache's build time from the returned wall clock (build cost is
    # amortized in production, so it must not steer the MAC comparison)
    function tuned_fmm!(P, leaf, mac)
        t = @elapsed result = fmm!(target_systems, source_systems, cache;
                                   expansion_order=P, leaf_size_source=leaf,
                                   multipole_acceptance=mac,
                                   error_tolerance, kwargs...,
                                   tune=true, update_target_systems=false,
                                   tune_nearfield_cache,
                                   nearfield_cache_max_bytes,
                                   nearfield_cache_max_build_time,
                                  )
        if tune_nearfield_cache
            optargs = result[1]
            if optargs.nearfield_cache_feasible
                last_feasible_leaf = leaf
            else
                isnothing(last_feasible_leaf) && error(
                    "tune_nearfield_cache: the first tuning trial " *
                    "(leaf_size_source=$leaf, multipole_acceptance=$mac) already " *
                    "exceeds nearfield_cache_max_bytes=$nearfield_cache_max_bytes " *
                    "or nearfield_cache_max_build_time=$nearfield_cache_max_build_time — " *
                    "raise the caps or start from a smaller leaf_size_source")
                cache_capped = true
                verbose && println("\tnearfield cache cap reached at leaf_size_source=$leaf; clamping to $last_feasible_leaf")
                leaf = last_feasible_leaf
                t = @elapsed result = fmm!(target_systems, source_systems, cache;
                                           expansion_order=P, leaf_size_source=leaf,
                                           multipole_acceptance=mac,
                                           error_tolerance, kwargs...,
                                           tune=true, update_target_systems=false,
                                           tune_nearfield_cache,
                                           nearfield_cache_max_bytes,
                                           nearfield_cache_max_build_time,
                                          )
            end
            t -= result[1].nearfield_cache_build_time
        end
        return t, result, leaf
    end

    #--- error tolerance selected ---#

    for multipole_acceptance in multipole_acceptances

        if verbose
            println("\nmultipole_acceptance = $multipole_acceptance...")
        end

        # initial fmm! call with max_expansion_order to get leaf_size
        t_fmm, result, leaf_used = tuned_fmm!(
            isnothing(error_tolerance) ? expansion_order : max_expansion_order,
            leaf_size_source, multipole_acceptance)
        optargs, _, _, _, m2l_list, _, _, error_success = result
        leaf_size_source = leaf_used

        # in case error is not satisfied
        if !error_success
            println("\terror tolerance not satisfied for max expansion order P=$max_expansion_order;")
            println("\tskipping this multipole_acceptance...")
            continue
        end

        leaf_size_source = optargs.leaf_size_source
        # once capped, pin to the last cache-feasible leaf (direction-agnostic:
        # at small scales the model can suggest SMALLER leaves with MORE bytes)
        cache_capped && (leaf_size_source = last_feasible_leaf)
        this_max_expansion_order = optargs.expansion_order + 2

        # second fmm! call with optimal leaf_size to get expansion order
        t_fmm, result, leaf_used = tuned_fmm!(
            isnothing(error_tolerance) ? expansion_order : this_max_expansion_order,
            leaf_size_source, multipole_acceptance)
        optargs, _, _, _, m2l_list, _, _, error_success = result
        leaf_size_source = leaf_used

        if !error_success # better run at the actual max_expansion_order
            max_expansion_order = original_max_expansion_order
            t_fmm, result, leaf_used = tuned_fmm!(
                isnothing(error_tolerance) ? expansion_order : max_expansion_order,
                leaf_size_source, multipole_acceptance)
            optargs, _, _, _, m2l_list, _, _, error_success = result
            leaf_size_source = leaf_used

        end

        expansion_order = optargs.expansion_order

        # final benchmark
        t_fmm, result, leaf_used = tuned_fmm!(expansion_order, leaf_size_source,
            multipole_acceptance)
        optargs, _, _, _, m2l_list, _, _, error_success = result
        leaf_size_source = leaf_used

        # track the best parameters for this multipole_acceptance
        if t_fmm < t_fmm_best
            t_fmm_best = t_fmm
            expansion_order_best = expansion_order
            leaf_size_source_best = leaf_size_source
            multipole_acceptance_best = multipole_acceptance
        end

        #=
        # iterate to (loose) convergence
        i = 1
        for _ in 1:max_iter
            println("\t~~~ iteration $i ~~~")

            # in case m2l list is empty (direct calculation is probably best)
            if length(m2l_list) == 0 # likely won't get much better
                println("\tM2L list is empty; \n\tending iterations for this multipole_acceptance...")
                if t_fmm < t_fmm_best
                    this_t_fmm_best = t_fmm_best = t_fmm
                    this_leaf_size_source_best = leaf_size_source_best = get_n_bodies_vec(source_systems)
                    this_expansion_order_best = expansion_order_best = 1
                    multipole_acceptance_best = multipole_acceptance
                end
                break # next multipole_acceptance
            end

            # save the expansion order
            expansion_order = optargs.expansion_order

            # predict optimal leaf size
            t_fmm = @elapsed optargs, cache, _, _, m2l_list, _, _, error_success = fmm!(target_systems, source_systems;
                                                                                        expansion_order,
                                                                                        leaf_size_source, multipole_acceptance,
                                                                                        error_tolerance, kwargs..., cache,
                                                                                        tune=true, update_target_systems=false
                                                                                       )

            # save leaf size
            leaf_size_source = optargs.leaf_size_source

            # benchmark and check for convergence
            t_fmm = @elapsed optargs, cache, _, _, m2l_list, _, _, error_success = fmm!(target_systems, source_systems;
                                                                                        expansion_order,
                                                                                        leaf_size_source, multipole_acceptance,
                                                                                        error_tolerance, kwargs..., cache,
                                                                                        tune=true, update_target_systems=false
                                                                                       )
            if error_success # (loosely) converged

                # check if this is our best yet for this MAC
                if t_fmm < this_t_fmm_best
                    this_t_fmm_best = t_fmm
                end

                # check if this is our best yet for all MAC's
                if t_fmm < t_fmm_best
                    t_fmm_best = t_fmm
                    this_leaf_size_source_best = leaf_size_source_best = leaf_size_source
                    this_expansion_order_best = expansion_order_best = expansion_order
                    multipole_acceptance_best = multipole_acceptance
                end

                break # move to the next multipole_acceptance
            end

            i += 1
        end
        =#

        if verbose
            println("\n\tBest Parameters: ")
            println("\t\tleaf_size_source:    ", leaf_size_source)
            println("\t\texpansion_order:     ", expansion_order)
            println("\t\tmultipole_acceptance: ", multipole_acceptance)
            println("\t\tcost:                $t_fmm seconds")
        end

    end

    if verbose
        println("\nFinished autotune!")
        println("\nParameters: ")
        println("\tleaf_size_source:    ", leaf_size_source_best)
        println("\texpansion_order:     ", expansion_order_best)
        println("\tmultipole_acceptance: ", multipole_acceptance_best)
        println("\tcost:                $t_fmm_best seconds")
        println("\n#===============================================#\n")
    end

    tuned_params = (
                    leaf_size_source = leaf_size_source_best,
                    expansion_order = expansion_order_best,
                    multipole_acceptance = multipole_acceptance_best,
                   )

    # tune_info is a THIRD return value (existing `tuned, cache = tune_fmm(...)`
    # destructuring ignores it) so tuned_params stays splat-able into fmm!;
    # cache_capped=true means leaf growth was clamped at the nearfield-cache
    # bytes/build-time caps rather than at the cost-model optimum
    tune_info = (; cache_capped)

    return tuned_params, cache, tune_info
end


#------- perturbation descent on tuned parameters -------#

"""
    tune_fmm_perturb(target_systems, source_systems;
        expansion_order, multipole_acceptance, leaf_size_source, optargs...)

Greedy one-at-a-time perturbation descent on measured `fmm!` wall time,
starting from already-tuned parameters (e.g. the first return value of
[`tune_fmm`](@ref), splatted). `tune_fmm` trusts the cost model to pick
`leaf_size_source` and `expansion_order` per multipole acceptance; this
routine instead *measures* each neighbor of the current point and moves while
a neighbor beats the incumbent by more than `improve_tol` (relative), so a
model-vs-reality gap at production scale is closed by experiment (BRAINSTORM
023, Ryan 2026-08-20).

**Keyword Arguments**

- `expansion_order::Int`, `multipole_acceptance::Float64`, `leaf_size_source`:
  the starting point (splat `tune_fmm`'s `tuned_params`)
- `error_tolerance`: same contract as `fmm!`/`tune_fmm`; candidates whose
  trial reports `error_success=false` are rejected. `nothing` disables the
  guard (pure cost descent at fixed `expansion_order` semantics)
- `max_expansion_order=20`, `mac_step=0.05`, `mac_bounds=(0.25, 0.85)`,
  `leaf_factor=1.5`: neighborhood definition (`P ± 1`, `MAC ± mac_step`
  clamped, `leaf ×/÷ leaf_factor`)
- `reps=2`: min-of-reps timing per candidate
- `tree_amortization::Real=1`: how many `fmm!` applies share ONE tree +
  interaction-list build in the workload being priced. The candidate cost is
  `t_build / tree_amortization + t_apply`.
    - `1` (default, legacy behavior): the tree is rebuilt for every apply, so
      the build is timed inside each trial via the `Cache` path. This is the
      correct objective whenever the geometry moves — e.g. tuning a particle
      field, which must be re-treed every timestep.
    - `Inf`: charge nothing for the build. Correct when the geometry is frozen
      and one tree is reused indefinitely — the panels-on-panels operator in a
      panel solve is built once a priori and reused across every iteration AND
      every timestep, so its build is a one-off that should not influence the
      choice of knobs at all (Ryan, BRAINSTORM 021, 2026-08-24).
    - finite `n > 1`: in between — a build reused over exactly `n` applies.
      Set it to the expected iteration count.

  For any `tree_amortization != 1` the plan is built ONCE per candidate, timed
  separately, and only its amortized share (zero, for `Inf`) is charged.

  This matters because tree and interaction-list construction get MORE
  expensive as `leaf_size_source` shrinks, so charging a full build to every
  apply adds a leaf-dependent penalty that biases the descent toward large
  leaves. Measured (BRAINSTORM 021, 2026-08-24) at R1, 8016 panels: the descent
  stalled at leaf 45 because leaf 30 timed 1.4% WORSE under the `n=1`
  objective, while a Krylov solve — which reuses one build across every
  iteration — is ~15-20% FASTER at leaf 30.
- `abandon_factor=1.3`: early-abandonment threshold. A trial is stopped as soon
  as its running min exceeds `abandon_factor x` the fastest COMPLETE,
  error-satisfying candidate measured so far. Such a candidate can no longer be
  accepted — acceptance requires `t < t0*(1 - improve_tol)` and the incumbent
  `t0` is never below that best — so the remaining reps are wasted. With
  `reps=5` this cuts a hopeless candidate from 5 trials to 1, and under
  `tree_amortization > 1` it can skip the applies entirely when the amortized
  build alone already loses. The threshold only tightens as the descent
  improves, so an abandoned point stays rejected and its (over-estimated) time
  is safe to memoize. Abandoned candidates never tighten the threshold
  themselves, and are flagged `abandoned=true` in `history` — note that a
  candidate abandoned at the build stage has NOT had its error tolerance
  verified. `Inf` disables early abandonment.
- `max_seconds=Inf`: absolute wall-clock backstop, checked between candidates.
  On expiry the descent returns the best point found so far, warns, and reports
  `timed_out=true` in the third return value. A timed-out descent must never
  be read as a converged one.
- `improve_tol=0.02`: relative improvement required to accept a move
- `max_iters=20`: maximum accepted moves
- `verbose=true`
- `kwargs...`: forwarded to `fmm!` (must match the production call's
  `scalar_potential`/`gradient`/`hessian` request). Under
  `tree_amortization > 1` the structural ones (derivative switches, `shrink`,
  `recenter`, `leaf_size_target`, `interaction_list_method`, `farfield`,
  `nearfield`, `self_induced`, `extra_outputs`, `metadata`) are routed to
  `FmmPlan` and the rest to the per-apply `fmm!`.

**Returns**

- `tuned_params::NamedTuple`: `(leaf_size_source, expansion_order,
  multipole_acceptance)` at the cost minimum, splat-able into `fmm!`
- `history::Vector{<:NamedTuple}`: every evaluated candidate with fields
  `(iter, expansion_order, multipole_acceptance, leaf_size_source, t,
  error_success, abandoned, accepted)`
- `info::NamedTuple`: `(timed_out, t_elapsed, n_candidates, n_abandoned,
  t_best)`
"""
function tune_fmm_perturb(target_systems, source_systems;
    expansion_order, multipole_acceptance, leaf_size_source,
    error_tolerance=nothing,
    max_expansion_order=20,
    mac_step=0.05, mac_bounds=(0.25, 0.85),
    leaf_factor=1.5,
    reps=2, tree_amortization::Real=1, max_seconds=Inf, abandon_factor=1.3,
    improve_tol=0.02, max_iters=20,
    verbose=true, kwargs...)

    tree_amortization >= 1 || throw(ArgumentError(
        "tree_amortization must be >= 1 (got $tree_amortization)"))
    abandon_factor > 1 || throw(ArgumentError(
        "abandon_factor must be > 1 (got $abandon_factor); use Inf to disable"))

    target_systems = to_tuple(target_systems)
    source_systems = to_tuple(source_systems)

    # kwargs routing for the amortized path: everything that shapes the trees
    # or the interaction lists belongs to FmmPlan (which has no catch-all), the
    # rest to the per-apply fmm!
    plan_kwargs = (; (k => v for (k, v) in pairs(kwargs)
                      if k in FMMPLAN_STRUCTURAL_KWARGS)...)
    apply_kwargs = (; (k => v for (k, v) in pairs(kwargs)
                       if !(k in FMMPLAN_STRUCTURAL_KWARGS))...)

    # preallocate cache with the same target buffer layout as the trials
    # (legacy tree_amortization==1 path only; the amortized path owns a plan
    # per candidate, each carrying its own cache)
    cache = if tree_amortization == 1
        _, c, _ = fmm!(target_systems, source_systems;
            expansion_order=1, leaf_size_source,
            nearfield=false, farfield=false, self_induced=false,
            kwargs..., tune=true, update_target_systems=false)
        c
    else
        nothing
    end

    scale_leaf(leaf::Integer, f) = max(1, round(Int, leaf * f))
    scale_leaf(leaf, f) = map(l -> max(1, round(Int, l * f)), leaf)
    key(P, mac, leaf) = (P, round(mac; digits=3), leaf)

    # fastest ERROR-SATISFYING, fully measured candidate so far. Any trial that
    # climbs past `t_best_ok[] * abandon_factor` is stopped where it stands:
    # it can no longer win, because acceptance needs t < t0*(1-improve_tol) and
    # t0 is never below t_best_ok[]. The threshold only ever tightens, so an
    # abandoned point stays rejected and its (over-estimated) time is safe to
    # memoize. Costs nothing when reps==1; saves reps-1 trials per hopeless
    # candidate otherwise, which is most of them once the descent gets going.
    t_best_ok = Ref(Inf)

    memo = Dict{Any, @NamedTuple{t::Float64, success::Bool, abandoned::Bool}}()
    function benchmark(P, mac, leaf)
        k = key(P, mac, leaf)
        haskey(memo, k) && return memo[k]
        cutoff = t_best_ok[] * abandon_factor    # Inf until the first success
        t_min = Inf
        success = true
        abandoned = false
        if tree_amortization == 1
            # trees + interaction lists are rebuilt inside every timed call
            for _ in 1:reps
                t = @elapsed result = fmm!(target_systems, source_systems, cache;
                    expansion_order=P, leaf_size_source=leaf,
                    multipole_acceptance=mac,
                    error_tolerance, kwargs...,
                    tune=true, update_target_systems=false)
                success = result[8]
                success || break
                t_min = min(t_min, t)
                if t_min > cutoff
                    abandoned = true
                    break
                end
            end
        else
            # build once (timed), then time the applies that reuse it
            t_build = @elapsed plan = FmmPlan(target_systems, source_systems;
                expansion_order=P, leaf_size_source=leaf,
                multipole_acceptance=mac, plan_kwargs...)
            t_amort = t_build / tree_amortization   # exactly 0.0 when Inf
            if t_amort > cutoff
                # the amortized build alone already loses; skip the applies.
                # error_success is left unverified — `abandoned` says so.
                abandoned = true
                t_min = t_amort
            else
                for _ in 1:reps
                    t = @elapsed result = fmm!(target_systems, source_systems, plan;
                        error_tolerance, apply_kwargs...,
                        tune=true, update_target_systems=false)
                    success = result[8]
                    success || break
                    t_min = min(t_min, t)
                    if t_min + t_amort > cutoff
                        abandoned = true
                        break
                    end
                end
                t_min += t_amort
            end
            plan = nothing
            GC.gc()     # a plan owns a full Cache; only one should be live
        end
        # only a COMPLETE, error-satisfying measurement may tighten the cutoff
        if success && !abandoned
            t_best_ok[] = min(t_best_ok[], t_min)
        end
        memo[k] = (; t=t_min, success, abandoned)
        return memo[k]
    end

    t_start = time()
    timed_out = false

    P0, mac0, leaf0 = expansion_order, multipole_acceptance, leaf_size_source
    r0 = benchmark(P0, mac0, leaf0)
    t0 = r0.t
    r0.success || error("tune_fmm_perturb: the starting parameters do not satisfy error_tolerance")

    history = [(iter=0, expansion_order=P0, multipole_acceptance=mac0,
                leaf_size_source=leaf0, t=t0, error_success=true,
                abandoned=false, accepted=true)]
    verbose && println("\n#======= Begin FastMultipole.tune_fmm_perturb() =======#")
    verbose && println("start: P=$P0 MAC=$mac0 leaf=$leaf0 t=$(round(t0; digits=3)) s " *
        "(reps=$reps, tree_amortization=$tree_amortization, " *
        "abandon_factor=$abandon_factor)")

    for iter in 1:max_iters
        neighbors = [
            (min(P0 + 1, max_expansion_order), mac0, leaf0),
            (max(P0 - 1, 1), mac0, leaf0),
            (P0, min(round(mac0 + mac_step; digits=3), mac_bounds[2]), leaf0),
            (P0, max(round(mac0 - mac_step; digits=3), mac_bounds[1]), leaf0),
            (P0, mac0, scale_leaf(leaf0, leaf_factor)),
            (P0, mac0, scale_leaf(leaf0, 1 / leaf_factor)),
        ]
        best_t = t0 * (1 - improve_tol)
        best = nothing
        for (P, mac, leaf) in neighbors
            key(P, mac, leaf) == key(P0, mac0, leaf0) && continue
            # guard between candidates: an already-memoized point is free, so
            # this only ever stops before a fresh (and possibly long) trial
            if !haskey(memo, key(P, mac, leaf)) && time() - t_start > max_seconds
                timed_out = true
                break
            end
            r = benchmark(P, mac, leaf)
            t, success, abandoned = r.t, r.success, r.abandoned
            push!(history, (iter=iter, expansion_order=P, multipole_acceptance=mac,
                            leaf_size_source=leaf, t=t, error_success=success,
                            abandoned=abandoned, accepted=false))
            verbose && println("  iter $iter: P=$P MAC=$mac leaf=$leaf " *
                (!success ? "REJECTED (error tolerance)" :
                 abandoned ? "ABANDONED at t>$(round(t; digits=3)) s " *
                             "(> $(abandon_factor)x best $(round(t_best_ok[]; digits=3)) s)" :
                 "t=$(round(t; digits=3)) s"))
            if success && !abandoned && t < best_t
                best_t = t
                best = (P, mac, leaf)
            end
        end
        timed_out && break
        best === nothing && break
        P0, mac0, leaf0 = best
        t0 = benchmark(P0, mac0, leaf0).t
        push!(history, (iter=iter, expansion_order=P0, multipole_acceptance=mac0,
                        leaf_size_source=leaf0, t=t0, error_success=true,
                        abandoned=false, accepted=true))
        verbose && println("  -> move to P=$P0 MAC=$mac0 leaf=$leaf0 t=$(round(t0; digits=3)) s")
    end

    t_elapsed = time() - t_start
    if timed_out
        msg = "tune_fmm_perturb: max_seconds EXPIRED after " *
              "$(round(t_elapsed; digits=1)) s (limit $max_seconds s) — the " *
              "descent was CUT SHORT and the returned point is best-so-far, " *
              "NOT a converged minimum"
        @warn msg
        verbose && println("  !! $msg")
    end
    verbose && println("minimum: P=$P0 MAC=$mac0 leaf=$leaf0 t=$(round(t0; digits=3)) s " *
        "($(length(memo)) distinct candidates, " *
        "$(count(v -> v.abandoned, values(memo))) abandoned early, " *
        "$(round(t_elapsed; digits=1)) s" *
        (timed_out ? ", TIMED OUT" : "") * ")")
    verbose && println("\n#===============================================#\n")

    tuned_params = (leaf_size_source=leaf0, expansion_order=P0,
                    multipole_acceptance=mac0)
    n_abandoned = count(v -> v.abandoned, values(memo))
    info = (; timed_out, t_elapsed, n_candidates=length(memo), n_abandoned,
            t_best=t0)
    return tuned_params, history, info
end
