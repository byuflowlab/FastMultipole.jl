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
- `reps=2`: min-of-reps timing per candidate (tree build included, as in
  production use)
- `improve_tol=0.02`: relative improvement required to accept a move
- `max_iters=20`: maximum accepted moves
- `verbose=true`
- `kwargs...`: forwarded to `fmm!` (must match the production call's
  `scalar_potential`/`gradient`/`hessian` request)

**Returns**

- `tuned_params::NamedTuple`: `(leaf_size_source, expansion_order,
  multipole_acceptance)` at the cost minimum, splat-able into `fmm!`
- `history::Vector{<:NamedTuple}`: every evaluated candidate with fields
  `(iter, expansion_order, multipole_acceptance, leaf_size_source, t,
  error_success, accepted)`
"""
function tune_fmm_perturb(target_systems, source_systems;
    expansion_order, multipole_acceptance, leaf_size_source,
    error_tolerance=nothing,
    max_expansion_order=20,
    mac_step=0.05, mac_bounds=(0.25, 0.85),
    leaf_factor=1.5,
    reps=2, improve_tol=0.02, max_iters=20,
    verbose=true, kwargs...)

    target_systems = to_tuple(target_systems)
    source_systems = to_tuple(source_systems)

    # preallocate cache with the same target buffer layout as the trials
    _, cache, _ = fmm!(target_systems, source_systems;
        expansion_order=1, leaf_size_source,
        nearfield=false, farfield=false, self_induced=false,
        kwargs..., tune=true, update_target_systems=false)

    scale_leaf(leaf::Integer, f) = max(1, round(Int, leaf * f))
    scale_leaf(leaf, f) = map(l -> max(1, round(Int, l * f)), leaf)
    key(P, mac, leaf) = (P, round(mac; digits=3), leaf)

    memo = Dict{Any, Tuple{Float64, Bool}}()
    function benchmark(P, mac, leaf)
        k = key(P, mac, leaf)
        haskey(memo, k) && return memo[k]
        t_min = Inf
        success = true
        for _ in 1:reps
            t = @elapsed result = fmm!(target_systems, source_systems, cache;
                expansion_order=P, leaf_size_source=leaf,
                multipole_acceptance=mac,
                error_tolerance, kwargs...,
                tune=true, update_target_systems=false)
            success = result[8]
            success || break
            t_min = min(t_min, t)
        end
        memo[k] = (t_min, success)
        return memo[k]
    end

    P0, mac0, leaf0 = expansion_order, multipole_acceptance, leaf_size_source
    t0, ok = benchmark(P0, mac0, leaf0)
    ok || error("tune_fmm_perturb: the starting parameters do not satisfy error_tolerance")

    history = [(iter=0, expansion_order=P0, multipole_acceptance=mac0,
                leaf_size_source=leaf0, t=t0, error_success=true, accepted=true)]
    verbose && println("\n#======= Begin FastMultipole.tune_fmm_perturb() =======#")
    verbose && println("start: P=$P0 MAC=$mac0 leaf=$leaf0 t=$(round(t0; digits=3)) s")

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
            t, success = benchmark(P, mac, leaf)
            push!(history, (iter=iter, expansion_order=P, multipole_acceptance=mac,
                            leaf_size_source=leaf, t=t, error_success=success,
                            accepted=false))
            verbose && println("  iter $iter: P=$P MAC=$mac leaf=$leaf " *
                (success ? "t=$(round(t; digits=3)) s" : "REJECTED (error tolerance)"))
            if success && t < best_t
                best_t = t
                best = (P, mac, leaf)
            end
        end
        best === nothing && break
        P0, mac0, leaf0 = best
        t0, _ = benchmark(P0, mac0, leaf0)
        push!(history, (iter=iter, expansion_order=P0, multipole_acceptance=mac0,
                        leaf_size_source=leaf0, t=t0, error_success=true,
                        accepted=true))
        verbose && println("  -> move to P=$P0 MAC=$mac0 leaf=$leaf0 t=$(round(t0; digits=3)) s")
    end

    verbose && println("minimum: P=$P0 MAC=$mac0 leaf=$leaf0 t=$(round(t0; digits=3)) s " *
        "($(length(memo)) distinct candidates)")
    verbose && println("\n#===============================================#\n")

    tuned_params = (leaf_size_source=leaf0, expansion_order=P0,
                    multipole_acceptance=mac0)
    return tuned_params, history
end
