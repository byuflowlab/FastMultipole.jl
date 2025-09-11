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

    t_fmm = @elapsed _, cache, _ = fmm!(target_systems, source_systems;
                       expansion_order=1, leaf_size_source,
                       nearfield=false, farfield=false, self_induced=false,
                       tune=true, update_target_systems=false
                      )

    #--- error tolerance selected ---#

    for multipole_acceptance in multipole_acceptances

        if verbose
            println("\nmultipole_acceptance = $multipole_acceptance...")
        end

        # initial fmm! call with max_expansion_order to get leaf_size
        t_fmm = @elapsed optargs, _, _, _, m2l_list, _, _, error_success = fmm!(target_systems, source_systems, cache;
                                                                                expansion_order=isnothing(error_tolerance) ? expansion_order : max_expansion_order,
                                                                                leaf_size_source, multipole_acceptance,
                                                                                error_tolerance, kwargs...,
                                                                                tune=true, update_target_systems=false,
                                                                               )

        # in case error is not satisfied
        if !error_success
            println("\terror tolerance not satisfied for max expansion order P=$max_expansion_order;")
            println("\tskipping this multipole_acceptance...")
            continue
        end

        leaf_size_source = optargs.leaf_size_source
        this_max_expansion_order = optargs.expansion_order + 2

        # second fmm! call with optimal leaf_size to get expansion order
        t_fmm = @elapsed optargs, _, _, _, m2l_list, _, _, error_success = fmm!(target_systems, source_systems, cache;
                                                                                 expansion_order=isnothing(error_tolerance) ? expansion_order : this_max_expansion_order,
                                                                                 leaf_size_source, multipole_acceptance,
                                                                                 error_tolerance, kwargs...,
                                                                                 tune=true, update_target_systems=false,
                                                                                )

        if !error_success # better run at the actual max_expansion_order
            max_expansion_order = original_max_expansion_order
            t_fmm = @elapsed optargs, _, _, _, m2l_list, _, _, error_success = fmm!(target_systems, source_systems, cache;
                                                                                     expansion_order=isnothing(error_tolerance) ? expansion_order : max_expansion_order,
                                                                                     leaf_size_source, multipole_acceptance,
                                                                                     error_tolerance, kwargs...,
                                                                                     tune=true, update_target_systems=false,
                                                                                    )

        end

        expansion_order = optargs.expansion_order

        # final benchmark
        t_fmm = @elapsed optargs, _, _, _, m2l_list, _, _, error_success = fmm!(target_systems, source_systems, cache;
                                                                                 expansion_order,
                                                                                 leaf_size_source, multipole_acceptance,
                                                                                 error_tolerance, kwargs...,
                                                                                 tune=true, update_target_systems=false,
                                                                                )

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

    return tuned_params, cache
end

