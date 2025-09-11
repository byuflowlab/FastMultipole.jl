#=
this file is used to generate Figure 4 in the paper (examples_verify_combined.pdf and examples_cost_combined.pdf)
=#

using FastMultipole
using Statistics
using Random
using BSON

include("../test/gravitational.jl")
include("../test/vortex.jl")

function get_gradient(system::Gravitational)
    return system.potential[5:7,:]
end

function get_gradient(system::VortexParticles)
    return system.velocity_stretching[1:3,:]
end

function benchmark_system(source, vortex, expansion_orders)
    # save direct velocity
    reset!(source)
    reset!(vortex)
    direct!((source, vortex))
    v_source_direct = get_gradient(source)
    v_vortex_direct = get_gradient(vortex)

    # storage containers
    max_errs_source_combined = Float64[]
    max_errs_vortex_combined = Float64[]
    max_errs_source_individual = Float64[]
    max_errs_vortex_individual = Float64[]
    ts_combined = Float64[]
    ts_source = Float64[]
    ts_vortex = Float64[]

    # run FMM
    println("\n#--- Benchmarking ---#\n")
    for expansion_order in expansion_orders
        println("Expansion order: $expansion_order")
        # reset systems
        reset!(source)
        reset!(vortex)

        # benchmark FMM combined
        println("\n\tbegin combined")
        optargs, cache, _ = fmm!((source, vortex); lamb_helmholtz=true, tune=true, expansion_order)
        t_combined_1 = @elapsed optargs, cache, _ = fmm!((source, vortex); lamb_helmholtz=true, tune=true, expansion_order, leaf_size_source=optargs.leaf_size_source, cache)
        reset!(source)
        reset!(vortex)
        t_combined_2 = @elapsed fmm!((source, vortex); lamb_helmholtz=true, expansion_order, leaf_size_source=optargs.leaf_size_source, cache)
        push!(ts_combined, min(t_combined_1, t_combined_2))
        
        # calculate errors
        v_source_fmm = get_gradient(source)
        v_vortex_fmm = get_gradient(vortex)
        err_source = maximum(sqrt.(sum((v_source_fmm - v_source_direct) .^2; dims=1)))
        err_vortex = maximum(sqrt.(sum((v_vortex_fmm - v_vortex_direct) .^2; dims=1)))
        push!(max_errs_source_combined, err_source)
        push!(max_errs_vortex_combined, err_vortex)

        # benchmark FMM source
        println("\n\tbegin source")
        optargs, source_cache, _ = fmm!((source, vortex), source; lamb_helmholtz=false, tune=true, expansion_order)
        t_source_1 = @elapsed optargs, source_cache, _ = fmm!((source, vortex), source; lamb_helmholtz=false, tune=true, expansion_order, leaf_size_source=optargs.leaf_size_source, source_cache)
        reset!(source)
        reset!(vortex)
        t_source_2 = @elapsed fmm!((source, vortex), source; lamb_helmholtz=false, expansion_order, leaf_size_source=optargs.leaf_size_source, source_cache)
        push!(ts_source, min(t_source_1, t_source_2))
        
        # save velocity
        v_source_fmm_source = get_gradient(source)
        v_vortex_fmm_source = get_gradient(vortex)

        # benchmark FMM vortex
        println("\n\tbegin vortex")
        optargs, cache, _ = fmm!((source, vortex), vortex; lamb_helmholtz=true, tune=true, expansion_order)
        t_vortex_1 = @elapsed optargs, cache, _ = fmm!((source, vortex), vortex; lamb_helmholtz=true, tune=true, expansion_order, leaf_size_source=optargs.leaf_size_source, cache)
        reset!(source)
        reset!(vortex)
        t_vortex_2 = @elapsed fmm!((source, vortex), vortex; lamb_helmholtz=true, expansion_order, leaf_size_source=optargs.leaf_size_source, cache)
        push!(ts_vortex, min(t_vortex_1, t_vortex_2))

        # calculate errors
        v_source_fmm = get_gradient(source) .+ v_source_fmm_source
        v_vortex_fmm = get_gradient(vortex) .+ v_vortex_fmm_source
        err_source = maximum(sqrt.(sum((v_source_fmm - v_source_direct) .^2; dims=1)))
        err_vortex = maximum(sqrt.(sum((v_vortex_fmm - v_vortex_direct) .^2; dims=1)))
        push!(max_errs_source_individual, err_source)
        push!(max_errs_vortex_individual, err_vortex)

    end

    return max_errs_source_combined, max_errs_vortex_combined, ts_combined,
           max_errs_source_individual, max_errs_vortex_individual, ts_source, ts_vortex
end

#--- create systems ---#

println("#--- Creating systems ---#")

n_bodies = 50_000
source = generate_gravitational(123, n_bodies; strength_scale=1.0/3798.955768976926)
vortex = generate_vortex(123, n_bodies; strength_scale=1.0/10566.33461495282)

direct!(vortex)
direct!(source)

v_vortex = vortex.velocity_stretching[1:3,:]
v_vortex_mag = sqrt.(sum(v_vortex .^2; dims=1))
v_vortex_mean = mean(v_vortex_mag)
@show v_vortex_mean

v_source = source.potential[5:7,:]
v_source_mag = sqrt.(sum(v_source .^2; dims=1))
v_source_mean = mean(v_source_mag)
@show v_source_mean

#--- FMM for combined systems ---#

expansion_orders = collect(1:15)
max_errs_source_combined, max_errs_vortex_combined, ts_combined,
    max_errs_source_individual, max_errs_vortex_individual, ts_source, ts_vortex = benchmark_system(source, vortex, expansion_orders)
println("max_errs_source_combined: ", max_errs_source_combined)
println("max_errs_vortex_combined: ", max_errs_vortex_combined)
println("ts_combined: ", ts_combined)
println("max_errs_source_individual: ", max_errs_source_individual) 
println("max_errs_vortex_individual: ", max_errs_vortex_individual)
println("ts_source: ", ts_source)
println("ts_vortex: ", ts_vortex)

#--- save results ---#

BSON.@save "results/20250404_combined_source_vortex.bson" max_errs_source_combined max_errs_vortex_combined ts_combined max_errs_source_individual max_errs_vortex_individual ts_source ts_vortex expansion_orders