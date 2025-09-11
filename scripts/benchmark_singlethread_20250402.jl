using FastMultipole
using Random
using Statistics

include("../test/gravitational.jl")
include("../test/vortex.jl")

function get_potential(system::Gravitational)
    return system.potential[1,:]
end

function benchmark_fmm_gravitational(sizes, rand_seed=123; fmm_args...)
    results = Float64[]
    println("Benchmarking FMM for Gravitational System")
    for n_bodies in sizes
        println("\n#--- Benchmarking n_bodies = $n_bodies ---#\n")
        system = generate_gravitational(rand_seed, n_bodies)

        # generate cache
        optargs, cache, _ = fmm!(system; tune=true, fmm_args...)

        # Warm-up call to reduce noise
        reset!(system)
        time1 = @elapsed fmm!(system; cache, optargs...)

        # Benchmark
        reset!(system)
        time2 = @elapsed fmm!(system; cache, optargs...)

        push!(results, min(time1, time2))
    end
    return results
end

function benchmark_fmm_gravitational2(sizes, rand_seed=123; fmm_args...)
    results = Float64[]
    results_direct = Float64[]
    errs = Float64[]
    println("Benchmarking FMM for Gravitational System")
    for n_bodies in sizes
        println("\n#--- Benchmarking n_bodies = $n_bodies ---#\n")
        system = generate_gravitational(rand_seed, n_bodies)

        # generate cache
        optargs, cache, _ = fmm!(system; lamb_helmholtz=false, tune=true, fmm_args...)

        # Warm-up call to reduce noise
        reset!(system)
        time1 = @elapsed fmm!(system; velocity=false, lamb_helmholtz=false, scalar_potential=true, cache, optargs...)

        # Benchmark
        reset!(system)
        time2 = @elapsed fmm!(system; velocity=false, lamb_helmholtz=false, scalar_potential=true, cache, optargs...)
        phi_fmm = get_potential(system)

        # get error
        if n_bodies <= 65536
            reset!(system)
            time3 = @elapsed direct!(system; velocity=false, scalar_potential=true)
            phi_direct = get_potential(system)
            time4 = @elapsed direct!(system; velocity=false, scalar_potential=true)

            err = phi_direct - phi_fmm
            err_norm = sqrt(sum(err .* err) / n_bodies)
            rel_err = err ./ phi_direct
            err_rel = sqrt(sum(rel_err .* rel_err) / n_bodies)
            push!(results_direct, min(time3,time4))
            push!(errs, err_rel)
        else
            push!(errs, NaN)
        end
        push!(results, min(time1, time2))
    end
    return results, results_direct, errs
end

function benchmark_fmm_vortex(sizes, rand_seed=123; fmm_args...)
    results = Float64[]
    println("Benchmarking FMM for Vortex System")
    for n_bodies in sizes
        println("\n#--- Benchmarking n_bodies = $n_bodies ---#\n")
        system = generate_vortex(rand_seed, n_bodies)

        # generate cache
        optargs, cache, _ = fmm!(system; lamb_helmholtz=true, fmm_args...)

        # Warm-up call to reduce noise
        reset!(system)
        time1 = @elapsed fmm!(system; cache, lamb_helmholtz=true, fmm_args...)

        # Benchmark
        reset!(system)
        time2 = @elapsed fmm!(system; cache, lamb_helmholtz=true, fmm_args...)

        push!(results, min(time1, time2))
    end
    return results
end

function benchmark_direct_gravitational(sizes, rand_seed=123)
    results = Float64[]
    println("Benchmarking Direct for Gravitational System")
    for n_bodies in sizes
        println("\n#--- Benchmarking n_bodies = $n_bodies ---#\n")
        system = generate_gravitational(rand_seed, n_bodies)

        # Warm-up call to reduce noise
        reset!(system)
        time1 = @elapsed direct!(system)

        # Benchmark
        reset!(system)
        time2 = @elapsed direct!(system)

        push!(results, min(time1, time2))
    end
    return results
end

function benchmark_direct_vortex(sizes, rand_seed=123)
    results = Float64[]
    println("Benchmarking Direct for Vortex System")
    for n_bodies in sizes
        println("\n#--- Benchmarking n_bodies = $n_bodies ---#\n")
        system = generate_vortex(rand_seed, n_bodies)

        # Warm-up call to reduce noise
        reset!(system)
        time1 = @elapsed direct!(system)

        # Benchmark
        reset!(system)
        time2 = @elapsed direct!(system)

        push!(results, min(time1, time2))
    end
    return results
end

#--- system sizes ---#

# sizes = [2^n for n in 4:2:22]

#--- benchmark fmm ---#

#=
for P in 1:3
    print("P = $P:")
    grav_results, grav_results_direct, errs = benchmark_fmm_gravitational2(sizes; expansion_order=P, multipole_acceptance=0.5, lamb_helmholtz=false, velocity=false, scalar_potential=true)
    println(grav_results)
    println(grav_results_direct)
    println(errs)
end
=#

# make plots for paper:

# sizes = [2^n for n in 8:2:23]
#=
grav_results = benchmark_fmm_gravitational(sizes; expansion_order=3, multipole_acceptance=0.5, leaf_size_source=50)
println(grav_results)
vort_results = benchmark_fmm_vortex(sizes; expansion_order=3, multipole_acceptance=0.5, leaf_size_source=50)
println(vort_results)
=#
#=
[0.000340917, 0.003108583, 0.029605666, 0.204325542, 1.123070834, 5.086285875, 21.709248084, 108.63555075]
[0.000496041, 0.003171125, 0.033046709, 0.305992458, 1.636601542, 5.664564125, 37.112525875, 153.982820042]
=#

#--- benchmark direct ---#

grav_results_direct = benchmark_direct_gravitational(sizes[1:5])
println(grav_results_direct)
vort_results_direct = benchmark_direct_vortex(sizes[1:5])
println(vort_results_direct)
#=
[0.000191708, 0.002213583, 0.033915041, 0.540281042, 8.677510666, 407.491280667]
[0.000435083, 0.006332833, 0.1002775, 1.594397833, 26.096670584, 414.648684791]
=#
