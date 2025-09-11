# Tuning Parameters

The FMM can be tuned for accuracy and performance by adjusting the following parameters: `multipole_acceptance`, `leaf_size`, and `expansion_order`. These parameters are passed to the `fmm!` function as keyword arguments. Let's take a look at how each affects the FMM. 

## Expansion Order

First we'll try varying the `expansion_order`, or the degree of the expansions used:

```@example guidedex
using FastMultipole # hide
using Random # hide
gravitational_path = normpath(joinpath(splitdir(pathof(FastMultipole))[1], "..", "test", "gravitational.jl")) # hide
include(gravitational_path) # hide
# create system
n_bodies, rand_seed = 10000, 123
system = generate_gravitational(rand_seed, n_bodies)

# compute potential directly
direct!(system; gradient=true)
gradient_direct = system.potential[5:7,:]

# try varying `expansion_order`
println("#--- varying expansion order ---#\n")
println("expansion_order = 1")
fmm!(system; expansion_order=1, multipole_acceptance=0.5, leaf_size=30, gradient=true)
system.potential .= 0.0
t_e_1 = @elapsed fmm!(system; expansion_order=3, multipole_acceptance=0.5, leaf_size=30, gradient=true)
println("\ttime cost: ", t_e_1, " seconds")
println("\tmax error: ", maximum(abs.(system.potential[5:7,:] .- gradient_direct)))

println("expansion_order = 4")
system.potential .= 0.0
t_e_4 = @elapsed fmm!(system; expansion_order=4, multipole_acceptance=0.5, leaf_size=30, gradient=true)
println("\ttime cost: ", t_e_4, " seconds")
println("\tmax error: ", maximum(abs.(system.potential[5:7,:] .- gradient_direct)))

println("expansion_order = 8")
system.potential .= 0.0
t_e_8 = @elapsed fmm!(system; expansion_order=8, multipole_acceptance=0.5, leaf_size=30, gradient=true)
println("\ttime cost: ", t_e_8, " seconds")
println("\tmax error: ", maximum(abs.(system.potential[5:7,:] .- gradient_direct)))
```
The `expansion_order` parameter is always positively correlated to accuracy but negatively correlated to cost. 

## Multipole Acceptance Criterion (MAC)

Now, let's explore the `multipole_acceptance` criterion. It must be between `0.0` and `1.0`, and indicates the smallest acceptable distance beyond which multipole expansions are allowed.

```@example guidedex
# try varying `multipole_acceptance`
println("#--- varying multipole acceptance ---#\n")

println("multipole_acceptance = 0.3")
system.potential .= 0.0
t_m_3 = @elapsed fmm!(system; expansion_order=4, multipole_acceptance=0.3, leaf_size=30, scalar_potential=true, gradient=true)
println("\ttime cost: ", t_m_3, " seconds")
println("\tmax error: ", maximum(abs.(system.potential[5:7,:] .- gradient_direct)))

println("multipole_acceptance = 0.5")
system.potential .= 0.0
t_m_5 = @elapsed fmm!(system; expansion_order=4, multipole_acceptance=0.5, leaf_size=30, scalar_potential=true, gradient=true)
println("\ttime cost: ", t_m_5, " seconds")
println("\tmax error: ", maximum(abs.(system.potential[5:7,:] .- gradient_direct)))

println("multipole_acceptance = 0.7")
system.potential .= 0.0
t_m_7 = @elapsed fmm!(system; expansion_order=4, multipole_acceptance=0.7, leaf_size=30, scalar_potential=true, gradient=true)
println("\ttime cost: ", t_m_7, " seconds")
println("\tmax error: ", maximum(abs.(system.potential[5:7,:] .- gradient_direct)))
```
The `multipole_acceptance` parameter is negatively correlated to accuracy. In this case, it is also negatively correlated to cost, but that won't always be true, depending on the cost of the `direct!` function.

## Leaf Size

Finally, let's explore the `leaf_size` parameter, which controls the size of the leaf nodes in the octree:

```@example guidedex
# try varying `leaf_size`
println("#--- varying leaf size ---#\n")

println("leaf_size = 5")
system.potential .= 0.0
t_l_5 = @elapsed fmm!(system; expansion_order=4, multipole_acceptance=0.5, leaf_size=5, gradient=true)
println("\ttime cost: ", t_l_5, " seconds")
println("\tmax error: ", maximum(abs.(system.potential[5:7,:] .- gradient_direct)))

println("leaf_size = 20")
system.potential .= 0.0
t_l_20 = @elapsed fmm!(system; expansion_order=4, multipole_acceptance=0.5, leaf_size=20, gradient=true)
println("\ttime cost: ", t_l_20, " seconds")
println("\tmax error: ", maximum(abs.(system.potential[5:7,:] .- gradient_direct)))

println("leaf_size = 100")
system.potential .= 0.0
t_l_80 = @elapsed fmm!(system; expansion_order=4, multipole_acceptance=0.5, leaf_size=80, gradient=true)
println("\ttime cost: ", t_l_80, " seconds")
println("\tmax error: ", maximum(abs.(system.potential[5:7,:] .- gradient_direct)))
```
The `leaf_size` parameter is positively correlated to accuracy, but its relation to cost is less predictable. 

The complicated relationship between these parameters, the cost, and the accuracy of the FMM motivates automated tuning of the parameters, which we'll explore in [Automated Tuning](advanced_usage_2.md).

!!! tip
    Tuning parameters can have an order-of-magnitude impact on the performance of FMM. The `expansion_order` parameter is positively correlated to accuracy but negatively correlated to cost. The `multipole_acceptance` and `leaf_size` parameters are still correlated, but less predictably so. This motivates automated tuning of the parameters, which we'll explore in the next example.

## Optimal Leaf Size

We can request `FastMultipole` to predict the optimal leaf size for our system by setting `tune=true` in the `fmm!` call. This performs best when `interaction_list_method=SelfTuning()` on a single thread, but still functions well for other choices. If `tune=true`, benchmarks will be used to estimate the leaf size at which the cost of direct calculations is approximately equal to expansion calculations. It is returned as part of a named tuple as the first returned value of `fmm!`, which can be splatted as a keyword argument in subsequent `fmm!` calls:

```@example guidedex
# get optimal leaf size
println("#--- default leaf size ---#\n")
t1 = @elapsed optargs, _ = fmm!(system; tune=true, gradient=true)
println("\ttime cost: ", t1, " seconds")

# run again with optimal leaf size
println("\n#--- optimal leaf size ---#\n")
t2 = @elapsed fmm!(system; scalar_potential=false, gradient=true, optargs...)
println("\ttime cost: ", t2, " seconds")
```
Note that `optargs` contains `leaf_size_source`, `expansion_order`, and `multipole_acceptance` parameters. Only `leaf_size_source` is tuned if `isnothing(error_tolerance) == true`. More complete auto-tuning will be discussed in [Advanced Usage](advanced_usage.md).
