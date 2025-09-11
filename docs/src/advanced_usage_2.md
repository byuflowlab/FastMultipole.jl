# Automated Tuning

In the this section, we will describe how to impose an error tolerance. Finally, we will show how to automatically tune the FMM parameters.

## Satisfying an Error Tolerance

`FastMultipole` can be configured to satisfy an error tolerance by dynamically adjusting the expansion order of each multipole-to-local transformation to the smallest integer that satisfies the tolerance, according to an error prediction. Users can indicate their desired error tolerance via the keyword argument `error_tolerance` in the `fmm!` function. A value of `nothing` indicates no error tolerance, in which case the expansion order is fixed at whatever is passed as the `expansion_order` keyword. Otherwise, `error_tolerance` should inherit from the `ErrorMethod` abstract type. The chosen type will determine how the error is predicted, and hence how the expansion order is chosen. Choices include:

- `PowerAbsolutePotential{tolerance, BE::Bool}`: Constrains the magnitude of the potential error to be less than `tolerance` using a radially invariant upper bound.
- `PowerAbsoluteGradient{tolerance, BE::Bool}`: Constrains the magnitude of the vector field error to be less than `tolerance` using a radially invariant upper bound.
- `PowerRelativePotential{relative_tolerance, absolute_tolerance, BE::Bool}`: Constrains the relative error of the potential to be less than `relative_tolerance` using a radially invariant upper bound. If the equivalent absolute tolerance of an interaction is less than `absolute_tolerance`, then `absolute_tolerance` is used instead.
- `PowerRelativeGradient{relative_tolerance, absolute_tolerance, BE::Bool}`: Constrains the relative error of the vector field to be less than `relative_tolerance` using a radially invariant upper bound. If the equivalent absolute tolerance of an interaction is less than `absolute_tolerance`, then `absolute_tolerance` is used instead.

Since each error 

!!! info
    For relative error methods, the [`get_previous_influence`](@ref FastMultipole.get_previous_influence) compatibility function must be overloaded for your target system. This function estimates the potential and gradient at the `i`th target body, which is used to compute the relative error. For example, estimates might be based on the previous step of a time-stepping or iterative algorithm.

Say I wanted to compute the gravitational potential to a tolerance of `1e-6` using the absolute potential error method. I would call the FMM as follows:

```@example advancedex2

using FastMultipole
using Random # needed for `gravitational.jl`

gravitational_path = normpath(joinpath(splitdir(pathof(FastMultipole))[1], "..", "test", "gravitational.jl"))
include(gravitational_path)

# create system
n_bodies, rand_seed = 5_000, 123
system = generate_gravitational(rand_seed, n_bodies)

# run FMM with error tolerance
fmm!(system; scalar_potential=true, gradient=false, hessian=false, error_tolerance=PowerAbsolutePotential(1e-6), expansion_order=8)

# print potential
println("gravitational potential:\n", system.potential[1,1:10], "...")
```
This keyword argument `error_tolerance=PowerAbsolutePotential(1e-6)` requests that the expansion order be dynamically adjusted to ensure that the potential error of each interaction is less than `1e-6` at each body. Note that this does ensure a perfectly conservative error bound, since many multipole-to-local transformations will be performed to any given target cell. However, the maximum error should be within an order of magnitude of the specified tolerance, and the average error will be much lower. It is also worth pointing out that the `expansion_order` keyword determines the maximum allowable expansion order; if it is too small, then `FastMultipole` will not be able to satisfy the error tolerance, and will show a warning. In this case, the expansion order is set to `8`, which is sufficient to satisfy the error tolerance.

```@example advancedex2
# verify error
phi_fmm = system.potential[1,:]
system.potential .= 0.0
direct!(system; scalar_potential=true, gradient=false, hessian=false)
phi_direct = system.potential[1,:]
println("max potential error: ", maximum(abs.(phi_direct .- phi_fmm)))
```
It is also worth noting that the expansion order is never allowed to be less than 1, which leads to "bottoming out" behavior. Let's see what happens when we request a tolerance of `1e-2`:

```@example advancedex2
# run FMM with error tolerance
system.potential .= 0.0
fmm!(system; scalar_potential=true, gradient=false, hessian=false, error_tolerance=PowerAbsolutePotential(1e-2))

# check error
phi_fmm = system.potential[1,:]
println("max error:\n", maximum(abs.(phi_direct .- phi_fmm)))
```
As a rule of thumb, `PowerAbsolutePotential` and `PowerAbsoluteGradient` are best when the maximum error is to be constrained, while `RotatedCoefficientsAbsoluteGradient` is better if the mean error is to be constrained.

!!! info
    When imposing an error tolerance, the `PowerAbsolutePotential` and `PowerAbsoluteGradient` methods are best when the maximum error is to be constrained, while `RotatedCoefficientsAbsoluteGradient` is better if the mean error is to be constrained.

## Fully Automatic Tuning of FMM Parameters

`FastMultipole` can automatically tune the FMM parameters to satisfy an error tolerance. This is done by running the [`tune_fmm`](@ref) function, which returns a named tuple of the optimal parameters and a preallocated buffer to reduce memory allocations.

Say I wanted the optimal tuning parameters for the gravitational potential of a system of point masses, with a maximum error tolerance of `1e-4`. I would call the function as follows:

```@example advancedex2
opt_params, cache = tune_fmm(system; error_tolerance=PowerAbsolutePotential(1e-4), scalar_potential=true, gradient=false, hessian=false)
println("Optimal parameters: ", opt_params)
```
This will return a named tuple of the optimal parameters, which can then be passed to the `fmm!` function. The `cache` is a preallocated buffer that can be used to reduce memory allocations during the FMM call.

```@example advancedex2
# run FMM without default parameters
println("Default Tuning Parameters:")
system.potential .= 0.0
t1 = @elapsed fmm!(system; scalar_potential=true, gradient=false, hessian=false, error_tolerance=PowerAbsolutePotential(1e-4))

# verify error
phi_fmm = system.potential[1,:]
println("\tmax error: ", maximum(abs.(phi_direct .- phi_fmm)))
println("\ttime cost: ", t1, " seconds")

# run FMM with optimal parameters
println("Optimal Tuning Parameters:")
t2 = @elapsed fmm!(system, cache; scalar_potential=true, gradient=false, hessian=false, error_tolerance=PowerAbsolutePotential(1e-4), opt_params...)

# verify error
println("\tmax error: ", maximum(abs.(phi_direct .- phi_fmm)))
println("\ttime cost: ", t2, " seconds")
```
Note that the `tune_fmm` function iterates over each requested `multipole_acceptance` criterion, so it is not optimal to call it before every FMM call. Instead, `tune_fmm` can be called once for a representative system primarily to choose `multipole_acceptance`. Since the optimal parameters depend on the size, distribution, and strengths of the sources and targets, it is recommended to set the keyword argument `tune=true` to iteratively update the `expansion_order` and `leaf_size_source` parameters each time `fmm!` is called. This will allow the FMM to adapt to the system as it evolves, and will ensure that the optimal parameters are used for each call.

!!! tip
    The `tune_fmm` function can be used to automatically tune the FMM parameters to satisfy an error tolerance. It returns a named tuple of the optimal parameters and a preallocated buffer to reduce memory allocations.

!!! warning
    The `tune_fmm` function is computationally expensive, so it should not be called before every FMM call. Instead, it should be called once for a representative system to choose the optimal `multipole_acceptance` parameter. Then, the optimal parameters will be updated in each `fmm!` call if the `tune=true` keyword argument is set. This allows the FMM to adapt to the system as it evolves.
