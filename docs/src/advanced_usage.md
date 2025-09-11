# Multiple Systems

The most advanced features of `FastMultipole` include:

- the ability to solve the $n$-body problem for multiple systems simultaneously in a single FMM call
- satisfy an error tolerance by dynamically adjusting the expansion order of multipole-to-local transformations
- predict the full array of FMM tuning parameters such that an error tolerance is met
In the this section, and continuing in [Automated Tuning](advanced_usage_2.md), we will first describe how to run the FMM for multi-system problems. Then, we will describe how to impose an error tolerance. Finally, we will show how to automatically tune the FMM parameters.

## Simultaneous Computation of Several Target and Source Systems

`FastMutipole` allows FMM to be performed efficiently on distinct target and source systems. This is done by creating two octrees: one for sources, and one for targets [yokota2013fmm2](@cite). For example, say we desire to know the influence of one system of point masses on another, but not vice versa. This is done by passing both systems to the `fmm!` function with the target system first, as:

```@example advancedex
using FastMultipole
using Random # needed for `gravitational.jl`

gravitational_path = normpath(joinpath(splitdir(pathof(FastMultipole))[1], "..", "test", "gravitational.jl"))
include(gravitational_path)

target_system = generate_gravitational(123, 1000)
source_system = generate_gravitational(321, 1000)

fmm!(target_system, source_system)
println("gravitational acceleration: ", target_system.potential[5:7,1:10], "...")
```

!!! tip
    The `fmm!` function can take any number of target and source systems, so long as they are passed as tuples. This allows for the simultaneous evaluation of multiple source systems on multiple target systems.

In practice, the source system might be a collection of systems, composed of a variety of datastructures. So long as the interface functions are defined for each system, we can pass a tuple of any number of systems as the source and or target. For example, we could evaluate the influence of a system of point masses, point vortices, and vortex filaments on two different target systems:

```@example advancedex
using LinearAlgebra

# include vortex filament and particle models and interface functions
vortex_path = normpath(joinpath(splitdir(pathof(FastMultipole))[1], "..", "test", "vortex.jl"))
include(vortex_path)
filament_path = normpath(joinpath(splitdir(pathof(FastMultipole))[1], "..", "test", "vortex_filament.jl"))
include(filament_path)

# generate systems
n_bodies = 2000
target_one = generate_gravitational(123, n_bodies)
target_two = generate_vortex(124, n_bodies)
source_one = generate_gravitational(125, n_bodies)
source_two = generate_vortex(126, n_bodies)
source_three = generate_filament_field(n_bodies, n_bodies^0.333, 127; strength_scale=1/n_bodies)

# run FMM
fmm!((target_one, target_two), (source_one, source_two, source_three);
    scalar_potential=false, gradient=true, hessian=(false, true))

v1 = target_one.potential[5:7,:]
v2 = target_two.gradient_stretching[1:3,:]

# run direct
target_one.potential .= 0.0
target_two.potential .= 0.0
target_two.gradient_stretching .= 0.0
direct!((target_one, target_two), (source_one, source_two, source_three); scalar_potential=false, gradient=true, hessian=(false, true))

# test accuracy
println("max error in target one: ", maximum(abs.(target_one.potential[5:7,:] .- v1)))
println("max error in target two: ", maximum(abs.(target_two.gradient_stretching[1:3,:] .- v2)))
```
Note that `scalar_potential`, `gradient`, and `hessian` can be passed as a single boolean or as a tuple of booleans, one for each target system. This allows the user to specify which values are desired for each target system, and avoids unnecessary calculations for values that are not needed. In this case, we have set `scalar_potential=false` and `gradient=true` to indicate all target systems, but a tuple `hessian=(false,true)` to indicate different settings for each. It is worth remembering that these switches must be implemented by the user when overloading the `direct!` function for each system to act as a source.

!!! tip
    The `fmm!` keyword arguments `scalar_potential`, `gradient`, and `hessian` can be passed as a single boolean or as a tuple of booleans, one for each target system. This allows the user to specify which values are desired for each target system, and avoids unnecessary calculations for values that are not needed.
