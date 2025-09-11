# Gravitational Example

`FastMultipole` is designed to incorporate easily into your existing Julia code with minimal effort. In this section and the [Vortex Filament Example](vortex_filament.md), we demonstrate how this can be done.

First, we'll review the interface functions used by the gravitational point mass model used in [Quick Start](@ref). This code can also be found under `FastMultipole/test/gravitational.jl`.

To better understand how the `FastMultipole` interface functions, let's take a look at the data structures we'll use to define our point masses:

```@example guidedex 
using FastMultipole
using FastMultipole.StaticArrays

const i_POTENTIAL = 1:1   # index of the gravitational potential
const i_GRADIENT = 5:7    # index of the gravitational acceleration
const i_HESSIAN = 8:16    # index of the hessian matrix

# a single point mass
struct Body{TF}
    position::SVector{3,TF}
    radius::TF
    strength::TF
end

# container for a system of `Body`'s
struct Gravitational{TF}
    bodies::Vector{Body{TF}}
    potential::Matrix{TF}
end

# constructor
function Gravitational(bodies::Matrix)
    nbodies = size(bodies)[2]
    bodies2 = [Body(SVector{3}(bodies[1:3,i]),bodies[4,i],bodies[5,i]) for i in 1:nbodies]
    potential = zeros(eltype(bodies), 16, nbodies)
    return Gravitational(bodies2,potential)
end
```

In short, `Body` objects represent individual point masses, and the `Gravitational` object contains a group of them, along with a matrix of the potential, velocity, and velocity gradient at each body. The observant reader will notice that the `strength` field of `Body` represents its mass. 

## Overloading `body_to_multipole!`

The function `body_to_multipole!` is used to generate multipole expansions for our particular system. This is done be overloading `FastMultipole.body_to_multipole!` for our data structure. Convenience functions exist within `FastMultipole` to simplify this complicated function into a one-liner:

```@example guidedex
FastMultipole.body_to_multipole!(system::Gravitational, args...) =
    FastMultipole.body_to_multipole!(Point{Source}, system, args...)
```

The `::Gravitational` type annotation is the key that allows `FastMultipole` to dispatch on `::Gravitational` systems. `Point{Source}` is used to indicate that our bodies are points and induce a source (i.e. ``1/r``) potential. Other convenience functions exist in `FastMultipole` for any combination of `Point`, `Filament`, or `Panel` geometries using `Source`, `Dipole`, or `Vortex` kernels, and are specified when overloading `body_to_multipole!` as `<geometry>{<kernel>}`. For example, if my model used constant vortex sheet panels, I would replace `Point{Source}` in the example above to `Panel{Vortex}`.

!!! tip
    `FastMultipole` provides convenience functions for generating multipole coefficients for any combination of `Point`, `Filament`, or `Panel` geometries using `Source`, `Dipole`, or `Vortex` kernels. These are specified when overloading `body_to_multipole!` as `<geometry>{<kernel>}`.

We use the fast recursive method of generating exact coefficients for panels as developed by [GUMEROV2023112118](@cite). We have derived our own formulae for vortex filaments and panels, which are in the process of publication.

`FastMultipole` also requires the user to indicate if a source system induces a vector potential. This is done by overloading the `has_vector_potential` function as follows:

```@example guidedex
function FastMultipole.has_vector_potential(system::Gravitational)
    return false
end
```
This will allow `FastMultipole` to avoid unnecessary calculations if the vector potential is not needed. Since `Gravitational` systems do not induce a vector potential, we return `false`.

## Buffer Interface Functions

`FastMultipole` allocates a buffer matrix in which to sort and store key system information. The buffer interface is created by overloading several functions for the user-defined system type. The first of these functions is [`FastMultipole.source_system_to_buffer!`](@ref), which populates a matrix with the important information about each body, column-wise. Overloading for `::Gravitational` systems looks like this:

```@example guidedex
function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::Gravitational, i_body)
    buffer[1:3, i_buffer] .= system.bodies[i_body].position
    buffer[4, i_buffer] = system.bodies[i_body].radius
    buffer[5, i_buffer] = system.bodies[i_body].strength
end
```
Here, the `i_body`th body position, radius, and strength are copied to the correct positions in the `i_buffer`th column of the buffer matrix.

It is worth noting that there are ways of defining these functions that would harm performance, e.g. by allocating an array each time the velocity is requested. If you notice `FastMultipole`'s performance is poor, this and the other interface functions are good places to check.

!!! warning
    The user-defined buffer interface functions are used frequently during an FMM call, so it is important to write them efficiently, avoiding allocations and following the Julia performance tips. If you notice that `FastMultipole`'s performance is poor, this might be the reason.

Because every system can have a unique buffer structure, we need to overload a few additional functions to tell `FastMultipole` how to allocate and access it. The first function, `data_per_body`, returns the number of rows in the buffer matrix that are used to define a single body. The second function, `get_position`, returns the position of the `i`th body in the system. The third function, `strength_dims`, returns the number of rows in the buffer matrix that are used to define the strength of a single body. Finally, we overload `get_n_bodies` to return the number of bodies in our system. For our `Gravitational` system, we define:

```@example guidedex
Base.eltype(::Gravitational{TF}) where TF = TF

function FastMultipole.data_per_body(system::Gravitational)
    return 5
end

function FastMultipole.get_position(system::Gravitational, i)
    return system.bodies[i].position
end

function FastMultipole.strength_dims(system::Gravitational)
    return 1
end

FastMultipole.get_n_bodies(system::Gravitational) = length(system.bodies)
```
These functions provide enough information for `FastMultipole` to allocate the buffer matrix and access the data it needs.

Finally, we need to tell `FastMultipole` how to transfer the results of the FMM call back to the user-defined system. This is done by overloading the [`FastMultipole.buffer_to_target_system!`](@ref) function. Overloading for `::Gravitational` systems looks like this:

```@example guidedex
function FastMultipole.buffer_to_target_system!(target_system::Gravitational, i_target, ::FastMultipole.DerivativesSwitch{PS,VS,GS}, target_buffer, i_buffer) where {PS,VS,GS}
    # get values
    TF = eltype(target_buffer)
    scalar_potential = PS ? FastMultipole.get_scalar_potential(target_buffer, i_buffer) : zero(TF)
    velocity = VS ? FastMultipole.get_gradient(target_buffer, i_buffer) : zero(SVector{3,TF})
    hessian = GS ? FastMultipole.get_hessian(target_buffer, i_buffer) : zero(SMatrix{3,3,TF,9})

    # update system
    target_system.potential[i_POTENTIAL[1], i_target] = scalar_potential
    target_system.potential[i_GRADIENT, i_target] .= velocity
    for (jj,j) in enumerate(i_HESSIAN)
        target_system.potential[j, i_target] = hessian[jj]
    end
end
```
We note that the convenience functions `get_gradient` and `get_hessian` return `::SVector{3}` and `::SMatrix{3,3}`, respectively, to reduce allocations.

## Overloading `direct!`

The last required interface function is [`FastMultipole.direct!`](@ref), which evaluates the potential at a target system using a source system without multipole acceleration. This is required in an FMM call for those interactions that are too close to be approximated by expansions. It is also useful for debugging and testing the accuracy of the FMM call. Overloading for `::Gravitational` systems, we have:

```@example guidedex
function FastMultipole.direct!(target_system, target_index, ::DerivativesSwitch{PS,VS,GS}, source_system::Gravitational, source_buffer, source_index) where {PS,VS,GS}
    @inbounds for i_source in source_index
        source_x, source_y, source_z = FastMultipole.get_position(source_buffer, i_source)
        source_strength = FastMultipole.get_strength(source_buffer, source_system, i_source)[1]
        @inbounds for j_target in target_index
            target_x, target_y, target_z = FastMultipole.get_position(target_system, j_target)
            dx = target_x - source_x
            dy = target_y - source_y
            dz = target_z - source_z
            r2 = dx*dx + dy*dy + dz*dz
            if r2 > 0
                r = sqrt(r2)
                if PS
                    dϕ = source_strength / r * FastMultipole.ONE_OVER_4π
                    FastMultipole.set_scalar_potential!(target_system, j_target, dϕ)
                end
                if VS
                    dF = SVector{3}(dx,dy,dz) * source_strength / (r2 * r) * FastMultipole.ONE_OVER_4π
                    FastMultipole.set_gradient!(target_system, j_target, dF)
                end
            end
        end
    end
end
```
Note that the velocity gradient is not calculated in this function. If the velocity gradient is requested, the contribution due to `::Gravitational` systems will then be zero.

!!! tip
    Use the boolean type parameters of the `::DerivativesSwitch{PS,GS,HS}` argument when overloading the `direct!` to know when to compute the scalar potential, its gradient, and its hessian, respectively. This can save cost by avoiding unnecessary calculations.

## Running the FMM

Now that we have defined the interface functions, we can run the FMM on our `Gravitational` system to obtain the gravitational acceleration at each body. The [`fmm!`](@ref) function is used to perform the FMM call as:

```@example guidedex
using Random

# create system
function generate_gravitational(seed, n_bodies; radius_factor=0.1, strength_scale=1/n_bodies, bodies_fun=(x)->x)
    # random seed
    Random.seed!(seed)

    # initialize bodies
    bodies = rand(5, n_bodies)

    # scale radius
    bodies[4,:] ./= (n_bodies^(1/3)*2)
    bodies[4,:] .*= radius_factor

    # scale strength
    bodies[5,:] .*= strength_scale

    # user-defined function for arbitrary transformation
    bodies_fun(bodies)

    # create system
    return Gravitational(bodies)
end

n_bodies, rand_seed = 5_000, 123
system = generate_gravitational(rand_seed, n_bodies)

# run FMM
fmm!(system; scalar_potential=false, gradient=true, hessian=true)

println("gravitational acceleration:\n", system.potential[5:7,1:10], "...")
```
The `scalar_potential` and `gradient` arguments are set to `false` and `true`, respectively, to indicate that we want to compute the vector field (i.e. the gravitational field lines that translate to force and their gradient) but not the scalar potential.

!!! warning
    The user must specify which fields they want to be computed by the FMM by setting the boolean keyword arguments `scalar_potential`, `gradient`, and `hessian` when calling `fmm!` (defaults are `scalar_potential=false`, `gradient=true`, and `hessian=false`). Note also that if the source system induces a vector potential (as indicated by the interface function `has_vector_potential(system)=true`), then the `scalar_potential` keyword should be set to `false`, since the scalar potential will be non-sensical in this case.

## Preallocating the Buffers

The `fmm!` function incurs a small overhead for allocating the buffer matrices. If our system is large and the FMM will be called more than once, we can preallocate the buffers to avoid this overhead. This is done by returning a `cache` of preallocated memory the first time `fmm!` is called, and then adding it as an additional argument to `fmm!` in subsequent calls as follows:

```@example guidedex
# allocate buffer cache
allocs_1 = @allocated _, cache, _ = fmm!(system; scalar_potential=false, gradient=true, hessian=true)

# run FMM with preallocated buffers
allocs_2 = @allocated fmm!(system, cache; scalar_potential=false, gradient=true, hessian=true)

println("memory allocated without the cache:  ", allocs_1, " bytes")
println("memory allocated with the cache:     ", allocs_2, " bytes")
```
Note that the second call to `fmm!` allocates less memory.

!!! tip
    If you plan to call `fmm!` multiple times on the same system, consider preallocating the buffers by saving the `cache` returned by the first call, and splatting it as a keyword argument for each subsequent call. This can reduce memory allocations and improve performance.
