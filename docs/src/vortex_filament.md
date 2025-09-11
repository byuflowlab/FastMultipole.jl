# Vortex Filament Example

In this example, we review the interface functions used by the vortex filament model found in `FastMultipole/test/vortex_filament.jl`. First, let's take a look at the data structure:

```@example guidedex
using FastMultipole # hide
using FastMultipole.StaticArrays # hide
struct VortexFilaments{TF}
    x::Matrix{SVector{3,TF}}
    strength::Vector{SVector{3,TF}}
    core_size::Vector{TF}
    error_tolerance::Vector{TF}
    potential::Vector{TF}
    gradient::Vector{SVector{3,TF}}
    hessian::Vector{SMatrix{3,3,TF,9}}
end
```
Note that `x` is a 2-row matrix, where the first row contains the start point of the filament and the second row contains the end point. The `strength` field contains the strength of the vortex filament, which is a vector quantity. The `core_size` field contains the regularization radius for each filament. The `potential`, `force`, and `gradient` fields are used to store the results of the FMM call.

## Overloading `body_to_multipole!`

Just like we did for the gravitational example, we need to overload `FastMultipole.body_to_multipole!` and `FastMultipole.has_vector_potential` for our vortex filament system. The recommended way to do this is:

```@example guidedex
FastMultipole.body_to_multipole!(system::VortexFilaments, args...) = 
    FastMultipole.body_to_multipole!(Filament{Vortex}, system, args...)

function FastMultipole.has_vector_potential(system::VortexFilaments)
    return true
end
```
Ordinarily, a 3-dimensional vector potential, such as the stream function in 3-dimensional fluid dynamics, requires 3 separate expansions (1 for each dimension). It is worth noting that when `Vortex` elements are used, `FastMultipole` never actually computes the vector potential ``\vec{\psi}``; rather, it computes two scalar components ``\phi`` and ``\chi`` of the Lamb-Helmholtz decomposition of the field as demonstrated by [gumerov2013efficient](@cite). This allows us to reduce the number of expansions required to express the scalar-plus-vector potential from 4 to 2, thus reducing computational cost. It is also worth noting that ``\phi`` and ``\chi`` are not origin-invariant fields, and thus cannot be used to evaluate scalar or vector potentials with FMM. In other words, when a `Vortex` kernel is used as a source in a FMM call, the `scalar_potential` should not be used.

!!! info
    When using `Vortex` elements, `FastMultipole` computes two scalar components ``\phi`` and ``\chi`` of the Lamb-Helmholtz decomposition of the field, rather than the vector potential ``\vec{\psi}``. This allows us to reduce the number of expansions required to express the scalar-plus-vector potential from 4 to 2, thus reducing computational cost.

!!! warning
    When a `Vortex` kernel is used as a source in a FMM call, the `scalar_potential` should not be used. This is because the `scalar_potential` is not origin-independent under the Lamb-Helmholtz decomposition.

## Buffer Interface Functions

The buffer interface functions are defined for `VortexFilament` systems as follows:

```@example guidedex
function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::VortexFilaments, i_body)
    # position
    buffer[1:3,i_buffer] .= (system.x[1,i_body]  + system.x[2,i_body]) * 0.5
    
    # get regularization radius
    Γ = system.strength[i_body]
    # Γmag = norm(Γ)
    core_size = system.core_size[i_body]
    buffer[4,i_buffer] = system.core_size[i_body] + 0.5 * norm(system.x[2,i_body] - system.x[1,i_body])

    # remainding quantities
    buffer[5:7,i_buffer] .= Γ
    buffer[8:10,i_buffer] .= system.x[1,i_body]
    buffer[11:13,i_buffer] .= system.x[2,i_body]
    buffer[14,i_buffer] = core_size
end

Base.eltype(::VortexFilaments{TF}) where TF = TF

FastMultipole.data_per_body(::VortexFilaments) = 14

FastMultipole.get_position(system::VortexFilaments, i) = (system.x[1,i] + system.x[2,i]) * 0.5

FastMultipole.strength_dims(system::VortexFilaments) = 3

FastMultipole.get_n_bodies(system::VortexFilaments) = length(system.strength)

function FastMultipole.buffer_to_target_system!(target_system::VortexFilaments, i_target, ::DerivativesSwitch{PS,VS,GS}, target_buffer, i_buffer) where {PS,VS,GS}

    # extract from buffer
    PS && (potential = FastMultipole.get_scalar_potential(target_buffer, i_buffer))
    VS && (velocity = FastMultipole.get_gradient(target_buffer, i_buffer))
    GS && (hessian = FastMultipole.get_hessian(target_buffer, i_buffer))

    # load into system
    PS && (target_system.potential[i_target] += potential)
    VS && (target_system.gradient[i_target] += velocity)
    GS && (target_system.hessian[i_target] += hessian)

end
```
First, we point out that the `source_system_to_buffer!` function stores a core size in row 14. Although it will not be used by any internal `FastMultipole` functions, it will be required by the `direct!` function. Since the columns of the buffer are sorted into an octree structure, they will not match the order of the original system; therefore, it is important to store essential body properties in additional rows of the buffer. We also note that the `get_position` function needs not return a previously-stored value; in this case, it computes the midpoint of the filament on the fly.

## Overloading `direct!`

The `FastMultipole.direct!` function is overloaded for `VortexFilaments` systems as follows:

```@example guidedex
function get_δ(distance, core_size)
    δ = distance < core_size ? (distance-core_size) * (distance-core_size) : zero(distance)
    return δ
end

function vortex_filament_finite_core_2(x1,x2,xt,q,core_size)
    # intermediate values
    r1 = xt - x1
    r2 = xt - x2

    nr1 = norm(r1)
    nr2 = norm(r2)

    num = cross(r1, r2)
    denom = nr1 * nr2 + dot(r1, r2)

    # core size comes into play here
    distance_1 = norm((x1+x2)*0.5 - xt)
    δ1 = get_δ(distance_1, core_size)

    distance_2 = norm(x1 - xt)
    δ2 = get_δ(distance_2, core_size)

    distance_3 = norm(x2 - xt)
    δ3 = get_δ(distance_3, core_size)

    # desingularized terms
    f1 = num/(denom + δ1)
    f2 = 1/(nr1+δ2)
    f3 = 1/(nr2+δ3)

    # evaluate velocity
    V = (f1*(f2+f3))/(4*pi) * q

    return V
end

function FastMultipole.direct!(target_system, target_index, derivatives_switch::DerivativesSwitch{PS,VS,GS}, source_system::VortexFilaments, source_buffer, source_index) where {PS,VS,GS}
    for i_source in source_index
        x1 = FastMultipole.get_vertex(source_buffer, source_system, i_source, 1)
        x2 = FastMultipole.get_vertex(source_buffer, source_system, i_source, 2)
        q = FastMultipole.get_strength(source_buffer, source_system, i_source)
        core_size = source_buffer[14, i_source]

        for i_target in target_index
            xt = FastMultipole.get_position(target_system, i_target)

            if VS
                # determine sign of q
                q_mag = norm(q) * sign(dot(q, x2-x1))

                # calculate velocity
                v = vortex_filament_finite_core_2(x1,x2,xt,q_mag,core_size)
                FastMultipole.set_gradient!(target_system, i_target, v)
            end
        end
    end
end
```
Note that `core_size` is retrieved from the buffer matrix rather than from `source_system` directly, as discussed previously.

!!! warning
    The bodies contained inside the `source_system` argument of the `direct!` function will not be in the correct order for use in the FMM. For this reason, it is not recommended to access the `source_system` in the `direct!` function except for global properties that apply to all bodies. Instead, use the `source_buffer` matrix as shown in the example above.

## Running the FMM

Let's try this out on a vortex filament system. The `fmm!` function is called in the same way as for the gravitational example, but with the `VortexFilaments` system:

```@example guidedex
using LinearAlgebra

# create system
function generate_filament_field(n_filaments, length_scale; strength_scale=1/n_filaments)
    centers = rand(SVector{3,Float64}, n_filaments)
    pts = zeros(SVector{3,Float64}, 2, n_filaments)
    strength_vec= zeros(SVector{3,Float64}, n_filaments)
    for (i,center) in enumerate(centers)
        dx = (rand(SVector{3,Float64}) * 2 .- 1.0) * 0.5 * length_scale
        pts[1,i] = center - dx
        pts[2,i] = center + dx
        Γ = dx / norm(dx) * rand() * strength_scale
        strength_vec[i] = Γ
    end

    # create filaments
    core_size = fill(1e-2, n_filaments)
    error_tolerance = fill(1e-4, n_filaments)
    potential = zeros(length(strength_vec))
    gradient = zeros(SVector{3,Float64}, length(strength_vec))
    hessian = zeros(SMatrix{3,3,Float64,9}, length(strength_vec))

    return VortexFilaments(pts, strength_vec, core_size, error_tolerance, potential, gradient, hessian)
end

n_filaments = 5000
length_scale = 1.0 / n_filaments^(1/3)
filaments = generate_filament_field(n_filaments, length_scale; strength_scale=1/n_filaments)

# run FMM
_, cache, _ = fmm!(filaments; scalar_potential=false, gradient=true)

# print results
println("Induced Velocity:\n", filaments.gradient[1:10], "...")
```
We set `scalar_potential=false` and `gradient=true` to indicate that we want to compute the vector field (i.e. the velocity field) but not the scalar potential.
