#=
Tests `farfield_extra_outputs!`: the user hook called at the end of L2B that accumulates
user-defined extra outputs from the FARFIELD (local expansion) contribution.

Setup: point-source gravity. Each target carries a unit normal in its metadata rows, and the
extra output is the normal flux `dot(∇φ, n̂)` -- a LINEAR functional of the field, so the
nearfield half (computed in the source's `direct!`) and the farfield half (computed in the hook)
must sum to the flux of the total gradient. That identity is the core assertion.

`FluxTargetNoHook` is identical to `FluxTarget` but does not overload the hook, giving a direct
"feature off" control.
=#

struct FluxTarget{TF}
    position::Matrix{TF}      # 3 x n
    normal::Matrix{TF}        # 3 x n, unit vectors
    gradient::Matrix{TF}      # 3 x n, output
    extra::Vector{TF}         # n, output: normal flux
end

struct FluxTargetNoHook{TF}
    position::Matrix{TF}
    normal::Matrix{TF}
    gradient::Matrix{TF}
    extra::Vector{TF}
end

const AnyFluxTarget{TF} = Union{FluxTarget{TF},FluxTargetNoHook{TF}}

function make_flux_target(::Type{T}, seed, n_bodies) where T
    Random.seed!(seed)
    position = rand(3, n_bodies)
    normal = randn(3, n_bodies)
    for i in 1:n_bodies
        normal[:, i] ./= norm(view(normal, :, i))
    end
    return T(position, normal, zeros(3, n_bodies), zeros(n_bodies))
end

Base.eltype(::AnyFluxTarget{TF}) where TF = TF
FastMultipole.get_n_bodies(system::AnyFluxTarget) = length(system.extra)
FastMultipole.get_position(system::AnyFluxTarget{TF}, i) where TF =
    SVector{3,TF}(system.position[1, i], system.position[2, i], system.position[3, i])

# the normal lives in metadata, so it is sorted along with positions and is visible to both the
# nearfield `direct!` and the farfield hook
FastMultipole.metadata_per_body(system::AnyFluxTarget) = 3

function FastMultipole.metadata_to_buffer!(buffer, switch, i_buffer, system::AnyFluxTarget, i_body)
    for j in 1:3
        buffer[FastMultipole.metadata_index(switch, j), i_buffer] = system.normal[j, i_body]
    end
end

@inline function buffer_normal(target_buffer, switch, i_buffer)
    TF = eltype(target_buffer)
    return SVector{3,TF}(
        target_buffer[FastMultipole.metadata_index(switch, 1), i_buffer],
        target_buffer[FastMultipole.metadata_index(switch, 2), i_buffer],
        target_buffer[FastMultipole.metadata_index(switch, 3), i_buffer],
    )
end

#--- THE FEATURE UNDER TEST: farfield half of the normal flux ---#

function FastMultipole.farfield_extra_outputs!(target_buffer, switch, i_buffer,
        target_system::FluxTarget, scalar_potential, gradient, hessian, third_derivative)
    n̂ = buffer_normal(target_buffer, switch, i_buffer)
    FastMultipole.set_extra_output!(target_buffer, switch, i_buffer, 1, dot(gradient, n̂))
end

# FluxTargetNoHook deliberately does NOT overload farfield_extra_outputs!

function FastMultipole.buffer_to_target_system!(target_system::AnyFluxTarget, i_target,
        switch::FastMultipole.DerivativesSwitch{PS,GS,HS,NO,NM,TS}, target_buffer, i_buffer) where {PS,GS,HS,NO,NM,TS}
    target_system.gradient[:, i_target] .+= FastMultipole.get_gradient(target_buffer, switch, i_buffer)
    # NO is a compile-time parameter; guard because get_extra_output is @inbounds and the extra
    # range is empty when no extra outputs were requested
    NO > 0 && (target_system.extra[i_target] += FastMultipole.get_extra_output(target_buffer, switch, i_buffer, 1))
end

#--- source: point masses whose direct! also accumulates the NEARFIELD half of the flux ---#

struct FluxSource{TF}
    position::Matrix{TF}      # 3 x n
    strength::Vector{TF}
end

function make_flux_source(seed, n_bodies)
    Random.seed!(seed + 7919)
    return FluxSource(rand(3, n_bodies) .+ 2.5, rand(n_bodies) ./ n_bodies)
end

Base.eltype(::FluxSource{TF}) where TF = TF
FastMultipole.get_n_bodies(system::FluxSource) = length(system.strength)
FastMultipole.get_position(system::FluxSource{TF}, i) where TF =
    SVector{3,TF}(system.position[1, i], system.position[2, i], system.position[3, i])
FastMultipole.data_per_body(system::FluxSource) = 5
FastMultipole.strength_dims(system::FluxSource) = 1
FastMultipole.has_vector_potential(system::FluxSource) = false
FastMultipole.body_to_multipole!(system::FluxSource, args...) =
    FastMultipole.body_to_multipole!(Point{Source}, system, args...; scale_strength=-1.0)

function FastMultipole.source_system_to_buffer!(buffer, i_buffer, system::FluxSource, i_body)
    buffer[1:3, i_buffer] .= FastMultipole.get_position(system, i_body)
    buffer[4, i_buffer] = 0.0
    buffer[5, i_buffer] = system.strength[i_body]
end

function FastMultipole.direct!(target_buffer, target_index,
        switch::FastMultipole.DerivativesSwitch{PS,GS,HS,NO,NM,TS},
        source_system::FluxSource, source_buffer, source_index) where {PS,GS,HS,NO,NM,TS}
    TF = eltype(target_buffer)
    @inbounds for j_target in target_index
        target_position = FastMultipole.get_position(target_buffer, j_target)
        d∇ϕ = zero(SVector{3,TF})
        for i_source in source_index
            source_position = FastMultipole.get_position(source_buffer, i_source)
            source_strength = FastMultipole.get_strength(source_buffer, source_system, i_source)[1]
            dx = target_position - source_position
            r2 = dot(dx, dx)
            if r2 > 0
                r = sqrt(r2)
                d∇ϕ -= dx * (source_strength / r * FastMultipole.ONE_OVER_4π) / r2
            end
        end
        GS && FastMultipole.set_gradient!(target_buffer, switch, j_target, d∇ϕ)
        # nearfield half of the same linear functional the farfield hook computes
        if NO > 0
            n̂ = buffer_normal(target_buffer, switch, j_target)
            FastMultipole.set_extra_output!(target_buffer, switch, j_target, 1, dot(d∇ϕ, n̂))
        end
    end
end

@testset "farfield extra outputs (L2B hook)" begin

    n_target, n_source = 1_500, 1_500
    source = make_flux_source(3, n_source)

    target = make_flux_target(FluxTarget, 11, n_target)
    control = make_flux_target(FluxTargetNoHook, 11, n_target)
    @test target.position == control.position && target.normal == control.normal

    optargs = (expansion_order=12, leaf_size_source=SVector{1}(50),
               multipole_acceptance=0.4, scalar_potential=false, gradient=true,
               extra_outputs=1)

    fmm!((target,), (source,); optargs...)
    fmm!((control,), (source,); optargs...)

    normals = [SVector{3}(target.normal[:, i]) for i in 1:n_target]
    gradients = [SVector{3}(target.gradient[:, i]) for i in 1:n_target]

    #--- 1. the identity: nearfield half + farfield half == flux of the total gradient ---#
    # both halves come from the same expression applied to the same field, so the only
    # discrepancy is floating-point summation order
    flux_from_gradient = [dot(g, n̂) for (g, n̂) in zip(gradients, normals)]
    @test isapprox(target.extra, flux_from_gradient; rtol=1e-10)

    #--- 2. the gradient itself is correct (sanity on the underlying FMM) ---#
    reference = make_flux_target(FluxTarget, 11, n_target)
    FastMultipole.direct!((reference,), (source,); scalar_potential=false, gradient=true,
        extra_outputs=1, n_threads=1)
    ref_gradients = [SVector{3}(reference.gradient[:, i]) for i in 1:n_target]
    grad_err = maximum(norm(g - r) for (g, r) in zip(gradients, ref_gradients))
    @test grad_err / maximum(norm, ref_gradients) < 1e-6

    #--- 3. the reference flux (all pairs direct) matches the FMM flux ---#
    @test isapprox(target.extra, reference.extra; rtol=1e-5)

    #--- 4. control: without the hook, only the nearfield half lands, so the result is WRONG ---#
    # (guards against the test passing for a reason other than the hook firing)
    @test !isapprox(control.extra, reference.extra; rtol=1e-2)
    farfield_share = maximum(abs.(target.extra .- control.extra)) / maximum(abs.(target.extra))
    @test farfield_share > 0.1

    #--- 5. the hook is a no-op when no extra outputs are requested ---#
    plain = make_flux_target(FluxTarget, 11, n_target)
    fmm!((plain,), (source,); expansion_order=12, leaf_size_source=SVector{1}(50),
        multipole_acceptance=0.4, scalar_potential=false, gradient=true)
    @test all(iszero, plain.extra)
    @test isapprox(plain.gradient, target.gradient; rtol=1e-12)
end
