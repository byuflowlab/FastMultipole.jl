#------- STAGE 0: real `fmm!` dispatch, through the src/ backend registry -------#
#
# `ka_fmm!` (above) still exists as the direct entry the bench and suites call.
# What follows wires the REAL one: `fmm!(targets, sources, cache)` reaches
# `FastMultipole._radix_cache_device_step!`, whose stub now consults the
# registry in `register_radix_device_backend!` instead of throwing. CUDA is
# unaffected -- its runtime `include` replaces the consulting stub outright, so
# a CUDA build never reaches the registry.



function _ka_radix_device_build_hook(sources::Tuple, args...;
        adaptive_policy=nothing, dpb_adaptive::Int=0, kwargs...)
    adaptive_policy === nothing || throw(ArgumentError(
        "the KA radix backend has no adaptive octree lifecycle; " *
        "build the cache with adaptive=nothing"))
    backend = FastMultipole.radix_sources_backend(sources)
    backend === nothing && throw(ArgumentError(
        "RadixFMMCache(device=true) resolved to the KA backend, but no source " *
        "system names one; define FastMultipole.device_backend(system) to " *
        "return the KernelAbstractions backend its storage lives on"))
    return ka_radix_cache_device_build(backend, sources, args...; kwargs...)
end

_ka_radix_device_step_hook(cache, targets, switches; nearfield_pass=nothing,
        extra_targets::Tuple=(), extra_target_switches::Tuple=(),
        extra_sources::Tuple=(), extra_tree_sources::Tuple=(),
        self_induce::Bool=true) =
    ka_radix_cache_device_step!(cache, targets, switches; nearfield_pass, extra_targets,
        extra_target_switches, extra_sources, extra_tree_sources, self_induce)

function __init__()
    FastMultipole.register_radix_device_backend!("KernelAbstractions",
        _ka_radix_device_build_hook, _ka_radix_device_step_hook)
    FastMultipole._RADIX_DEVICE_UPDATE_HOOK[] = ka_update_radix_state!
    return nothing
end


#------- direct_rectangular! on device arrays (all pairs, one work-item per target) -------#
#
# Mirrors _rect_points_host! / _rect_panels_host! (src/direct_rectangular.jl):
# the pair functions are device-compilable as written, so the device methods
# only supply the target-parallel loop. out, targets and sources must all be
# device arrays of the same element type.

@kernel function ka_rect_points_kernel!(out, @Const(targets), @Const(sources), n_sources, ::Val{GRAD}) where GRAD
    i = @index(Global)
    T = eltype(out)
    @inbounds begin
        tx = targets[1, i]; ty = targets[2, i]; tz = targets[3, i]
        u1 = zero(T); u2 = zero(T); u3 = zero(T)
        j1 = zero(T); j2 = zero(T); j3 = zero(T); j4 = zero(T); j5 = zero(T)
        j6 = zero(T); j7 = zero(T); j8 = zero(T); j9 = zero(T)
        for q in 1:n_sources
            Ux, Uy, Uz, a1, a2, a3, a4, a5, a6, a7, a8, a9 =
                FastMultipole._rect_point_pair(FastMultipole.RectangularGaussianErfVortex(), tx, ty, tz,
                    sources[1, q], sources[2, q], sources[3, q],
                    sources[4, q], sources[5, q], sources[6, q],
                    sources[7, q], Val(GRAD))
            u1 += Ux; u2 += Uy; u3 += Uz
            if GRAD
                j1 += a1; j2 += a2; j3 += a3; j4 += a4; j5 += a5
                j6 += a6; j7 += a7; j8 += a8; j9 += a9
            end
        end
        out[1, i] += u1; out[2, i] += u2; out[3, i] += u3
        if GRAD
            out[4, i] += j1; out[5, i] += j2; out[6, i] += j3
            out[7, i] += j4; out[8, i] += j5; out[9, i] += j6
            out[10, i] += j7; out[11, i] += j8; out[12, i] += j9
        end
    end
end

@kernel function ka_rect_panels_kernel!(out, @Const(targets), @Const(sources), n_sources,
        ::Val{GRAD}, ::Val{POT}, ::Val{REG}) where {GRAD,POT,REG}
    i = @index(Global)
    T = eltype(out)
    @inbounds begin
        target = SVector{3,T}(targets[1, i], targets[2, i], targets[3, i])
        u = zero(SVector{3,T})
        g = zero(SMatrix{3,3,T,9})
        p = zero(T)
        for q in 1:n_sources
            tag, nv, v1, v2, v3, v4, s1, s2, koff =
                FastMultipole._rect_load_panel_source(sources, q, T)
            uq, gq, pq = FastMultipole._rect_panel_pair(FastMultipole.RectangularPanelInfluence(), target,
                tag, nv, v1, v2, v3, v4, s1, s2, koff, Val(GRAD), Val(REG), Val(POT))
            u += uq
            GRAD && (g += gq)
            POT && (p += pq)
        end
        out[1, i] += u[1]; out[2, i] += u[2]; out[3, i] += u[3]
        if GRAD
            for j in 1:3, k in 1:3
                out[3 + (j-1)*3 + k, i] += g[k, j]
            end
        end
        POT && (out[FastMultipole.rect_potential_row(GRAD), i] += p)
    end
end

function _ka_rect_assert_device(out, targets, sources)
    (targets isa AnyGPUMatrix && sources isa AnyGPUMatrix) || throw(ArgumentError(
        "direct_rectangular! on a device output needs device targets and sources " *
        "on the same backend; got $(typeof(targets)) and $(typeof(sources))"))
    KA.get_backend(out) == KA.get_backend(targets) == KA.get_backend(sources) ||
        throw(ArgumentError("direct_rectangular!: out, targets and sources must share one backend"))
    return nothing
end

function FastMultipole.direct_rectangular!(out::AnyGPUMatrix{T}, targets::AbstractMatrix{T},
        kernel::FastMultipole.RectangularGaussianErfVortex, sources::AbstractMatrix{T};
        gradient::Bool=false, scalar_potential::Bool=false, workgroup::Int=64) where T
    FastMultipole._rect_check_args(out, targets, kernel, sources, gradient, scalar_potential)
    _ka_rect_assert_device(out, targets, sources)
    n_targets = size(targets, 2)
    n_targets == 0 && return out
    backend = KA.get_backend(out)
    kern = ka_rect_points_kernel!(backend, workgroup)
    kern(out, targets, sources, size(sources, 2), Val(gradient); ndrange=n_targets)
    KA.synchronize(backend)
    return out
end

function FastMultipole.direct_rectangular!(out::AnyGPUMatrix{T}, targets::AbstractMatrix{T},
        kernel::FastMultipole.RectangularPanelInfluence, sources::AbstractMatrix{T};
        gradient::Bool=false, scalar_potential::Bool=false, workgroup::Int=64) where T
    FastMultipole._rect_check_args(out, targets, kernel, sources, gradient, scalar_potential)
    _ka_rect_assert_device(out, targets, sources)
    n_targets = size(targets, 2)
    n_targets == 0 && return out
    backend = KA.get_backend(out)
    regv = FastMultipole._rect_reg_val(kernel.filament_reg)
    kern = ka_rect_panels_kernel!(backend, workgroup)
    kern(out, targets, sources, size(sources, 2), Val(gradient), Val(scalar_potential), regv; ndrange=n_targets)
    KA.synchronize(backend)
    return out
end

