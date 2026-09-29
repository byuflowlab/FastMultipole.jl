#------- STAGE 0: real `fmm!` dispatch, through the src/ backend registry -------#
#
# `fmm!(targets, sources, cache)` reaches
# `FastMultipole._radix_cache_device_step!`, whose stub now consults the
# registry in `register_radix_device_backend!` instead of throwing. CUDA is
# unaffected -- its runtime `include` replaces the consulting stub outright, so
# a CUDA build never reaches the registry.



function _ka_radix_device_build_hook(sources::Tuple, args...; kwargs...)
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
# One kernel for every AbstractRectangularKernel: each work-item runs the same
# per-target sum as the host loop (FastMultipole._rect_target!), which calls the
# kernel type's `rect_pair`. A consumer package that defines `rect_pair` for its
# own isbits kernel type gets this device method without further code. out,
# targets and sources must all be device arrays on one backend.

@kernel function ka_rect_kernel!(out, @Const(targets), kernel, @Const(sources), n_sources,
        grad::Val, pot::Val)
    i = @index(Global)
    FastMultipole._rect_target!(out, targets, kernel, sources, i, n_sources, grad, pot)
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
        kernel::FastMultipole.AbstractRectangularKernel, sources::AbstractMatrix{T};
        gradient::Bool=false, scalar_potential::Bool=false, workgroup::Int=64) where T
    FastMultipole._rect_check_args(out, targets, kernel, sources, gradient, scalar_potential)
    _ka_rect_assert_device(out, targets, sources)
    n_targets = size(targets, 2)
    n_targets == 0 && return out
    backend = KA.get_backend(out)
    kern = _cached_kernel(ka_rect_kernel!, backend, workgroup)
    kern(out, targets, kernel, sources, size(sources, 2), Val(gradient), Val(scalar_potential);
        ndrange=n_targets)
    KA.synchronize(backend)
    return out
end
