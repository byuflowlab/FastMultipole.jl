module FastMultipoleKAExt

using FastMultipole
using KernelAbstractions
using LinearAlgebra
using StaticArrays: SVector, SMatrix
using GPUArraysCore: AnyGPUMatrix
const KA = KernelAbstractions

# Constructing a KA kernel object (`some_kernel!(backend, workgroup)`) redoes
# generic dispatch/partitioning work on every call; caching it per (kernel
# function, backend type, workgroup) avoids repeating that on the hot path.
const _KERNEL_CACHE = Dict{Tuple{Any,DataType,Int},Any}()
function _cached_kernel(f, backend, workgroup::Int)
    wg = resolve_workgroup(backend, workgroup)
    key = (f, typeof(backend), wg)
    return get!(() -> f(backend, wg), _KERNEL_CACHE, key)
end


# The extension is one module split by responsibility; the files are included in
# dependency order (definitions depend on earlier ones).
include("ka/ka_primitives.jl")
include("ka/ka_body_kernels.jl")
include("ka/ka_lifecycle.jl")
include("ka/ka_hierarchical_m2l.jl")
include("ka/ka_workspace_routes.jl")
include("ka/ka_grid_refresh.jl")
include("ka/ka_finalize_refresh.jl")
include("ka/ka_device_step.jl")
include("ka/ka_cache_build.jl")
include("ka/ka_extra_systems.jl")
include("ka/ka_functors.jl")
include("ka/ka_entry.jl")

end # module
