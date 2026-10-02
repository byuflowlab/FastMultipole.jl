# internals the suites exercise directly (not part of the exported surface)
using FastMultipole: RadixGrid, DeviceRadixGrid, ConstantPStencilConfig, RadixSeparationPolicy, ParentNeighborM2L, RigidHierarchicalTables, RadixLevelOccupancy, RadixM2LBatch, RadixInteractionList, TreeRole, SourceTree, TargetTree, element_strength_dims, DeviceResidentRadixState, AbstractOperatorBasis, CompressedComplexBasis, OperatorOrders, OperatorBasisInfo, OperatorInvariantCache, OperatorScratch, FlatCoefficientBuffer, AbstractM2LOperator, MaterializedYRotationM2L, FactoredRotationM2L, M2LOperatorScratch, rigid_stencil_epsilon, constant_p_stencil_bound, RadixRouteSelection, run_host_radix_lifecycle!, finalize_radix_output!, snapshot_locked_radix_settings, verify_locked_radix_settings, rect_source_rows, rect_output_rows
# test-only radix reference code (standalone RadixGrid, ParentNeighborM2L list,
# host_radix_state) shared with the host suite
include(joinpath(@__DIR__, "..", "helpers", "radix_reference.jl"))
# Shared backend selection for the ka_*_correctness.jl suites: include it
# instead of `using Metal` / `using CUDA`, and the same suite text runs on
# either backend.
#
# Metal/CUDA are mutually exclusive by package availability, not by choice: a
# non-Apple environment has no Metal installed, so `using Metal` must not even
# be attempted off-Apple or package resolution fails before any code runs.

using KernelAbstractions
import Base.Sys: isapple

const HAS_METAL = isapple()

@static if isapple()
    using Metal
    const DEV_NAME = "Metal"
    const DEV_BACKEND = Metal.MetalBackend()
    devarray(x) = Metal.MtlArray(x)
    const devmatrix = Metal.MtlMatrix
    dev_functional() = Metal.functional()
else
    using CUDA
    const DEV_NAME = "CUDA"
    const DEV_BACKEND = CUDABackend()
    devarray(x) = CUDA.CuArray(x)
    const devmatrix = CUDA.CuMatrix
    dev_functional() = CUDA.functional()
end
