# internals the suites exercise directly (not part of the exported surface)
using FastMultipole: RadixGrid, DeviceRadixGrid, ConstantPStencilConfig, RadixSeparationPolicy, ParentNeighborM2L, RigidHierarchicalTables, RadixLevelOccupancy, RadixM2LBatch, RadixInteractionList, TreeRole, SourceTree, TargetTree, element_strength_dims, DeviceResidentRadixState, AbstractOperatorBasis, CompressedComplexBasis, OperatorOrders, OperatorBasisInfo, OperatorInvariantCache, OperatorScratch, FlatCoefficientBuffer, AbstractM2LOperator, MaterializedYRotationM2L, FactoredRotationM2L, M2LOperatorScratch, rigid_stencil_epsilon, constant_p_stencil_bound, build_radix_interaction_list, RadixRouteSelection, host_radix_state, host_resident_radix_grid, run_host_radix_lifecycle!, finalize_radix_output!, snapshot_locked_radix_settings, verify_locked_radix_settings, rect_source_rows, rect_output_rows
# Shared backend selection for the ka_*_correctness.jl suites.
#
# These suites were written against Metal (the only GPU on the dev machine) with
# `Metal.MtlArray`/`Metal.MetalBackend()` hardcoded. Step (v) of the dispatch-
# wiring plan -- handing `actx.grid` to a `DeviceResidentRadixState` -- runs into
# the CUDA-only resident lifecycle, so from here on correctness has to be gated
# on real H200 hardware, not Metal. This file is that gate's front end: include
# it instead of `using Metal`, and the same suite text runs on either backend.
#
# Metal/CUDA are mutually exclusive by package availability, not by choice: the
# HPC env has no Metal installed (no Apple GPU there), so `using Metal` must not
# even be attempted off-Apple or package resolution fails before any code runs.
# Same `@static if` reasoning as tree_build_benchmark_4way.jl.

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
