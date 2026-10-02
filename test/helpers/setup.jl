# Everything a test file may rely on: packages, the internals the tests exercise,
# the test systems, and two expansion helpers. Loaded once per process (the
# serial run, or each worker of the parallel run in runtests.jl).
using FastMultipole
using KernelAbstractions, GPUArraysCore   # together, the trigger of the device extension
# internals the tests exercise directly (not part of the exported surface)
using FastMultipole: RadixGrid, DeviceRadixGrid, ConstantPStencilConfig, RadixSeparationPolicy, ParentNeighborM2L, RigidHierarchicalTables, RadixLevelOccupancy, RadixM2LBatch, RadixInteractionList, TreeRole, SourceTree, TargetTree, element_strength_dims, DeviceResidentRadixState, AbstractOperatorBasis, CompressedComplexBasis, OperatorOrders, OperatorBasisInfo, OperatorInvariantCache, OperatorScratch, FlatCoefficientBuffer, AbstractM2LOperator, MaterializedYRotationM2L, FactoredRotationM2L, M2LOperatorScratch, rigid_stencil_epsilon, constant_p_stencil_bound, RadixRouteSelection, run_host_radix_lifecycle!, finalize_radix_output!, snapshot_locked_radix_settings, verify_locked_radix_settings, rect_source_rows, rect_output_rows
# test-only radix reference code (standalone RadixGrid, ParentNeighborM2L list,
# host_radix_state, reference direct-pair kernels)
include("radix_reference.jl")

using FLOWMath
using ForwardDiff
using LegendrePolynomials
using FastMultipole.LinearAlgebra
using Random
using SpecialFunctions
using FastMultipole.StaticArrays
using Test

#--- define gravitational kernel and mass elements ---#

include("gravitational.jl")
include("vortex.jl")
include("vortex_filament.jl")
include("panels.jl")
include("evaluate_multipole.jl")
include("bodytolocal.jl")
include("metadata_system.jl")
include("rigid_motion.jl")

#--- helper functions ---#

function vector_to_expansion!(expansion, vector, index, expansion_order)
    len = length(expansion)
    i_comp = 1
    i = 1
    for n in 0:expansion_order
        for m in -n:n
            if m >= 0
                expansion[1,index,i_comp] = real(vector[i])
                expansion[2,index,i_comp] = imag(vector[i])
                i_comp += 1
            end
            i += 1
        end
    end
end

function test_expansion!(expansion1, expansion2, index, expansion_order; throwme=false)
    i = 1
    for n in 0:expansion_order
        for m in 0:n
            if throwme && !isapprox(expansion1[1,index,i], expansion2[1,index,i]; atol=1e-12)
                throw("n=$n, m=$m")
            end
            @test isapprox(expansion1[1,index,i], expansion2[1,index,i]; atol=1e-12)
            @test isapprox(expansion1[2,index,i], expansion2[2,index,i]; atol=1e-12)
            i += 1
        end
    end
end
