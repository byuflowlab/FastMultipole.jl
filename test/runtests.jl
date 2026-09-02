using FastMultipole

using FLOWMath
using ForwardDiff
using LegendrePolynomials
using FastMultipole.LinearAlgebra
using Random
using SpecialFunctions
using FastMultipole.StaticArrays
using Test

#--- define gravitational kernel and mass elements ---#

include("./gravitational.jl")
include("./vortex.jl")
include("./vortex_filament.jl")
include("./panels.jl")
include("evaluate_multipole.jl")
include("bodytolocal.jl")

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

#--- run tests ---#

include("auxilliary_test.jl")
include("metadata_extra_test.jl")
include("operator_cache_types_test.jl")
include("coefficient_buffer_layout_test.jl")
include("real_solid_harmonic_basis_test.jl")
include("direct_conditioning_test.jl")
include("direct_test.jl")
include("direct_rectangular_test.jl")
include("harmonics_test.jl")
include("rotate_test.jl")
include("rotate_batched_test.jl")
include("bodytomultipole_test.jl")
include("multipole_power_test.jl")
include("translate_multipole_test.jl")
include("translate_multipole_to_local_test.jl")
include("translate_batched_test.jl")
include("m2l_operator_test.jl")
include("m2m_l2l_operator_test.jl")
include("resident_m2m_gemm_test.jl")
include("precomputed_y_resident_m2l_test.jl")
include("dense_translation_m2l_test.jl")
include("radix_settings_test.jl")
include("cuda_radix_lifecycle_test.jl")
include("cuda_radix_integration_test.jl")
include("cuda_radix_hierarchical_test.jl")
include("cuda_radix_interface_test.jl")
include("cuda_radix_graph_test.jl")
include("cuda_radix_adaptive_test.jl")
include("translate_local_test.jl")
include("evaluate_expansions_test.jl")
include("lamb_helmholtz_test.jl")
include("tree_test.jl")
include("radix_grid_clustering_test.jl")
include("radix_interaction_list_test.jl")
include("hierarchical_m2l_host_test.jl")
include("radix_fmm_integration_test.jl")
include("radix_trimming_test.jl")
include("adaptive_octree_test.jl")
include("adaptive_lifecycle_test.jl")
include("radix_fmm_timestepping_test.jl")
include("device_system_interface_test.jl")
include("dynamic_expansion_order_test.jl")
include("interaction_list_test.jl")
include("fmm_test.jl")
include("fmm_plan_test.jl")
include("transform_tree_test.jl")
include("transform_plan_test.jl")
include("nearfield_cache_test.jl")
include("autotune_cost_test.jl")
include("solve_test.jl")
include("transform_solver_test.jl")
include("fgs_coloring_test.jl")

#--- GPU correctness suites ---#
# The ka_*_correctness.jl suites live in their own env (test/metal_env) so the
# main test env stays free of CUDA/Metal. They run only when a device is
# plausibly present; each suite still checks `dev_functional()` itself.
# run_suites.sh exits with the number of failing suites; per-suite logs go to
# test/metal_env/logs/. FASTMULTIPOLE_GPU_TESTS=0|1 overrides detection and
# FASTMULTIPOLE_GPU_TEST_PROJECT points the suites at another env (the H200
# checkout runs them under ~/fmauto_env).
gpu_present = Sys.isapple() || Sys.which("nvidia-smi") !== nothing
if get(ENV, "FASTMULTIPOLE_GPU_TESTS", gpu_present ? "1" : "0") == "1"
    @testset "GPU correctness suites" begin
        @test success(`bash $(joinpath(@__DIR__, "metal_env", "run_suites.sh"))`)
    end
end
