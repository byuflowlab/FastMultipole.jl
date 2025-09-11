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
include("direct_test.jl")
include("harmonics_test.jl")
include("rotate_test.jl")
include("bodytomultipole_test.jl")
include("multipole_power_test.jl")
include("translate_multipole_test.jl")
include("translate_multipole_to_local_test.jl")
include("translate_local_test.jl")
include("evaluate_expansions_test.jl")
include("lamb_helmholtz_test.jl")
include("tree_test.jl")
include("dynamic_expansion_order_test.jl")
include("interaction_list_test.jl")
include("fmm_test.jl")
include("solve_test.jl")

