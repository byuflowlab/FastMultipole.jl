ENV["FASTMULTIPOLE_FORCE_CUDA_LOAD"] = "1"
ENV["FASTMULTIPOLE_REQUIRE_CUDA_TESTS"] = "1"

include(joinpath(@__DIR__, "..", "cuda_radix_lifecycle_test.jl"))

include(joinpath(@__DIR__, "..", "cuda_radix_integration_test.jl"))

include(joinpath(@__DIR__, "..", "cuda_radix_hierarchical_test.jl"))
