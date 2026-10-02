using BenchmarkTools

module TestMe

include("../test/helpers/gravitational.jl")

function bm_fmm(element)
    options = fmm.Options(4,1,4.0)
    fmm.fmm!((element,), options)
    return nothing
end

function bm_direct(element)
    fmm.direct!((element,))
    return nothing
end

n_bodies = 100
bodies = rand(8,n_bodies)
element = Gravitational(bodies)
options = fmm.Options(4,1,4.0)

end

n_bodies = 10
bodies = rand(8,n_bodies)
element = TestMe.Gravitational(bodies)
TestMe.bm_fmm(element) # for precompilation
TestMe.bm_direct(element) # for precompilation

println("n_bodies = 100")
n_bodies = 100
bodies = rand(8,n_bodies)
element = TestMe.Gravitational(bodies)
@btime TestMe.bm_fmm(element) # for testing
# @btime TestMe.bm_direct(element) # for testing

println("n_bodies = 1000")
n_bodies = 1000
bodies = rand(8,n_bodies)
element = TestMe.Gravitational(bodies)
@btime TestMe.bm_fmm(element) # for testing
# @btime TestMe.bm_direct(element) # for testing

println("n_bodies = 10000")
n_bodies = 10000
bodies = rand(8,n_bodies)
element = TestMe.Gravitational(bodies)
@btime TestMe.bm_fmm(element) # for testing
# @btime TestMe.bm_direct(element) # for testing
