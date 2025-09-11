using FastMultipole
using Random

include("../test/gravitational.jl")
include("../test/vortex.jl")
include("../test/panels.jl")
include("../test/evaluate_multipole.jl")

function test_direct()
    # multithreaded test
    @assert Threads.nthreads() > 1
    leaf_size, multipole_acceptance, n_bodies = 1, 0.5, 10000
    system = generate_gravitational(123, n_bodies; radius_factor=0.0)
    tree, m2l_list, direct_list, derivatives = fmm!(system; expansion_order=4, leaf_size, multipole_acceptance, farfield=false, unsort_bodies=false)

    # single-threaded check
    system2 = generate_gravitational(123, n_bodies; radius_factor=0.0)
    FastMultipole.resort!(system2, tree)
    FastMultipole.nearfield_singlethread!(system2, tree.branches, derivatives, system2, tree.branches, direct_list)

    ϕ_multithread = system.potential[1:7,:]
    ϕ_singlethread = system2.potential[1:7,:]
    println("Test direct:")
    @show maximum(abs.(ϕ_multithread - ϕ_singlethread))

end

#test_direct()

function test_expansions()
    # multithreaded test
    @assert Threads.nthreads() > 1
    leaf_size, multipole_acceptance, expansion_order, n_bodies = 1, 0.5, 1, 10
    system = generate_gravitational(123, n_bodies; radius_factor=0.0)
    tree, m2l_list, direct_list, derivatives = fmm!(system; expansion_order, leaf_size, multipole_acceptance, nearfield=false, unsort_bodies=false)

    # single-threaded check
    system2 = generate_gravitational(123, n_bodies; radius_factor=0.0)
    FastMultipole.resort!(system2, tree)
    FastMultipole.reset_expansions!(tree)

    Pmax = expansion_order
    lamb_helmholtz = Val(false)
    one_thread = 1
    FastMultipole.upward_pass_multithread!(tree.branches, system2, Val(Pmax), lamb_helmholtz, tree.levels_index, tree.leaf_index, one_thread)
    println("\n\nSingle Threaded tests:")
    println("\tpost upward pass")
    @show tree.branches[end].multipole_expansion
    FastMultipole.horizontal_pass_multithread!(tree.branches, tree.branches, m2l_list, lamb_helmholtz, Val(Pmax), one_thread)
    println("\tpost horizontal pass")
    @show tree.branches[end].local_expansion
    FastMultipole.downward_pass_multithread!(tree.branches, system2, derivatives, Val(Pmax), lamb_helmholtz, tree.levels_index, tree.leaf_index, one_thread)
    println("\tpost downward pass")
    @show tree.branches[end].local_expansion

    ϕ_multithread = system.potential[1:7,:]
    ϕ_singlethread = system2.potential[1:7,:]
    println("Test expansions:")
    @show maximum(abs.(ϕ_multithread - ϕ_singlethread))

end

#test_expansions()

function test_full()

    # multithreaded test
    @assert Threads.nthreads() > 1
    leaf_size, multipole_acceptance, expansion_order, n_bodies = 100, 0.4, 16, 10000
    system = generate_gravitational(123, n_bodies; radius_factor=0.0)
    tree, m2l_list, direct_list, derivatives = fmm!(system; expansion_order, leaf_size, multipole_acceptance, nearfield=true, unsort_bodies=false)

    # single-threaded check
    system2 = generate_gravitational(123, n_bodies; radius_factor=0.0)
    FastMultipole.resort!(system2, tree)
    FastMultipole.reset_expansions!(tree)

    Pmax = expansion_order
    lamb_helmholtz = Val(false)
    one_thread = 1
    FastMultipole.nearfield_singlethread!(system2, tree.branches, derivatives, system2, tree.branches, direct_list)
    FastMultipole.upward_pass_multithread!(tree.branches, system2, Val(Pmax), lamb_helmholtz, tree.levels_index, tree.leaf_index, one_thread)
    FastMultipole.horizontal_pass_multithread!(tree.branches, tree.branches, m2l_list, lamb_helmholtz, Val(Pmax), one_thread)
    FastMultipole.downward_pass_multithread!(tree.branches, system2, derivatives, Val(Pmax), lamb_helmholtz, tree.levels_index, tree.leaf_index, one_thread)

    ϕ_multithread = system.potential[1:7,:]
    ϕ_singlethread = system2.potential[1:7,:]

    # check error
    system3 = generate_gravitational(123, n_bodies; radius_factor=0.0)
    FastMultipole.resort!(system3, tree)
    FastMultipole.direct!(system3)
    ϕ_direct = system3.potential[1:7,:]
    println("Test Direct")
    @show maximum(abs.(ϕ_multithread - ϕ_singlethread))
    @show maximum(abs.(ϕ_direct - ϕ_singlethread))

end

test_full()

function test_vorton_full()

    # multithreaded test
    @assert Threads.nthreads() > 1
    leaf_size, multipole_acceptance, expansion_order, n_bodies = 1, 0.4, 16, 10000
    system = generate_vortex(123, n_bodies)
    tree, m2l_list, direct_list, derivatives = fmm!(system; expansion_order, leaf_size, multipole_acceptance, nearfield=true, unsort_bodies=false, lamb_helmholtz=true)

    @assert length(m2l_list) > 0

    # single-threaded check
    system2 = generate_vortex(123, n_bodies)
    FastMultipole.resort!(system2, tree)
    FastMultipole.reset_expansions!(tree)

    Pmax = expansion_order
    lamb_helmholtz = Val(true)
    one_thread = 1
    FastMultipole.nearfield_singlethread!(system2, tree.branches, derivatives, system2, tree.branches, direct_list)
    FastMultipole.upward_pass_multithread!(tree.branches, system2, Val(Pmax), lamb_helmholtz, tree.levels_index, tree.leaf_index, one_thread)
    FastMultipole.horizontal_pass_multithread!(tree.branches, tree.branches, m2l_list, lamb_helmholtz, Val(Pmax), one_thread)
    FastMultipole.downward_pass_multithread!(tree.branches, system2, derivatives, Val(Pmax), lamb_helmholtz, tree.levels_index, tree.leaf_index, one_thread)

    ϕ_multithread = system.potential[1:7,:]
    ϕ_singlethread = system2.potential[1:7,:]

    # check error
    system3 = generate_vortex(123, n_bodies)
    FastMultipole.resort!(system3, tree)
    FastMultipole.direct!(system3)
    ϕ_direct = system3.potential[1:7,:]
    println("Test Direct")
    @show maximum(abs.(ϕ_multithread - ϕ_singlethread))
    @show maximum(abs.(ϕ_direct - ϕ_singlethread))

end

test_vorton_full()
