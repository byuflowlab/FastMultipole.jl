@testset "direct" begin

    function V(xi, xj, mj; G=1)
        Rho_ij = xi - xj
        rho_ij = sqrt(Rho_ij' * Rho_ij)
        vij = G * mj / rho_ij / 4 / pi
        if isinf(vij); vij = 0.0; end
        return Rho_ij, rho_ij, vij
    end

    x = [
        -5.0 -4.8 -5.1 4.9 5.2 5.1 5.3;
        0.4 0.2 0.1 0.3 0.2 -0.1 0.1;
        -0.1 0.2 0.1 0.1 0.0 -0.1 -0.2
    ]

    m = [
        1.4,
        2.0,
        2.3,
        4.1,
        1.1,
        3.4,
        4.5
    ]

    Rho_ijs = zeros(3,length(m),length(m))
    rho_ijs = zeros(length(m),length(m))
    V_ijs = zeros(length(m),length(m))

    for i in 1:length(m)
        for j in 1:length(m)
            Rho_ij, rho_ij, vij = V(x[:,i], x[:,j], m[j])
            Rho_ijs[:,i,j] .= Rho_ij
            rho_ijs[i,j] = rho_ij
            V_ijs[i,j] = vij
        end
    end

    V_tots = zeros(length(m))
    for i = 1:length(m)
        V_tots[i] = sum(V_ijs[i,:])
    end

    bodies = vcat(x,rand(1,length(m)),m',zeros(3,length(m)))
    mass = Gravitational(bodies)

    FastMultipole.direct!(mass; scalar_potential=true)
    V_tots_direct = mass.potential[1,:]

    for i in 1:length(V_tots)
        @test isapprox(V_tots[i], V_tots_direct[i]; atol=1e-4)
    end
end

@testset "big direct" begin
    
n_bodies = 1000
mass = generate_gravitational(123, n_bodies)
source_tree = Tree((mass,), false; leaf_size=SVector{1}(1))
target_tree = Tree((mass,), true; leaf_size=SVector{1}(1))
n_leaves_target = length(target_tree.leaf_index)
n_leaves_source = length(source_tree.leaf_index)
direct_list = Vector{SVector{2,Int}}(undef, n_leaves_target*n_leaves_source)
k = 1
for n in target_tree.leaf_index
    for m in source_tree.leaf_index
        direct_list[k] = SVector{2,Int}(n,m)
        k += 1
    end
end

derivatives_switches = FastMultipole.DerivativesSwitch(true, false, false, (mass,))
interaction_list_method = FastMultipole.SelfTuning()
FastMultipole.nearfield_multithread!(target_tree.buffers, target_tree.branches, (mass,), source_tree.buffers, source_tree.branches, derivatives_switches, direct_list, interaction_list_method, Threads.nthreads())
# copy results to target systems
FastMultipole.buffer_to_target!((mass,), target_tree, derivatives_switches)

potential_multithread = mass.potential[1,:]

# single-threaded approach
FastMultipole.reset!(target_tree.buffers)

FastMultipole.nearfield_singlethread!(target_tree.buffers, target_tree.branches, (mass,), source_tree.buffers, source_tree.branches, derivatives_switches, direct_list)
# copy results to target systems
FastMultipole.buffer_to_target!((mass,), target_tree, derivatives_switches)

potential_singlethread = mass.potential[1,:]

@test isapprox(potential_multithread, potential_singlethread; atol=1e-12)

end
