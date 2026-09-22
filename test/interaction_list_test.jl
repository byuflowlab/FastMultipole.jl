@testset "interaction list: sort convenience function" begin

Random.seed!(456)
n_bodies = 1_000_000
bodies = rand(8,n_bodies)
masses = Gravitational(bodies)
switches = DerivativesSwitch(true, true, true, (masses,))
tree = Tree((masses,), false, switches; leaf_size = SVector{1}(100))
mac = 0.5

m2l_list, direct_list = build_interaction_lists(tree.branches, tree.branches, tree.leaf_index, mac, true, true, true)

source_sorted_direct_list = FastMultipole.sort_by_source(direct_list, tree.branches)
for i in 2:length(source_sorted_direct_list)
    _, i_source = source_sorted_direct_list[i]
    _, im1_source = source_sorted_direct_list[i-1]
    @test i_source >= im1_source
end

target_sorted_direct_list = FastMultipole.sort_by_target(direct_list, tree.branches)
for i in 2:length(target_sorted_direct_list)
    i_target, _ = target_sorted_direct_list[i]
    im1_target, _ = target_sorted_direct_list[i-1]
    @test i_target >= im1_target
end

end
