# @testset "octree creation" begin

    # build list of elements to sort
    xs = [
        1.2 1.1 0.8;
        0.8 0.9 0.2;
        0.1 0.2 0.9;
        0.1 0.3 0.2;
        0.2 0.25 0.4
    ]
    radii = zeros(Float64,size(xs)[1])
    ms = rand(size(xs)[1])
    bodies = vcat(xs',radii',ms',zeros(3,length(ms)))
    elements = Gravitational(bodies)

    # test center_radius function
    center, box = FastMultipole.center_box((elements,), Float64)

    # manually
    test_center = [0.65, 0.65, 0.55]
    x_min, x_max = 0.1, 1.2
    y_min, y_max = 0.2, 1.1
    z_min, z_max = 0.2, 0.9
    test_box = [max(x_max-test_center[1], test_center[1]-x_min),
    max(y_max-test_center[2], test_center[2]-y_min),
    max(z_max-test_center[3], test_center[3]-z_min)]

    for i in 1:3
        @test isapprox(center[i], test_center[i]; atol=1e-12)
    end
    for i in 1:3
        @test isapprox(test_box[i], box[i]; atol=1e-12)
    end

    # test branch! function
    expansion_order, leaf_size, multipole_acceptance = 2, SVector{1}(1), 0.5
    switches = DerivativesSwitch(true, true, true, (elements,))
    tree = FastMultipole.Tree((elements,), false, switches; expansion_order, leaf_size, shrink=false, recenter=false)

    r1 = max(max(tree.branches[1].box[1],tree.branches[1].box[2]),tree.branches[1].box[3])
    r2 = max(max(tree.branches[2].box[1],tree.branches[2].box[2]),tree.branches[2].box[3])

    test_branches = [
        5 0.65 0.65 0.55 sqrt(3)*r1;
        2 0.65-r1/2 0.65-r1/2 0.55-r1/2 sqrt(3)*r2;
        1 0.65+r1/2 0.65+r1/2 0.55-r1/2 sqrt(3)*r2;
        1 0.65-r1/2 0.65-r1/2 0.55+r1/2 sqrt(3)*r2;
        1 0.65+r1/2 0.65+r1/2 0.55+r1/2 sqrt(3)*r2;
        1 tree.branches[2].center[1]-r2/2 tree.branches[2].center[2]-r2/2 tree.branches[2].center[3]-r2/2 sqrt(3)*r2/2;
        1 tree.branches[2].center[1]-r2/2 tree.branches[2].center[2]-r2/2 tree.branches[2].center[3]+r2/2 sqrt(3)*r2/2
    ]

    @test length(tree.branches) == size(test_branches)[1]

    for i_branch in 1:length(tree.branches)
        @test isapprox(length(tree.branches[i_branch].bodies_index[1]), test_branches[i_branch,1]; atol=1e-8)
        for i in 1:3
            @test isapprox(tree.branches[i_branch].center[i], test_branches[i_branch,1+i]; atol=1e-7)
        end
        @test isapprox(tree.branches[i_branch].radius, test_branches[i_branch,5]; atol=1e-7)
    end
# end

@testset "octree creation: shrinking" begin

    # build list of elements to sort
    xs = [
        1.2 1.1 0.8;
        0.8 0.9 0.2;
        0.1 0.2 0.9;
        0.1 0.3 0.2;
        0.2 0.25 0.4
    ]
    Random.seed!(123)
    radii = rand(size(xs,1))
    ms = rand(size(xs)[1])
    bodies = vcat(xs',radii',ms',zeros(3,length(ms)))
    elements = Gravitational(bodies)
    buffer = FastMultipole.system_to_buffer(elements)

    # test target box
    center, box = FastMultipole.center_box((buffer,), Float64)
    source_center, source_box = FastMultipole.source_center_box((buffer,), FastMultipole.get_bodies_index((buffer,)), Float64)

    # test source box
    source_radius = FastMultipole.shrink_radius_source(source_center, (buffer,), SVector{1}((1:5,)))
    target_radius = FastMultipole.shrink_radius_target(center, (buffer,), SVector{1}((1:5,)))

    # manually
    x_min, x_max, y_min, y_max, z_min, z_max = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in 1:5
        x_min = min(x_min, elements.bodies[i].position[1] - elements.bodies[i].radius)
        y_min = min(y_min, elements.bodies[i].position[2] - elements.bodies[i].radius)
        z_min = min(z_min, elements.bodies[i].position[3] - elements.bodies[i].radius)
        x_max = max(x_max, elements.bodies[i].position[1] + elements.bodies[i].radius)
        y_max = max(y_max, elements.bodies[i].position[2] + elements.bodies[i].radius)
        z_max = max(z_max, elements.bodies[i].position[3] + elements.bodies[i].radius)
    end

    test_source_center = SVector{3}((x_min+x_max)*0.5, (y_min+y_max)*0.5, (z_min+z_max)*0.5)
    test_source_box = [(x_max-x_min)*0.5, (y_max-y_min)*0.5, (z_max-z_min)*0.5]
    test_source_radius = 0.0
    for i_body in 1:5
        test_source_radius = max(test_source_radius, norm(elements.bodies[i_body].position - test_source_center) + elements.bodies[i_body].radius)
    end

    @test isapprox(test_source_center, source_center; atol=1e-12)
    @test isapprox(test_source_radius, source_radius; atol=1e-12)
    @test isapprox(test_source_box, source_box; atol=1e-12)

    # test branch! function
    expansion_order, leaf_size, multipole_acceptance = 2, SVector{1}(1), 0.5
    switches = DerivativesSwitch(true, true, true, (elements,))
    source_tree = FastMultipole.Tree((elements,), false, switches; expansion_order, leaf_size, shrink=true, recenter=true)
    target_tree = FastMultipole.Tree((elements,), true, switches; expansion_order, leaf_size, shrink=true, recenter=true)

    # test branch 3-5 (leaf branches)
    function test_leaf(source_branches, target_branches, buffer::Matrix, i_leaf)
        source_branch = source_branches[i_leaf]
        target_branch = target_branches[i_leaf]
        i_body = source_branch.bodies_index[1][1]
        test_target_center = FastMultipole.get_position(buffer, i_body)
        test_source_center = FastMultipole.get_position(buffer, i_body)
        test_target_radius = 0.0
        test_source_radius = FastMultipole.get_radius(buffer, i_body)
        test_target_box = SVector{3}(0.0,0.0,0.0)
        test_source_box = SVector{3}(test_source_radius for _ in 1:3)

        @test isapprox(test_target_center, target_branch.center; atol=1e-12)
        @test isapprox(test_source_center, source_branch.center; atol=1e-12)
        @test isapprox(test_target_radius, target_branch.radius; atol=1e-12)
        @test isapprox(test_source_radius, source_branch.radius; atol=1e-12)
        @test isapprox(test_target_box, target_branch.box; atol=1e-12)
        @test isapprox(test_source_box, source_branch.box; atol=1e-12)
    end

    test_leaf(source_tree.branches, target_tree.branches, source_tree.buffers[1], 3)
    test_leaf(source_tree.branches, target_tree.branches, source_tree.buffers[1], 4)
    test_leaf(source_tree.branches, target_tree.branches, source_tree.buffers[1], 5)
    test_leaf(source_tree.branches, target_tree.branches, source_tree.buffers[1], 6)
    test_leaf(source_tree.branches, target_tree.branches, source_tree.buffers[1], 7)

    # test branch 2 (mid level)
    # NOTE: this test works by comparing to body locations
    #       it only works because the child branches of
    #       branch 2 have a target radius of zero
    #       (exactly 1 body each at their centers)
    source_branch = source_tree.branches[2]
    target_branch = target_tree.branches[2]
    bodies_index = source_branch.bodies_index[1]
    centers = [FastMultipole.get_position(source_tree.buffers[1], i) for i in bodies_index]
    test_target_center = sum(centers) / length(centers)

    # source center
    min_x_source = minimum([centers[i][1] - FastMultipole.get_radius(source_tree.buffers[1],i) for i in bodies_index])
    min_y_source = minimum([centers[i][2] - FastMultipole.get_radius(source_tree.buffers[1],i) for i in bodies_index])
    min_z_source = minimum([centers[i][3] - FastMultipole.get_radius(source_tree.buffers[1],i) for i in bodies_index])
    max_x_source = maximum([centers[i][1] + FastMultipole.get_radius(source_tree.buffers[1],i) for i in bodies_index])
    max_y_source = maximum([centers[i][2] + FastMultipole.get_radius(source_tree.buffers[1],i) for i in bodies_index])
    max_z_source = maximum([centers[i][3] + FastMultipole.get_radius(source_tree.buffers[1],i) for i in bodies_index])

    test_source_center = SVector{3}((min_x_source+max_x_source)*0.5, (min_y_source+max_y_source)*0.5, (min_z_source+max_z_source)*0.5)
    test_source_box = SVector{3}((max_x_source-min_x_source)*0.5, (max_y_source-min_y_source)*0.5, (max_z_source-min_z_source)*0.5)

    test_target_radius = maximum([norm(FastMultipole.get_position(source_tree.buffers[1],i) - target_branch.center) for i in bodies_index])
    test_source_radius = maximum([norm(FastMultipole.get_position(source_tree.buffers[1],i) - source_branch.center) + FastMultipole.get_radius(source_tree.buffers[1],i) for i in bodies_index])
    bodies_x = [c[1] for c in centers]
    bodies_y = [c[2] for c in centers]
    bodies_z = [c[3] for c in centers]
    min_x = minimum(bodies_x)
    max_x = maximum(bodies_x)
    min_y = minimum(bodies_y)
    max_y = maximum(bodies_y)
    min_z = minimum(bodies_z)
    max_z = maximum(bodies_z)
    test_target_box = SVector{3}((max_x - min_x)/2, (max_y - min_y)/2, (max_z - min_z)/2)

    @test isapprox(test_target_center, target_branch.center; atol=1e-12)
    @test isapprox(test_source_center, source_branch.center; atol=1e-12)
    @test isapprox(test_target_radius, target_branch.radius; atol=1e-12)
    @test isapprox(test_source_radius, source_branch.radius; atol=1e-12)
    @test isapprox(test_target_box, target_branch.box; atol=1e-12)
    @test isapprox(test_source_box, source_branch.box; atol=1e-12)

end

@testset "octree creation: leaf index" begin

n_bodies = 101
bodies = rand(8,n_bodies)
system = (Gravitational(bodies),)

switches = DerivativesSwitch(true, true, true, (elements,))
tree = fmm.Tree(system, false, switches; leaf_size=SVector{1}(5))
leaf_index = tree.leaf_index

i_leaf = 1
for branch in tree.branches
    if branch.n_branches == 0
        @test i_leaf == branch.i_leaf
        i_leaf += 1
    end
end

end

@testset "octree creation: buffer_to_target!" begin

n_bodies = 5
bodies = rand(8,n_bodies)
system = (Gravitational(bodies),)
switches = DerivativesSwitch(true, true, true, (system,))
tree = fmm.Tree(system, true, switches; leaf_size=SVector{1}(5))

# fill potential with index
for i in 1:FastMultipole.get_n_bodies(system)
    tree.buffers[1][4,i] = i
end

# copy back to system
FastMultipole.buffer_to_target!(system, tree.buffers, (DerivativesSwitch(),), tree.sort_index_list)

# check that potential matches indices
check_system = zeros(4, n_bodies)
check_buffer = zeros(4, n_bodies)
for i in 1:n_bodies
    check_system[1,i] = system[1].potential[1,i]
    check_system[2:4,i] .= system[1].bodies[i].position
    check_buffer[1,i] = tree.buffers[1][4,i]
    check_buffer[2:4,i] .= tree.buffers[1][1:3,i]
end

check_system_sorted = sortslices(check_system, dims=2)
check_buffer_sorted = sortslices(check_buffer, dims=2)

for i in 1:n_bodies
    for j in 1:4
        @test isapprox(check_system_sorted[j,i], check_buffer_sorted[j,i]; atol=1e-12)
    end
end

end
