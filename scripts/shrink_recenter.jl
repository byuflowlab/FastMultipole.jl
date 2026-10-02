using FastMultipole
using Random

include("../test/helpers/gravitational.jl")

function generate_gravitational(seed, n_bodies; radius_factor=1.0)
    Random.seed!(123)
    bodies = rand(8,n_bodies)
    # bodies[1:3,3] .=  0.811770914672987, 0.15526131946379113, 0.30656077208169424
    # bodies[1:3,3] .=   0.7427186184997012, 0.2351893322824516, 0.3380666354208596
    bodies[4,:] ./= (n_bodies^(1/3)*2)
    bodies[4,:] .*= radius_factor
    system = Gravitational(bodies)
end

system = generate_gravitational(123, 10; radius_factor=1.0)
tree = Tree(system; leaf_size=2, shrink=true, recenter=true)

visualize("test_shrink_recenter", system, tree; toggle_bodies=true)

# it seems to work well 20241014

