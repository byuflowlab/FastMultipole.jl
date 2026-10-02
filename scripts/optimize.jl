using FastMultipole, Random

include("../test/helpers/gravitational.jl")

function optimize(n_bodies; seed = 123, radius_factor=0.1, p_bounds=(1,15), theta_bounds=(0.0,0.1,1.0), ncrit_bound=(0,500))
    # generate system
    Random.seed!(seed)
    bodies = rand(8,n_bodies)
    bodies[4,:] ./= (n_bodies^(1/3)*2)
    bodies[4,:] .*= radius_factor
    bodies[5,:] ./= (n_bodies^(1/3)*2) # attempt to preserve similar potential at each body, rather than a constant total strength
    system = Gravitational(bodies)

    # 
end
