using FastMultipole
using FastMultipole.StaticArrays
using LinearAlgebra
using PythonPlot
using BSON
using Test
using Random

include("../test/helpers/gravitational.jl")
include("../test/helpers/evaluate_multipole.jl")

function spherical_to_cartesian(ρ,θ,ϕ)
    z = ρ * cos(θ)
    x = ρ * sin(θ) * cos(ϕ)
    y = ρ * sin(θ) * sin(ϕ)
    return x, y, z
end

function build_system(center, ρs, θs, ϕs, radii, qs)
    source_bodies = zeros(8,length(ρs))
    for (i,(ρ,θ,ϕ,r,q)) in enumerate(zip(ρs, θs, ϕs, radii, qs))
        x, y, z = spherical_to_cartesian(ρ,θ,ϕ)
        source_bodies[1:3,i] .= center + SVector{3}(x,y,z)
        source_bodies[4,i] = r
        source_bodies[5,i] = q
    end
    system = Gravitational(source_bodies)
    return system
end

function generate_multipole(center, ρs, θs, ϕs, radii, qs, expansion_order, shrink)
    # build system
    system = build_system(center, ρs, θs, ϕs, radii, qs)

    # build branch
    branch = Branch(1:length(ρs), 0, 1:0, 0, 1, center, sqrt(3.0), sqrt(3.0), SVector{6}(-1.0,1.0,-1.0,1.0,-1.0,1.0), SVector{3}(1.0,1.0,1.0), expansion_order)

    # shrink branch
    branches = [branch]
    shrink && FastMultipole.shrink_recenter!(branches, [1:1], system)
    branch = branches[1]

    # multipole coefficients
    body_to_multipole!(branch, system, branch.harmonics, Val(expansion_order))

    return branch, system
end

function evaluate_multipole(xt, branch::Branch, expansion_order)
    ϕ_m2b, v_m2b, g_m2b = evaluate_multipole(xt, branch.center, branch.multipole_expansion, DerivativesSwitch(), Val(expansion_order))
    return ϕ_m2b
end

function evaluate_direct(xt, system::Gravitational)
    u = 0.0
    for body in system.bodies
        xs = body.position
        q = body.strength
        u += q / norm(xt-xs)
    end
    return u / 4 / pi
end

function multipole_to_local!(local_branch, multipole_branch, expansion_order)
    # reset expansion
    local_branch.local_expansion .= zero(eltype(local_branch.local_expansion))

    # preallocate containers
    lamb_helmholtz = Val(false)
    Hs_π2 = [1.0]
    FastMultipole.update_Hs_π2!(Hs_π2, Val(expansion_order))
    Ts = zeros(FastMultipole.length_Ts(expansion_order))
    eimϕs = zeros(2, expansion_order+1)
    weights_tmp_1 = initialize_expansion(expansion_order, eltype(Ts))
    weights_tmp_2 = initialize_expansion(expansion_order, eltype(Ts))

    # normalization
    ζs_mag = zeros(FastMultipole.length_ζs(expansion_order))
    FastMultipole.update_ζs_mag!(ζs_mag, 0, expansion_order)
    ηs_mag = zeros(FastMultipole.length_ηs(expansion_order))
    FastMultipole.update_ηs_mag!(ηs_mag, 0, expansion_order)

    # local coefficients
    FastMultipole.multipole_to_local!(local_branch, multipole_branch, weights_tmp_1, weights_tmp_2, Ts, eimϕs, ζs_mag, ηs_mag, Hs_π2, expansion_order, lamb_helmholtz)

    return nothing
end

function evaluate_local(xt, branch::Branch, expansion_order)
    Δx = xt - branch.center
    velocity_n_m = zeros(2,3,size(branch.multipole_expansion,3))
    lamb_helmholtz = Val(false)
    u, velocity, gradient = FastMultipole.evaluate_local(Δx, branch.harmonics, velocity_n_m, branch.local_expansion, Val(expansion_order), lamb_helmholtz, DerivativesSwitch())
    return u
end

#--- RUN TESTS ---#

function solve_p_old(ε, ρ, am, pmax)
    c = ρ/am - 1.0
    p_old = ceil(-log(c,ε))
    if sign(p_old) < 0 || isinf(p_old)
        p_old = maxintfloat()
    end
    return min(Integer(p_old),pmax)
end

function get_body_positions(x_set, y_set, z_set, r_range, q_range)
    ρs = []
    θs = []
    ϕs = []
    radii = []
    qs = []
    Random.seed!(123)
    for x in x_set
        for y in y_set
            for z in z_set
                r = r_range[1] + rand() * (r_range[2]-r_range[1])
                q = q_range[1] + rand() * (q_range[2]-q_range[1])
                ρ, θ, ϕ = FastMultipole.cartesian_to_spherical(x,y,z)
                push!(ρs, ρ)
                push!(θs, θ)
                push!(ϕs, ϕ)
                push!(radii, r)
                push!(qs, q)
            end
        end
    end
    return ρs, θs, ϕs, radii, qs
end

function relative_error(u_true, u_test)
    return (u_test .- u_true) ./ u_true
end

function test_p(epsilon::Number, pmax=50;
        multipole_center = SVector{3}(0.0,0,0),
        local_center = SVector{3}(4.0,0,0),
        source_xs = [1.0],
        source_ys = [0.0],
        source_zs = [0.0],
        source_q_range = [1.0, 1.0],
        source_r_range = [0.0, 0.0],
        target_xs = [-1.0],
        target_ys = [0.0],
        target_zs = [0.0],
        target_q_range = [1.0, 1.0],
        target_r_range = [0.0, 0.0],
        shrink=true,
    )
    # define multipole (sources)
    ρs, θs, ϕs, rs, qs = get_body_positions(source_xs, source_ys, source_zs, source_r_range, source_q_range)
    multipole, multipole_system = generate_multipole(multipole_center, ρs, θs, ϕs, rs, qs, pmax, shrink)

    # define local expansion (targets)
    ρs, θs, ϕs, rs, qs = get_body_positions(target_xs, target_ys, target_zs, target_r_range, target_q_range)
    local_branch, local_system = generate_multipole(local_center, ρs, θs, ϕs, rs, qs, pmax, shrink)

    # translate for local expansion
    multipole_to_local!(local_branch, multipole, pmax)

    # determine expansion order using Pringle method
    ρ = norm(local_branch.center - multipole.center)
    am = multipole.source_radius
    println("\t\tEqual Spheres...")
    p_equal_spheres_check = solve_p_old(epsilon, ρ, am, pmax)

    # FastMultipole old error method (Equal spheres)
    p_equal_spheres = FastMultipole.get_P(local_branch, multipole, ρ, FastMultipole.EqualSpheres(), pmax, epsilon)

    # adapted for unequal spheres
    println("\t\tUnequal Spheres...")
    p_unequal_spheres = FastMultipole.get_P(local_branch, multipole, ρ, FastMultipole.UnequalSpheres(), pmax, epsilon)

    if !(shrink && local_branch.source_radius > multipole.source_radius) # can break if target branch has a larger radius than source branch
        @test p_unequal_spheres <= p_equal_spheres
    end

    # adapted for unequal boxes
    println("\t\tUnequal Boxes...")
    p_unequal_boxes = FastMultipole.get_P(local_branch, multipole, ρ, FastMultipole.UnequalBoxes(), pmax, epsilon)

    @test p_unequal_boxes <= p_unequal_spheres

    #--- evaluate potential ---#

    # equal spheres
    xts = (body.position for body in local_system.bodies)
    u_equal_spheres_multipole = [evaluate_multipole(xt, multipole, p_equal_spheres) for xt in xts]
    u_equal_spheres_local = [evaluate_local(xt, local_branch, p_equal_spheres) for xt in xts]

    # unequal spheres
    u_unequal_spheres_multipole = [evaluate_multipole(xt, multipole, p_unequal_spheres) for xt in xts]
    u_unequal_spheres_local = [evaluate_local(xt, local_branch, p_unequal_spheres) for xt in xts]

    # unequal boxes
    u_unequal_boxes_multipole = [evaluate_multipole(xt, multipole, p_unequal_boxes) for xt in xts]
    u_unequal_boxes_local = [evaluate_local(xt, local_branch, p_unequal_boxes) for xt in xts]

    # direct
    u_direct = [evaluate_direct(xt, multipole_system) for xt in xts]

    # prepare return values
    multipole_error_equal_spheres = relative_error(u_direct, u_equal_spheres_multipole)
    total_error_equal_spheres = relative_error(u_direct, u_equal_spheres_local)

    if !(shrink && local_branch.source_radius > multipole.source_radius) # can break if target branch has a larger radius than source branch
        @test maximum(abs.(total_error_equal_spheres)) <= epsilon
    end

    multipole_error_unequal_spheres = relative_error(u_direct, u_unequal_spheres_multipole)
    total_error_unequal_spheres = relative_error(u_direct, u_unequal_spheres_local)

    @test maximum(abs.(total_error_unequal_spheres)) <= epsilon

    multipole_error_unequal_boxes = relative_error(u_direct, u_unequal_boxes_multipole)
    total_error_unequal_boxes = relative_error(u_direct, u_unequal_boxes_local)

    @test maximum(abs.(total_error_unequal_boxes)) <= epsilon

    return p_equal_spheres, multipole_error_equal_spheres, total_error_equal_spheres, p_unequal_spheres, multipole_error_unequal_spheres, total_error_unequal_spheres, p_unequal_boxes, multipole_error_unequal_boxes, total_error_unequal_boxes
end

function test_p_set(epsilon, shrink; pmax=50,
        multipole_center = SVector{3}(0.0,0,0),
        local_center = SVector{3}(4.0,0,0),
    )

    println("===== BEGIN TEST =====")

    println("\tε = $epsilon")
    println("\tshrink = $shrink")

    println("======================\n")

    # single source, single target: highest error
    println("\tSingle source, single target: highest error")
    source_xs = [1.0]
    source_ys = [0.0]
    source_zs = [0.0]
    source_q_range = [1.0, 1.0]
    source_r_range = [0.0, 0.0]
    target_xs = [-1.0]
    target_ys = [0.0]
    target_zs = [0.0]
    target_q_range = [1.0, 1.0]
    target_r_range = [0.0, 0.0]
    pes, mes, tes, pus, mus, tus, pub, mub, tub = test_p(epsilon, pmax;
        multipole_center, local_center, source_xs,  source_ys,  source_zs,  source_q_range,  source_r_range,  target_xs,  target_ys,  target_zs,  target_q_range,  target_r_range,  shrink)

    # single source, single target: lowest error
    println("\tSingle source, single target: lowest error")
    source_xs = [-1.0]
    source_ys = [0.0]
    source_zs = [0.0]
    source_q_range = [1.0, 1.0]
    source_r_range = [0.0, 0.0]
    target_xs = [1.0]
    target_ys = [0.0]
    target_zs = [0.0]
    target_q_range = [1.0, 1.0]
    target_r_range = [0.0, 0.0]
    pes, mes, tes, pus, mus, tus, pub, mub, tub = test_p(epsilon, pmax;
        multipole_center, local_center, source_xs,  source_ys,  source_zs,  source_q_range,  source_r_range,  target_xs,  target_ys,  target_zs,  target_q_range,  target_r_range,  shrink)

    # single source, many targets: highest error
    println("\tSingle source, many targets: highest error")
    source_xs = [1.0]
    source_ys = [0.0]
    source_zs = [0.0]
    source_q_range = [1.0, 1.0]
    source_r_range = [0.0, 0.0]
    target_xs = range(-1,stop=1,length=10)
    target_ys = range(-1,stop=1.0,length=10)
    target_zs = range(-1,stop=1.0,length=10)
    target_q_range = [1.0, 1.0]
    target_r_range = [0.0, 0.0]
    pes, mes, tes, pus, mus, tus, pub, mub, tub = test_p(epsilon, pmax;
        multipole_center, local_center, source_xs,  source_ys,  source_zs,  source_q_range,  source_r_range,  target_xs,  target_ys,  target_zs,  target_q_range,  target_r_range,  shrink)

    # single source, many targets: lowest error
    println("\tSingle source, many targets: lowest error")
    source_xs = [-1.0]
    source_ys = [0.0]
    source_zs = [0.0]
    source_q_range = [1.0, 1.0]
    source_r_range = [0.0, 0.0]
    target_xs = range(-1,stop=1,length=10)
    target_ys = range(-1,stop=1.0,length=10)
    target_zs = range(-1,stop=1.0,length=10)
    target_q_range = [1.0, 1.0]
    target_r_range = [0.0, 0.0]
    pes, mes, tes, pus, mus, tus, pub, mub, tub = test_p(epsilon, pmax;
        multipole_center, local_center, source_xs,  source_ys,  source_zs,  source_q_range,  source_r_range,  target_xs,  target_ys,  target_zs,  target_q_range,  target_r_range,  shrink)

    # many sources, single target: highest error
    println("\tMany sources, single target: highest error")
    source_xs = range(-1,stop=1.0,length=10)
    source_ys = range(-1,stop=1.0,length=10)
    source_zs = range(-1,stop=1.0,length=10)
    source_q_range = [1.0, 1.0]
    source_r_range = [0.0, 0.0]
    target_xs = [-1.0]
    target_ys = [0.0]
    target_zs = [0.0]
    target_q_range = [1.0, 1.0]
    target_r_range = [0.0, 0.0]
    pes, mes, tes, pus, mus, tus, pub, mub, tub = test_p(epsilon, pmax;
        multipole_center, local_center, source_xs,  source_ys,  source_zs,  source_q_range,  source_r_range,  target_xs,  target_ys,  target_zs,  target_q_range,  target_r_range,  shrink)

    # many sources, single target: lowest error
    println("\tMany sources, single target: lowest error")
    source_xs = range(-1,stop=1.0,length=10)
    source_ys = range(-1,stop=1.0,length=10)
    source_zs = range(-1,stop=1.0,length=10)
    source_q_range = [1.0, 1.0]
    source_r_range = [0.0, 0.0]
    target_xs = [1.0]
    target_ys = [0.0]
    target_zs = [0.0]
    target_q_range = [1.0, 1.0]
    target_r_range = [0.0, 0.0]
    pes, mes, tes, pus, mus, tus, pub, mub, tub = test_p(epsilon, pmax;
        multipole_center, local_center, source_xs,  source_ys,  source_zs,  source_q_range,  source_r_range,  target_xs,  target_ys,  target_zs,  target_q_range,  target_r_range,  shrink)

    # many sources, many targets
    println("\tMany sources, many targets")
    source_xs = range(-1,stop=1.0,length=10)
    source_ys = range(-1,stop=1.0,length=10)
    source_zs = range(-1,stop=1.0,length=10)
    source_q_range = [1.0, 1.0]
    source_r_range = [0.0, 0.0]
    target_xs = range(-1.0, stop=1,length=10)
    target_ys = range(-1,stop=1.0,length=10)
    target_zs = range(-1,stop=1.0,length=10)
    target_q_range = [1.0, 1.0]
    target_r_range = [0.0, 0.0]
    pes, mes, tes, pus, mus, tus, pub, mub, tub = test_p(epsilon, pmax;
        multipole_center, local_center, source_xs,  source_ys,  source_zs,  source_q_range,  source_r_range,  target_xs,  target_ys,  target_zs,  target_q_range,  target_r_range,  shrink)

    println("===== END TEST =====\n")
    return nothing
end

@testset "dynamic expansion order" begin

test_p_set(1e-2, false)
test_p_set(1e-2, true)
test_p_set(1e-4, false)
test_p_set(1e-4, true)
test_p_set(1e-11, true)

end

#=
fig = figure("choose p")
fig.clear()
fig.add_subplot(211, xlabel="relative error tolerance", ylabel="resulting error")
fig.add_subplot(212, xlabel="relative error tolerance", ylabel="expansion order")
ax1 = fig.get_axes()[0]
ax0 = fig.get_axes()[1]

distance = 0.5
eps = [10.0^x for x in -12:0.005:0]
m, l, old_ps, m_new, l_new, new_ps = choose_p(eps, distance)

ax0.plot(eps, old_ps)
ax0.plot(eps, new_ps)
#ax1.plot(eps, m)
ax1.plot(eps, l)
ax1.plot(eps, l_new)
ax1.plot(eps, eps, "red")

ax0.set_xscale("log")
ax0.legend(["Pringle", "novel"])

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend(["Pringle", "novel", "tolerance"])

tight_layout()

BSON.@save "choose_p2_d$distance.bson" m l old_ps m_new l_new new_ps
savefig("choose_p2_d$distance.png")
=#
