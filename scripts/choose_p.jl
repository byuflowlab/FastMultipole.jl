using FastMultipole
using FastMultipole.StaticArrays
using LinearAlgebra
using PythonPlot
using BSON

include("../test/helpers/gravitational.jl")
include("../test/helpers/evaluate_multipole.jl")

function spherical_to_cartesian(ρ,θ,ϕ)
    z = ρ * cos(θ)
    x = ρ * sin(θ) * cos(ϕ)
    y = ρ * sin(θ) * sin(ϕ)
    return x, y, z
end

function build_system(center, ρs, θs, ϕs, qs)
    source_bodies = zeros(8,length(ρs))
    for (i,(ρ,θ,ϕ,q)) in enumerate(zip(ρs, θs, ϕs, qs))
        x, y, z = spherical_to_cartesian(ρ,θ,ϕ)
        source_bodies[1:3,i] .= center + SVector{3}(x,y,z)
        source_bodies[5,i] = q
    end
    system = Gravitational(source_bodies)
    return system
end

function generate_multipole(center, ρs, θs, ϕs, qs, expansion_order)
    # build system
    system = build_system(center, ρs, θs, ϕs, qs)

    # build branch
    branch = Branch(1:length(ρs), 0, 1:0, 0, 1, center, 0.0, 0.0, SVector{6}(0.0,0,0,0,0,0), SVector{3}(0.0,0,0), expansion_order)

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

function generate_local(center, multipole_branch::Branch, expansion_order)
    local_branch = Branch(1:1, 0, 1:0, 0, 1, center, 0.0, 0.0, SVector{6}(0.0,0,0,0,0,0), SVector{3}(0.0,0,0), expansion_order)

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

    return local_branch
end

function evaluate_local(xt, branch::Branch, expansion_order)
    Δx = xt - branch.center
    velocity_n_m = zeros(2,3,size(branch.multipole_expansion,3))
    lamb_helmholtz = Val(false)
    u, velocity, gradient = FastMultipole.evaluate_local(Δx, branch.harmonics, velocity_n_m, branch.local_expansion, Val(expansion_order), lamb_helmholtz, DerivativesSwitch())
    return u
end

function upper_bound_error(center_separation, am, al, expansion_order)
    c̃ = (center_separation - am) / al
    ε = 1/(c̃ - 1) * (1/c̃)^expansion_order
    return ε
end

function old_upper_bound_error(center_separation, am, al, expansion_order)
    c = center_separation / am - 1.0
    ε = 1/(c - 1) * (1/c)^expansion_order
    return ε*2
end

function porter_error(center_separation, am, al, expansion_order)
    d = center_separation
    multi_err = (d - am) / (d-al-am) * (am / (d-al))^(expansion_order + 1)
    local_err = (d - am) / (d-am-al) * (al / (d-am))^(expansion_order + 1)
    return multi_err, local_err
end

#--- RUN TESTS ---#

function solve_p_new(ε, ρ, am, al)
    target = ε * (ρ - am - al)
    f1 = al / (ρ-am)
    f2 = am / (ρ-al)
    t1 = al
    t2 = am
    for p in 0:50
        (t1 + t2 < target) && (return p)
        t1 *= f1
        t2 *= f2
    end
    return 51
end

function solve_p_new_multipole(ε, ρ, am, al)
    target = ε * (ρ-am-al)
    f1 = am / (ρ-al)
    t1 = 3 * am
    for p in 0:50
        this_target = target * (2*p+8)# * sqrt(4*p+6)
        t1 < this_target && (return p)
        t1 *= f1
    end
    return 51
end

function solve_p_old(ε, ρ, am)
    c = ρ/am - 1.0
    p_old = Int(ceil(-log(c,ε)))
    return p_old
end

function solve_p_old_multipole(ε, ρ, am, al)
    target = ε * (ρ - am - al)
    f2 = am / (ρ-al)
    t2 = am
    for p in 0:50
        t2 < target && (return p)
        t2 *= f2
    end
    return 51
end

function choose_p(epsilon::Number, dx=0.0, pmax=51; multipole_only=false,
        ρs = [1.0],
        θs = [π/2],
        ϕs = [0.0],
        qs = [1.0],
    )
    # define multipole (source)
    multipole_center = SVector{3}(0.0,0,0)
    multipole, system = generate_multipole(multipole_center, ρs, θs, ϕs, qs, pmax)

    # define target
    xt = SVector{3}(2.0+dx,0.0,0.0)

    # define local expansion
    local_center = SVector{3}(3.0,0,0)
    local_branch = generate_local(local_center, multipole, pmax)

    # determine expansion order
    ρ = norm(local_center - multipole.center)
    al = 1.0 - dx
    am = ρs[1]
    if multipole_only
        p_old = solve_p_old_multipole(epsilon, ρ, am, al)
    else
        p_old = solve_p_old(epsilon, ρ, am)
    end

    # use new error function
    if multipole_only
        p_new = solve_p_new_multipole(epsilon, ρ, am, al)
    else
        p_new = solve_p_new(epsilon, ρ, am, al)
    end

    # potential
    u_multipole = evaluate_multipole(xt, multipole, p_old)
    u_local = evaluate_local(xt, local_branch, p_old)
    u_direct = evaluate_direct(xt, system)

    # new potential
    u_multipole_new = evaluate_multipole(xt, multipole, p_new)
    u_local_new = evaluate_local(xt, local_branch, p_new)

    return abs((u_multipole - u_direct)/u_direct), abs((u_local - u_direct)/u_direct), p_old, abs((u_multipole_new - u_direct)/u_direct), abs((u_local_new - u_direct)/u_direct), p_new
end

function choose_p(εs::Vector, dx=0.0, pmax=51; multipole_only=false,
        ρs = [1.0],
        θs = [π/2],
        ϕs = [0.0],
        qs = [1.0],
    )
    multipole_error = zeros(length(εs))
    local_error = zeros(length(εs))
    new_multipole_error = zeros(length(εs))
    new_local_error = zeros(length(εs))
    old_ps = zeros(Int,length(εs))
    new_ps = zeros(Int,length(εs))
    for (i,ε) in enumerate(εs)
        m, l, p_old, m_new, l_new, p_new = choose_p(ε, dx, pmax; multipole_only, ρs, θs, ϕs, qs)
        multipole_error[i] = m
        local_error[i] = l
        old_ps[i] = p_old
        new_multipole_error[i] = m_new
        new_local_error[i] = l_new
        new_ps[i] = p_new
    end
    return multipole_error, local_error, old_ps, new_multipole_error, new_local_error, new_ps
end

function uniform_sphere(nx, A)
    ρs = Float64[]
    θs = Float64[]
    ϕs = Float64[]
    for x in range(-1, stop=1.0, length= nx)
        for y in range(-1, stop=1.0, length=nx)
            for z in range(-1, stop=1.0, length=nx)
                ρ = x*x+y*y+z*z
                if ρ <= 1.0
                    r, θ, ϕ = FastMultipole.cartesian_to_spherical(SVector{3}(x,y,z))
                    push!(ρs, r)
                    push!(θs, θ)
                    push!(ϕs, ϕ)
                end
            end
        end
    end
    qs = zeros(length(ρs))
    qs .= A / length(qs)
    return ρs, θs, ϕs, qs
end

function uniform_box(nx, A)
    ρs = Float64[]
    θs = Float64[]
    ϕs = Float64[]
    for x in range(-1, stop=1.0, length= nx)
        for y in range(-1, stop=1.0, length=nx)
            for z in range(-1, stop=1.0, length=nx)
                r, θ, ϕ = FastMultipole.cartesian_to_spherical(SVector{3}(x,y,z))
                push!(ρs, r)
                push!(θs, θ)
                push!(ϕs, ϕ)
            end
        end
    end
    qs = zeros(length(ρs))
    qs .= A / length(qs)
    return ρs, θs, ϕs, qs
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

fig = figure("choose p multipole")
fig.clear()
fig.add_subplot(121, xlabel="relative error tolerance", ylabel="multipole error")
fig.add_subplot(122, xlabel="relative error tolerance", ylabel="expansion order")
ax1 = fig.get_axes()[0]
ax0 = fig.get_axes()[1]

distance = -0.25
eps = [10.0^x for x in -12:0.005:0]

nx = 5
A = 1.0
ρs, θs, ϕs, qs = uniform_sphere(nx, A)
#ρs, θs, ϕs, qs = uniform_box(nx, A)
m, l, old_ps, m_new, l_new, new_ps = choose_p(eps, distance; multipole_only=true, ρs, θs, ϕs, qs)

ax0.plot(eps, new_ps)
ax0.plot(eps, old_ps)
ax1.plot(eps, m_new)
ax1.plot(eps, m)
ax1.plot(eps, eps, "red")

ax0.set_xscale("log")
ax0.legend(["uniform source", "worst case"])

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.legend(["uniform source", "worst case", "tolerance"])

tight_layout()

#BSON.@save "choose_p2_d$distance.bson" m l old_ps m_new l_new new_ps
#savefig("choose_p2_d$distance.png")
