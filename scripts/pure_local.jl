using FastMultipole
using Random
using LinearAlgebra
using Statistics
using PythonPlot

include("../test/bodytolocal.jl")
include("../test/gravitational.jl")
include("../test/vortex.jl")

function get_strength(system::Gravitational, i)
    return system.bodies[i].strength
end

function set_gradient!(system::Gravitational, i, v)
    system.potential[5:7,i] .+= v
end

function pure_local!(targets, system, branch, expansion_order)

    # preallocate memory
    expansions = FastMultipole.initialize_expansions(expansion_order, 1, Float64)
    local_expansion = view(expansions, :,:,:,1)
    harmonics = FastMultipole.initialize_harmonics(expansion_order)

    # local expansion
    body_type = typeof(system) <: Gravitational ? Point{Source} : Point{Vortex}
    for i_body in 1:get_n_bodies(system)
        strength = get_strength(system, i_body)
        Δx = FastMultipole.get_position(system, i_body) - branch.target_center
        body_to_local_point!(body_type, local_expansion, harmonics, Δx, strength, expansion_order)
    end

    # evaluate at all targets
    lamb_helmholtz = typeof(system) <: Gravitational ? Val(false) : Val(true)
    velocity_n_m = FastMultipole.initialize_gradient_n_m(expansion_order, Float64)
    for i_target in 1:get_n_bodies(targets)
        Δx = FastMultipole.get_position(targets, i_target) - branch.target_center
        _, v, _ = FastMultipole.evaluate_local(Δx, harmonics, velocity_n_m, local_expansion, expansion_order, lamb_helmholtz, DerivativesSwitch(false, true, false))
        set_gradient!(targets, i_target, v)
    end

    return branch, expansions
end

function sweep_pure_local(targets, x::Real, expansion_order)

    # create branch
    bodies_index = SVector{1}([1:get_n_bodies(targets)])
    n_branches = 0
    branch_index = 1:0
    i_parent = 0
    i_leaf_index = 1
    center, box = FastMultipole.center_box((targets,), Float64)
    center += SVector{3}(-1.0,0.0,0.0)
    radius = norm(box)
    branch = Branch(bodies_index, n_branches, branch_index, i_parent, i_leaf_index, center, center, radius, radius, box, box, expansion_order)

    # create systems
    n_bodies = 100
    ε_abs = 1e-3
    lamb_helmholtz = false
    bodies = zeros(5,1)
    bodies[1:3,1] .= SVector{3}(x,0.5,0.5) #+ SVector{3}(1.0,0,0) * x
    bodies[5,1] = 1.0
    system = Gravitational(bodies)

    # direct
    reset!(targets)
    direct!(targets, system)
    v_true = targets.potential[5:7,:]
    reset!(targets)

    # fmm
    expansions = pure_local!(targets, system, branch, expansion_order)
    v_local = targets.potential[5:7,:]

    diff = (v_true - v_local) ./ v_true
    diff2 = diff .* diff
    diffnorm = sqrt.(sum(diff2, dims=1))

    m = mean(diffnorm)
    s = std(diffnorm)
    mi = minimum(diffnorm)
    ma = maximum(diffnorm)

    return m, s, mi, ma
end

function sweep_pure_local(targets, xs, expansion_order)

    # preallocate results
    n = length(xs)
    ms = zeros(n)
    ss = zeros(n)
    mis = zeros(n)
    mas = zeros(n)

    # run sweep
    for (i,x) in enumerate(xs)
        m, s, mi, ma = sweep_pure_local(targets, x, expansion_order)
        ms[i] = m
        ss[i] = s
        mis[i] = mi
        mas[i] = ma
    end

    return ms, ss, mis, mas
end

targets = generate_gravitational(123, n_bodies; strength_scale=0.0)
xs = range(1.0, stop=5, length=1000)
expansion_order = 25

ms, ss, mis, mas = sweep_pure_local(targets, xs, expansion_order)

fig = figure("sweep")
fig.clear()
fig.add_subplot(111, xlabel="distance", ylabel="relative norm velocity error")
ax = fig.get_axes()[0]
ax.plot(xs, mas, "-.")
ax.plot(xs, ms .+ ss, "--")
ax.plot(xs, ms)
ax.plot(xs, mis, ":")
ax.set_yscale("log")
ax.legend(["max", "1 std", "mean", "min"])
savefig("local_convergence.png")

println("done")
