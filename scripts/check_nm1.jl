using FastMultipole
using Random
using Statistics
using PythonPlot
using LaTeXStrings

include("../test/gravitational.jl")
include("../test/vortex.jl")

n_bodies = 30_000
ε_abs = 1e-5
system = generate_gravitational(123, n_bodies; strength_scale=1/n_bodies/0.07891333941819026)
# system = generate_vortex(123, n_bodies; strength_scale=1/n_bodies/0.21254115544924337)
lamb_helmholtz = typeof(system) <: Gravitational ? false : true
direct!(system)
if typeof(system) <: Gravitational
    v_true = system.potential[5:7,:]
else
    v_true = system.velocity_stretching[1:3,:]
end
v_true_mags = sqrt.(sum(v_true .* v_true; dims=1))
@show mean(v_true_mags)

reset!(system)
# @time tuned_params, cache = FastMultipole.tune_fmm!(system; ε_abs, lamb_helmholtz, verbose=true, )#multipole_acceptances=0.4:0.1:0.5)

FastMultipole.DEBUG[] = true
resize!(FastMultipole.DEBUG_DISTRIBUTION, 0)
@time fmm!(system; ε_abs, lamb_helmholtz, tuned_params..., cache)
FastMultipole.DEBUG[] = false

v_fmm = similar(v_true)
if typeof(system) <: Gravitational
    for i in 1:get_n_bodies(system)
        v_fmm[:,i] .= system.potential[5:7,i]
    end
else
    for i in 1:get_n_bodies(system)
        v_fmm[:,i] .= system.velocity_stretching[1:3,i]
    end
end

diff = v_fmm - v_true
diff2 = diff .* diff
diffmags = sqrt.(sum(diff2, dims=1))
@show mean(diffmags)
@show std(diffmags)
@show minimum(diffmags)
@show maximum(diffmags)
println()

@show tuned_params
@show mean(FastMultipole.DEBUG_DISTRIBUTION)
@show std(FastMultipole.DEBUG_DISTRIBUTION)
@show minimum(FastMultipole.DEBUG_DISTRIBUTION)
@show maximum(FastMultipole.DEBUG_DISTRIBUTION)

dist = deepcopy(FastMultipole.DEBUG_DISTRIBUTION)

d_high = dist .> 1000
d_low = dist .< 1.0e-1

dist_clean = deepcopy(dist) # dist[.!(d_high .|| d_low)]

fig = figure()
fig.add_subplot(111)
ax = fig.get_axes()[0]
ax.hist(dist_clean, bins=1:10)

# xlabel(L"\varepsilon_l/\varepsilon_m")
xlabel("number of times in local prediction loop")
ylabel("density")

println("done")

