using FastMultipole
using Random
using Statistics
using PythonPlot
using LaTeXStrings
using DelimitedFiles

include("../test/gravitational.jl")
include("../test/vortex.jl")

function get_gradient(system::Gravitational)
    return system.potential[5:7,:]
end

function get_gradient(system::VortexParticles)
    return system.velocity_stretching[1:3,:]
end

function check_error(system, v_true, ε_abs, lamb_helmholtz, bonus_expansion)
    reset!(system)

    # get optimal tuning parameters
    optargs, cache = tune_fmm(system; ε_abs, lamb_helmholtz, bonus_expansion)

    # benchmark
    t_fmm = @elapsed fmm!(system; optargs..., cache, ε_abs, lamb_helmholtz, bonus_expansion)

    # evaluate error
    v_fmm = get_gradient(system)
    diff = v_true - v_fmm
    diff .*= diff
    diff = sum(diff; dims=1)
    diff .= sqrt.(diff)

    # average benchmarks
    t_fmm2 = @elapsed fmm!(system; optargs..., cache, ε_abs, lamb_helmholtz, bonus_expansion)
    t_fmm3 = @elapsed fmm!(system; optargs..., cache, ε_abs, lamb_helmholtz, bonus_expansion)
    t_fmm4 = @elapsed fmm!(system; optargs..., cache, ε_abs, lamb_helmholtz, bonus_expansion)
    t_fmm5 = @elapsed fmm!(system; optargs..., cache, ε_abs, lamb_helmholtz, bonus_expansion)
    t_fmm6 = @elapsed fmm!(system; optargs..., cache, ε_abs, lamb_helmholtz, bonus_expansion)
    t_fmm7 = @elapsed fmm!(system; optargs..., cache, ε_abs, lamb_helmholtz, bonus_expansion)
    t_fmm = (t_fmm + t_fmm2 + t_fmm3 + t_fmm4 + t_fmm5 + t_fmm6 + t_fmm7 - max(t_fmm, t_fmm2, t_fmm3, t_fmm4, t_fmm5, t_fmm6, t_fmm7)) / 6

    # evaluate quartiles
    p0, p25, p50, p75, p100 = quantile(diff, (0.0, 0.25, 0.5, 0.75, 1.0))

    return t_fmm, p0, p25, p50, p75, p100
end

function check_error(system, εs, bonus_expansion)
    # preliminary calcs
    reset!(system)
    direct!(system; hessian=false, velocity=true)
    v_true = get_gradient(system)
    @show mean(sqrt.(sum(v_true .* v_true; dims=1)))

    # evaluate error
    res = zeros(length(εs), 5)
    ts = zeros(length(εs))
    for (i,ε) in enumerate(εs)
        println("\n--- ε=$ε ---\n")
        t, p0, p25, p50, p75, p100 = check_error(system, v_true, ε, typeof(system) <: Vortex, bonus_expansion)
        res[i,1] = p0
        res[i,2] = p25
        res[i,3] = p50
        res[i,4] = p75
        res[i,5] = p100
        ts[i] = t
        println("\n------------")
    end

    return res, ts
end

function write_csv(name, out)
    q, t = out
    writedlm(name*"_quartiles.csv", q, ',')
    writedlm(name*"_time.csv", t, ',')
end

#--- create systems ---#

n_bodies = 30_000
grav = generate_gravitational(123, n_bodies; strength_scale=1/n_bodies/0.07891333941819026)
vort = generate_vortex(123, n_bodies; strength_scale=1/n_bodies/0.21254115544924337)

#--- run tests ---#

εs = 10.0 .^collect(-9:0.5:-0.5)
bonus_expansion = true

println("\n===== Gravitational, bonus=true =====\n")
q_grav_n = check_error(grav, εs, bonus_expansion)
write_csv("grav_n", q_grav_n)

println("\n===== Vortex, bonus=true =====\n")
q_vort_n = check_error(vort, εs, bonus_expansion)
write_csv("vort_n", q_vort_n)

bonus_expansion = false
println("\n===== Gravitational, bonus=false =====\n")
q_grav_nm1 = check_error(grav, εs, bonus_expansion)
write_csv("grav_nm1", q_grav_nm1)

println("\n===== Vortex, bonus=false =====\n")
q_vort_nm1 = check_error(vort, εs, bonus_expansion)
write_csv("vort_nm1", q_vort_nm1)

