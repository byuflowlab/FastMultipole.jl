using FastMultipole
using FastMultipole.StaticArrays
using FastMultipole.LinearAlgebra
using BenchmarkTools
using PythonPlot
using Statistics
using Random

include(joinpath(@__DIR__, "gravitational.jl"))

"""
    benchmark_m2l(; n_bodies, expansion_order, dv, error_methods, tolerances, seed)

Benchmark the M2L operator across multiple error methods and tolerances.

Returns `(data, fields)` where:
- `data::Array{Float64, 3}`: axes are `(tolerance, error_method, field)`
- `fields::Dict{Symbol, Int}`: maps field name to third-axis index

Fields (all times in seconds):
  `:tolerance`, `:expansion_order_used`,
  `:benchmark_median`, `:benchmark_mean`, `:benchmark_std`,
  `:error_min_p`, `:error_q25_p`, `:error_q50_p`, `:error_q75_p`, `:error_max_p`,
  `:error_min`, `:error_q25`, `:error_q50`, `:error_q75`, `:error_max`
"""
function benchmark_m2l(;
    n_bodies=50,
    expansion_order=20,
    dv=SVector{3}(2.0, 0.0, 0.0),
    error_methods=[FastMultipole.HeuristicRelativePotential, FastMultipole.PringleAbsolutePotential, FastMultipole.PowerAbsolutePotential],
    tolerances=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    seed=123,
    scale_target_box=1
)
    #--- shared setup ---#

    # create source system in [0,1)^3
    source_system = generate_gravitational(seed, n_bodies; strength_scale=1.0)

    # create target system in dv .+ [0,1)^3
    target_system = generate_gravitational(seed + 1, n_bodies;
        bodies_fun = bodies -> (bodies[1:3, :] .+= dv; bodies[1:3, :] .*= scale_target_box; bodies)
    )

    # separate copy for direct reference
    target_direct = deepcopy(target_system)

    # build trees with leaf_size > n_bodies so each is a single branch (root = leaf)
    lamb_helmholtz = Val(false)
    switches = (DerivativesSwitch(true, true, false, source_system),)
    source_tree = Tree((source_system,), false, switches;
        expansion_order, leaf_size=SVector{1,Int}(n_bodies + 1),
        interaction_list_method=Barba()
    )
    target_tree = Tree((target_system,), true, switches;
        expansion_order, leaf_size=SVector{1,Int}(n_bodies + 1),
        interaction_list_method=Barba()
    )

    @assert length(target_tree.leaf_index) == 1 "target tree should have exactly 1 leaf, got $(length(target_tree.leaf_index))"
    @assert length(source_tree.leaf_index) == 1 "source tree should have exactly 1 leaf, got $(length(source_tree.leaf_index))"

    # upward pass: B2M (M2M is a no-op for single branch)
    FastMultipole.upward_pass_singlethread!(source_tree, (source_system,), expansion_order, lamb_helmholtz)

    # direct reference
    direct!(target_direct, source_system; scalar_potential=true, gradient=true, n_threads=1)
    @show mean(target_direct.potential[1,:])

    # pre-allocate M2L temporaries
    TF = Float64
    weights_tmp_1 = initialize_expansion(expansion_order, TF)
    weights_tmp_2 = initialize_expansion(expansion_order, TF)
    Ts = zeros(TF, FastMultipole.length_Ts(expansion_order))
    eimϕs = zeros(TF, 2, expansion_order + 1)
    harmonics = initialize_harmonics(expansion_order, TF)
    gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order)

    # extract branches and expansions
    i_leaf = target_tree.leaf_index[1]
    source_branch = source_tree.branches[source_tree.leaf_index[1]]
    target_branch = target_tree.branches[i_leaf]
    source_expansion = view(source_tree.expansions, :, :, :, source_tree.leaf_index[1])
    target_expansion = view(target_tree.expansions, :, :, :, i_leaf)
    target_buffer = target_tree.buffers[1]
    bodies_index = target_branch.bodies_index[1]
    derivatives_switch = switches[1]

    # copy for benchmark (so benchmark doesn't clobber the real expansion)
    target_exp_bench = similar(target_expansion)

    #--- output layout ---#

    fields = Dict{Symbol, Int}(
        :tolerance => 1, :expansion_order_used => 2,
        :benchmark_median => 3, :benchmark_mean => 4, :benchmark_std => 5,
        :error_min_p => 6, :error_q25_p => 7, :error_q50_p => 8, :error_q75_p => 9, :error_max_p => 10,
        :error_min => 11, :error_q25 => 12, :error_q50 => 13, :error_q75 => 14, :error_max => 15,
    )
    n_tol = length(tolerances)
    n_methods = length(error_methods)
    data = zeros(Float64, n_tol, n_methods, 15)

    #--- loop over error methods and tolerances ---#

    println("Benchmarking M2L...\n")
    for (i_method, ErrorMethodType) in enumerate(error_methods)

        println("\tBenchmarking M2L with error method: $(nameof(ErrorMethodType))")
        for (i_tol, tol) in enumerate(tolerances)

            # check if using a relative error method and compensate if so
            if ErrorMethodType <: FastMultipole.RelativeErrorMethod
                tol = tol * maximum(abs.(target_direct.potential[1, :]))
                error_method = ErrorMethodType(tol, tol, true)
            else
                error_method = ErrorMethodType(tol, false)
            end

            # reset target expansion and buffer potential/gradient rows
            target_tree.expansions .= 0
            target_buffer[4:7, :] .= 0
            reset!(target_system)

            # M2L
            P_used, _ = FastMultipole.multipole_to_local!(
                target_expansion, target_branch, source_expansion, source_branch,
                weights_tmp_1, weights_tmp_2, Ts, eimϕs,
                FastMultipole.ζs_mag, FastMultipole.ηs_mag, FastMultipole.Hs_π2,
                FastMultipole.M̃, FastMultipole.L̃,
                expansion_order, lamb_helmholtz, error_method
            )

            # L2B: evaluate local expansion at target body positions
            FastMultipole.evaluate_local!(
                target_buffer, bodies_index, harmonics, gradient_n_m,
                target_expansion, target_branch.center,
                expansion_order, lamb_helmholtz, derivatives_switch
            )

            # transfer results back to target system
            FastMultipole.buffer_to_target!((target_system,), target_tree, switches)

            # compute absolute gradient error per body
            errs_g = [
                norm(SVector{3}(target_system.potential[5:7, i]) - SVector{3}(target_direct.potential[5:7, i]))
                for i in 1:n_bodies
            ]
            q = quantile(errs_g, [0.0, 0.25, 0.50, 0.75, 1.0])

            # absolute potential error per body
            errs_p = [
                abs(target_system.potential[1, i] - target_direct.potential[1, i])
                for i in 1:n_bodies
            ]
            q_p = quantile(errs_p, [0.0, 0.25, 0.50, 0.75, 1.0])

            # benchmark M2L only
            bench = @benchmark FastMultipole.multipole_to_local!(
                $target_exp_bench, $target_branch, $source_expansion, $source_branch,
                $weights_tmp_1, $weights_tmp_2, $$Ts, $eimϕs,
                FastMultipole.ζs_mag, FastMultipole.ηs_mag, FastMultipole.Hs_π2,
                FastMultipole.M̃, FastMultipole.L̃,
                $expansion_order, $lamb_helmholtz, $error_method
            ) setup=($target_exp_bench .= 0)

            data[i_tol, i_method, 1]  = tol
            data[i_tol, i_method, 2]  = Float64(P_used)
            data[i_tol, i_method, 3]  = median(bench.times) * 1e-9
            data[i_tol, i_method, 4]  = mean(bench.times) * 1e-9
            data[i_tol, i_method, 5]  = std(bench.times) * 1e-9
            data[i_tol, i_method, 6]  = q_p[1]
            data[i_tol, i_method, 7]  = q_p[2]
            data[i_tol, i_method, 8]  = q_p[3]
            data[i_tol, i_method, 9]  = q_p[4]
            data[i_tol, i_method, 10] = q_p[5]
            data[i_tol, i_method, 11] = q[1]
            data[i_tol, i_method, 12] = q[2]
            data[i_tol, i_method, 13] = q[3]
            data[i_tol, i_method, 14] = q[4]
            data[i_tol, i_method, 15] = q[5]
        end
    end
    println("Done benchmarking M2L.")

    return data, fields
end

"""
    plot_m2l(data, fields, quantity::Symbol, error_methods; kwargs...)

Plot `quantity` (a key from `fields`) vs tolerance for each error method.

# Arguments
- `data::Array{Float64,3}`: output of `benchmark_m2l`
- `fields::Dict{Symbol,Int}`: output of `benchmark_m2l`
- `quantity::Symbol`: field to plot on the y-axis (e.g. `:benchmark_median`, `:error_q50`)
- `error_methods`: the same vector of error method types passed to `benchmark_m2l`

# Keyword Arguments
- `logx::Bool=true`: use log scale on x-axis
- `logy::Bool=true`: use log scale on y-axis
- `figname::String="m2l_benchmark"`: PythonPlot figure name
- `ylabel::String=string(quantity)`: y-axis label
- `savepath::Union{String,Nothing}=nothing`: if not `nothing`, save figure to this path
"""
function plot_m2l(data, fields, quantity::Symbol, error_methods;
    logx::Bool=true,
    logy::Bool=true,
    figname::String="m2l_benchmark",
    ylabel::String=string(quantity),
    savepath::Union{String,Nothing}=nothing,
)
    @assert haskey(fields, quantity) "Unknown quantity :$quantity. Valid keys: $(sort(collect(keys(fields))))"

    fig = PythonPlot.figure(figname)
    fig.clear()
    ax = fig.add_subplot(111)

    i_field = fields[quantity]
    tolerances = data[:, 1, fields[:tolerance]]

    for (i_method, ErrorMethodType) in enumerate(error_methods)
        label = string(nameof(ErrorMethodType))
        ax.plot(tolerances, data[:, i_method, i_field]; label, marker="o", markersize=4)
    end

    if logx
        ax.set_xscale("log")
    end
    if logy
        ax.set_yscale("log")
    end
    ax.set_xlabel("tolerance")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.invert_xaxis()

    if savepath !== nothing
        fig.savefig(savepath; dpi=250, bbox_inches="tight")
    end

    return fig
end

"""
    plot_m2l_xy(data, fields, x::Symbol, y::Symbol, error_methods; kwargs...)

Plot field `x` vs field `y` for each error method.

# Arguments
- `data::Array{Float64,3}`: output of `benchmark_m2l`
- `fields::Dict{Symbol,Int}`: output of `benchmark_m2l`
- `x::Symbol`: field for the x-axis
- `y::Symbol`: field for the y-axis
- `error_methods`: the same vector of error method types passed to `benchmark_m2l`

# Keyword Arguments
- `logx::Bool=true`: use log scale on x-axis
- `logy::Bool=true`: use log scale on y-axis
- `figname::String="m2l_xy"`: PythonPlot figure name
- `xlabel::String=string(x)`: x-axis label
- `ylabel::String=string(y)`: y-axis label
- `savepath::Union{String,Nothing}=nothing`: if not `nothing`, save figure to this path
"""
function plot_m2l_xy(data, fields, x::Symbol, y::Symbol, error_methods;
    logx::Bool=true,
    logy::Bool=true,
    figname::String="m2l_xy",
    xlabel::String=string(x),
    ylabel::String=string(y),
    savepath::Union{String,Nothing}=nothing,
)
    @assert haskey(fields, x) "Unknown x field :$x. Valid keys: $(sort(collect(keys(fields))))"
    @assert haskey(fields, y) "Unknown y field :$y. Valid keys: $(sort(collect(keys(fields))))"

    fig = PythonPlot.figure(figname)
    fig.clear()
    ax = fig.add_subplot(111)

    i_x = fields[x]
    i_y = fields[y]

    for (i_method, ErrorMethodType) in enumerate(error_methods)
        label = string(nameof(ErrorMethodType))
        ax.plot(data[:, i_method, i_x], data[:, i_method, i_y]; label, marker="o", markersize=4)
    end

    if logx
        ax.set_xscale("log")
    end
    if logy
        ax.set_yscale("log")
    end
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if savepath !== nothing
        fig.savefig(savepath; dpi=250, bbox_inches="tight")
    end

    return fig
end

# settings
error_methods = [FastMultipole.HeuristicRelativePotential, FastMultipole.PringleAbsolutePotential, FastMultipole.PowerAbsolutePotential]
# error_methods = [FastMultipole.PowerAbsolutePotential]
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

# run benchmarks
# FastMultipole.DEBUG[] = true
data, fields = benchmark_m2l(; error_methods, tolerances, 
                                scale_target_box=1,
                                n_bodies=50,
                                expansion_order=30,
                                dv=SVector{3}(0.0, 0.0, 0.0),
                            )
# FastMultipole.DEBUG[] = false

# plot results
plot_symbols = [:benchmark_median, :error_max_p]

# plot_m2l(data, fields, quantity::Symbol, error_methods;
#     logx::Bool=true,
#     logy::Bool=true,
#     figname::String="m2l_benchmark",
#     ylabel::String=string(quantity),
#     savepath::Union{String,Nothing}=nothing,
# )

for (i, sym) in enumerate(plot_symbols)
    plot_m2l(data, fields, sym, error_methods;
        logx=true,
        logy=true,
        figname="m2l_benchmark_$(sym)",
        ylabel=string(sym),
        savepath="m2l_benchmark_$(sym).png"
    )
end

# plot_m2l_xy(data, fields, :error_max_p, :benchmark_median, error_methods;
#     logx=true,
#     logy=true,
#     figname="m2l_xy",
#     savepath=nothing,
# )