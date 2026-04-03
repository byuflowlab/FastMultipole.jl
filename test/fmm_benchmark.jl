using FastMultipole
using FastMultipole.StaticArrays
using FastMultipole.LinearAlgebra
using Statistics
using PythonPlot

include(joinpath(@__DIR__, "gravitational.jl"))

"""
    benchmark_fmm_passes(; kwargs...)

Benchmark each FMM pass (upward, horizontal, downward, nearfield) across
multiple error methods and tolerances.

Returns `(data, fields)` where:
- `data::Array{Float64, 3}`: axes are `(tolerance, error_method, field)`
- `fields::NamedTuple`: maps field names to third-axis indices
"""
function benchmark_fmm_passes(;
    n_bodies=1000,
    seed=123,
    expansion_order=20,
    error_methods=[
        FastMultipole.HeuristicRelativePotential, 
        FastMultipole.PringleAbsolutePotential,
        FastMultipole.PowerAbsolutePotential,
    ],
    tolerances=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    leaf_size=10,
    multipole_acceptance=0.5,
    n_samples=5,
    interaction_list_method=FastMultipole.Barba(),
)
    # generate system and direct reference
    system = generate_gravitational(seed, n_bodies; strength_scale=1.0/742.8504160733446)
    system_direct = deepcopy(system)
    direct!(system_direct; scalar_potential=true, gradient=true, n_threads=1)
    @show mean(system_direct.potential[1,:])

    # output layout
    fields = (
        tolerance             = 1,
        expansion_order_used  = 2,
        t_upward              = 3,
        t_horizontal          = 4,
        t_downward            = 5,
        t_nearfield           = 6,
        t_total               = 7,
        err_min_phi           = 8,
        err_q25_phi           = 9,
        err_q50_phi           = 10,
        err_q75_phi           = 11,
        err_max_phi           = 12,
        err_min_grad          = 13,
        err_q25_grad          = 14,
        err_q50_grad          = 15,
        err_q75_grad          = 16,
        err_max_grad          = 17,
    )
    n_fields = length(fields)
    n_tol = length(tolerances)
    n_methods = length(error_methods)
    data = zeros(Float64, n_tol, n_methods, n_fields)

    lamb_helmholtz = Val(FastMultipole.has_vector_potential(system))
    source_systems = (system,)

    println("Benchmarking FMM passes (n_bodies=$n_bodies, P=$expansion_order, n_samples=$n_samples)...\n")

    for (i_method, ErrorMethodType) in enumerate(error_methods)
        println("  Error method: $(nameof(ErrorMethodType))")

        for (i_tol, tol) in enumerate(tolerances)

            # construct error method instance
            if ErrorMethodType <: FastMultipole.RelativeErrorMethod
                error_method = ErrorMethodType(tol, tol, true)
            else
                error_method = ErrorMethodType(tol, false)
            end

            # --- Phase 1: full FMM for error computation ---
            reset!(system)
            _, cache, target_tree, source_tree, m2l_list, direct_list, derivatives_switches, _ = fmm!(
                system;
                expansion_order, leaf_size, multipole_acceptance,
                error_tolerance=error_method,
                scalar_potential=true, gradient=true,
                silence_warnings=true,
                interaction_list_method,
            )

            # compute errors vs direct reference
            errs_phi = [abs(system.potential[1, i] - system_direct.potential[1, i]) for i in 1:n_bodies]
            errs_grad = [
                norm(SVector{3}(system.potential[5:7, i]) - SVector{3}(system_direct.potential[5:7, i]))
                for i in 1:n_bodies
            ]
            q_phi = quantile(errs_phi, [0.0, 0.25, 0.50, 0.75, 1.0])
            q_grad = quantile(errs_grad, [0.0, 0.25, 0.50, 0.75, 1.0])

            # --- Phase 2: timed per-pass benchmarks ---
            # run n_samples iterations, discard first (warmup), average the rest
            t_up_samples = zeros(n_samples)
            t_hz_samples = zeros(n_samples)
            t_dp_samples = zeros(n_samples)
            t_nf_samples = zeros(n_samples)
            Pmax = 0

            for i_sample in 1:n_samples
                # reset state
                reset!(system)
                FastMultipole.reset_expansions!(target_tree)
                FastMultipole.reset_expansions!(source_tree)
                for buf in target_tree.buffers
                    buf[4:7, :] .= 0
                end

                # upward pass
                t_up_samples[i_sample] = @elapsed FastMultipole.upward_pass_singlethread!(
                    source_tree, source_systems, expansion_order, lamb_helmholtz
                )

                # horizontal pass
                t_hz_samples[i_sample] = @elapsed begin
                    Pmax_i, _ = FastMultipole.horizontal_pass_singlethread!(
                        target_tree, source_tree, m2l_list, lamb_helmholtz,
                        expansion_order, error_method
                    )
                    Pmax = Pmax_i
                end

                # downward pass
                t_dp_samples[i_sample] = @elapsed begin
                    FastMultipole.downward_pass_singlethread_1!(target_tree, expansion_order, lamb_helmholtz)
                    gradient_n_m = FastMultipole.initialize_gradient_n_m(expansion_order, eltype(target_tree.branches[1]))
                    FastMultipole.downward_pass_singlethread_2!(
                        target_tree, target_tree.buffers, expansion_order, lamb_helmholtz,
                        derivatives_switches, gradient_n_m
                    )
                end

                # nearfield
                t_nf_samples[i_sample] = @elapsed FastMultipole.nearfield_singlethread!(
                    target_tree.buffers, target_tree.branches,
                    source_systems, source_tree.buffers, source_tree.branches,
                    derivatives_switches, direct_list
                )
            end

            # discard first sample (warmup), average the rest
            t_upward     = mean(@view t_up_samples[2:end])
            t_horizontal = mean(@view t_hz_samples[2:end])
            t_downward   = mean(@view t_dp_samples[2:end])
            t_nearfield  = mean(@view t_nf_samples[2:end])
            t_total      = t_upward + t_horizontal + t_downward + t_nearfield

            # store results
            data[i_tol, i_method, fields.tolerance]            = tol
            data[i_tol, i_method, fields.expansion_order_used] = Float64(Pmax)
            data[i_tol, i_method, fields.t_upward]             = t_upward
            data[i_tol, i_method, fields.t_horizontal]         = t_horizontal
            data[i_tol, i_method, fields.t_downward]           = t_downward
            data[i_tol, i_method, fields.t_nearfield]          = t_nearfield
            data[i_tol, i_method, fields.t_total]              = t_total
            data[i_tol, i_method, fields.err_min_phi]          = q_phi[1]
            data[i_tol, i_method, fields.err_q25_phi]          = q_phi[2]
            data[i_tol, i_method, fields.err_q50_phi]          = q_phi[3]
            data[i_tol, i_method, fields.err_q75_phi]          = q_phi[4]
            data[i_tol, i_method, fields.err_max_phi]          = q_phi[5]
            data[i_tol, i_method, fields.err_min_grad]         = q_grad[1]
            data[i_tol, i_method, fields.err_q25_grad]         = q_grad[2]
            data[i_tol, i_method, fields.err_q50_grad]         = q_grad[3]
            data[i_tol, i_method, fields.err_q75_grad]         = q_grad[4]
            data[i_tol, i_method, fields.err_max_grad]         = q_grad[5]

            println("    tol=$tol  Pmax=$Pmax  t_total=$(round(t_total; sigdigits=3))s  err_max=$(round(q_phi[5]; sigdigits=3))")
        end
    end

    println("\nDone.")
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
    plot_benchmark_bars(data, fields, error_methods; kwargs...)

Stacked bar chart comparing per-pass FMM timings across error methods.

Each tolerance level is a group of bars (one bar per error method), stacked
by pass: nearfield, upward, horizontal, downward.

# Arguments
- `data::Array{Float64,3}`: output of `benchmark_fmm_passes`
- `fields::NamedTuple`: output of `benchmark_fmm_passes`
- `error_methods`: the same vector of error method types passed to `benchmark_fmm_passes`

# Keyword Arguments
- `figname::String="fmm_pass_benchmark"`: PythonPlot figure name
- `savepath::Union{String,Nothing}=nothing`: if not `nothing`, save figure to this path
"""
function plot_benchmark_bars(data, fields, error_methods;
    figname::String="fmm_pass_benchmark",
    savepath::Union{String,Nothing}=nothing,
)
    fig = PythonPlot.figure(figname; figsize=(max(10, 2*size(data,1)), 5))
    fig.clear()
    ax = fig.add_subplot(111)

    n_tol = size(data, 1)
    n_methods = length(error_methods)
    tolerances = data[:, 1, fields.tolerance]

    pass_fields = [:t_nearfield, :t_upward, :t_horizontal, :t_downward]
    pass_labels = ["nearfield", "upward", "horizontal", "downward"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    bar_width = 0.8 / n_methods
    group_positions = collect(1:n_tol)

    for (i_method, ErrorMethodType) in enumerate(error_methods)
        x_offset = (i_method - (n_methods + 1) / 2) * bar_width
        x_pos = group_positions .+ x_offset
        bottoms = zeros(n_tol)

        for (i_pass, field) in enumerate(pass_fields)
            vals = data[:, i_method, getfield(fields, field)]
            label = i_method == 1 ? pass_labels[i_pass] : nothing
            ax.bar(x_pos, vals; width=bar_width, bottom=bottoms,
                   color=colors[i_pass], label=label, edgecolor="white", linewidth=0.5)
            bottoms .+= vals
        end
    end

    # x-axis: tolerance group labels
    ax.set_xticks(group_positions)
    ax.set_xticklabels(["$(tol)" for tol in tolerances])
    ax.set_xlabel("tolerance")
    ax.set_ylabel("time (s)")

    # method labels along the top
    for (i_method, ErrorMethodType) in enumerate(error_methods)
        x_offset = (i_method - (n_methods + 1) / 2) * bar_width
        x = group_positions[1] + x_offset
        y_top = sum(data[1, i_method, getfield(fields, f)] for f in pass_fields)
        ax.text(x, y_top, string(nameof(ErrorMethodType));
                ha="center", va="bottom", fontsize=7, rotation=45)
    end

    ax.legend(; loc="upper left", fontsize=8)
    ax.set_title("FMM pass timing by error method")
    fig.tight_layout()

    if savepath !== nothing
        fig.savefig(savepath; dpi=250, bbox_inches="tight")
    end

    return fig
end

error_methods=[
        FastMultipole.HeuristicAbsolutePotential, 
        FastMultipole.PringleAbsolutePotential,
        FastMultipole.PowerAbsolutePotential,
    ]

data, fields = benchmark_fmm_passes(;
    n_bodies=40000,
    seed=123,
    expansion_order=16,
    error_methods,
    tolerances=[1e-3, 1e-5, 1e-7, 1e-9],
    leaf_size=50,
    multipole_acceptance=0.5,
    n_samples=5,
)

plot_benchmark_bars(data, fields, error_methods;
    figname="fmm_pass_benchmark",
    savepath=nothing,
)