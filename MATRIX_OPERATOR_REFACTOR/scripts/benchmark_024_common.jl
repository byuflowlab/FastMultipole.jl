# Shared, dependency-free utilities for task 024's host and CUDA drivers.
#
# A "case" is one (distribution, precision, P, LH, N, ell, BLAS/device)
# combination.  The run scripts launch every case in a fresh Julia process so a
# failed dense materialization cannot contaminate the other measurements.

using FastMultipole
using FastMultipole.StaticArrays
using Dates
using LinearAlgebra
using Random
using Statistics

const FM024_REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(FM024_REPO, "test", "gravitational.jl"))

const FM024_SCHEMA_VERSION = "024.1"
const FM024_WARMUPS = parse(Int, get(ENV, "FM024_WARMUPS", "2"))
const FM024_SAMPLES = parse(Int, get(ENV, "FM024_SAMPLES", "7"))
const FM024_MAX_BYTES = parse(Int, get(ENV, "FM024_MAX_PERSISTENT_BYTES",
    string(12 << 30)))

_envbool(name, default=false) =
    lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")
_readcmd(cmd, fallback="unknown") = try strip(read(cmd, String)) catch; fallback end
_env_or(name, fallback::Function) = haskey(ENV, name) ? ENV[name] : fallback()

const FM024_STRATEGIES = (
    (label="concat", operator=MaterializedYRotationM2L(),
        strategy=ConcatenatedFixedZM2L()),
    (label="factored", operator=FactoredRotationM2L(),
        strategy=ConcatenatedFixedZM2L()),
    (label="precomputed_y", operator=FactoredRotationM2L(),
        strategy=PrecomputedFactoredYM2L()),
    (label="dense", operator=MaterializedYRotationM2L(),
        strategy=DenseTranslationM2L(max_persistent_bytes=FM024_MAX_BYTES)),
)

function fm024_system(seed::Integer, n::Integer, distribution::AbstractString)
    sys = generate_gravitational(seed, n)
    distribution == "uniform" && return sys
    distribution == "clustered" ||
        throw(ArgumentError("FM024_DISTRIBUTION must be uniform or clustered"))
    rng = MersenneTwister(seed)
    centers = (
        SVector(0.16, 0.18, 0.22), SVector(0.78, 0.20, 0.69),
        SVector(0.23, 0.77, 0.73), SVector(0.72, 0.75, 0.25),
        SVector(0.51, 0.49, 0.51),
    )
    for i in eachindex(sys.bodies)
        body = sys.bodies[i]
        center = centers[1 + mod(i - 1, length(centers))]
        # A log-spread deterministic mixture produces both heavy and sparse
        # displacement classes while remaining inside the fixed [0,1]^3 box.
        sigma = iszero(mod(i, 11)) ? 0.075 : 0.018
        pos = clamp.(center .+ sigma .* SVector{3}(randn(rng, 3)), 0.002, 0.998)
        sys.bodies[i] = Body(pos, body.radius, body.strength)
    end
    return sys
end

fm024_copy_system(sys) =
    Gravitational(copy(sys.bodies), zeros(eltype(sys.potential), size(sys.potential)))

function fm024_jitter!(sys, seed, step)
    rng = MersenneTwister(seed + 1009 * step)
    for i in eachindex(sys.bodies)
        body = sys.bodies[i]
        δ = 2e-4 .* (rand(rng, SVector{3,Float64}) .- 0.5)
        sys.bodies[i] = Body(clamp.(body.position .+ δ, 0.001, 0.999),
            body.radius, body.strength)
    end
    return sys
end

function fm024_stats_ms(samples)
    values = Float64.(samples)
    isempty(values) && return (median=NaN, minimum=NaN, iqr=NaN)
    return (median=median(values), minimum=minimum(values),
        iqr=quantile(values, 0.75) - quantile(values, 0.25))
end

function fm024_time(f; sync=()->nothing)
    f()
    sync()
    return nothing
end

function fm024_samples(f; sync=()->nothing)
    for _ in 1:FM024_WARMUPS
        fm024_time(f; sync)
    end
    samples = Float64[]
    for _ in 1:FM024_SAMPLES
        t0 = time_ns()
        f()
        sync()
        push!(samples, (time_ns() - t0) / 1e6)
    end
    return fm024_stats_ms(samples)
end

_payload_bytes(x) = try sizeof(x) catch; Base.summarysize(x) end

function fm024_class_counts(plan)
    raw = if plan isa FastMultipole.ResidentM2LDensePlan
        plan.class_counts
    elseif plan isa FastMultipole.ResidentM2LDenseCUDAPlan
        plan.host_class_counts
    elseif plan isa FastMultipole.ResidentM2LPrecomputedYPlan
        plan.host_class_counts === nothing ? plan.offset_counts : plan.host_class_counts
    elseif plan isa FastMultipole.ResidentM2LFactoredPlan
        plan.host_class_counts === nothing ?
            [g.count[] for g in plan.groups] : plan.host_class_counts
    else
        Int[]
    end
    return Int[x for x in raw if x > 0]
end

function fm024_occupancy(counts)
    isempty(counts) && return (nonempty_classes=0, mean_occupancy=0.0,
        max_occupancy=0, p50_occupancy=0.0, p90_occupancy=0.0,
        p95_occupancy=0.0, p99_occupancy=0.0, occupancy_skew=0.0)
    xs = Float64.(counts)
    μ = mean(xs)
    σ = std(xs; corrected=false)
    skew = σ == 0 ? 0.0 : mean(((xs .- μ) ./ σ) .^ 3)
    return (nonempty_classes=length(xs), mean_occupancy=μ,
        max_occupancy=maximum(counts), p50_occupancy=quantile(xs, 0.50),
        p90_occupancy=quantile(xs, 0.90), p95_occupancy=quantile(xs, 0.95),
        p99_occupancy=quantile(xs, 0.99), occupancy_skew=skew)
end

function fm024_memory(state; device=false, used0=0, used_now=0)
    plan = state.scratch.m2l_concat
    operator_bytes = hasproperty(plan, :operator_bytes) ? plan.operator_bytes : 0
    metadata_bytes = hasproperty(plan, :route_metadata_bytes) ?
        plan.route_metadata_bytes :
        sum(_payload_bytes, (state.route_levels, state.route_offsets,
            state.route_targets, state.route_sources))
    scratch_bytes = hasproperty(plan, :scratch_bytes) ?
        plan.scratch_bytes : Base.summarysize(state.scratch)
    persistent_bytes = hasproperty(plan, :persistent_bytes) ?
        plan.persistent_bytes : Base.summarysize(state.scratch)
    expansion_bytes = sum(_payload_bytes, (state.multipoles.phi,
        state.multipoles.chi, state.locals.phi, state.locals.chi))
    peak_bytes = device ? max(used_now - used0, 0) :
        (try Int(Sys.maxrss()) catch; 0 end)
    return (; operator_bytes, metadata_bytes, expansion_bytes, scratch_bytes,
        persistent_bytes, peak_bytes)
end

function fm024_direct_errors(sys, ref, lh)
    potential_error = lh ? NaN :
        maximum(abs.(Float64.(sys.potential[1, :]) .- ref.potential[1, :]))
    gradient_error =
        maximum(abs.(Float64.(sys.potential[5:7, :]) .- ref.potential[5:7, :]))
    return potential_error, gradient_error
end

function fm024_direct_pass(platform, TF, lh, potential_error, gradient_error)
    ptol, gtol = TF === Float32 ? (1e-3, 1e-1) :
        platform == "cuda" ? (1e-4, 1e-2) : (1e-6, 1e-4)
    return (lh || potential_error < ptol) && gradient_error < gtol
end

function fm024_coeff_pass(TF, phi, phi_ref, chi, chi_ref)
    rtol, atol = TF === Float32 ? (5e-3, 5e-4) : (1e-9, 1e-10)
    phi_ok = isapprox(phi, phi_ref; rtol, atol)
    chi_ok = isempty(chi_ref) || isapprox(chi, chi_ref; rtol, atol)
    maxerr = max(maximum(abs.(Float64.(phi) .- Float64.(phi_ref)); init=0.0),
        maximum(abs.(Float64.(chi) .- Float64.(chi_ref)); init=0.0))
    return phi_ok && chi_ok, maxerr
end

# Reconstructed per-column Ts(theta) oracle for fixed-box RadixFMMCache states.
# Production cache states intentionally omit the legacy materialized interaction
# list, so this uses their current valid route prefix directly.  Chunking bounds
# oracle memory without changing coverage or route order.
function fm024_reconstructed_ts_oracle!(state::FastMultipole.DeviceResidentRadixState{
        TF,B,LH}; chunk=4096) where {TF,B,LH}
    cache = state.invariant_cache
    op = MaterializedYRotationM2L()
    width = max(min(chunk, max(state.counts.n_routes, 1)), 1)
    scratch = M2LOperatorScratch(TF, cache.basis_info, width)
    fill!(state.locals.phi, zero(TF))
    LH && fill!(state.locals.chi, zero(TF))
    for c0 in 1:width:state.counts.n_routes
        cols = c0:min(c0 + width - 1, state.counts.n_routes)
        nbatch = length(cols)
        targets = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
        sources = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
        phis = Vector{TF}(undef, nbatch)
        thetas = Vector{TF}(undef, nbatch)
        rs = Vector{TF}(undef, nbatch)
        @inbounds for (j, route_i) in enumerate(cols)
            target = state.route_targets[route_i]
            source = state.route_sources[route_i]
            sources.phi[:, j] .= state.multipoles.phi[:, source]
            LH && (sources.chi[:, j] .= state.multipoles.chi[:, source])
            dx = state.grid.node_centers[1, target] - state.grid.node_centers[1, source]
            dy = state.grid.node_centers[2, target] - state.grid.node_centers[2, source]
            dz = state.grid.node_centers[3, target] - state.grid.node_centers[3, source]
            r, theta, phi =
                FastMultipole.cartesian_to_spherical(SVector{3,TF}(dx, dy, dz))
            rs[j] = TF(r)
            thetas[j] = TF(theta)
            phis[j] = TF(phi)
        end
        FastMultipole.m2l_operator_batch!(op, targets, sources, phis, thetas, rs,
            cache, scratch, Val(LH))
        @inbounds for (j, route_i) in enumerate(cols)
            target = state.route_targets[route_i]
            state.locals.phi[:, target] .+= targets.phi[:, j]
            LH && (state.locals.chi[:, target] .+= targets.chi[:, j])
        end
    end
    return state
end

function fm024_metadata(platform)
    return (schema_version=FM024_SCHEMA_VERSION, campaign="resident_m2l_024",
        platform, host=gethostname(),
        gpu=get(ENV, "FM024_GPU_NAME", "none"),
        kernel=_readcmd(`uname -sr`), cpu_model=get(ENV, "FM024_CPU_MODEL", Sys.CPU_NAME),
        julia_version=string(VERSION), package_project=Base.active_project(),
        blas_vendor=string(BLAS.vendor()), blas_threads=BLAS.get_num_threads(),
        julia_threads=Threads.nthreads(),
        git_commit=_env_or("FM024_GIT_COMMIT", () -> _readcmd(`git rev-parse HEAD`)),
        git_tree=_env_or("FM024_GIT_TREE",
            () -> _readcmd(`git rev-parse "HEAD^{tree}"`)),
        git_worktree=get(ENV, "FM024_GIT_WORKTREE", "unknown"),
        source_manifest=get(ENV, "FM024_SOURCE_MANIFEST", "unknown"),
        slurm_job_id=get(ENV, "SLURM_JOB_ID", "none"),
        warmups=FM024_WARMUPS, samples=FM024_SAMPLES)
end

function fm024_csvfield(x)
    s = replace(string(x), '\n' => ' ', '\r' => ' ')
    return occursin(r"[\",]", s) ? "\"" * replace(s, '"' => "\"\"") * "\"" : s
end

function fm024_write_rows(path, rows)
    isempty(rows) && error("refusing to write an empty task-024 result")
    mkpath(dirname(path))
    open(path, "w") do io
        names = propertynames(first(rows))
        println(io, join(names, ','))
        for row in rows
            propertynames(row) == names ||
                error("task-024 rows do not share the common schema")
            println(io, join((fm024_csvfield(getproperty(row, n)) for n in names), ','))
        end
    end
    println("wrote ", path)
    return path
end

function fm024_infeasible_note(err)
    return replace(sprint(showerror, err), ',' => ';', '\n' => ' ')
end
