# Task 033: generate sampled-direct accuracy references for both cases at all
# grid sizes, in Float64, using the baseline FLOWVPM direct kernel
# (fmm.direct! with hessian=true, i.e. the UJ_direct path). Cheap relative to
# the benchmark itself: <=512 sampled targets (all targets for n<=10^4)
# against all n sources. Writes CSVs plus a sha256 manifest.
#
# Env: FM033_REFERENCE_DIR (output directory; default alongside data),
# FM033_CASES (comma list, default "cube,wake"), FM033_NS (comma list of n,
# default the full FM033_N_GRID). The manifest merge is additive over the
# requested set (see the comment above the manifest block).
#
# NOTE: run with -t 1. FastMultipole 2.0.4 direct_multithread! is broken on
# the (target, source) path (UndefVarError: n_source_bodies, direct.jl:111);
# single-thread direct is correct and cheap at <=512 samples per case.

include(joinpath(@__DIR__, "benchmark_033_common.jl"))
using Dates

refdir = get(ENV, "FM033_REFERENCE_DIR",
    joinpath(@__DIR__, "..", "data", "flowvpm_baseline", "references"))
mkpath(refdir)
host = gethostname()
timestamp = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")

cases = Tuple(split(get(ENV, "FM033_CASES", "cube,wake"), ','))
ns = Tuple(parse.(Int, split(get(ENV, "FM033_NS", join(FM033_N_GRID, ',')), ',')))

println("=== 033 reference generation on $host with $(Threads.nthreads()) threads")

for case in cases, n_target in ns
    path = fm033_reference_path(refdir, case, n_target)
    if isfile(path) && filesize(path) > 0
        println("skip existing $path")
        continue
    end
    t_build = @elapsed source = fm033_build(case, n_target)
    n_actual = vpm.get_np(source)
    indices = fm033_reference_indices(n_actual)

    # Target field holding only the sampled particles (same X and sigma;
    # target strength is irrelevant for induced U/J).
    target = vpm.ParticleField(length(indices))
    for bi in indices
        vpm.add_particle(target, collect(vpm.get_X(source, bi)),
            collect(vpm.get_Gamma(source, bi)), vpm.get_sigma(source, bi)[])
    end
    vpm._reset_particles(target)
    t_direct = @elapsed fmm.direct!(target, source;
        scalar_potential=false, hessian=true)

    tmp = path * ".tmp"
    isfile(tmp) && rm(tmp)
    for (si, bi) in enumerate(indices)
        U = vpm.get_U(target, si)
        J = vpm.get_J(target, si)
        fm033_append_row(tmp, FM033_REFERENCE_HEADER, (
            case, n_target, n_actual, FM033_SEED, FM033_SAMPLER_SEED,
            length(indices), host, timestamp, bi,
            U[1], U[2], U[3], ntuple(k -> J[k], 9)...,
        ))
    end
    mv(tmp, path; force=true)
    println("wrote $path (n_actual=$n_actual samples=$(length(indices)) " *
        "build=$(round(t_build, digits=2))s direct=$(round(t_direct, digits=2))s)")
end

# Manifest update, additive (widened for task 037b): once every REQUESTED
# (case, n) reference exists, merge into the manifest — existing lines are
# kept byte-identical (and in place), lines for requested files not yet
# listed are appended in request order, and a requested file whose checksum
# no longer matches its line is updated in place (reported). The default
# invocation (cases=cube,wake over FM033_N_GRID) reproduces the previous
# behavior and rewrites the manifest byte-identically.
if all(isfile(fm033_reference_path(refdir, case, n)) for case in cases, n in ns)
    manifest = joinpath(refdir, "direct_reference_checksums.sha256")
    lines = isfile(manifest) ? readlines(manifest) : String[]
    lineno = Dict{String,Int}()   # basename => index in `lines`
    for (i, line) in enumerate(lines)
        parts = split(line)
        length(parts) == 2 && (lineno[parts[2]] = i)
    end
    for case in cases, n_target in ns
        path = fm033_reference_path(refdir, case, n_target)
        entry = string(fm033_file_checksum(path), "  ", basename(path))
        i = get(lineno, basename(path), 0)
        if i == 0
            push!(lines, entry)
            lineno[basename(path)] = length(lines)
        elseif lines[i] != entry
            println("updating stale manifest line for $(basename(path))")
            lines[i] = entry
        end
    end
    tmp = manifest * ".tmp"
    open(tmp, "w") do io
        foreach(line -> println(io, line), lines)
    end
    mv(tmp, manifest; force=true)
    println("wrote $manifest")
else
    println("partial requested reference set; manifest not written")
end
