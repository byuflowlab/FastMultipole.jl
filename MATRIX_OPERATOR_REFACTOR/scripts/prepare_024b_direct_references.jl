using FastMultipole
using Dates
using Random
using Sockets

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(REPO, "test", "gravitational.jl"))
include(joinpath(@__DIR__, "benchmark_024b_common.jl"))

const CAMPAIGN_NS = [1000, 3162, 10000, 31623, 100000, 316228, 1000000]
const NS = parse.(Int, split(get(ENV, "FM024B_REFERENCE_NS",
    join(CAMPAIGN_NS, ',')), ','))
const SEED = parse(Int, get(ENV, "FM024B_SEED", "24025"))
const SAMPLER_SEED = parse(Int, get(ENV, "FM024B_SAMPLER_SEED", "24026"))
const DIRECT_THREADS = parse(Int,
    get(ENV, "FM024B_DIRECT_THREADS", string(Threads.nthreads())))
const OUTDIR = get(ENV, "FM024B_REFERENCE_DIR",
    joinpath(REPO, "MATRIX_OPERATOR_REFACTOR", "data", "cpu_gpu_scaling",
        "references"))

1 <= DIRECT_THREADS <= Threads.nthreads() ||
    error("FM024B_DIRECT_THREADS must be in 1:Threads.nthreads()")

function sample_indices(n)
    return fm024b_expected_reference_indices(n, SAMPLER_SEED)
end

function write_reference(n)
    sources = generate_gravitational(SEED, n)
    indices = sample_indices(n)
    target_bodies = [Body(body.position, body.radius, body.strength)
        for body in sources.bodies[indices]]
    targets = Gravitational(target_bodies,
        zeros(eltype(sources.potential), size(sources.potential, 1), length(indices)))

    direct!(targets, sources; scalar_potential=true, gradient=true,
        n_threads=DIRECT_THREADS)

    path = fm024b_reference_path(OUTDIR, n)
    mkpath(dirname(path))
    timestamp = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
    open(path, "w") do io
        println(io, join(FM024B_REFERENCE_HEADER, ','))
        for (sample_i, body_i) in enumerate(indices)
            row = (n, SEED, SAMPLER_SEED, length(indices), gethostname(), timestamp,
                body_i, targets.potential[1, sample_i],
                targets.potential[5, sample_i], targets.potential[6, sample_i],
                targets.potential[7, sample_i])
            println(io, join(fm024b_csvfield.(row), ','))
        end
    end

    checked = fm024b_read_reference(path, n)
    println("reference n=$n samples=$(checked.samples) checksum=$(checked.checksum)")
    return checked.checksum
end

checksums = Dict(n => write_reference(n) for n in NS)
manifest = joinpath(OUTDIR, "direct_reference_checksums.sha256")
open(manifest, "w") do io
    for n in sort(NS)
        println(io, checksums[n], "  direct_reference_n", n, ".csv")
    end
end
println("024b direct-reference manifest wrote $manifest")
