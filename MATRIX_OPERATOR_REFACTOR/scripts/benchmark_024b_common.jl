using SHA
using Random

const FM024B_REFERENCE_HEADER = (
    "n", "seed", "sampler_seed", "reference_samples", "host", "timestamp",
    "index", "reference_potential", "reference_gradient_x",
    "reference_gradient_y", "reference_gradient_z",
)

function fm024b_csvfield(x)
    s = replace(string(x), '"' => "\"\"")
    return occursin(r"[,\"]", s) ? "\"$s\"" : s
end

function fm024b_append_row(path, header, row)
    mkpath(dirname(path))
    newfile = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        newfile && println(io, join(header, ','))
        println(io, join(fm024b_csvfield.(row), ','))
    end
end

fm024b_reference_path(reference_dir, n) =
    joinpath(reference_dir, "direct_reference_n$(n).csv")

fm024b_file_checksum(path) = bytes2hex(sha256(read(path)))

function fm024b_expected_reference_indices(n, sampler_seed)
    n <= 10_000 && return collect(1:n)
    rng = MersenneTwister(sampler_seed + n)
    return sort!(randperm(rng, n)[1:512])
end

function fm024b_read_reference(path, expected_n;
        expected_seed=24025, expected_sampler_seed=24026)
    isfile(path) || error("missing 024b direct reference: $path")
    lines = readlines(path)
    isempty(lines) && error("empty 024b direct reference: $path")
    header = Tuple(split(lines[1], ','))
    header == FM024B_REFERENCE_HEADER ||
        error("unexpected reference schema in $path: $(join(header, ','))")

    indices = Int[]
    potential = Float64[]
    gradient = Vector{NTuple{3,Float64}}()
    declared_samples = 0
    reference_host = ""
    reference_timestamp = ""
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) == length(header) ||
            error("malformed reference row in $path")
        parse(Int, fields[1]) == expected_n ||
            error("reference n mismatch in $path")
        parse(Int, fields[2]) == expected_seed ||
            error("reference seed mismatch in $path")
        parse(Int, fields[3]) == expected_sampler_seed ||
            error("reference sampler seed mismatch in $path")
        declared_samples = parse(Int, fields[4])
        isempty(reference_host) && (reference_host = fields[5])
        isempty(reference_timestamp) && (reference_timestamp = fields[6])
        fields[5] == reference_host || error("inconsistent reference host in $path")
        fields[6] == reference_timestamp ||
            error("inconsistent reference timestamp in $path")
        push!(indices, parse(Int, fields[7]))
        push!(potential, parse(Float64, fields[8]))
        push!(gradient, (parse(Float64, fields[9]), parse(Float64, fields[10]),
            parse(Float64, fields[11])))
    end
    length(indices) == declared_samples ||
        error("reference sample count mismatch in $path")
    issorted(indices) || error("reference indices are not sorted in $path")
    length(unique(indices)) == length(indices) ||
        error("reference indices are not unique in $path")
    all(i -> 1 <= i <= expected_n, indices) ||
        error("reference index out of range in $path")
    indices == fm024b_expected_reference_indices(expected_n, expected_sampler_seed) ||
        error("reference indices do not match deterministic sampling in $path")

    return (; indices, potential, gradient, samples=length(indices),
        checksum=fm024b_file_checksum(path), seed=expected_seed,
        sampler_seed=expected_sampler_seed, host=reference_host,
        timestamp=reference_timestamp)
end

function fm024b_accuracy_metrics(sys, reference)
    nsample = reference.samples
    potential_error2 = 0.0
    potential_reference2 = 0.0
    gradient_error2 = 0.0
    gradient_reference2 = 0.0
    gradient_max = 0.0

    for sample_i in 1:nsample
        body_i = reference.indices[sample_i]
        p = Float64(sys.potential[1, body_i])
        pref = reference.potential[sample_i]
        dp = p - pref
        potential_error2 += dp * dp
        potential_reference2 += pref * pref

        gx = Float64(sys.potential[5, body_i])
        gy = Float64(sys.potential[6, body_i])
        gz = Float64(sys.potential[7, body_i])
        gxref, gyref, gzref = reference.gradient[sample_i]
        dx = gx - gxref
        dy = gy - gyref
        dz = gz - gzref
        error2 = dx * dx + dy * dy + dz * dz
        gradient_error2 += error2
        gradient_reference2 += gxref * gxref + gyref * gyref + gzref * gzref
        gradient_max = max(gradient_max, sqrt(error2))
    end

    return (
        potential_abs_rms=sqrt(potential_error2 / nsample),
        potential_rel_rms=sqrt(potential_error2 /
            max(potential_reference2, eps(Float64))),
        gradient_abs_rms=sqrt(gradient_error2 / nsample),
        gradient_rel_rms=sqrt(gradient_error2 /
            max(gradient_reference2, eps(Float64))),
        gradient_max=gradient_max,
    )
end
