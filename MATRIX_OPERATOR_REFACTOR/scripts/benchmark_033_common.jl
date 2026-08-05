# Task 033 shared harness: case builders, reference IO, and accuracy metrics
# for the FLOWVPM baseline CPU benchmarks (FLOWVPM at gpu-full branch point
# e2bd487, FastMultipole pinned to 2.0.x).
#
# Case definitions (recorded in 033-flowvpm-baseline-benchmarks.md):
#
# cube: n particles uniformly random in the unit cube, seed 33025+n.
#   Gamma components uniform in [-1,1]/n. Smoothing radius sigma =
#   2*(1/n)^(1/3) (overlap factor 2 with mean spacing d=(V/n)^(1/3), V=1).
#   Solver: rVPM, gaussianerf kernel, default relaxation, inviscid, no SFS.
#
# wake: a helical wake cylinder (user decision 2026-08-05, replacing the vortex
#   ring). Solid cylinder of diameter D=1 and length 5D about the z axis,
#   n positions uniform in its volume at seed 33025+7919+n. Strength direction
#   is the local helix tangent at pitch p = D (5 turns over the length),
#   magnitude |Gamma| = (r/R)/n (tip-weighted). sigma = 2*(V_cyl/n)^(1/3):
#   overlap 2 against the local mean spacing, the same convention as the cube,
#   so both cases share one unambiguous beta. Solver settings are identical to
#   the cube, so the two cases differ only in geometry and strength coherence.
#
#   Why not the ring (031/031a review, 2026-08-05): the torus filled 1.9% of
#   its bounding cube AND was only a few cells thick, so near sets were
#   half-empty and the 031a cost model's occupancy assumption broke; its
#   overlap was also direction-dependent (sigma/rl=3 radially, sigma/dS~1.58
#   azimuthally), leaving no single beta. The wake cylinder is locally dense and
#   isotropic with a single beta. At AR=5 it still fills only 3.1% of its
#   bounding CUBE -- that residual box waste is a real production lever, staged
#   in 035, not a defect of the case.
#
# FMM settings (user decision 2026-08-04): p=4, ncrit=50, theta=0.4, all
# autotuning off; remaining fields at struct defaults.

using Random
using SHA
using Statistics
import FLOWVPM
const vpm = FLOWVPM
const fmm = FLOWVPM.fmm

const FM033_FLOWVPM_DIR = dirname(dirname(pathof(FLOWVPM)))

const FM033_SEED = 33025
const FM033_SAMPLER_SEED = 33026
const FM033_N_GRID = (1000, 3162, 10000, 31623, 100000, 316228, 1000000)
const FM033_CASES = ("cube", "wake")

fm033_settings() = vpm.FMM(; p=4, ncrit=50, theta=0.4,
    autotune_p=false, autotune_ncrit=false, autotune_reg_error=false)

fm033_cube_sigma(n) = 2.0 * (1.0 / n)^(1 / 3)

function fm033_build_cube(n)
    rng = MersenneTwister(FM033_SEED + n)
    sigma = fm033_cube_sigma(n)
    pfield = vpm.ParticleField(n;
        formulation=vpm.rVPM,
        kernel=vpm.gaussianerf,
        viscous=vpm.Inviscid(),
        SFS=vpm.noSFS,
        transposed=true,
        integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm,
        fmm=fm033_settings())
    for _ in 1:n
        X = rand(rng, 3)
        Gamma = (2 .* rand(rng, 3) .- 1) ./ n
        vpm.add_particle(pfield, X, Gamma, sigma)
    end
    return pfield
end

# --- wake cylinder ----------------------------------------------------------
# Diameter D = 2*FM033_WAKE_R, length FM033_WAKE_LEN = 5D, axis = z, centred at
# the origin. Helix pitch p = D, so the tangent at radius r and azimuth theta is
# proportional to (-r*sin(theta), r*cos(theta), p/(2pi)).
const FM033_WAKE_R = 0.5
const FM033_WAKE_LEN = 5.0                       # = 5 * (2*FM033_WAKE_R)
const FM033_WAKE_PITCH = 2 * FM033_WAKE_R        # p = D
const FM033_WAKE_SEED_OFFSET = 7919              # keeps the RNG stream distinct from cube
fm033_wake_volume() = pi * FM033_WAKE_R^2 * FM033_WAKE_LEN
fm033_wake_sigma(n) = 2.0 * (fm033_wake_volume() / n)^(1 / 3)

function fm033_build_wake(n)
    rng = MersenneTwister(FM033_SEED + FM033_WAKE_SEED_OFFSET + n)
    sigma = fm033_wake_sigma(n)
    pfield = vpm.ParticleField(n;
        formulation=vpm.rVPM,
        kernel=vpm.gaussianerf,
        viscous=vpm.Inviscid(),
        SFS=vpm.noSFS,
        transposed=true,
        integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm,
        fmm=fm033_settings())
    for _ in 1:n
        # exactly uniform in the cylinder volume
        r = FM033_WAKE_R * sqrt(rand(rng))
        theta = 2pi * rand(rng)
        z = FM033_WAKE_LEN * (rand(rng) - 0.5)
        X = (r * cos(theta), r * sin(theta), z)
        # local helix tangent, tip-weighted magnitude
        t = (-r * sin(theta), r * cos(theta), FM033_WAKE_PITCH / (2pi))
        tnrm = sqrt(sum(abs2, t))
        mag = (r / FM033_WAKE_R) / n
        Gamma = (mag * t[1] / tnrm, mag * t[2] / tnrm, mag * t[3] / tnrm)
        vpm.add_particle(pfield, collect(X), collect(Gamma), sigma)
    end
    return pfield
end

function fm033_build(case, n_target)
    case == "cube" && return fm033_build_cube(n_target)
    case == "wake" && return fm033_build_wake(n_target)
    error("unknown 033 case: $case")
end

function fm033_sigma(case, n_target)
    case == "cube" && return fm033_cube_sigma(n_target)
    case == "wake" && return fm033_wake_sigma(n_target)
    error("unknown 033 case: $case")
end

# --- sampled-direct references (024b conventions, new seeds) -----------------

function fm033_reference_indices(n)
    n <= 10_000 && return collect(1:n)
    rng = MersenneTwister(FM033_SAMPLER_SEED + n)
    return sort!(randperm(rng, n)[1:512])
end

const FM033_REFERENCE_HEADER = (
    "case", "n_target", "n_actual", "seed", "sampler_seed", "samples", "host",
    "timestamp", "index", "ux", "uy", "uz",
    "j11", "j21", "j31", "j12", "j22", "j32", "j13", "j23", "j33",
)

fm033_reference_path(reference_dir, case, n_target) =
    joinpath(reference_dir, "direct_reference_$(case)_n$(n_target).csv")

fm033_file_checksum(path) = bytes2hex(sha256(read(path)))

function fm033_csvfield(x)
    s = replace(string(x), '"' => "\"\"")
    return occursin(r"[,\"]", s) ? "\"$s\"" : s
end

function fm033_append_row(path, header, row)
    mkpath(dirname(path))
    newfile = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        newfile && println(io, join(header, ','))
        println(io, join(fm033_csvfield.(row), ','))
    end
end

function fm033_read_reference(path, case, n_target, n_actual)
    isfile(path) || error("missing 033 direct reference: $path")
    lines = readlines(path)
    isempty(lines) && error("empty 033 direct reference: $path")
    header = Tuple(split(lines[1], ','))
    header == FM033_REFERENCE_HEADER ||
        error("unexpected reference schema in $path: $(join(header, ','))")

    indices = Int[]
    U = Vector{NTuple{3,Float64}}()
    J = Vector{NTuple{9,Float64}}()
    declared_samples = 0
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ',')
        length(fields) == length(header) || error("malformed reference row in $path")
        fields[1] == case || error("reference case mismatch in $path")
        parse(Int, fields[2]) == n_target || error("reference n_target mismatch in $path")
        parse(Int, fields[3]) == n_actual || error("reference n_actual mismatch in $path")
        parse(Int, fields[4]) == FM033_SEED || error("reference seed mismatch in $path")
        parse(Int, fields[5]) == FM033_SAMPLER_SEED ||
            error("reference sampler seed mismatch in $path")
        declared_samples = parse(Int, fields[6])
        push!(indices, parse(Int, fields[9]))
        push!(U, ntuple(k -> parse(Float64, fields[9 + k]), 3))
        push!(J, ntuple(k -> parse(Float64, fields[12 + k]), 9))
    end
    length(indices) == declared_samples ||
        error("reference sample count mismatch in $path")
    issorted(indices) || error("reference indices not sorted in $path")
    indices == fm033_reference_indices(n_actual) ||
        error("reference indices do not match deterministic sampling in $path")

    return (; indices, U, J, samples=length(indices),
        checksum=fm033_file_checksum(path))
end

# Relative RMS errors of the pfield's accumulated U (rows 10:12) and J
# (rows 16:24) against the sampled direct reference.
function fm033_accuracy_metrics(pfield, reference)
    u_err2 = u_ref2 = j_err2 = j_ref2 = 0.0
    u_max = 0.0
    for (si, bi) in enumerate(reference.indices)
        Ux, Uy, Uz = vpm.get_U(pfield, bi)
        uref = reference.U[si]
        du = (Ux - uref[1], Uy - uref[2], Uz - uref[3])
        e2 = du[1]^2 + du[2]^2 + du[3]^2
        u_err2 += e2
        u_ref2 += uref[1]^2 + uref[2]^2 + uref[3]^2
        u_max = max(u_max, sqrt(e2))
        Jval = vpm.get_J(pfield, bi)
        jref = reference.J[si]
        for k in 1:9
            dj = Jval[k] - jref[k]
            j_err2 += dj * dj
            j_ref2 += jref[k]^2
        end
    end
    ns = reference.samples
    return (
        u_rel_rms=sqrt(u_err2 / max(u_ref2, eps(Float64))),
        u_abs_rms=sqrt(u_err2 / ns),
        u_max_err=u_max,
        j_rel_rms=sqrt(j_err2 / max(j_ref2, eps(Float64))),
    )
end
