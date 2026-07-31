using FastMultipole
using FastMultipole.StaticArrays
using LinearAlgebra

const THETA = 0.5
const P_LITERATURE = 4
const P_CODE = P_LITERATURE - 1
const SOURCE_STRENGTH = 1.0
const H0 = 0.51
const ELLS = 2:7
const EPSILON_AT_ELL4 = 0.19542385331034917
const OUT = normpath(joinpath(@__DIR__, "..", "data", "cpu_gpu_scaling",
    "references", "compatibility_verification.csv"))

stencil_epsilon(ell) = EPSILON_AT_ELL4 * 2.0^(ell - 4)
cell_half_width(ell) = H0 / (1 << ell)
cell_radius(ell) = sqrt(3.0) * cell_half_width(ell)

function representative_offset(norm2::Int)
    limit = isqrt(norm2)
    for k in 0:limit, j in 0:limit, i in 0:limit
        i * i + j * j + k * k == norm2 && return SVector(i, j, k)
    end
    error("no integer offset represents squared norm $norm2")
end

function csvfield(x)
    s = replace(string(x), '"' => "\"\"")
    return occursin(r"[,\"]", s) ? "\"$s\"" : s
end

function write_csv(path, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(csvfield.(row), ','))
        end
    end
end

function verify_ell(ell)
    G = 1 << ell
    half_width = cell_half_width(ell)
    rho = cell_radius(ell)
    epsilon = stencil_epsilon(ell)
    offset12 = representative_offset(12)
    offset13 = representative_offset(13)
    upper = constant_p_stencil_bound(P_CODE, offset12, SOURCE_STRENGTH, half_width)
    lower = constant_p_stencil_bound(P_CODE, offset13, SOURCE_STRENGTH, half_width)

    lower <= epsilon < upper ||
        error("ell=$ell epsilon=$epsilon is outside [$lower, $upper)")

    accepted = 0
    mismatches = 0
    nearest_rejected_norm2 = -1
    nearest_accepted_norm2 = typemax(Int)
    legacy_cutoff = sqrt(3.0) / THETA

    for k in -(G - 1):(G - 1), j in -(G - 1):(G - 1), i in -(G - 1):(G - 1)
        offset = SVector(i, j, k)
        norm2 = i * i + j * j + k * k
        legacy_accepts = norm(offset) > legacy_cutoff
        radix_accepts =
            constant_p_stencil_bound(P_CODE, offset, SOURCE_STRENGTH, half_width) <=
            epsilon
        accepted += radix_accepts
        mismatches += legacy_accepts != radix_accepts
        if legacy_accepts
            nearest_accepted_norm2 = min(nearest_accepted_norm2, norm2)
        else
            nearest_rejected_norm2 = max(nearest_rejected_norm2, norm2)
        end
    end

    mismatches == 0 || error("ell=$ell has $mismatches predicate mismatches")
    nearest_rejected_norm2 == 12 ||
        error("ell=$ell nearest rejected squared norm is $nearest_rejected_norm2")
    nearest_accepted_norm2 == 13 ||
        error("ell=$ell nearest accepted squared norm is $nearest_accepted_norm2")

    total_offsets = (2G - 1)^3
    return (ell, rho, lower, upper, epsilon, total_offsets, accepted,
        nearest_rejected_norm2, nearest_accepted_norm2, mismatches)
end

P_CODE == 3 || error("literature P=4 must map to expansion_order=3")
rows = verify_ell.(ELLS)
header = ("ell", "rho", "epsilon_lower_inclusive", "epsilon_upper_exclusive",
    "epsilon_chosen", "total_offset_count", "accepted_offset_count",
    "nearest_rejected_squared_norm", "nearest_accepted_squared_norm",
    "mismatch_count")
write_csv(OUT, header, rows)
println("024b compatibility verification wrote $OUT")
foreach(row -> println("ell=$(row[1]) accepted=$(row[7])/$(row[6]) mismatches=$(row[10])"),
    rows)
