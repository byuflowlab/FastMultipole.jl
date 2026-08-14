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
# rotor: DJI 9443 prescribed helical rotor wake with per-particle sigma
#   (task 037b, 2026-08-13) — see the rotor section below and
#   data/rotor_wake/PROVENANCE.md for the full definition and derived numbers.
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
const FM033_CASES = ("cube", "wake", "rotor")

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

# --- DJI 9443 rotor wake (task 037b, added 2026-08-13) -----------------------
# A prescribed helical rotor wake with per-particle sigma, built from the
# measured DJI 9443 bound-circulation distribution (FLOWPanel phase 02c mesh
# convergence, finest capped mesh dji121c, RPM 5400), vendored at
# data/rotor_wake/dji9443_fixed_bin_circulation.csv so cluster runs never need
# FLOWPanel. Full derivation and derived numbers: data/rotor_wake/PROVENANCE.md.
#
# Wake model (all decisions fixed by the 037b task spec):
# * Per blade b (B=2), trail three filament classes from Gamma(r/R):
#   root (+Gamma_1 at r_1), a thinned inboard sheet (bin-boundary filaments of
#   strength Gamma_{i+1}-Gamma_i, lumped in groups of k so inboard particles
#   are ~35% of the total), and the tip vortex (-Gamma_end at r_end). Total
#   trailed circulation sums to zero; the tip filament is the strongest single
#   structure. Because every filament shares one azimuthal step (below), the
#   ~35% inboard budget forces exactly one lumped inboard filament per blade
#   (k=34 of 34 boundaries, inboard fraction 1/3); it is placed at the
#   |dGamma|-weighted mean boundary radius rather than the every-34th boundary
#   (which would sit adjacent to the tip and cancel half the tip vortex).
# * Kinematics: azimuth psi = Omega*t + 2pi*b/B; axial convection from
#   momentum theory on the Kutta-Joukowski thrust of the measured
#   distribution, T = B*rho*Omega*trapz(Gamma(r)*r dr), v_i = sqrt(T/(2 rho pi
#   R^2)), far-wake doubling blended as v(age) = v_i*(2 - exp(-age/1rev));
#   radial contraction r(age) = r0*(0.78 + 0.22*exp(-age/0.25rev)). Wake
#   convects in -z; rotor plane z=0. Base age 12 rev, extended in 0.5-rev
#   steps until aspect ratio length/(2R) >= 5.
# * Discretization: one azimuthal step dpsi for ALL filaments, chosen so the
#   total count is ~n_target (per-filament count = ceil(n_target/n_filaments);
#   emission stops deterministically at exactly n_target, trimming the oldest
#   end of the last blade's tip filament by at most n_filaments-1 particles).
#   Particle strength Gamma_p = filament strength * ds * unit tangent (ds =
#   midpoint-rule arc length of its segment), position = segment midpoint +
#   0.15*sigma_p*randn(rng,3) jitter, rng = MersenneTwister(FM033_SEED +
#   104729 + n_target) drawn in a fixed order (sole RNG use).
# * Per-particle sigma: sigma0 = beta*ds with beta=2 (ds is the local
#   inter-particle spacing along the filament), core-spreading growth
#   sigma(age) = sigma0*sqrt(1 + (2/3)*age_rev) so sigma(12 rev) = 3*sigma0.
#   No global floor/cap: the natural range (see rotor_case_stats.csv) is
#   well-behaved. This is the intended sigma heterogeneity: tight young tip
#   vortices near the rotor plane, diffuse old inboard wake.
const FM033_ROTOR_R = 0.119                     # rotor radius [m]
const FM033_ROTOR_B = 2                         # blade count
const FM033_ROTOR_RPM = 5400.0
const FM033_ROTOR_OMEGA = 2pi * FM033_ROTOR_RPM / 60    # 565.4867 rad/s
const FM033_ROTOR_RHO = 1.225                   # air [kg/m^3]; thrust doc only
const FM033_ROTOR_SEED_OFFSET = 104729          # keeps RNG stream distinct
const FM033_ROTOR_GAMMA_CASE = "dji121c"        # finest capped FLOWPanel mesh
const FM033_ROTOR_GAMMA_CSV = joinpath(@__DIR__, "..", "data", "rotor_wake",
    "dji9443_fixed_bin_circulation.csv")
const FM033_ROTOR_BETA = 2.0                    # sigma0 = beta * local spacing
const FM033_ROTOR_JITTER = 0.15                 # position jitter, units of sigma_p
const FM033_ROTOR_AGE_BASE = 12.0               # wake age [rev] before AR extension
const FM033_ROTOR_AR_MIN = 5.0                  # required length/(2R)
const FM033_ROTOR_CONTRACT_FAR = 0.78           # far-wake contraction ratio
const FM033_ROTOR_CONTRACT_TAU = 0.25           # contraction time constant [rev]
const FM033_ROTOR_SIGMA_GROWTH = 2.0 / 3.0      # per rev; sigma(12rev)=3*sigma0
const FM033_ROTOR_INBOARD_FRACTION = 0.35       # target inboard particle share

# Cached (r/R, gamma_mean) table for FM033_ROTOR_GAMMA_CASE.
const _fm033_rotor_gamma_cache = Ref{Union{Nothing,NTuple{2,Vector{Float64}}}}(nothing)
function fm033_rotor_gamma()
    if _fm033_rotor_gamma_cache[] === nothing
        isfile(FM033_ROTOR_GAMMA_CSV) ||
            error("missing rotor circulation table: $(FM033_ROTOR_GAMMA_CSV)")
        lines = readlines(FM033_ROTOR_GAMMA_CSV)
        header = split(lines[1], ',')
        ci = findfirst(==("case"), header)
        xi = findfirst(==("abs_r_over_R"), header)
        gi = findfirst(==("gamma_mean"), header)
        (ci === nothing || xi === nothing || gi === nothing) &&
            error("unexpected rotor circulation schema in $(FM033_ROTOR_GAMMA_CSV)")
        x = Float64[]
        g = Float64[]
        for line in lines[2:end]
            isempty(strip(line)) && continue
            f = split(line, ',')
            f[ci] == FM033_ROTOR_GAMMA_CASE || continue
            push!(x, parse(Float64, f[xi]))
            push!(g, parse(Float64, f[gi]))
        end
        length(x) >= 3 || error("rotor circulation case $(FM033_ROTOR_GAMMA_CASE) not found")
        issorted(x) || error("rotor circulation stations not sorted")
        _fm033_rotor_gamma_cache[] = (x, g)
    end
    return _fm033_rotor_gamma_cache[]
end

# All deterministic derived quantities of the rotor case at a given n_target.
function fm033_rotor_spec(n_target)
    xs, gs = fm033_rotor_gamma()
    R = FM033_ROTOR_R
    r = xs .* R                                  # dimensional station radii
    ns = length(r)
    # Kutta-Joukowski thrust of the measured distribution (trapezoid over bins)
    T = FM033_ROTOR_B * FM033_ROTOR_RHO * FM033_ROTOR_OMEGA *
        sum((gs[i] * r[i] + gs[i+1] * r[i+1]) / 2 * (r[i+1] - r[i]) for i in 1:ns-1)
    v_i = sqrt(T / (2 * FM033_ROTOR_RHO * pi * R^2))
    T_rev = 2pi / FM033_ROTOR_OMEGA
    # axial displacement after `a` revs under v(age) = v_i*(2 - exp(-age/1rev))
    wake_z(a) = v_i * T_rev * (2a - 1 + exp(-a))
    age = FM033_ROTOR_AGE_BASE
    while wake_z(age) / (2R) < FM033_ROTOR_AR_MIN
        age += 0.5
    end
    # thinned inboard sheet: nb = ns-1 boundaries, keep m lumped filaments with
    # m minimizing |m/(m+2) - 0.35| (equal particles per filament makes the
    # inboard share m/(m+2)); k = ceil(nb/m) boundaries lumped per kept filament
    nb = ns - 1
    m = argmin(m -> abs(m / (m + 2) - FM033_ROTOR_INBOARD_FRACTION), 1:nb)
    k = cld(nb, m)
    # per-blade filament list (root, inboard lumped groups, tip), each as
    # (r0, strength, inboard::Bool); strengths sum to zero
    filaments = Tuple{Float64,Float64,Bool}[]
    push!(filaments, (r[1], gs[1], false))                   # root: +Gamma_1
    for j0 in 1:k:nb
        j1 = min(j0 + k - 1, nb)
        dG = gs[j1+1] - gs[j0]                               # summed bin deltas
        wsum = rsum = 0.0
        for j in j0:j1                                       # |dGamma|-weighted radius
            w = abs(gs[j+1] - gs[j])
            wsum += w
            rsum += w * (r[j] + r[j+1]) / 2
        end
        push!(filaments, (wsum > 0 ? rsum / wsum : (r[j0] + r[j1+1]) / 2, dG, true))
    end
    push!(filaments, (r[ns], -gs[ns], false))                # tip: -Gamma_end
    @assert abs(sum(f[2] for f in filaments)) < 1e-12
    n_filaments = FM033_ROTOR_B * length(filaments)
    n_per = cld(n_target, n_filaments)
    da = age / n_per                                         # segment age span [rev]
    dpsi = 2pi * da                                          # azimuthal step [rad]
    return (; T, v_i, T_rev, age, k, m, filaments, n_filaments, n_per, da, dpsi)
end

# Iterate the rotor particles in their fixed deterministic order, calling
# f(X, Gamma_p, sigma_p) for each. `rng === nothing` skips the position jitter
# (used by the sigma_max query); otherwise rng is consumed in emission order.
function fm033_rotor_foreach(f, n_target; rng=nothing)
    spec = fm033_rotor_spec(n_target)
    count = 0
    for b in 0:FM033_ROTOR_B-1
        psi0 = 2pi * b / FM033_ROTOR_B
        for (r0, G, _) in spec.filaments
            for j in 1:spec.n_per
                count == n_target && return count
                a = (j - 0.5) * spec.da                      # midpoint age [rev]
                decay = exp(-a / FM033_ROTOR_CONTRACT_TAU)
                c = FM033_ROTOR_CONTRACT_FAR + (1 - FM033_ROTOR_CONTRACT_FAR) * decay
                rc = r0 * c
                drda = -r0 * (1 - FM033_ROTOR_CONTRACT_FAR) / FM033_ROTOR_CONTRACT_TAU * decay
                psi = 2pi * a + psi0
                sp, cp = sincos(psi)
                # dX/da of the contracting helix (a in revs)
                tx = drda * cp - rc * 2pi * sp
                ty = drda * sp + rc * 2pi * cp
                tz = -spec.v_i * spec.T_rev * (2 - exp(-a))
                speed = sqrt(tx * tx + ty * ty + tz * tz)
                ds = speed * spec.da                         # segment arc length
                sigma = FM033_ROTOR_BETA * ds *
                        sqrt(1 + FM033_ROTOR_SIGMA_GROWTH * a)
                gscale = G * ds / speed
                z = -spec.v_i * spec.T_rev * (2a - 1 + exp(-a))
                x1, x2, x3 = rc * cp, rc * sp, z
                if rng !== nothing
                    jit = FM033_ROTOR_JITTER * sigma
                    x1 += jit * randn(rng)
                    x2 += jit * randn(rng)
                    x3 += jit * randn(rng)
                end
                f((x1, x2, x3), (gscale * tx, gscale * ty, gscale * tz), sigma)
                count += 1
            end
        end
    end
    return count
end

function fm033_build_rotor(n_target)
    rng = MersenneTwister(FM033_SEED + FM033_ROTOR_SEED_OFFSET + n_target)
    pfield = vpm.ParticleField(n_target;
        formulation=vpm.rVPM,
        kernel=vpm.gaussianerf,
        viscous=vpm.Inviscid(),
        SFS=vpm.noSFS,
        transposed=true,
        integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm,
        fmm=fm033_settings())
    emitted = fm033_rotor_foreach(n_target; rng=rng) do X, Gamma, sigma
        vpm.add_particle(pfield, collect(X), collect(Gamma), sigma)
    end
    emitted == n_target || error("rotor case emitted $emitted != $n_target particles")
    return pfield
end

# The rotor sigma is heterogeneous; fm033_sigma keeps its scalar-return
# contract by returning sigma_max, which is what the FastMultipole geometry
# gate consumes (it gates on the live sigma_max of the field). Deterministic
# (no RNG involved: jitter does not affect sigma).
function fm033_rotor_sigma_max(n_target)
    smax = Ref(0.0)
    fm033_rotor_foreach(n_target) do _, _, sigma
        smax[] = max(smax[], sigma)
    end
    return smax[]
end

function fm033_build(case, n_target)
    case == "cube" && return fm033_build_cube(n_target)
    case == "wake" && return fm033_build_wake(n_target)
    case == "rotor" && return fm033_build_rotor(n_target)
    error("unknown 033 case: $case")
end

function fm033_sigma(case, n_target)
    case == "cube" && return fm033_cube_sigma(n_target)
    case == "wake" && return fm033_wake_sigma(n_target)
    case == "rotor" && return fm033_rotor_sigma_max(n_target)   # sigma_max (heterogeneous)
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
