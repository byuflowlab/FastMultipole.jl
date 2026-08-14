# Task 035: separate smoothing-cutoff error from FMM approximation error.
#
# For the checksummed 033 sample targets, independently evaluate the exact
# fully regularized field R and exact partitioned fields P(rho_t).  Compare
# those with the actual FMM field F using the exact identity
#
#   F - R = (P - R) + (F - P).
#
# Both component norms use ||R|| as denominator, so cutoff_rel + fmm_rel is a
# rigorous triangle bound on the regularized-relative error.  The globally
# singular field is only formed as the prefix-sum base of the partitioned
# fields (it is never compared against anything).
#
# Two-pass semantics (task 037b).  In exact arithmetic the shipped
# TwoPassVortex field (pass 1: regularized pair math for rho <= rho_c and
# exact singular beyond, wherever the pair is routed; pass 2: the additive
# regularization deficit on the shell rho_c < rho <= rho_t) equals the SAME
# partitioned intermediate P(rho_t) this oracle already computes,
# INDEPENDENT of rho_c — rho_c only moves pairs between the two passes.
# Equivalently, the general truncated-shell field
#     P_shell(rho_c, rho_x) = regularized (rho <= rho_c)
#                             + deficit-corrected on (rho_c, rho_x]
#                             + singular beyond rho_x
# satisfies P_shell(rho_c, rho_x) == P(rho_x) for every rho_c < rho_x, so no
# separate two-threshold machinery exists (or is needed).  For
# kernel=twopass configs the "cutoff" component ||P(rho_t) - R|| is
# therefore exactly the truncated-deficit error — the correction dropped for
# pairs beyond the pass-2 truncation radius rho_t — and P is looked up at
# rho_t.  The component_semantics CSV column labels each row accordingly
# ("cutoff" for regularized/partitioned, "truncated_deficit" for twopass).
#
# FM035D_EXACT_ONLY=1 turns the script into a CPU-only cutoff-curve
# instrument: no CUDA import, no GPU field; one exact O(samples*n) pass per
# (case, n) serves every requested rho at once (per-pair deficit binning +
# prefix sum), and each config emits a single tf=Float64 row whose GPU/F
# columns are empty.  When FM035D_CONFIG_FILE is set, the config file is
# authoritative for the (case, n) grid; otherwise FM035D_CASES x FM035D_NS
# selects among the built-in configs.  The output CSV is rewritten from
# scratch on every run (no resume-append), so schema changes are safe.

const FM035D_EXACT_ONLY = get(ENV, "FM035D_EXACT_ONLY", "0") == "1"
if !FM035D_EXACT_ONLY
    import CUDA
    CUDA.functional() || error("CUDA is not functional on this node")
end
using Printf
using Statistics
using Base.Threads

const FM035D_FMDIR = abspath(get(ENV, "FM035_FMDIR",
    get(ENV, "HOME", "") * "/FastMultipole-034"))
const FM035D_OUT = abspath(get(ENV, "FM035D_OUT", joinpath(FM035D_FMDIR,
    "MATRIX_OPERATOR_REFACTOR", "data", "flowvpm_gpu_campaign",
    "fm035_error_decomposition.csv")))
const FM035D_REFDIR = joinpath(FM035D_FMDIR, "MATRIX_OPERATOR_REFACTOR",
    "data", "flowvpm_baseline", "references")
const FM035D_TOTAL_GATE = 1e-3
const FM035D_CASES = Tuple(split(get(ENV, "FM035D_CASES", "cube,wake"), ','))
const FM035D_NS = Tuple(parse.(Int, split(get(ENV, "FM035D_NS",
    "100000,1000000"), ',')))
const FM035D_CONFIG_FILE = get(ENV, "FM035D_CONFIG_FILE", "")

include(joinpath(FM035D_FMDIR, "MATRIX_OPERATOR_REFACTOR", "scripts",
    "benchmark_033_common.jl"))
const FM = vpm.fmm

# Same independent erf implementation used by the task-032 validation oracle.
function fm035d_ref_erf(x::Float64)
    ax = abs(x)
    if ax < 2.0
        s = 0.0
        term = ax
        n = 0
        while true
            add = term / (2n + 1)
            s += add
            n += 1
            term *= -ax * ax / n
            abs(add) <= eps() * max(abs(s), 1.0) && break
        end
        return sign(x) * (2 / sqrt(pi)) * s
    end
    u = 1 / (2 * ax * ax)
    cf = 1.0
    for k in 60:-1:1
        cf = 1 + k * u / cf
    end
    return sign(x) * (1 - exp(-ax * ax) / (ax * sqrt(pi)) / cf)
end

@inline function fm035d_addpair!(U, J, k, dx, dy, dz, r2, cr3,
        cx, cy, cz, gx, gy, gz, g, h)
    a = h/r2
    b = -g*cr3
    U[1,k] += g*cx; U[2,k] += g*cy; U[3,k] += g*cz
    J[1,k] += a*cx*dx
    J[2,k] += a*cy*dx - b*gz
    J[3,k] += a*cz*dx + b*gy
    J[4,k] += a*cx*dy + b*gz
    J[5,k] += a*cy*dy
    J[6,k] += a*cz*dy - b*gx
    J[7,k] += a*cx*dz - b*gy
    J[8,k] += a*cy*dz + b*gx
    J[9,k] += a*cz*dz
    return nothing
end

# One O(samples*n) pass produces R and every requested exact partitioned field
# P(rho_t), for U and J. This keeps cutoff selection independent of FMM error.
function fm035d_exact_fields(pfield, indices, rhos)
    A = pfield.particles
    n = vpm.get_np(pfield)
    S = length(indices)
    UR = zeros(3, S); JR = zeros(9, S)
    US = zeros(3, S); JS = zeros(9, S)
    # Each pair's regularized-minus-singular correction is placed in the first
    # cutoff bin that contains it. Prefix-summing the bins after traversal
    # yields every P(rho) while evaluating each pair only twice (R and S), not
    # once per candidate cutoff.
    delta_U = [zeros(3, S) for _ in rhos]
    delta_J = [zeros(9, S) for _ in rhos]
    Aconst = sqrt(2 / pi)
    @threads for k in 1:S
        i = indices[k]
        xi = Float64(A[vpm.X_INDEX[1], i])
        yi = Float64(A[vpm.X_INDEX[2], i])
        zi = Float64(A[vpm.X_INDEX[3], i])
        for j in 1:n
            j == i && continue
            dx = xi - Float64(A[vpm.X_INDEX[1], j])
            dy = yi - Float64(A[vpm.X_INDEX[2], j])
            dz = zi - Float64(A[vpm.X_INDEX[3], j])
            r2 = dx*dx + dy*dy + dz*dz
            r2 == 0 && continue
            r = sqrt(r2); invr = inv(r)
            sigma = Float64(A[vpm.SIGMA_INDEX, j])
            gx = Float64(A[vpm.GAMMA_INDEX[1], j])
            gy = Float64(A[vpm.GAMMA_INDEX[2], j])
            gz = Float64(A[vpm.GAMMA_INDEX[3], j])
            rho = r / sigma
            e = exp(-rho*rho/2)
            g = fm035d_ref_erf(rho/sqrt(2.0)) - Aconst*rho*e
            gp = Aconst*rho*rho*e
            h = rho*gp - 3g
            cr3 = inv(4pi) * invr^3
            cx = (dz*gy - dy*gz)*cr3
            cy = (dx*gz - dz*gx)*cr3
            cz = (dy*gx - dx*gy)*cr3

            fm035d_addpair!(UR, JR, k, dx, dy, dz, r2, cr3,
                cx, cy, cz, gx, gy, gz, g, h)
            fm035d_addpair!(US, JS, k, dx, dy, dz, r2, cr3,
                cx, cy, cz, gx, gy, gz, 1.0, -3.0)
            irho = searchsortedfirst(rhos, rho)
            if irho <= length(rhos)
                fm035d_addpair!(delta_U[irho], delta_J[irho], k,
                    dx, dy, dz, r2, cr3, cx, cy, cz, gx, gy, gz,
                    g - 1.0, h + 3.0)
            end
        end
    end
    partitioned = Dict{eltype(rhos),NamedTuple}()
    running_U = copy(US); running_J = copy(JS)
    for (i, cutoff) in enumerate(rhos)
        running_U .+= delta_U[i]
        running_J .+= delta_J[i]
        partitioned[cutoff] = (U=copy(running_U), J=copy(running_J))
    end
    return (; UR, JR, US, JS, partitioned)
end

function fm035d_to_gpu(cpu)
    n = vpm.get_np(cpu)
    gpu = vpm.ParticleField(n, Float64;
        formulation=vpm.rVPM, kernel=vpm.gaussianerf, viscous=vpm.Inviscid(),
        SFS=vpm.noSFS, transposed=true, integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm, fmm=fm033_settings(), arraytype=CUDA.CuArray)
    gpu.np = n
    gpu.particles .= CUDA.CuArray{Float64}(Array(cpu.particles)[:, 1:n])
    return gpu
end

norm2(A) = sum(abs2, A)
function fm035d_metrics(F, P, R)
    ec = P .- R
    ef = F .- P
    et = F .- R
    den = sqrt(max(norm2(R), eps(Float64)))
    nc = sqrt(norm2(ec)); nf = sqrt(norm2(ef)); nt = sqrt(norm2(et))
    corrden = nc*nf
    corr = corrden == 0 ? 0.0 : sum(ec .* ef)/corrden
    sumden = nc + nf
    cancellation = sumden == 0 ? 0.0 : nt/sumden
    identity = sqrt(norm2(et .- (ec .+ ef))) / den
    return (; cutoff_rel=nc/den, fmm_rel=nf/den, total_rel=nt/den,
        triangle_rel=sumden/den, correlation=corr,
        cancellation_ratio=cancellation, identity_rel=identity)
end

function fm035d_reference_matrix(values, rows)
    out = zeros(rows, length(values))
    for k in eachindex(values), j in 1:rows
        out[j,k] = values[k][j]
    end
    return out
end

function fm035d_config(label, order, rho, case, n, ell, q;
        kernel=:partitioned, rho_c=nothing, rectangular=false,
        gh_mode=:shipped)
    return (; label, order, rho, case, n, ell, q, kernel, rho_c, rectangular,
        gh_mode)
end

function fm035d_parse_config(line)
    kv = Dict{Symbol,String}()
    for token in split(strip(line))
        m = match(r"^([A-Za-z_0-9]+)=(.*)$", token)
        m === nothing && error("bad decomposition config token: $token")
        kv[Symbol(m.captures[1])] = m.captures[2]
    end
    return fm035d_config(kv[:label], parse(Int, kv[:expansion_order]),
        parse(Float64, kv[:rho_t]), kv[:case], parse(Int, kv[:n]),
        parse(Int, kv[:ell]), parse(Int, kv[:q]);
        kernel=Symbol(get(kv, :kernel, "partitioned")),
        rho_c=haskey(kv, :rho_c) ? parse(Float64, kv[:rho_c]) : nothing,
        rectangular=get(kv, :rectangular, "0") == "1",
        gh_mode=Symbol(get(kv, :gh_mode, "shipped")))   # task 037f
end

# Honest component labeling (header note above): for twopass rows the "cutoff"
# component ||P(rho_t) - R|| is the truncated-deficit error of the pass-2
# truncation radius; for regularized/partitioned rows it is the smoothing-
# cutoff error proper.
fm035d_component_semantics(kernel) =
    kernel === :twopass ? "truncated_deficit" : "cutoff"

const CONFIGS = isempty(FM035D_CONFIG_FILE) ? [
    fm035d_config("p4", 3, 4.252, "cube", 100000, 4, 17),
    fm035d_config("p5", 4, 3.668, "cube", 100000, 4, 12),
    fm035d_config("p4", 3, 4.252, "cube", 1000000, 5, 17),
    fm035d_config("p5", 4, 3.668, "cube", 1000000, 5, 12),
    fm035d_config("p4", 3, 4.252, "wake", 100000, 5, 12),
    fm035d_config("p5", 4, 3.668, "wake", 100000, 5, 6),
    fm035d_config("p4", 3, 4.252, "wake", 1000000, 6, 12),
    fm035d_config("p5", 4, 3.668, "wake", 1000000, 6, 6),
] : [fm035d_parse_config(line) for line in readlines(FM035D_CONFIG_FILE)
     if !isempty(strip(line)) && !startswith(strip(line), '#')]
const HEADER = ("label","job","host","case","n","tf","literature_P",
    "expansion_order","ell","q","kernel","rho_t","rho_c","rectangular",
    "gh_mode","component_semantics",
    "reference_checksum",
    "reference_u_rel","reference_j_rel","u_cutoff_rel","u_fmm_rel",
    "u_total_rel","u_triangle_rel","u_correlation","u_cancellation_ratio",
    "u_identity_rel","u_triangle_pass","j_cutoff_rel",
    "j_fmm_rel","j_total_rel","j_triangle_rel","j_correlation",
    "j_cancellation_ratio","j_identity_rel")

# With a config file the file is authoritative for the (case, n) grid; the
# built-in default grid is still filtered by FM035D_CASES/FM035D_NS.
const FM035D_CASE_POINTS = isempty(FM035D_CONFIG_FILE) ?
    [(case, n) for case in FM035D_CASES for n in FM035D_NS] :
    unique((cfg.case, cfg.n) for cfg in CONFIGS)

mkpath(dirname(FM035D_OUT))
open(FM035D_OUT, "w") do io
    println(io, join(HEADER, ','))
    for (case, n) in FM035D_CASE_POINTS
        case_configs = [cfg for cfg in CONFIGS if cfg.case == case && cfg.n == n]
        isempty(case_configs) && continue
        cpu = fm033_build(case, n)
        refpath = fm033_reference_path(FM035D_REFDIR, case, n)
        ref = fm033_read_reference(refpath, case, n, n)
        rhos = sort!(unique(cfg.rho for cfg in case_configs))
        exact_s = @elapsed exact = fm035d_exact_fields(cpu, ref.indices, rhos)
        Ucheck = fm035d_reference_matrix(ref.U, 3)
        Jcheck = fm035d_reference_matrix(ref.J, 9)
        urefrel = sqrt(norm2(exact.UR .- Ucheck)/max(norm2(Ucheck), eps()))
        jrefrel = sqrt(norm2(exact.JR .- Jcheck)/max(norm2(Jcheck), eps()))
        urefrel <= 5e-12 || error("independent U_R mismatch: $case n=$n rel=$urefrel")
        jrefrel <= 5e-12 || error("independent J_R mismatch: $case n=$n rel=$jrefrel")
        @printf("[exact] %s n=%d samples=%d %.2fs Ucheck=%.3e Jcheck=%.3e\n",
            case, n, ref.samples, exact_s, urefrel, jrefrel)
        if FM035D_EXACT_ONLY
            # CPU-only cutoff-curve rows (037b instrument): exact components
            # only, one tf=Float64 row per config, GPU/F columns left empty.
            den_u = sqrt(max(norm2(exact.UR), eps(Float64)))
            den_j = sqrt(max(norm2(exact.JR), eps(Float64)))
            for cfg in case_configs
                P_U, P_J = exact.partitioned[cfg.rho]
                ucut = sqrt(norm2(P_U .- exact.UR)) / den_u
                jcut = sqrt(norm2(P_J .- exact.JR)) / den_j
                sem = fm035d_component_semantics(cfg.kernel)
                row = (cfg.label, get(ENV,"SLURM_JOB_ID",""), gethostname(),
                    case, n, Float64, cfg.order+1, cfg.order, cfg.ell, cfg.q,
                    cfg.kernel, cfg.rho, something(cfg.rho_c, ""),
                    cfg.rectangular, cfg.gh_mode, sem, ref.checksum,
                    urefrel, jrefrel,
                    ucut, "", "", "", "", "", "", "",
                    jcut, "", "", "", "", "", "")
                println(io, join(row, ',')); flush(io)
                @printf("[exact-row] %s %s n=%d %s rho_t=%.4g U %s=%.3e J %s=%.3e\n",
                    cfg.label, case, n, cfg.kernel, cfg.rho, sem, ucut, sem, jcut)
            end
            continue
        end

        for cfg in case_configs
            P_U, P_J = exact.partitioned[cfg.rho]
            for TF in (Float32, Float64)
                gpu = fm035d_to_gpu(cpu)
                # task 037f: gh mode is read inside the lifecycle (graph-baked
                # at record time) — set BEFORE cache construction.  Note
                # :fp32/:reduced_fp32 are documented no-ops on the TF=Float32
                # row of a config (they equal :shipped/:reduced there).
                FM.CUDA_NEARFIELD_GH_MODE[] = cfg.gh_mode
                vpm.radix_fmm_settings!(gpu; expansion_order=cfg.order,
                    ell=cfg.ell, near_radius2=cfg.q, precision=TF,
                    direct_kernel=cfg.kernel, rho_t=cfg.rho, rho_c=cfg.rho_c,
                    m2l_strategy=:dense, rectangular=cfg.rectangular)
                vpm.UJ_fmm(gpu); CUDA.synchronize()
                A = Array(gpu.particles)
                FU = Float64.(A[vpm.U_INDEX, ref.indices])
                FJ = Float64.(A[vpm.J_INDEX, ref.indices])
                um = fm035d_metrics(FU, P_U, exact.UR)
                jm = fm035d_metrics(FJ, P_J, exact.JR)
                row = (cfg.label, get(ENV,"SLURM_JOB_ID",""), gethostname(), case,
                    n, TF, cfg.order+1, cfg.order, cfg.ell, cfg.q, cfg.kernel,
                    cfg.rho, something(cfg.rho_c, ""), cfg.rectangular,
                    cfg.gh_mode, fm035d_component_semantics(cfg.kernel),
                    ref.checksum,
                    urefrel, jrefrel, um.cutoff_rel, um.fmm_rel, um.total_rel,
                    um.triangle_rel, um.correlation, um.cancellation_ratio,
                    um.identity_rel, um.triangle_rel < FM035D_TOTAL_GATE,
                    jm.cutoff_rel,
                    jm.fmm_rel, jm.total_rel, jm.triangle_rel, jm.correlation,
                    jm.cancellation_ratio, jm.identity_rel)
                println(io, join(row, ',')); flush(io)
                @printf("[row] %s %s n=%d %s U cutoff=%.3e fmm=%.3e total=%.3e bound=%.3e\n",
                    cfg.label, case, n, TF, um.cutoff_rel, um.fmm_rel,
                    um.total_rel, um.triangle_rel)
                vpm.clear_radix_fmm_cache!(gpu)
                gpu = nothing; GC.gc(); CUDA.reclaim()
            end
        end
        cpu = nothing; GC.gc()
    end
end
println("035 error decomposition complete: $FM035D_OUT")
