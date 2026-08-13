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
# singular field is deliberately not formed.

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
const FM035D_COMPONENT_GATE = 5e-4
const FM035D_CASES = Tuple(split(get(ENV, "FM035D_CASES", "cube,wake"), ','))
const FM035D_NS = Tuple(parse.(Int, split(get(ENV, "FM035D_NS",
    "100000,1000000"), ',')))

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

# One O(samples*n) pass produces R, P(4.252), and P(3.668), for U and J.
function fm035d_exact_fields(pfield, indices)
    A = pfield.particles
    n = vpm.get_np(pfield)
    S = length(indices)
    UR = zeros(3, S); JR = zeros(9, S)
    UP4 = zeros(3, S); JP4 = zeros(9, S)
    UP5 = zeros(3, S); JP5 = zeros(9, S)
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
            rho <= 4.252 ?
                fm035d_addpair!(UP4, JP4, k, dx, dy, dz, r2, cr3,
                    cx, cy, cz, gx, gy, gz, g, h) :
                fm035d_addpair!(UP4, JP4, k, dx, dy, dz, r2, cr3,
                    cx, cy, cz, gx, gy, gz, 1.0, -3.0)
            rho <= 3.668 ?
                fm035d_addpair!(UP5, JP5, k, dx, dy, dz, r2, cr3,
                    cx, cy, cz, gx, gy, gz, g, h) :
                fm035d_addpair!(UP5, JP5, k, dx, dy, dz, r2, cr3,
                    cx, cy, cz, gx, gy, gz, 1.0, -3.0)
        end
    end
    return (; UR, JR, UP4, JP4, UP5, JP5)
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

const CONFIGS = [
    ("p4", 3, 4.252, "cube", 100000, 4, 17),
    ("p5", 4, 3.668, "cube", 100000, 4, 12),
    ("p4", 3, 4.252, "cube", 1000000, 5, 17),
    ("p5", 4, 3.668, "cube", 1000000, 5, 12),
    ("p4", 3, 4.252, "wake", 100000, 5, 12),
    ("p5", 4, 3.668, "wake", 100000, 5, 6),
    ("p4", 3, 4.252, "wake", 1000000, 6, 12),
    ("p5", 4, 3.668, "wake", 1000000, 6, 6),
]
const HEADER = ("label","job","host","case","n","tf","literature_P",
    "expansion_order","ell","q","rho_t","reference_checksum",
    "reference_u_rel","reference_j_rel","u_cutoff_rel","u_fmm_rel",
    "u_total_rel","u_triangle_rel","u_correlation","u_cancellation_ratio",
    "u_identity_rel","u_cutoff_pass","u_fmm_pass","j_cutoff_rel",
    "j_fmm_rel","j_total_rel","j_triangle_rel","j_correlation",
    "j_cancellation_ratio","j_identity_rel")

mkpath(dirname(FM035D_OUT))
open(FM035D_OUT, "w") do io
    println(io, join(HEADER, ','))
    for case in FM035D_CASES, n in FM035D_NS
        cpu = fm033_build(case, n)
        refpath = fm033_reference_path(FM035D_REFDIR, case, n)
        ref = fm033_read_reference(refpath, case, n, n)
        exact_s = @elapsed exact = fm035d_exact_fields(cpu, ref.indices)
        Ucheck = fm035d_reference_matrix(ref.U, 3)
        Jcheck = fm035d_reference_matrix(ref.J, 9)
        urefrel = sqrt(norm2(exact.UR .- Ucheck)/max(norm2(Ucheck), eps()))
        jrefrel = sqrt(norm2(exact.JR .- Jcheck)/max(norm2(Jcheck), eps()))
        urefrel <= 5e-12 || error("independent U_R mismatch: $case n=$n rel=$urefrel")
        jrefrel <= 5e-12 || error("independent J_R mismatch: $case n=$n rel=$jrefrel")
        @printf("[exact] %s n=%d samples=%d %.2fs Ucheck=%.3e Jcheck=%.3e\n",
            case, n, ref.samples, exact_s, urefrel, jrefrel)
        FM035D_EXACT_ONLY && continue

        for (label, order, rho, ccase, nn, ell, q) in CONFIGS
            ccase == case && nn == n || continue
            P_U, P_J = label == "p4" ? (exact.UP4, exact.JP4) : (exact.UP5, exact.JP5)
            for TF in (Float32, Float64)
                gpu = fm035d_to_gpu(cpu)
                vpm.radix_fmm_settings!(gpu; expansion_order=order, ell,
                    near_radius2=q, precision=TF, direct_kernel=:partitioned,
                    rho_t=rho, m2l_strategy=:dense)
                vpm.UJ_fmm(gpu); CUDA.synchronize()
                A = Array(gpu.particles)
                FU = Float64.(A[vpm.U_INDEX, ref.indices])
                FJ = Float64.(A[vpm.J_INDEX, ref.indices])
                um = fm035d_metrics(FU, P_U, exact.UR)
                jm = fm035d_metrics(FJ, P_J, exact.JR)
                row = (label, get(ENV,"SLURM_JOB_ID",""), gethostname(), case,
                    n, TF, order+1, order, ell, q, rho, ref.checksum,
                    urefrel, jrefrel, um.cutoff_rel, um.fmm_rel, um.total_rel,
                    um.triangle_rel, um.correlation, um.cancellation_ratio,
                    um.identity_rel, um.cutoff_rel <= FM035D_COMPONENT_GATE,
                    um.fmm_rel <= FM035D_COMPONENT_GATE, jm.cutoff_rel,
                    jm.fmm_rel, jm.total_rel, jm.triangle_rel, jm.correlation,
                    jm.cancellation_ratio, jm.identity_rel)
                println(io, join(row, ',')); flush(io)
                @printf("[row] %s %s n=%d %s U cutoff=%.3e fmm=%.3e total=%.3e bound=%.3e\n",
                    label, case, n, TF, um.cutoff_rel, um.fmm_rel,
                    um.total_rel, um.triangle_rel)
                vpm.clear_radix_fmm_cache!(gpu)
                gpu = nothing; GC.gc(); CUDA.reclaim()
            end
        end
        cpu = nothing; GC.gc()
    end
end
println("035 error decomposition complete: $FM035D_OUT")
