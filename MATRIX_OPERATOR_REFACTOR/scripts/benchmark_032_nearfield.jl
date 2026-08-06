# Task 032 stage 2, H200 nearfield benchmarks (benchmark-only, no src changes):
#
#   A. Functor-abstraction cost (031 sign-off (b)): the generic
#      `_cuda_direct_pairs_functor_kernel!` instantiated with `SingularSource()`
#      vs the shipped hard-coded `_cuda_direct_pairs_output_kernel!`, on the 028
#      scalar workload (n = 1e6, shipped hierarchical defaults, ell = 5), both
#      output widths (4/13 rows), Float32 and Float64.
#   B. Vortex kernel ladder at fixed adequate geometry (ell = 4, near_radius2 =
#      12, sigma = beta * n^{-1/3} with beta = 2, so g_min*h_leaf > rho_t*sigma):
#      hard-coded singular vortex vs functor SingularVortex vs the shipped
#      erf-free RegularizedVortex vs a script-local FDLIBM `custom_erf` port of
#      the same regularized kernel (CustomErfVortex). Decides the 032 task-file
#      "benchmark erf-free vs custom_erf, ship the faster at equal accuracy"
#      question. Host-side accuracy of both g/h forms is printed against a
#      BigFloat reference alongside the timings.
#
# Timing method: median CUDA.@elapsed over FM032_REPS launches after warmup,
# production launch geometry (warp-per-pair, DIRECT_CUDA_MAX_BLOCKS cap).
# Output: data/feasibility_1m_10ms/nearfield032_<host>_<stamp>.csv

using FastMultipole
using FastMultipole.StaticArrays
using Statistics
using Dates
using Printf
using Random

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

FastMultipole.load_cuda_radix_lifecycle!() ||
    error("CUDA radix lifecycle failed to load: $(FastMultipole.cuda_radix_status())")
using CUDA

const FM = FastMultipole
const N = parse(Int, get(ENV, "FM032_N", "1000000"))
const REPS = parse(Int, get(ENV, "FM032_REPS", "9"))
const SEED = 24025
const STAMP = Dates.format(now(), "yyyymmdd-HHMMSS")
const OUTDIR = get(ENV, "FM032_OUTDIR", joinpath(@__DIR__, "..", "data",
    "feasibility_1m_10ms"))
const JOBID = get(ENV, "SLURM_JOB_ID", "")
const CSVPATH = joinpath(OUTDIR, "nearfield032_$(gethostname())_$(STAMP).csv")

function median_gpu_ms(f, reps=REPS)
    f(); CUDA.synchronize()
    samples = Float64[]
    for _ in 1:reps
        push!(samples, Float64(CUDA.@elapsed f()) * 1e3)
    end
    return median(samples), minimum(samples), maximum(samples)
end

function write_csv(path, rows)
    isempty(rows) && return
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            println(io, join(string.(values(row)), ','))
        end
    end
end

#------- script-local systems -------#

# 028-equivalent scalar bodies (rows 1:3 pos, 4 radius, 5 strength)
struct BM032Scalar{TF}
    bodies::Matrix{TF}
end
FM.get_n_bodies(s::BM032Scalar) = size(s.bodies, 2)
FM.data_per_body(::BM032Scalar) = 5
FM.strength_dims(::BM032Scalar) = 1
FM.get_position(s::BM032Scalar{TF}, i) where TF =
    SVector{3,TF}(s.bodies[1, i], s.bodies[2, i], s.bodies[3, i])
FM.has_vector_potential(::BM032Scalar) = false
function FM.source_system_to_buffer!(buffer, i_buffer, s::BM032Scalar, i_body)
    for r in 1:5
        buffer[r, i_buffer] = s.bodies[r, i_body]
    end
    return nothing
end
FM.buffer_to_target_system!(s::BM032Scalar, i_target, switch, buffer, i_buffer) = nothing

# vortex bodies with sigma in extra row 8 (rows 5:7 Gamma, 4 MAC radius)
struct BM032Vortex{TF}
    bodies::Matrix{TF}     # 8 x n
end
FM.get_n_bodies(s::BM032Vortex) = size(s.bodies, 2)
FM.data_per_body(::BM032Vortex) = 8
FM.strength_dims(::BM032Vortex) = 3
FM.get_position(s::BM032Vortex{TF}, i) where TF =
    SVector{3,TF}(s.bodies[1, i], s.bodies[2, i], s.bodies[3, i])
FM.has_vector_potential(::BM032Vortex) = true
FM.body_type(::BM032Vortex) = Point{Vortex}
FM.direct_kernel(::BM032Vortex) = RegularizedVortex(; sigma_row=8)
function FM.source_system_to_buffer!(buffer, i_buffer, s::BM032Vortex, i_body)
    for r in 1:8
        buffer[r, i_buffer] = s.bodies[r, i_body]
    end
    return nothing
end
FM.buffer_to_target_system!(s::BM032Vortex, i_target, switch, buffer, i_buffer) = nothing

#------- script-local FDLIBM custom_erf port (candidate, not shipped) -------#
# Verbatim rational-polynomial structure from ../FLOWVPM.jl/src/FLOWVPM_gpu_erf.jl,
# made type-generic in TF (the FLOWVPM original closes over per-precision consts).

@inline function bm_custom_erf(x::T) where T<:AbstractFloat
    xabs = abs(x)
    sgn = sign(x)
    val = sgn * one(T)
    if xabs < T(0.84375)
        z = x * x
        r = muladd(z, muladd(z, muladd(z, muladd(z, T(-2.37630166566501626084e-05),
            T(-5.77027029648944159157e-03)), T(-2.84817495755985104766e-02)),
            T(-3.25042107247001499370e-01)), T(1.28379167095512558561e-01))
        s = muladd(z, muladd(z, muladd(z, muladd(z, muladd(z,
            T(-3.96022827877536812320e-06), T(1.32494738004321644526e-04)),
            T(5.08130628187576562776e-03)), T(6.50222499887672944485e-02)),
            T(3.97917223959155352819e-01)), one(T))
        val = sgn * (xabs + xabs * (r / s))
    elseif xabs < T(1.25)
        s = xabs - one(T)
        P = muladd(s, muladd(s, muladd(s, muladd(s, muladd(s, muladd(s,
            T(-2.16637559486879084300e-03), T(3.54783043256182359371e-02)),
            T(-1.10894694282396677476e-01)), T(3.18346619901161753674e-01)),
            T(-3.72207876035701323847e-01)), T(4.14856118683748331666e-01)),
            T(-2.36211856075265944077e-03))
        Q = muladd(s, muladd(s, muladd(s, muladd(s, muladd(s, muladd(s,
            T(1.19844998467991074170e-02), T(1.36370839120290507362e-02)),
            T(1.26171219808761642112e-01)), T(7.18286544141962662868e-02)),
            T(5.40397917702171048937e-01)), T(1.06420880400844228286e-01)), one(T))
        val = sgn * (T(8.45062911510467529297e-01) + P / Q)
    elseif xabs < T(2.857142857142857)
        s = one(T) / (x * x)
        R = muladd(s, muladd(s, muladd(s, muladd(s, muladd(s, muladd(s, muladd(s,
            T(-9.81432934416914548592e+00), T(-8.12874355063065934246e+01)),
            T(-1.84605092906711035994e+02)), T(-1.62396669462573470355e+02)),
            T(-6.23753324503260060396e+01)), T(-1.05586262253232909814e+01)),
            T(-6.93858572707181764372e-01)), T(-9.86494403484714822705e-03))
        S = muladd(s, muladd(s, muladd(s, muladd(s, muladd(s, muladd(s, muladd(s,
            muladd(s, T(-6.04244152148580987438e-02), T(6.57024977031928170135e+00)),
            T(1.08635005541779435134e+02)), T(4.29008140027567833386e+02)),
            T(6.45387271733267880336e+02)), T(4.34565877475229228821e+02)),
            T(1.37657754143519042600e+02)), T(1.96512716674392571292e+01)), one(T))
        r = exp(-x * x - T(0.5625) + R / S)
        val = sgn * (one(T) - r / xabs)
    elseif xabs < T(6)
        s = one(T) / (x * x)
        R = muladd(s, muladd(s, muladd(s, muladd(s, muladd(s, muladd(s,
            T(-4.83519191608651397019e+02), T(-1.02509513161107724954e+03)),
            T(-6.37566443368389627722e+02)), T(-1.60636384855821916062e+02)),
            T(-1.77579549177547519889e+01)), T(-7.99283237680523006574e-01)),
            T(-9.86494292470009928597e-03))
        S = muladd(s, muladd(s, muladd(s, muladd(s, muladd(s, muladd(s, muladd(s,
            T(-2.24409524465858183362e+01), T(4.74528541206955367215e+02)),
            T(2.55305040643316442583e+03)), T(3.19985821950859553908e+03)),
            T(1.53672958608443695994e+03)), T(3.25792512996573918826e+02)),
            T(3.03380607434824582924e+01)), one(T))
        r = exp(-x * x - T(0.5625) + R / S)
        val = sgn * (one(T) - r / xabs)
    end
    return val
end

# Regularized kernel candidate using the erf port: series below the theory-§3
# switch rho = 0.5 (cancellation safety), custom_erf + naive h above (the
# FLOWVPM structure).
struct CustomErfVortex <: FM.AbstractDirectKernel
    sigma_row::Int
end
FM._emits_potential(::CustomErfVortex) = false

@inline function bm_erf_g_h(rho::T) where T<:AbstractFloat
    if rho <= T(0.5)
        return FM._gaussianerf_g_h(rho)     # identical series in both candidates
    end
    A = T(FM._GAUSSERF_A)
    e = exp(-rho * rho / 2)
    g = bm_custom_erf(rho / sqrt(T(2))) - A * rho * e
    return g, A * rho * rho * rho * e - 3 * g
end

@inline function FM._direct_pair_ug(kernel::CustomErfVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    if sigma > zero(T)
        g, _ = bm_erf_g_h(r2 * invr / sigma)
    end
    return FM._vortex_pair_ug(dx, dy, dz, invr, gsx, gsy, gsz, g)
end

@inline function FM._direct_pair_ugh(kernel::CustomErfVortex, dx, dy, dz, r2, invr,
        source_bodies, j)
    T = typeof(r2)
    @inbounds gsx = source_bodies[5, j]
    @inbounds gsy = source_bodies[6, j]
    @inbounds gsz = source_bodies[7, j]
    @inbounds sigma = source_bodies[kernel.sigma_row, j]
    g = one(T)
    h = -T(3)
    if sigma > zero(T)
        g, h = bm_erf_g_h(r2 * invr / sigma)
    end
    return FM._vortex_pair_ugh(dx, dy, dz, r2, invr, gsx, gsy, gsz, g, h)
end

#------- host-side accuracy of the two g/h forms (BigFloat reference) -------#

setprecision(BigFloat, 256)
function gh_big(rho::Float64)
    rb = big(rho)
    x = rb / sqrt(big(2))
    s = zero(BigFloat); term = x; n = 0
    while true
        add = term / (2n + 1); s += add; n += 1; term *= -x * x / n
        abs(add) <= eps(BigFloat) * max(abs(s), one(BigFloat)) && break
    end
    erfb = 2 / sqrt(big(pi)) * s
    Ab = sqrt(big(2) / big(pi))
    gb = erfb - Ab * rb * exp(-rb * rb / 2)
    return Float64(gb), Float64(Ab * rb^3 * exp(-rb * rb / 2) - 3 * gb)
end

println("--- host accuracy of the two g/h forms (abs err vs BigFloat, rho in (0, 6]) ---")
for TF in (Float32, Float64)
    e_free_g = 0.0; e_free_h = 0.0; e_erf_g = 0.0; e_erf_h = 0.0
    for rho in range(1e-3, 6.0, length=6001)
        gb, hb = gh_big(rho)
        g1, h1 = FM._gaussianerf_g_h(TF(rho))
        g2, h2 = bm_erf_g_h(TF(rho))
        e_free_g = max(e_free_g, abs(Float64(g1) - gb))
        e_free_h = max(e_free_h, abs(Float64(h1) - hb))
        e_erf_g = max(e_erf_g, abs(Float64(g2) - gb))
        e_erf_h = max(e_erf_h, abs(Float64(h2) - hb))
    end
    @printf("%s  erf-free: |dg|=%.2e |dh|=%.2e   custom_erf: |dg|=%.2e |dh|=%.2e\n",
        TF, e_free_g, e_free_h, e_erf_g, e_erf_h)
end

#------- benchmark drivers -------#

rows = Any[]

function bench_direct_kernels!(rows, label, state, npairs, entries)
    threads = 128
    blocks = min(cld(npairs, threads ÷ 32), FM.DIRECT_CUDA_MAX_BLOCKS[])
    for (name, launch) in entries
        med, lo, hi = median_gpu_ms(launch)
        push!(rows, (; job=JOBID, workload=label, kernel=name,
            precision=string(eltype(state.output)), n=N, pairs=npairs,
            blocks, median_ms=med, min_ms=lo, max_ms=hi))
        @printf("%-10s %-24s %8.3f ms  (min %.3f)\n", label, name, med, lo)
    end
end

# A. scalar functor A/B at the 028 shipped defaults (ell = 5)
for TF in (Float32, Float64), hessian in (false, true)
    Random.seed!(SEED)
    bodies = rand(8, N)
    bodies[4, :] .= 0
    bodies[5, :] ./= N
    sys = BM032Scalar(Matrix{TF}(bodies[1:5, :]))
    cache = RadixFMMCache(sys; expansion_order=3, ell=5, hessian,
        bounds=(SVector{3,TF}(-0.01, -0.01, -0.01), TF(1.02)), device=true,
        options=CUDARadixLifecycleOptions(; precision=TF,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    state = cache.state
    npairs = state.counts.n_direct
    hsv = hessian ? Val(true) : Val(false)
    threads = 128
    blocks = min(cld(npairs, threads ÷ 32), FM.DIRECT_CUDA_MAX_BLOCKS[])
    label = "scalar-" * (hessian ? "h13" : "h4")
    bench_direct_kernels!(rows, label, state, npairs, (
        ("hardcoded", () -> CUDA.@cuda threads=threads blocks=blocks FM._cuda_direct_pairs_output_kernel!(
            state.output, state.source_bodies, state.cell_ranges,
            state.direct_targets, state.direct_sources, npairs, hsv)),
        ("functor_singular", () -> CUDA.@cuda threads=threads blocks=blocks FM._cuda_direct_pairs_functor_kernel!(
            SingularSource(), state.output, state.source_bodies, state.cell_ranges,
            state.direct_targets, state.direct_sources, npairs, hsv)),
    ))
    CUDA.reclaim()
end

# B. vortex ladder at fixed adequate geometry (ell = 4, q = 12, beta = 2)
for TF in (Float32, Float64)
    Random.seed!(SEED)
    n3 = N^(1 / 3)
    sigma = 2 / n3                       # beta = 2 overlap, uniform
    bodies = zeros(8, N)
    bodies[1:3, :] .= rand(3, N)
    bodies[4, :] .= 4.789 * sigma        # MAC radius (inflated, FLOWVPM-style)
    bodies[5:7, :] .= randn(3, N) ./ N
    bodies[8, :] .= sigma
    sys = BM032Vortex(Matrix{TF}(bodies))
    cache = RadixFMMCache(sys; expansion_order=3, ell=4, hessian=true,
        near_radius2=12,
        bounds=(SVector{3,TF}(-0.01, -0.01, -0.01), TF(1.02)), device=true,
        options=CUDARadixLifecycleOptions(; precision=TF,
            m2l_strategy=FastMultipole.ConcatenatedFixedZM2L()))
    state = cache.state
    npairs = state.counts.n_direct
    threads = 128
    blocks = min(cld(npairs, threads ÷ 32), FM.DIRECT_CUDA_MAX_BLOCKS[])
    hsv = Val(true)
    reg = RegularizedVortex(; sigma_row=8)
    cerf = CustomErfVortex(8)
    bench_direct_kernels!(rows, "vortex-h13", state, npairs, (
        ("hardcoded_vortex", () -> CUDA.@cuda threads=threads blocks=blocks FM._cuda_direct_pairs_vortex_kernel!(
            state.output, state.source_bodies, state.cell_ranges,
            state.direct_targets, state.direct_sources, npairs, hsv)),
        ("functor_singular_vortex", () -> CUDA.@cuda threads=threads blocks=blocks FM._cuda_direct_pairs_functor_kernel!(
            SingularVortex(), state.output, state.source_bodies, state.cell_ranges,
            state.direct_targets, state.direct_sources, npairs, hsv)),
        ("regularized_erffree", () -> CUDA.@cuda threads=threads blocks=blocks FM._cuda_direct_pairs_functor_kernel!(
            reg, state.output, state.source_bodies, state.cell_ranges,
            state.direct_targets, state.direct_sources, npairs, hsv)),
        ("regularized_customerf", () -> CUDA.@cuda threads=threads blocks=blocks FM._cuda_direct_pairs_functor_kernel!(
            cerf, state.output, state.source_bodies, state.cell_ranges,
            state.direct_targets, state.direct_sources, npairs, hsv)),
    ))
    # numerical agreement of the two regularized candidates on device
    fill!(state.output, zero(TF))
    CUDA.@cuda threads=threads blocks=blocks FM._cuda_direct_pairs_functor_kernel!(
        reg, state.output, state.source_bodies, state.cell_ranges,
        state.direct_targets, state.direct_sources, npairs, hsv)
    CUDA.synchronize()
    out_free = Array(state.output)
    fill!(state.output, zero(TF))
    CUDA.@cuda threads=threads blocks=blocks FM._cuda_direct_pairs_functor_kernel!(
        cerf, state.output, state.source_bodies, state.cell_ranges,
        state.direct_targets, state.direct_sources, npairs, hsv)
    CUDA.synchronize()
    out_erf = Array(state.output)
    scale = maximum(abs.(out_erf))
    @printf("%s device candidate agreement: max|erffree - customerf| = %.3e (rel %.3e)\n",
        TF, maximum(abs.(out_free .- out_erf)), maximum(abs.(out_free .- out_erf)) / scale)
    CUDA.reclaim()
end

write_csv(CSVPATH, rows)
println("wrote ", CSVPATH)
