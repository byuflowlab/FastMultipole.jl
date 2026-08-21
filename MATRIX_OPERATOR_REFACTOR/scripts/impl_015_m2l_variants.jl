# =============================================================================
# 015 Axis-Swap (M2L variant) Benchmarks
# =============================================================================
#
# Isolated-operator benchmark of the two near-term full-M2L variants shipped by
# task 014, behind the single entry point `m2l_operator_batch!`:
#
#   * MaterializedYRotationM2L : rebuilds the Wigner Ts(theta) y-rotation from the
#       013 building blocks and applies it (materialized angle-dependent operator).
#   * FactoredRotationM2L      : explicit factored Z_phi -> S -> Z_theta -> S_inv
#       stages (013c Plain-H) with cached fixed per-degree mode matrices U_n/V_n.
#
# This is task 015's MEASUREMENT deliverable (no new operators). It compares the
# two variants against each other and against the CURRENT PRODUCTION recurrence
# M2L (rotation-trick `multipole_to_local!`), sweeping expansion order P, the
# Lamb-Helmholtz policy, and batch width, and records timing + allocation/storage
# footprint + a batch-composition (shared-direction / shared-norm) probe.
#
# It does NOT touch src/. CPU only. Actual GPU benchmarks are STAGED FOR LATER
# tasks 022 (device-resident GPU M2L) and 024 (definitive end-to-end CPU+GPU
# comparison) -- there is no GPU operator path yet, so 015's GPU recommendation is
# analytical (see 015-results.md), reasoning from the 008c GPU baseline.
#
# -----------------------------------------------------------------------------
# HOW TO RUN  (from the repository root)
# -----------------------------------------------------------------------------
# Like 008c, BLAS threading MUST be set at process start via the env var (runtime
# BLAS.set_num_threads() is unreliable for OpenBLAS). Run the script TWICE to get
# the single- vs multi-thread-CPU regimes:
#
#   Single-thread BLAS:
#     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
#       julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_015_m2l_variants.jl
#
#   Multi-thread BLAS (all cores):
#     OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu) OMP_NUM_THREADS=$(sysctl -n hw.ncpu) \
#       julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_015_m2l_variants.jl
#   (Linux: use $(nproc); MKL: MKL_NUM_THREADS)
#
# Override the sweeps from the shell if desired:
#     P_LIST="4,8,12" BATCH_LIST="1,16,256,4096" SAMPLES=200 \
#       julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_015_m2l_variants.jl
#
# Output is written to a MACHINE-TAGGED directory:
#     MATRIX_OPERATOR_REFACTOR/data/axis_swap/<hostname>/
#       env.md                          (environment + BLAS metadata)
#       m2l_variants_blas<N>.csv        (variant vs production timing sweep,
#                                        N = actual BLAS thread count)
#       footprint.csv                   (per-call allocation + cache/scratch bytes)
#       batch_composition_blas<N>.csv   (shared-direction / shared-norm probe)
#
# Self-contained: no plotting / BenchmarkTools deps.
# =============================================================================

using FastMultipole
using FastMultipole.LinearAlgebra
using FastMultipole.LinearAlgebra: BLAS
using FastMultipole.StaticArrays
using Statistics
using Printf
using Dates
using Random

const FM = FastMultipole

# -----------------------------------------------------------------------------
# Self-contained timing helper (identical policy to the 008c baseline): calibrate
# an inner eval count so even sub-microsecond ops are measurable, then report the
# minimum per-call time over `samples` repeats (min is robust to OS noise).
# `setup` runs (untimed) before each sample for in-place ops (e.g. zeroing the
# accumulated target buffer).
# -----------------------------------------------------------------------------
function timeit(f; setup=nothing, samples::Int=SAMPLES)
    setup === nothing || setup()
    f()  # warmup / compile
    evals = 1
    while true
        setup === nothing || setup()
        dt = @elapsed for _ in 1:evals; f(); end
        dt > 1e-4 && break
        evals *= 10
        evals > 10^8 && break
    end
    best = Inf
    for _ in 1:samples
        setup === nothing || setup()
        dt = @elapsed for _ in 1:evals; f(); end
        best = min(best, dt / evals)
    end
    return best
end

# ---- parameters (overridable via ENV) ---------------------------------------
_parse_int_list(s) = parse.(Int, split(s, ","))
function _parse_prec_list(s)
    map(split(s, ",")) do tok
        t = lowercase(strip(tok))
        t in ("float64", "f64", "double") ? Float64 :
        t in ("float32", "f32", "single") ? Float32 :
        error("unknown precision '$tok' (use Float64 / Float32)")
    end
end
const P_LIST     = haskey(ENV, "P_LIST")     ? _parse_int_list(ENV["P_LIST"])     : [4, 8, 12, 20]
const BATCH_LIST = haskey(ENV, "BATCH_LIST") ? _parse_int_list(ENV["BATCH_LIST"]) : [1, 8, 64, 512, 4096]
const SAMPLES    = haskey(ENV, "SAMPLES")    ? parse(Int, ENV["SAMPLES"])         : 50
const SEED       = 1234
# Precision sweep. Default Float64 (production precision); the operator pipeline
# is TF-parametric, so Float32 can be added via PREC_LIST="f64,f32" if desired.
const PREC_LIST  = haskey(ENV, "PREC_LIST")  ? _parse_prec_list(ENV["PREC_LIST"]) : [Float64]
# The production recurrence per-pair time is batch-independent, so cap how many
# pairs we actually time (running it B times at huge B is pointless).
const RECURRENCE_BATCH_CAP = haskey(ENV, "RECURRENCE_BATCH_CAP") ? parse(Int, ENV["RECURRENCE_BATCH_CAP"]) : 2048

# ---- batch-composition probe sweep (overridable via ENV) --------------------
# The probe sweeps order, batch size, the number of DISTINCT directions and norms
# present in the batch (capped at the batch size), and the Lamb-Helmholtz policy.
# It is a relative-shape probe (does per-expansion cost depend on angle/norm
# diversity?), so it uses fewer samples than the absolute timing sweep.
const COMP_P_LIST     = haskey(ENV, "COMP_P_LIST")     ? _parse_int_list(ENV["COMP_P_LIST"])     : [8, 20]
const COMP_BATCH_LIST = haskey(ENV, "COMP_BATCH_LIST") ? _parse_int_list(ENV["COMP_BATCH_LIST"]) : [64, 4096]
const DISTINCT_LIST   = haskey(ENV, "DISTINCT_LIST")   ? _parse_int_list(ENV["DISTINCT_LIST"])   : [1, 8, 64]
const COMP_SAMPLES    = haskey(ENV, "COMP_SAMPLES")    ? parse(Int, ENV["COMP_SAMPLES"])         : 15

# ---- output location (machine-tagged) ---------------------------------------
const HOST = gethostname()
const OUTDIR = normpath(joinpath(@__DIR__, "..", "data", "axis_swap", HOST))
mkpath(OUTDIR)

function progress(msg)
    println("[", Dates.format(Dates.now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

# -----------------------------------------------------------------------------
# Environment metadata + weak-BLAS detection (mirrors 008c).
# -----------------------------------------------------------------------------
function blas_description()
    libs = String[]
    try
        cfg = BLAS.get_config()
        for lib in cfg.loaded_libs
            push!(libs, basename(String(lib.libname)))
        end
    catch err
        push!(libs, "unknown ($(err))")
    end
    return libs
end

git_head() = try strip(read(`git rev-parse HEAD`, String)) catch; "unknown" end
git_dirty() = try !isempty(strip(read(`git status --porcelain`, String))) catch; false end

function blas_is_optimized(libs)
    known = ("openblas", "mkl", "blis", "accelerate", "veclib", "armpl")
    return any(l -> any(k -> occursin(k, lowercase(l)), known), libs)
end

function write_env(io)
    libs = blas_description()
    optimized = blas_is_optimized(libs)
    println(io, "# 015 axis-swap (M2L variant) benchmarks -- environment")
    println(io)
    println(io, "- date: ", Dates.now())
    println(io, "- hostname: ", HOST)
    println(io, "- julia: ", VERSION)
    cpu = Sys.cpu_info()
    println(io, "- cpu_model: ", isempty(cpu) ? "unknown" : cpu[1].model)
    println(io, "- physical/logical cores: ", Sys.CPU_THREADS)
    println(io, "- Threads.nthreads(): ", Threads.nthreads())
    println(io, "- BLAS.get_num_threads(): ", BLAS.get_num_threads())
    println(io, "- BLAS libs: ", join(libs, ", "))
    println(io, "- BLAS optimized?: ", optimized)
    if !optimized
        println(io)
        println(io, "> WARNING: no tuned BLAS detected. The materialized variant's dense GEMM")
        println(io, "> timings below are a LOWER bound on achievable performance. Re-run on a box")
        println(io, "> with a tuned BLAS before drawing the final CPU recommendation.")
    end
    println(io, "- git HEAD: ", git_head())
    println(io, "- git dirty: ", git_dirty())
    println(io, "- P_LIST: ", P_LIST)
    println(io, "- BATCH_LIST: ", BATCH_LIST)
    println(io, "- PREC_LIST: ", PREC_LIST)
    println(io, "- SAMPLES: ", SAMPLES)
    println(io, "- RECURRENCE_BATCH_CAP: ", RECURRENCE_BATCH_CAP)
    println(io, "- COMP_P_LIST: ", COMP_P_LIST)
    println(io, "- COMP_BATCH_LIST: ", COMP_BATCH_LIST)
    println(io, "- DISTINCT_LIST: ", DISTINCT_LIST)
    println(io, "- COMP_SAMPLES: ", COMP_SAMPLES)
    println(io)
    println(io, "> GPU: no GPU operator path exists yet (task 022 deferred). Actual GPU")
    println(io, "> benchmarks are staged for tasks 022 and 024; the 015 GPU recommendation is")
    println(io, "> analytical (see 015-results.md), reasoning from the 008c GPU baseline.")
    return optimized
end

# -----------------------------------------------------------------------------
# Make sure the production global precomputed tables are sized for max P (the
# production recurrence reference and the cross-check use them).
# -----------------------------------------------------------------------------
function ensure_globals!(Pmax)
    FM.update_Hs_π2!(FM.Hs_π2, Pmax)
    FM.update_ζs_mag!(FM.ζs_mag, Pmax)
    FM.update_ηs_mag!(FM.ηs_mag, Pmax)
    FM.update_M̃!(FM.M̃, Pmax)
    FM.update_L̃!(FM.L̃, Pmax)
    return nothing
end

# A representative non-axis offset (general direction; nonzero phi and theta).
const REF_OFFSET = SVector{3}(1.5, -2.0, 2.5)

# -----------------------------------------------------------------------------
# Physical random source [2,2,nh,B] (m = 0 rows have zero imaginary part, as every
# real expansion does -- required for FactoredRotationM2L's rank-1 mode path).
# -----------------------------------------------------------------------------
function fill_physical_source!(sources, P, ::Val{LH}) where {LH}
    TF = eltype(sources)
    B = size(sources, 4)
    fill!(sources, zero(TF))
    for j in 1:B, n in 0:P, m in 0:n
        i = FM.harmonic_index(n, m)
        sources[1, 1, i, j] = randn(TF)
        sources[2, 1, i, j] = m == 0 ? zero(TF) : randn(TF)
        if LH
            sources[1, 2, i, j] = randn(TF)
            sources[2, 2, i, j] = m == 0 ? zero(TF) : randn(TF)
        end
    end
    return sources
end

# -----------------------------------------------------------------------------
# Production recurrence reference: a closure running ONE full M2L for a fixed pair
# (the rotation-trick `multipole_to_local!`). All work buffers preallocated, so
# only the operator work is timed. The per-pair cost is angle/distance independent
# in shape, so a single representative offset is sufficient.
# -----------------------------------------------------------------------------
function make_production_ref(P, TF, ::Val{LH}, Δx) where {LH}
    box = SVector{3}(zero(TF), zero(TF), zero(TF))
    src_branch = FM.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(zero(TF), zero(TF), zero(TF)), zero(TF), box)
    tgt_branch = FM.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(Δx), zero(TF), box)
    Hs = TF[1.0]; FM.update_Hs_π2!(Hs, P)
    Ts = zeros(TF, FM.length_Ts(P))
    eimϕs = zeros(TF, 2, P + 1)
    w1 = FM.initialize_expansion(P, TF)
    w2 = FM.initialize_expansion(P, TF)
    w3 = FM.initialize_expansion(P, TF)
    ζ = zeros(TF, FM.length_ζs(P)); FM.update_ζs_mag!(ζ, 0, P)
    η = zeros(TF, FM.length_ηs(P)); FM.update_ηs_mag!(η, 0, P)
    Mt = zeros(TF, length(FM.M̃)); copyto!(Mt, FM.M̃)
    Lt = zeros(TF, length(FM.L̃)); copyto!(Lt, FM.L̃)
    src = FM.initialize_expansion(P, TF)
    for n in 0:P, m in 0:n
        i = FM.harmonic_index(n, m)
        src[1, 1, i] = randn(TF); src[2, 1, i] = m == 0 ? zero(TF) : randn(TF)
        if LH
            src[1, 2, i] = randn(TF); src[2, 2, i] = m == 0 ? zero(TF) : randn(TF)
        end
    end
    local_exp = FM.initialize_expansion(P, TF)
    lh = Val(LH)
    f = function ()
        FM.multipole_to_local!(local_exp, tgt_branch, src, src_branch,
            w1, w2, w3, Ts, eimϕs, ζ, η, Hs, FM.M̃, FM.L̃, P, lh)
        return nothing
    end
    return f
end

# -----------------------------------------------------------------------------
# Correctness gate: confirm the operators we time are the parity-validated ones.
# Compares one batch column of MaterializedYRotationM2L against the production
# `multipole_to_local!` for the reference offset (Val(false)). Aborts on mismatch
# so we never publish timings for a broken build.
# -----------------------------------------------------------------------------
function sanity_check(P, TF)
    lh = Val(false)
    cache = OperatorInvariantCache(TF, P, lh)
    nh = ((P + 1) * (P + 2)) >> 1
    scratch = M2LOperatorScratch(TF, cache.basis_info, 1)
    src = FM.initialize_expansion(P, TF)
    for n in 0:P, m in 0:n
        i = FM.harmonic_index(n, m)
        src[1, 1, i] = randn(TF); src[2, 1, i] = m == 0 ? zero(TF) : randn(TF)
    end
    sources = zeros(TF, 2, 2, nh, 1); sources[:, :, :, 1] .= src
    r, θ, ϕ = FM.cartesian_to_spherical(REF_OFFSET)
    targets = zeros(TF, 2, 2, nh, 1)
    FM.m2l_operator_batch!(MaterializedYRotationM2L(), targets, sources, [ϕ], [θ], [r], cache, scratch, lh)

    # production reference for this exact source/offset
    box = SVector{3}(zero(TF), zero(TF), zero(TF))
    src_branch = FM.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(zero(TF), zero(TF), zero(TF)), zero(TF), box)
    tgt_branch = FM.Branch(2:2, 0, 1:0, 0, 1, SVector{3}(REF_OFFSET), zero(TF), box)
    Hs = TF[1.0]; FM.update_Hs_π2!(Hs, P)
    Ts = zeros(TF, FM.length_Ts(P)); eimϕs = zeros(TF, 2, P + 1)
    w1 = FM.initialize_expansion(P, TF); w2 = FM.initialize_expansion(P, TF); w3 = FM.initialize_expansion(P, TF)
    ζ = zeros(TF, FM.length_ζs(P)); FM.update_ζs_mag!(ζ, 0, P)
    η = zeros(TF, FM.length_ηs(P)); FM.update_ηs_mag!(η, 0, P)
    ref = FM.initialize_expansion(P, TF)
    FM.multipole_to_local!(ref, tgt_branch, src, src_branch, w1, w2, w3, Ts, eimϕs, ζ, η, Hs, FM.M̃, FM.L̃, P, lh)

    maxerr = 0.0
    for i in 1:nh, c in 1:2
        maxerr = max(maxerr, abs(targets[c, 1, i, 1] - ref[c, 1, i]))
    end
    if maxerr > 1e-6
        error("015 sanity check FAILED at P=$P: max |operator - production| = $maxerr (> 1e-6). Aborting; timings would be meaningless.")
    end
    progress("sanity check OK at P=$P (max abs diff vs production = $(@sprintf("%.2e", maxerr)))")
    return nothing
end

# -----------------------------------------------------------------------------
# (1) Variant-vs-production timing sweep.
# -----------------------------------------------------------------------------
function bench_variants(io, blas_threads)
    println(io, "variant,form,precision,blas_threads,P,lamb_helmholtz,batch,measured_batch,scaled,seconds,seconds_per_expansion")
    variants = (("materialized", MaterializedYRotationM2L()), ("factored", FactoredRotationM2L()))
    for T in PREC_LIST
        for P in P_LIST
            for LHbool in (false, true)
                lh = Val(LHbool)
                progress("variants: precision=$(T) P=$P lamb_helmholtz=$LHbool")
                cache = OperatorInvariantCache(T, P, lh)
                P_active = cache.basis_info.orders.P_active
                nh = ((P_active + 1) * (P_active + 2)) >> 1
                r, θ, ϕ = FM.cartesian_to_spherical(SVector{3}(T.(REF_OFFSET)))

                for B in BATCH_LIST
                    scratch = M2LOperatorScratch(T, cache.basis_info, B)
                    sources = zeros(T, 2, 2, nh, B)
                    fill_physical_source!(sources, P, lh)
                    targets = zeros(T, 2, 2, nh, B)
                    phis = fill(T(ϕ), B); thetas = fill(T(θ), B); rs = fill(T(r), B)

                    for (vname, vop) in variants
                        tt = timeit(() -> FM.m2l_operator_batch!(vop, targets, sources, phis, thetas, rs, cache, scratch, lh);
                                    setup=() -> fill!(targets, zero(T)))
                        @printf(io, "%s,batched_operator,%s,%d,%d,%s,%d,%d,false,%.6e,%.6e\n",
                                vname, string(T), blas_threads, P, string(LHbool), B, B, tt, tt / B)
                    end
                    flush(io)
                end

                # production recurrence reference (Float64 only -- production precision;
                # per-pair cost is batch independent, so cap and scale like 008c).
                if T === Float64
                    ref = make_production_ref(P, T, lh, SVector{3}(REF_OFFSET))
                    for B in BATCH_LIST
                        Brec = min(B, RECURRENCE_BATCH_CAP)
                        scaled = Brec != B
                        tr = timeit(() -> (for _ in 1:Brec; ref(); end))
                        tr_pe = tr / Brec
                        @printf(io, "production_recurrence,recurrence,Float64,%d,%d,%s,%d,%d,%s,%.6e,%.6e\n",
                                blas_threads, P, string(LHbool), B, Brec, string(scaled), tr_pe * B, tr_pe)
                    end
                    flush(io)
                end
            end
        end
    end
end

# -----------------------------------------------------------------------------
# (2) Allocation / storage footprint.
#   call_bytes    : @allocated for one steady-state m2l_operator_batch! call
#                   (expected ~0 -- scratch is preallocated; nonzero is a finding).
#   cache_bytes   : Base.summarysize(OperatorInvariantCache)  (per (P, lh))
#   scratch_bytes : Base.summarysize(M2LOperatorScratch)      (per (P, lh, batch))
# -----------------------------------------------------------------------------
function bench_footprint(io)
    println(io, "variant,precision,P,lamb_helmholtz,batch,call_bytes,cache_bytes,scratch_bytes")
    T = Float64
    variants = (("materialized", MaterializedYRotationM2L()), ("factored", FactoredRotationM2L()))
    for P in P_LIST
        for LHbool in (false, true)
            lh = Val(LHbool)
            cache = OperatorInvariantCache(T, P, lh)
            cache_bytes = Base.summarysize(cache)
            P_active = cache.basis_info.orders.P_active
            nh = ((P_active + 1) * (P_active + 2)) >> 1
            r, θ, ϕ = FM.cartesian_to_spherical(REF_OFFSET)
            for B in BATCH_LIST
                scratch = M2LOperatorScratch(T, cache.basis_info, B)
                scratch_bytes = Base.summarysize(scratch)
                sources = zeros(T, 2, 2, nh, B); fill_physical_source!(sources, P, lh)
                targets = zeros(T, 2, 2, nh, B)
                phis = fill(T(ϕ), B); thetas = fill(T(θ), B); rs = fill(T(r), B)
                for (vname, vop) in variants
                    FM.m2l_operator_batch!(vop, targets, sources, phis, thetas, rs, cache, scratch, lh)  # warm
                    fill!(targets, zero(T))
                    call_bytes = @allocated FM.m2l_operator_batch!(vop, targets, sources, phis, thetas, rs, cache, scratch, lh)
                    @printf(io, "%s,%s,%d,%s,%d,%d,%d,%d\n",
                            vname, string(T), P, string(LHbool), B, call_bytes, cache_bytes, scratch_bytes)
                end
                flush(io)
            end
        end
    end
end

# -----------------------------------------------------------------------------
# (3) Batch-composition probe (offset-class / shared-direction / shared-norm).
# Sweeps order, batch size, the number of DISTINCT directions (phi,theta) and the
# number of DISTINCT norms r present in the batch, and the Lamb-Helmholtz policy.
# Directions/norms are drawn from pools of `Bmax` genuinely distinct offsets, and a
# requested count `n` is realized EXACTLY as `pool[mod1(j, n)]` (so `n_dir`/`n_norm`
# in the CSV are the true distinct counts present, never the batch size as a label).
# `n_offset_class` is the actual number of distinct (direction, norm) pairs realized
# across the batch -- the requested offset-class-count axis. The current API
# recomputes per column, so per-expansion time is expected to be independent of all
# three diversity axes; recording that is itself the finding that a future
# shared-angle optimization (one materialized operator per offset class) is
# unrealized headroom.
# -----------------------------------------------------------------------------
# distinct-count levels for a batch B: the requested levels that are <= B, plus the
# fully-shared (1) and fully-distinct (B) endpoints.
function distinct_levels(B)
    levels = filter(<=(B), DISTINCT_LIST)
    return sort(unique(vcat(1, levels, B)))
end

function bench_batch_composition(io, blas_threads)
    println(io, "variant,precision,P,lamb_helmholtz,batch,n_dir,n_norm,n_offset_class,seconds,seconds_per_expansion")
    T = Float64
    variants = (("materialized", MaterializedYRotationM2L()), ("factored", FactoredRotationM2L()))

    # Genuinely distinct direction/norm pools, sized to the largest batch, drawn from
    # a dedicated deterministic RNG so re-runs reproduce the same offsets.
    Bmax = maximum(COMP_BATCH_LIST)
    rng = MersenneTwister(SEED + 7)
    dirs = Vector{Tuple{T,T}}(undef, Bmax)   # (theta, phi)
    norms = Vector{T}(undef, Bmax)           # r
    for k in 1:Bmax
        x = SVector{3}(randn(rng, T), randn(rng, T), randn(rng, T))
        r, θ, ϕ = FM.cartesian_to_spherical(x)
        dirs[k] = (θ, ϕ)
        norms[k] = one(T) + 3 * rand(rng, T)   # distinct positive distances
    end

    for P in COMP_P_LIST
        for LHbool in (false, true)
            lh = Val(LHbool)
            cache = OperatorInvariantCache(T, P, lh)
            P_active = cache.basis_info.orders.P_active
            nh = ((P_active + 1) * (P_active + 2)) >> 1
            for B in COMP_BATCH_LIST
                progress("batch composition: P=$P lamb_helmholtz=$LHbool batch=$B")
                scratch = M2LOperatorScratch(T, cache.basis_info, B)
                sources = zeros(T, 2, 2, nh, B); fill_physical_source!(sources, P, lh)
                targets = zeros(T, 2, 2, nh, B)
                phis = Vector{T}(undef, B); thetas = Vector{T}(undef, B); rs = Vector{T}(undef, B)

                levels = distinct_levels(B)
                for n_dir in levels, n_norm in levels
                    classes = Set{Tuple{Int,Int}}()
                    for j in 1:B
                        di = mod1(j, n_dir); ni = mod1(j, n_norm)
                        thetas[j], phis[j] = dirs[di]
                        rs[j] = norms[ni]
                        push!(classes, (di, ni))
                    end
                    n_offset_class = length(classes)
                    for (vname, vop) in variants
                        tt = timeit(() -> FM.m2l_operator_batch!(vop, targets, sources, phis, thetas, rs, cache, scratch, lh);
                                    setup=() -> fill!(targets, zero(T)), samples=COMP_SAMPLES)
                        @printf(io, "%s,%s,%d,%s,%d,%d,%d,%d,%.6e,%.6e\n",
                                vname, string(T), P, string(LHbool), B, n_dir, n_norm, n_offset_class, tt, tt / B)
                    end
                end
                flush(io)
            end
        end
    end
end

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
function main()
    Random.seed!(SEED)
    ensure_globals!(maximum(P_LIST))

    progress("Writing 015 benchmarks to: $OUTDIR")
    progress("Sweep: P_LIST=$P_LIST BATCH_LIST=$BATCH_LIST PREC_LIST=$PREC_LIST SAMPLES=$SAMPLES")

    optimized = open(joinpath(OUTDIR, "env.md"), "w") do io
        write_env(io)
    end
    progress("Wrote environment metadata: $(joinpath(OUTDIR, "env.md"))")
    optimized || @warn "No tuned BLAS detected -- materialized dense GEMM numbers are a lower bound (see env.md)."

    # correctness gate before any timing
    sanity_check(minimum(P_LIST), Float64)
    sanity_check(maximum(P_LIST), Float64)

    blas_threads = BLAS.get_num_threads()

    fname = "m2l_variants_blas$(blas_threads).csv"
    progress("[1/3] variant-vs-production timing sweep (BLAS threads = $blas_threads) ...")
    open(joinpath(OUTDIR, fname), "w") do io
        bench_variants(io, blas_threads)
    end
    progress("[1/3] wrote $fname")

    progress("[2/3] allocation / storage footprint ...")
    open(joinpath(OUTDIR, "footprint.csv"), "w") do io
        bench_footprint(io)
    end
    progress("[2/3] wrote footprint.csv")

    cname = "batch_composition_blas$(blas_threads).csv"
    progress("[3/3] batch-composition probe ...")
    open(joinpath(OUTDIR, cname), "w") do io
        bench_batch_composition(io, blas_threads)
    end
    progress("[3/3] wrote $cname")

    progress("Done. Results in: $OUTDIR")
end

main()
