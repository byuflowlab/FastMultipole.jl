# =============================================================================
# 008c Implementation Performance Baseline -- CPU
# =============================================================================
#
# Pre-implementation performance + allocation baseline for the Matrix Operator
# Refactor (task 008c). This script:
#
#   1. Records reproducible environment metadata (Julia, CPU, threads, BLAS
#      vendor/threads, git HEAD) and FLAGS a weak/reference BLAS -- because a
#      fast BLAS is the whole reason to consider dense/GEMM operator forms, a
#      weak local BLAS means the dense numbers below UNDER-state the GEMM path.
#
#   2. Baselines the CURRENT PRODUCTION recurrence-wrapped operator stages
#      (the rotation-trick pieces in src/translate.jl + src/rotate.jl):
#        - z-rotation              rotate_z!
#        - axis-swap (Wigner y)    rotate_multipole_y!
#        - M2M  z-translation      translate_multipole_z!
#        - M2L  z-translation      translate_multipole_to_local_z!
#        - L2L  z-translation      translate_local_z!
#        - Lamb-Helmholtz xform    transform_lamb_helmholtz_{multipole,local}!
#
#   3. Builds THROWAWAY DENSE-MATRIX prototypes of the two GEMM-relevant stages
#      (block-diagonal-over-m z-translation, block-diagonal-over-n axis-swap)
#      and measures them head-to-head against the recurrences, sweeping the
#      batch count (1 .. many expansions) and toggling single- vs multi-thread
#      BLAS, so the GEMM-vs-loop crossover is visible per regime.
#
# This is a TEMPORARY benchmark harness. It does NOT touch src/. The dense
# prototypes are representative cost models (correct block dimensions, random
# data); they are not the production operators.
#
# -----------------------------------------------------------------------------
# HOW TO RUN  (from the repository root)
# -----------------------------------------------------------------------------
# BLAS threading MUST be set at process start via the env var -- runtime
# BLAS.set_num_threads() does NOT reliably change OpenBLAS execution (verified:
# get_num_threads() updates but GEMM time does not). To get the single- vs
# multi-thread-CPU regimes, run the script TWICE:
#
#   Single-thread BLAS (apples-to-apples vs the scalar recurrence):
#     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
#       julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_cpu.jl
#
#   Multi-thread BLAS (all cores):
#     OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu) OMP_NUM_THREADS=$(sysctl -n hw.ncpu) \
#       julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_cpu.jl
#   (on Linux use: OPENBLAS_NUM_THREADS=$(nproc) ...; for MKL use MKL_NUM_THREADS)
#
# Override the sweeps from the shell if desired:
#     P_LIST="4,8,12" BATCH_LIST="1,16,256,4096" SAMPLES=200 \
#       julia --project=. MATRIX_OPERATOR_REFACTOR/scripts/impl_baseline_cpu.jl
#
# Output is written to a MACHINE-TAGGED directory so a later run on a stronger
# box does not clobber these numbers:
#     MATRIX_OPERATOR_REFACTOR/data/impl_performance_baseline/<hostname>/
#       env.md                      (environment + BLAS metadata)
#       stage_recurrence.csv        (production stage timings)
#       dense_vs_loop_blas<N>.csv   (dense-prototype head-to-head, batch sweep,
#                                    N = actual BLAS thread count for that run)
#
# The script is self-contained: drop the repo on a new machine and the two run
# commands above just work (no plotting / external deps).
# =============================================================================

using FastMultipole
using FastMultipole.LinearAlgebra
using FastMultipole.LinearAlgebra: BLAS
using Statistics
using Printf
using Dates
using Random

const FM = FastMultipole

# -----------------------------------------------------------------------------
# Self-contained timing helper (no BenchmarkTools dependency -> portable).
# Calibrates an inner eval count so even sub-microsecond ops are measurable,
# then reports the minimum per-call time over `samples` repeats (min is robust
# to OS noise). `setup` runs (untimed) before each sample for in-place ops.
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
# Expansion orders for the dense head-to-head (sweeps so the operator-form
# decision is validated across low and high order).
const P_DENSE_LIST = haskey(ENV, "P_DENSE_LIST") ? _parse_int_list(ENV["P_DENSE_LIST"]) : [2, 3, 4, 5, 6, 7, 10, 14, 20]
# Dense head-to-head sweeps both precisions (recurrence stays Float64 = production).
const PREC_LIST  = haskey(ENV, "PREC_LIST")  ? _parse_prec_list(ENV["PREC_LIST"]) : [Float64, Float32]

# ---- output location (machine-tagged) ---------------------------------------
const HOST = gethostname()
const OUTDIR = normpath(joinpath(@__DIR__, "..", "data", "impl_performance_baseline", HOST))
mkpath(OUTDIR)

# -----------------------------------------------------------------------------
# Environment metadata + weak-BLAS detection
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

function git_head()
    try
        return strip(read(`git rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

function git_dirty()
    try
        return !isempty(strip(read(`git status --porcelain`, String)))
    catch
        return false
    end
end

"""
Heuristic: an optimized BLAS exposes a known tuned kernel name (openblas, mkl,
blis, apple accelerate). If none is detected we flag the numbers as suspect for
the dense/GEMM comparison.
"""
function blas_is_optimized(libs)
    known = ("openblas", "mkl", "blis", "accelerate", "veclib", "armpl")
    return any(l -> any(k -> occursin(k, lowercase(l)), known), libs)
end

function write_env(io)
    libs = blas_description()
    optimized = blas_is_optimized(libs)
    println(io, "# 008c CPU baseline -- environment")
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
        println(io, "> WARNING: no tuned BLAS detected. Dense/GEMM timings below are a LOWER")
        println(io, "> bound on achievable GEMM performance and must NOT be used as the sole")
        println(io, "> basis for the dense-vs-recurrence decision. Re-run on a box with a")
        println(io, "> tuned BLAS (and run impl_baseline_gpu.jl on a GPU) before deciding.")
    end
    println(io, "- git HEAD: ", git_head())
    println(io, "- git dirty: ", git_dirty())
    println(io, "- P_LIST: ", P_LIST)
    println(io, "- BATCH_LIST: ", BATCH_LIST)
    println(io, "- P_DENSE_LIST: ", P_DENSE_LIST)
    println(io, "- PREC_LIST: ", PREC_LIST)
    println(io, "- SAMPLES: ", SAMPLES == 0 ? "default" : SAMPLES)
    return optimized
end

# -----------------------------------------------------------------------------
# Make sure the production global precomputed tables are sized for max P
# -----------------------------------------------------------------------------
function ensure_globals!(Pmax)
    FM.update_Hs_π2!(FM.Hs_π2, Pmax)
    FM.update_ζs_mag!(FM.ζs_mag, Pmax)
    FM.update_ηs_mag!(FM.ηs_mag, Pmax)
    FM.update_M̃!(FM.M̃, Pmax)
    FM.update_L̃!(FM.L̃, Pmax)
    return nothing
end

# -----------------------------------------------------------------------------
# (2) Production recurrence-wrapped operator-STAGE baselines
# -----------------------------------------------------------------------------
function bench_stages(io)
    println(io, "stage,P,lamb_helmholtz,seconds")
    for P in P_LIST
        for LHbool in (false, true)
            LH = Val(LHbool)
            TF = Float64
            src   = FM.initialize_expansion(P, TF)
            dst   = FM.initialize_expansion(P, TF)
            tmp   = FM.initialize_expansion(P, TF)
            Ts    = zeros(TF, FM.length_Ts(P))
            eimϕs = zeros(TF, 2, P + 1)
            rand!(src)
            r, θ, ϕ = 2.3, 0.7, 0.9

            # z-rotation
            t = timeit(() -> FM.rotate_z!(dst, src, eimϕs, ϕ, P, LH))
            @printf(io, "rotate_z,%d,%s,%.6e\n", P, LHbool, t)

            # axis-swap (Wigner y-rotation; builds Ts internally)
            t = timeit(() -> FM.rotate_multipole_y!(dst, src, Ts, FM.Hs_π2, FM.ζs_mag, θ, P, LH))
            @printf(io, "rotate_multipole_y,%d,%s,%.6e\n", P, LHbool, t)

            # M2M z-translation
            t = timeit(() -> FM.translate_multipole_z!(dst, src, r, P, LH))
            @printf(io, "translate_multipole_z,%d,%s,%.6e\n", P, LHbool, t)

            # M2L z-translation
            t = timeit(() -> FM.translate_multipole_to_local_z!(dst, src, r, P, LH))
            @printf(io, "translate_multipole_to_local_z,%d,%s,%.6e\n", P, LHbool, t)

            # L2L z-translation
            t = timeit(() -> FM.translate_local_z!(dst, src, r, P, LH))
            @printf(io, "translate_local_z,%d,%s,%.6e\n", P, LHbool, t)

            if LHbool
                # Lamb-Helmholtz transforms (in place; reset before each sample)
                t = timeit(() -> FM.transform_lamb_helmholtz_multipole!(tmp, r, P); setup=() -> (tmp .= src))
                @printf(io, "transform_lamb_helmholtz_multipole,%d,%s,%.6e\n", P, LHbool, t)
                t = timeit(() -> FM.transform_lamb_helmholtz_local!(tmp, r, P); setup=() -> (tmp .= src))
                @printf(io, "transform_lamb_helmholtz_local,%d,%s,%.6e\n", P, LHbool, t)
            end
            flush(io)
        end
    end
end

# -----------------------------------------------------------------------------
# (3) Dense-prototype head-to-head vs recurrence  (GEMM-relevant stages)
# -----------------------------------------------------------------------------
#
# The two stages with a real dense/GEMM analog are:
#   * fixed-m z-translation : block-diagonal over m, block size (P+1-m)
#   * axis-swap (y rotation): block-diagonal over n, block size (2n+1)
#
# We build random dense blocks of the correct dimensions and apply them to a
# batch of `B` expansions. The "recurrence" comparison applies the production
# recurrence `B` times. Re/im lanes are modeled as 2 columns per expansion
# (the compressed-complex interleaving from theory 007), i.e. 2*B GEMM columns.
# -----------------------------------------------------------------------------

# dense block sizes for the two stages at order P
mblocks(P) = [P + 1 - m for m in 0:P]        # z-translation, one block per m
nblocks(P) = [2n + 1 for n in 0:P]           # y-rotation, one block per degree n

"""
Apply a list of dense blocks (element type `T`) to per-block matrices of width
`cols` via mul!. Returns the closure; total work ~ sum(block^2) * cols.
"""
function make_dense_apply(::Type{T}, blocks, cols) where {T}
    As = [randn(T, b, b) for b in blocks]
    Xs = [randn(T, b, cols) for b in blocks]
    Ys = [zeros(T, b, cols) for b in blocks]
    f = function ()
        @inbounds for k in eachindex(As)
            mul!(Ys[k], As[k], Xs[k])
        end
        return nothing
    end
    return f
end

# NOTE on thread control: runtime BLAS.set_num_threads() was found to be
# UNRELIABLE for OpenBLAS on this platform -- get_num_threads() changes but the
# actual GEMM execution does not (verified with a large reference GEMM). The
# only reliable lever is the process-start env var (OPENBLAS_NUM_THREADS /
# OMP_NUM_THREADS). Therefore this script does NOT call set_num_threads; it runs
# at whatever threading the process was launched with, records the actual
# BLAS.get_num_threads(), and tags the output filename by it. Run the script
# twice (env var = 1, then = ncores) to get the single- vs multi-thread regimes.
function bench_dense_vs_loop(io, blas_threads)
    TF = Float64
    LH = Val(false)
    r = 2.3

    println(io, "stage,form,precision,blas_threads,P,batch,seconds,seconds_per_expansion")

    for P in P_DENSE_LIST
        src = FM.initialize_expansion(P, TF); rand!(src)
        dst = FM.initialize_expansion(P, TF)
        Ts = zeros(TF, FM.length_Ts(P)); θ = 0.7

        for B in BATCH_LIST
            cols = 2 * B   # re + im lanes

            # recurrence side is Float64 only (production precision); time once per B.
            # pure Julia scalar code -> always single-threaded.
            trz = timeit(() -> (for _ in 1:B; FM.translate_multipole_to_local_z!(dst, src, r, P, LH); end))
            @printf(io, "m2l_z_translation,recurrence,Float64,%d,%d,%d,%.6e,%.6e\n",
                    blas_threads, P, B, trz, trz / B)
            try_ = timeit(() -> (for _ in 1:B; FM.rotate_multipole_y!(dst, src, Ts, FM.Hs_π2, FM.ζs_mag, θ, P, LH); end))
            @printf(io, "axis_swap,recurrence,Float64,%d,%d,%d,%.6e,%.6e\n",
                    blas_threads, P, B, try_, try_ / B)

            # dense side: sweep precisions (Float64 matches production; Float32
            # shows the single-precision speedup).
            for T in PREC_LIST
                apply_dense_z = make_dense_apply(T, mblocks(P), cols)
                td = timeit(apply_dense_z)
                @printf(io, "m2l_z_translation,dense,%s,%d,%d,%d,%.6e,%.6e\n",
                        T, blas_threads, P, B, td, td / B)

                apply_dense_y = make_dense_apply(T, nblocks(P), cols)
                td = timeit(apply_dense_y)
                @printf(io, "axis_swap,dense,%s,%d,%d,%d,%.6e,%.6e\n",
                        T, blas_threads, P, B, td, td / B)
            end

            flush(io)
        end
    end
end

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
function main()
    Random.seed!(SEED)
    ensure_globals!(max(maximum(P_LIST), maximum(P_DENSE_LIST)))

    println("Writing baseline to: ", OUTDIR)

    optimized = open(joinpath(OUTDIR, "env.md"), "w") do io
        write_env(io)
    end
    if !optimized
        @warn "No tuned BLAS detected -- dense/GEMM numbers are a lower bound (see env.md)."
    end

    println("[1/2] production recurrence stage baselines ...")
    open(joinpath(OUTDIR, "stage_recurrence.csv"), "w") do io
        bench_stages(io)
    end

    # Threading is controlled by the launch env var, NOT runtime calls (see note
    # above bench_dense_vs_loop). Tag the output by the actual BLAS thread count
    # so the env-var=1 and env-var=ncores runs land in separate files.
    blas_threads = BLAS.get_num_threads()
    fname = "dense_vs_loop_blas$(blas_threads).csv"
    println("[2/2] dense-prototype vs recurrence head-to-head (BLAS threads = $blas_threads) ...")
    open(joinpath(OUTDIR, fname), "w") do io
        bench_dense_vs_loop(io, blas_threads)
    end

    println("Done. Results in: ", OUTDIR, " (dense file: ", fname, ")")
end

main()
